import os
import random
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from sklearn.metrics import f1_score

from architecture.extended_cbm import ExtendedCBMOutput
from cbm_datasets import Batch
from utils.segmentation import dice, foreground_dice, iou, part_level_dice


def get_blob_dir():
    """
    Retrieves and creates the base directory for storing artifacts.

    Checks for the 'BLOB_DIR' environment variable. If not set, raises an error.
    Creates the directory if it does not exist.

    Returns:
        Path: The path object to the blob directory.

    Raises:
        RuntimeError: If the 'BLOB_DIR' environment variable is not set.
    """
    blob_dir = os.getenv("BLOB_DIR")
    if blob_dir is None:
        raise RuntimeError("BLOB_DIR must be set before starting training")

    blob_dir = Path(blob_dir)
    blob_dir.mkdir(parents=True, exist_ok=True)
    return blob_dir

def seed_everything(seed: int = 42) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_preds_and_targets(
    pred_logits: torch.Tensor,
    targets: torch.Tensor,
    mode: Literal["multilabel", "multiclass"],
    threshold: float = 0.5,
):
    """Helper function to convert logits and targets to numpy arrays for sklearn"""

    # 1. Process targets
    # If targets are one-hot (B, C), convert to indices (B,) for multiclass
    if mode == "multiclass" and targets.ndim > 1 and targets.shape[1] > 1:
        targets_np = targets.argmax(dim=1).cpu().numpy()
    else:
        # For multilabel or if targets are already indices
        targets_np = targets.cpu().numpy()

    # 2. Process predictions
    if mode == "multiclass":
        # Argmax over the class dimension
        preds_np = pred_logits.argmax(dim=1).cpu().numpy()
    else:  # multilabel
        # Sigmoid + threshold
        preds_np = pred_logits.sigmoid().gt(threshold).float().cpu().numpy()

    return preds_np, targets_np


def f1_score_concepts(
    pred_logits: torch.Tensor,  # [B, C]
    targets: torch.Tensor,  # [B, C]
    threshold: float = 0.5,
) -> float:
    preds: torch.Tensor = pred_logits.detach().cpu()
    labels = targets.detach().cpu().numpy()

    probs = torch.sigmoid(preds).numpy()
    preds_binary = (probs >= threshold).astype(int)

    f1_score_concepts = f1_score(labels, preds_binary, average="macro", zero_division=0)
    return float(f1_score_concepts)


def f1_score_classes(
    pred_logits: torch.Tensor,  # [B, C]
    targets: torch.Tensor,  # [B,]
) -> float:
    preds = pred_logits.detach().cpu()
    labels = targets.detach().cpu().numpy()

    preds_binary = torch.argmax(preds, dim=1).numpy()

    f1_score_classes =  f1_score(labels, preds_binary, average="macro", zero_division=0)
    return float(f1_score_classes)


def f1_score_classes_funnybirds(
    pred_logits: torch.Tensor,  # [B, 50]
    target: torch.Tensor,  # [B, 50] (multi-hot: 1 for valid classes)
) -> float:
    """
    Computes the macro F1-score for FunnyBirds, where multiple classes
    can be valid per image.
    """
    # 1. Get the model's top-1 prediction
    preds_argmax = torch.argmax(pred_logits, dim=1)  # [B]

    # 2. Check if the prediction is in the list of valid classes
    # For each sample, extract whether the predicted index has a '1' in the target
    batch_indices = torch.arange(pred_logits.size(0))
    is_correct = target[batch_indices, preds_argmax] == 1  # [B] (bool tensor)

    # 3. Create dummy targets for sklearn
    # Since sklearn expects a comparison between y_true and y_pred,
    # we "simulate" a hit by setting pred = true if is_correct is True.

    # We simply use the predicted indices
    y_pred = preds_argmax.cpu().numpy()

    # For y_true, use the predicted index if correct,
    # otherwise pick ANY other valid index from the target
    y_true = y_pred.copy()

    # Where the prediction was wrong, we need to set a "true" label
    incorrect_indices = torch.where(~is_correct)[0]
    for idx in incorrect_indices:
        # Find all valid class indices for this sample
        valid_indices = torch.nonzero(target[idx] == 1).flatten()
        # Take the first available valid label as representative
        y_true[idx] = valid_indices[0].cpu().item()

    # 4. Compute macro F1
    return f1_score(y_true, y_pred, average="macro", zero_division=0)  # type: ignore


def accuracy_score(pred_logits: torch.Tensor, targets: torch.Tensor) -> float:
    """
    Computes accuracy for multiclass problems.
    Expects logits and targets (either indices or one-hot).
    """
    # Prediction is the index with the highest logit
    preds = pred_logits.argmax(dim=1)

    # If targets are one-hot encoded (B, Num_Classes), convert to indices (B,)
    if targets.ndim > 1 and targets.shape[1] > 1:
        truth = targets.argmax(dim=1)
    else:
        truth = targets

    correct = (preds == truth).float().sum()
    return (correct / len(truth)).item()

@torch.no_grad()
def calculate_metrics(
    pred: ExtendedCBMOutput,
    target: Batch,
    mode: Literal["train", "val", "test"],
    concept_names: list[str],
    part_names: list[str],
    calculate_dice: bool,
    calculate_iou: bool,
    calculate_fg_dice: bool,
    calculate_f1_concepts: bool = True,
    dice_thr=0.5,
):
    metrics: dict[str, float] = {}

    

        # --- Segmentation Metrics ---
    if calculate_dice:
        # Your existing implementation is fine
        dice_scores = dice(
            pred_logits=pred.segmentation_module.mask_logits,
            targets=target.mask_concepts,
            threshold=dice_thr,
            reduction="none",
        )
        # Add part-level Dice
        part_dices = part_level_dice(dice_scores, mode, concept_names, part_names)
        metrics.update(part_dices)
        metrics[f"{mode}/dice_mean"] = dice_scores.mean().item()

    if calculate_iou:
        iou_scores = iou(
            pred_logits=pred.segmentation_module.mask_logits,
            targets=target.mask_concepts,
            threshold=dice_thr,
            reduction="none",
        )
        metrics[f"{mode}/iou_mean"] = iou_scores.mean().item()

    if calculate_fg_dice:
        fg_dice = foreground_dice(
            pred.segmentation_module.mask_logits,
            target.mask_concepts,
            threshold=dice_thr,
            reduction="none",
        )
        metrics[f"{mode}/foreground_dice_mean"] = fg_dice.mean().item()

    # --- Concept Metrics (Multilabel) ---
    if target.concepts is not None and calculate_f1_concepts:
        f1_concepts = f1_score_concepts(
            pred_logits=pred.concept_module.concept_logits,
            targets=(target.concepts>0.5).int(), 
            threshold=0.5,
        )
        metrics[f"{mode}/f1_concept_activations"] = f1_concepts

    # --- Object Metrics (Multiclass) ---
    # IMPORTANT: Object prediction is multiclass (1-of-N)

    if target.labels.ndim > 1:
        # F1 score (macro average over all classes)
        f1_obj = f1_score_classes_funnybirds(
            pred.classification_module.labels_logits,
            target.labels,
        )
    else:
        # F1 score (macro average over all classes)
        f1_obj = f1_score_classes(
            pred.classification_module.labels_logits,
            target.labels,
        )

    # Accuracy
    acc_obj = accuracy_score(pred.classification_module.labels_logits, target.labels)

    metrics[f"{mode}/f1_labels"] = f1_obj
    metrics[f"{mode}/accuracy_labels"] = acc_obj

    return metrics
