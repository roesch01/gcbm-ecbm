from typing import Literal

import torch
from segmentation_models_pytorch.metrics import get_stats, iou_score


def iou(
    pred_logits: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
    reduction: Literal["full", "batch", "channel", "none"] | None = None,
) -> torch.Tensor:
    """Compute IoU using segmentation_models_pytorch functions.
    Args:
        pred_logits (torch.Tensor): Predicted logits of shape (B, C, H, W).
        targets (torch.Tensor): Ground truth masks of shape (B, C, H, W).
        threshold (float, optional): Threshold to binarize predictions. Defaults to 0.5.
        reduction (str, optional): Reduction method ('full', 'batch', 'channel', 'none', None). Defaults to None.
    Returns:
        torch.Tensor: IoU score. If reduction is 'none', 'channel' or None shape is (B, C), if 'batch' shape is (B,), if 'full' shape is (1,).
    """

    assert reduction in ["full", "batch", "channel", "none", None], (
        f"Invalid reduction: {reduction}. Must be 'full', 'batch', 'channel', 'none', or None."
    )

    targets_bin = (targets > 0.5).long()

    probs = torch.sigmoid(pred_logits)

    tp, fp, fn, tn = get_stats(probs, targets_bin, mode="multilabel", threshold=threshold)  # type: ignore
    iou_scores = iou_score(tp, fp, fn, tn, reduction="none")

    if reduction == "full":
        return iou_scores.mean()
    if reduction == "batch":
        return iou_scores.mean(dim=1)
    
    if reduction == "channel":
        raise NotImplementedError("Channel-wise IoU reduction is not implemented yet.")

    return iou_scores


def dice(
    pred_logits: torch.Tensor,
    targets: torch.Tensor,
    eps: float = 1e-6,
    threshold: float = 0.5,
    reduction: Literal["full", "batch", "channel", "none"] | None = None,
) -> torch.Tensor:
    """Compute Dice coefficient for multi-label segmentation.
    Args:
        pred_logits (torch.Tensor): Predicted logits of shape (B, C, H, W).
        targets (torch.Tensor): Ground truth masks of shape (B, C, H, W).
        eps (float, optional): Small value to avoid division by zero. Defaults to 1e-6.
        threshold (float, optional): Threshold to binarize predictions. Defaults to 0.5.
        reduction (str, optional): Reduction method ('full', 'batch', 'channel', 'none', None). Defaults to None.
    Returns:
        torch.Tensor: Dice score. If reduction is 'none', None or 'channel' shape is (B, C), if 'batch' shape is (B,), if 'full' shape is (1,).
    """

    assert reduction in ["full", "batch", "channel", "none", None], (
        f"Invalid reduction: {reduction}. Must be 'mean', 'none', or None."
    )

    targets_bin = (targets > 0.5).long()

    probs = torch.sigmoid(pred_logits)  # binarized predictions

    tp, fp, fn, tn = get_stats(probs, targets_bin, mode="multilabel", threshold=threshold)  # type: ignore
    dice_scores = (2 * tp + eps) / (2 * tp + fp + fn + eps)  # shape: (B, C)

    if reduction == "full":
        return dice_scores.mean()

    if reduction == "batch":
        return dice_scores.mean(dim=1)

    return dice_scores


def foreground_dice(
    pred_logits: torch.Tensor,
    target: torch.Tensor,
    eps: float = 1e-6,
    threshold: float = 0.5,
    reduction: Literal["mean", "none"] | None = "mean",
) -> torch.Tensor:
    """
    Compute per-sample mean Dice only over foreground channels (channels that have any target pixels).
    - pred_logits: Tensor (B, C, H, W)
    - target:      Tensor (B, C, H, W)
    Returns:
    - per_sample_fg_dice: Tensor of shape (B,) with mean foreground dice for each sample (0.0 if no foreground).
    """

    assert reduction in ["mean", "none", None], (
        f"Invalid reduction: {reduction}. Must be 'mean', 'none', or None."
    )

    if reduction is None:
        reduction = "none"

    # per-channel dice (B, C)
    dice_pc = dice(pred_logits, target, eps=eps, threshold=threshold)

    # foreground mask: channel is foreground if target has any positive pixel
    fg_mask_positive = (target.float().sum(dim=(2, 3)) > 0).to(dice_pc.dtype)  # (B, C)

    # Count only channels>0
    numerator = (dice_pc * fg_mask_positive).sum(dim=1)  # Sum over Channels
    denominator = fg_mask_positive.sum(dim=1).clamp_min(1)  # Number of valid channels

    dice_per_image = numerator / denominator  # (B,)

    if reduction == "mean":
        return dice_per_image.mean()

    return dice_per_image


def part_level_dice(
    dices: torch.Tensor,
    mode: Literal["train", "val", "test"],
    concept_names: list[str],
    part_names: list[str],
):
    """
    Computes the average Dice coefficient per part (beak, eye, foot, tail, wing)
    from the Dice values per concept.

    Args:
        dices (torch.Tensor): Tensor of shape (Batches x Concepts) with Dice values per concept.
        mode (str): Mode, e.g., "train", "val", "test" for naming the keys in the output.

    Returns:
        dict: Dictionary with average Dice values per part, e.g., {"beak-dice": 0.85, ...}
    """

    assert isinstance(dices, torch.Tensor), f"dices must be a torch.Tensor, got {type(dices)}"
    assert dices.ndim == 2, f"dices must be 2D (Batches x Concepts), got {dices.ndim}D"

    n_concepts = dices.shape[1]

    assert n_concepts == len(concept_names), (
        f"Expected dices for {len(concept_names)} concepts, got {n_concepts}"
    )

    parts_dice: dict[str, float] = {}

    for part in part_names:
        part_indices = [j for j, concept in enumerate(concept_names) if concept.startswith(part)]
        parts_dice[f"{mode}/dice/{part}"] = dices[:, part_indices].mean().item()

    return parts_dice