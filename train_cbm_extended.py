import os
from pathlib import Path
from typing import Any, Literal

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import f1_score, precision_score, recall_score
from tqdm import tqdm

import wandb
from architecture.extended_cbm import CBMWrapper, ExtendedCBMOutput, get_cbm_wrapper
from cbm_datasets import Batch, get_dataloader, get_datasets
from utils.argparser import cbm_extended_argument_parser
from utils.loss import FromOutput, FromTarget, MultiTaskLoss, Task, get_criterion
from utils.others import calculate_metrics, get_blob_dir, seed_everything
from utils.visualization import show_segmentation_sample

seed_everything(seed=42)

matplotlib.use("Agg")


def train_one_epoch(
    model: CBMWrapper,
    train_loader: torch.utils.data.DataLoader[Batch],
    val_loader: torch.utils.data.DataLoader[Batch],
    optimizer: torch.optim.Optimizer,
    criterion: MultiTaskLoss,
    device: str,
    run: wandb.Run,
    calculate_dice: bool,
    calculate_fg_dice: bool,
    calculate_iou: bool,
    actual_batch_size: int,
    concept_names: list,
    part_names: list[str],
    dice_thr: float = 0.5,
    epoch: int = 0,
    image_every_n_steps: int = 100,
    n_gifs: int = 3,
    target_batch_size: int = 64,
    
) -> tuple[float, float]:
    """
    Trains the model for one epoch using gradient accumulation and logs metrics to WandB.

    Args:
        model: The CBMWrapper or neural network to train.
        train_loader: DataLoader for the training set.
        val_loader: DataLoader for the validation set (used for visualization).
        optimizer: PyTorch optimizer.
        criterion: Multi-task loss function.
        device: Device to run the training on.
        run: Active Weights & Biases run.
        actual_batch_size: The physical batch size processed in one forward pass.
        target_batch_size: The effective batch size simulated via gradient accumulation.
        image_every_n_steps: Frequency (in virtual steps) to log visualizations.

    Returns:
        tuple: (average_epoch_loss, average_dice_score)
    """
    model.train()

    # --- 1. Calculate Accumulation Steps ---
    accumulation_steps = max(1, target_batch_size // actual_batch_size)

    if target_batch_size % actual_batch_size != 0:
        print(
            f"Warning: Target batch size ({target_batch_size}) is not "
            f"evenly divisible by actual batch size ({actual_batch_size})."
        )

    # Initialize accumulators for metrics and losses
    accum_metrics = {}
    accum_loss_details = {}
    total_loss_epoch = 0.0

    optimizer.zero_grad()

    pbar = tqdm(train_loader, desc=f"[Train e{epoch}]")
    num_batches = len(train_loader)

    for step, batch in enumerate(pbar):
        
        batch:Batch

        batch = batch.to(device)

        # --- 2. Forward Pass ---
        predictions: ExtendedCBMOutput = model(batch.images)
        loss_total, loss_details = criterion(predictions, batch)

        # --- 3. Backward Pass (Scaled) ---
        # Scale loss by accumulation steps to ensure consistent gradient magnitude
        (loss_total / accumulation_steps).backward()

        # Update epoch loss tracking
        total_loss_epoch += loss_total.item()

        # Accumulate loss details for logging
        for k, v in loss_details.items():
            val = v.item() if isinstance(v, torch.Tensor) else v
            accum_loss_details[k] = accum_loss_details.get(k, 0.0) + val

        # Calculate and accumulate metrics for the current micro-batch
        metrics = calculate_metrics(
            pred=predictions,
            target=batch,
            concept_names=concept_names,
            part_names=part_names,
            mode="train",
            calculate_dice=calculate_dice,
            calculate_fg_dice=calculate_fg_dice,
            calculate_iou=calculate_iou,
            calculate_f1_concepts=calculate_dice
        )

        for k, v in metrics.items():
            val = v.item() if isinstance(v, torch.Tensor) else v
            accum_metrics[k] = accum_metrics.get(k, 0.0) + val

        # --- 4. Optimizer Step (Every accumulation_steps) ---
        is_update_step = ((step + 1) % accumulation_steps == 0) or ((step + 1) == num_batches)

        if is_update_step:
            optimizer.step()
            optimizer.zero_grad()

            # Calculate global and virtual steps for consistent WandB X-axis
            global_actual_step = epoch * num_batches + step
            virtual_step = global_actual_step // accumulation_steps

            # Compute averages over the accumulated steps
            # Note: The last batch might have fewer steps, we normalize by actual steps taken
            steps_taken = accumulation_steps if (step + 1) % accumulation_steps == 0 else (step + 1) % accumulation_steps
            
            avg_metrics = {k: v / steps_taken for k, v in accum_metrics.items()}
            avg_loss_details = {k: v / steps_taken for k, v in accum_loss_details.items()}

            # --- 5. Visualization ---
            if virtual_step % image_every_n_steps == 0:
                model.eval()
                # Generate and log GIFs
                create_test_imgs_for_gifs(
                    model=model,
                    n_gifs=n_gifs,
                    run=run,
                    val_loader=val_loader,
                    global_step=virtual_step,
                    device=device,
                    concept_names=concept_names
                )

                # Generate segmentation sample figure
                # fig_orig, fig_masks = show_segmentation_sample(
                #     gt_image_normalized=batch["image"][0].cpu(),
                #     gt_masks=batch["mask_concepts"][0].cpu() if "mask_concepts" in batch else None,
                #     gt_concepts_activated=batch["concepts"][0].cpu(),
                #     predicted_masks=predictions["segmentation.mask_logits"][0].detach().sigmoid().cpu(),
                #     predicted_concepts_activated=torch.sigmoid(predictions["concepts.concept_logits"][0]).detach().cpu(),
                #     overlay=True,
                #     title_texts={
                #         "Max Pixel": predictions["segmentation.mask_logits"][0].sigmoid().amax(dim=(1, 2)).cpu(),
                #         "Pix>0": (predictions["segmentation.mask_logits"][0].sigmoid() > 1 / 255).sum(dim=(1, 2)).cpu(),
                #     },
                #     show_masks=False,
                #     concept_names=train_loader.dataset.concepts,
                #     only_active_concepts=False,
                #     separator=train_loader.dataset.parts_separator,
                # )
                
                # Log visualization to WandB
                # run.log({"train/segmentation": wandb.Image(fig_masks)}, step=virtual_step)

                # # Clean up matplotlib resources to prevent memory leaks
                # plt.close(fig_orig)
                # plt.close(fig_masks)
                # plt.close("all")
                model.train()

            # --- 6. Logging ---
            run.log(avg_metrics, step=virtual_step)
            run.log({f"train/{key}": value for key, value in avg_loss_details.items()}, step=virtual_step)

            # Log model-specific parameters (e.g., Learnable Scalers/Shifts)
            log_params = {}
            if predictions.concept_module.shift is not None:
                shifts = predictions.concept_module.shift.detach().cpu().squeeze()
                scales = predictions.concept_module.shift.detach().cpu().squeeze()
                
                for i, concept_name in enumerate(concept_names):
                    log_params[f"train/concept_shift/{concept_name}"] = shifts[i].item()
                    log_params[f"train/concept_scale/{concept_name}"] = scales[i].item()
                log_params["train/concept_shift/mean"] = shifts.mean().item()

            if log_params:
                run.log(log_params, step=virtual_step)

            # Reset accumulators for the next virtual batch
            accum_metrics = {}
            accum_loss_details = {}

    avg_loss = total_loss_epoch / max(1, num_batches)
    return avg_loss, 0.0


def create_test_imgs_for_gifs(
    model: CBMWrapper,
    n_gifs: int,
    run: wandb.Run,
    val_loader: torch.utils.data.DataLoader,
    global_step: int,
    device: str,
    concept_names: list[str]
):
    for i, batch in enumerate(val_loader):
        if i >= n_gifs:
            break

        batch:Batch
        # batch = batch.to(device=device)

        # predictions: ExtendedCBMOutput = model(batch.images)

        # imagenet_mean = torch.tensor([0.485, 0.456, 0.406]).to(device).view(-1, 1, 1)
        # imagenet_std = torch.tensor([0.229, 0.224, 0.225]).to(device).view(-1, 1, 1)

        # gt_image_normalized = batch["image"]
        # gt_image = gt_image_normalized * imagenet_std + imagenet_mean

        # fig_orig, fig_masks = show_segmentation_sample(
        #     gt_image_normalized=batch.images[0].cpu(),
        #     gt_masks=batch.mask_concepts[0].cpu() if batch.mask_concepts is not None else None,
        #     gt_concepts_activated=batch.concepts[0].cpu(),
        #     predicted_masks=predictions.segmentation_module.mask_logits[0].detach().sigmoid().cpu(),
        #     predicted_concepts_activated=predictions.concept_module.concept_logits[0]
        #     .detach()
        #     .sigmoid()
        #     .cpu(),
        #     # image_features=predictions["upsampler.image_features"][0].detach().cpu(),
        #     overlay=True,
        #     title_texts={
        #         "Max Pixel": predictions.segmentation_module.mask_logits[0]
        #         .sigmoid()
        #         .amax(dim=(1, 2))
        #         .cpu(),
        #         "Pix>0": (predictions.segmentation_module.mask_logits[0].sigmoid() > 1 / 255)
        #         .sum(dim=(1, 2))
        #         .cpu(),
        #     },  # type: ignore
        #     show_masks=False,
        #     concept_names=concept_names,
        #     only_active_concepts=len(batch.concepts[0]) > 30,
        #     separator="-"#val_loader.dataset.parts_separator,
        # )

        # run.log(
        #     {f"eval/img-all/img-{i}": wandb.Image(fig_masks)},
        #     step=global_step,
        # )

        # plt.close(fig_orig)
        # plt.close(fig_masks)
        # plt.close("all")

        # create_vis_active_concepts(predictions, model)


@torch.no_grad()
def evaluate(
    model: CBMWrapper,
    val_loader: torch.utils.data.DataLoader,
    device: str,
    run: wandb.Run,
    criterion: MultiTaskLoss,
    calculate_dice: bool,
    calculate_fg_dice: bool,
    calculate_iou: bool,
    concept_names: list[str],
    part_names: list[str],
    dice_thr: float = 0.5,
):
    model.eval()
    total_loss = 0.0
    total_samples = 0  # Trackt die tatsächliche Anzahl der Bilder

    dice_scores: list[float] = []
    iou_scores: list[float] = []
    foreground_dice_scores: list[float] = []
    f1_labels: list[float] = []

    all_concept_logits = []
    all_concept_targets = []

    all_label_preds = []
    all_label_targets = []


    total_loss_details = {}
    correct = 0

    pbar = tqdm(val_loader, desc="[Eval]")
    for step, batch in enumerate(pbar):
        
        batch: Batch
        batch = batch.to(device)
        
        # Batch Größe ermitteln (wichtig für batch_size > 1)
        current_batch_size = batch.images.size(0)
        total_samples += current_batch_size

        predictions: ExtendedCBMOutput = model(batch.images)
        loss_total, loss_details = criterion(predictions, batch)

        # Loss wird meist als Mean über den Batch berechnet,
        # daher mit Batch-Größe gewichten für exakten Gesamtdurchschnitt
        total_loss += loss_total.item() * current_batch_size

        for key, value in loss_details.items():
            if key not in total_loss_details:
                total_loss_details[key] = 0.0
            total_loss_details[key] += value * current_batch_size

        metrics = calculate_metrics(
            predictions,
            batch,
            mode="val",
            dice_thr=dice_thr,
            concept_names=concept_names,
            part_names=part_names,
            calculate_dice=calculate_dice,
            calculate_fg_dice=calculate_fg_dice,
            calculate_iou=calculate_iou,
            calculate_f1_concepts=False
        )

        # Accuracy Berechnung korrigiert für Batches
        pred_labels = predictions.classification_module.labels_logits.argmax(dim=1)
        gt_labels = batch.labels.argmax(dim=1)
        correct += (pred_labels == gt_labels).sum().item()

        # Metriken (calculate_metrics gibt meist schon Durchschnitte pro Batch zurück)
        if calculate_dice:
            dice_scores.append(metrics["val/dice_mean"])
        if calculate_fg_dice:
            foreground_dice_scores.append(metrics["val/foreground_dice_mean"])

        if calculate_iou:
            iou_scores.append(metrics["val/iou_mean"])
        
        all_concept_logits.append(
            predictions.concept_module.concept_logits.detach().cpu()
        )
        if batch.concepts is not None:
            all_concept_targets.append(
                batch.concepts.detach().cpu()
            )

        all_label_preds.append(pred_labels.cpu())
        all_label_targets.append(gt_labels.cpu())


    # Finale Berechnungen basierend auf total_samples statt len(val_loader)
    avg_loss = total_loss / total_samples
    avg_loss_details = {
        f"eval/{key}": val / total_samples for key, val in total_loss_details.items()
    }

    avg_accuracy_labels = correct / total_samples

    # Da metrics meist Batch-Averages sind, ist np.mean über die Liste okay,
    # solange die Batches etwa gleich groß sind.
    avg_dice = float(np.mean(dice_scores)) if dice_scores else 0.0
    avg_iou = float(np.mean(iou_scores)) if iou_scores else 0.0
    avg_foreground_dice = float(np.mean(foreground_dice_scores)) if foreground_dice_scores else 0.0

    if all_concept_logits and False:
        concept_logits = torch.cat(all_concept_logits, dim=0)
        concept_targets = torch.cat(all_concept_targets, dim=0)

        # Wahrscheinlichkeiten und Vorhersagen (Threshold 0.5)
        probs = torch.sigmoid(concept_logits)
        preds = (probs >= 0.5).int().cpu().numpy()
        targets = (concept_targets >= 0.5).int().cpu().numpy()

        # Wir berechnen 'macro', um jedes Konzept gleich zu gewichten
        prec_concepts = float(precision_score(targets, preds, average="macro", zero_division=0))
        rec_concepts = float(recall_score(targets, preds, average="macro", zero_division=0))
        f1_concepts = float(f1_score(targets, preds, average="macro", zero_division=0))

        acc_per_concept = (preds == targets).mean(axis=0)
        acc_concepts = float(acc_per_concept.mean())
    else:
        prec_concepts = rec_concepts = f1_concepts = acc_concepts = 0.0

    if all_label_preds:

        label_preds = torch.cat(all_label_preds, dim=0)  # [N,]
        label_targets = torch.cat(all_label_targets, dim=0)  # [N,]
        f1_labels = f1_score(label_targets, label_preds, average="macro") # type: ignore

    print(f"Correct: {correct} of {total_samples} (Acc: {avg_accuracy_labels:.4f})")

    run.log(
        {
            "eval/loss": avg_loss,
            "eval/dice_mean": avg_dice,
            "eval/iou_mean": avg_iou,
            "eval/f1_concept_activations": f1_concepts,
            "eval/precision_concepts": prec_concepts,
            "eval/recall_concepts": rec_concepts,
            "eval/accuracy_concepts": acc_concepts,
            "eval/f1_labels": f1_labels,
            "eval/accuracy_labels": avg_accuracy_labels,
            "eval/foreground_dice_scores": avg_foreground_dice,
            **avg_loss_details,
        },
        step=run.step,  # Optional: Epoch als Step mitgeben
    )

    print(
        f"[Eval] loss={avg_loss:.4f}  dice={avg_dice:.4f}  iou={avg_iou:.4f}  f1_concepts={f1_concepts:.4f}  f1_labels={f1_labels:.4f}  accuracy_labels={avg_accuracy_labels:.4f}"
    )
    return (
        avg_loss,
        avg_dice,
        avg_iou,
        f1_concepts,
        f1_labels,
        avg_accuracy_labels,
    )


def save_checkpoint(
    model: CBMWrapper,
    optimizer: torch.optim.Optimizer,
    dataset: str,
    epoch: int,
    config: dict[str, Any],
    run: wandb.Run,
    checkpoint_dir: Path,
):
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    filename = f"cbm_anyup_dinov3_{dataset.lower()}_run_{run.id}_epoch{epoch}.pth"
    checkpoint_path = checkpoint_dir / filename

    # 3. Lokal auf den Shared Storage speichern
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "config": config,
        },
        checkpoint_path,
    )
    print(f"Model checkpoint saved locally at {checkpoint_path}")

    # 4. WandB Reference Artifact erstellen (Lösung 3)
    # Erstellt einen Link im Dashboard, ohne die Datei hochzuladen
    # artifact = wandb.Artifact(
    #     name=f"checkpoint-{run.id}", type="model", metadata={"epoch": epoch, "dataset": dataset}
    # )

    # # WICHTIG: file:// Präfix für lokale Referenzen
    # artifact.add_reference(f"file://{checkpoint_path}")

    # # Artifact dem Run hinzufügen
    # run.log_artifact(artifact)
    # print(f"WandB reference for checkpoint {filename} created.")


def load_checkpoint(
    model: CBMWrapper, optimizer: torch.optim.Optimizer, checkpoint_path: Path, device
) -> int:
    # load checkpoint of the model
    # checkpoint_dir = "/pfs/work8/workspace/ffhk/scratch/ma_faroesch-master-thesis-shared/playground-uc3/master-thesis/checkpoints"
    # checkpoint_path = f"{checkpoint_dir}/cbm_anyup_dinov3_funnybirds_epoch0.pth"
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    start_epoch = checkpoint["epoch"] + 1
    print(f"Model checkpoint loaded from {checkpoint_path}, starting from epoch {start_epoch}")

    return start_epoch





def main(
    encoder_name: str | None,
    upsampler_name: str | None,
    segmentation_module_name: str | None,
    concept_module_name: str,
    classification_module_name: str,
    unified_name: str | None,
    freeze_encoder: bool,
    freeze_upsampler: bool,
    checkpoint_path: Path | None,
    dataset: Literal["FunnyBirds", "CUB_312", "CUB_112"],
    weight_decay: float,
    batch_size: int,
    num_workers: int,
    img_size: int,
    lr: float,
    affinity_num_samples: int,
    affinity_sim_threshold: float,
    top_k_percent: float,
    epochs: int,
    lambda_concept_loss: float,
    lambda_concept_reg_loss: float,
    lambda_affinity_loss: float,
    lambda_mask_reg_loss: float,
    lambda_mask_loss: float,
    lambda_tv_loss: float,
    lambda_classification_loss: float,
    concept_criterion_name: str,
    concept_reg_criterion_name: str,
    mask_criterion_name: str,
    mask_reg_criterion_name: str,
    classification_criterion_name: str,
    affinity_criterion_name: str,
    tv_criterion_name: str,
    calculate_dice: bool,
    calculate_iou: bool,
    calculate_fg_dice: bool,
    image_every_n_steps: int,
    dino_ckpt_segdino: str | None,
    blob_dir: Path,
    concept_masks_scale: Literal["small", "medium", "large"] | None,
    test_id: int,
    use_soft_labels: bool,
    n_concepts_implicitly_learned: int | None,
    attr_level: Literal["image", "class"]
):
    root_dir_dataset = os.path.join(os.getenv('ROOT_DIR_WORKSPACE', '.'), 'datasets')
    print(f"Using dataset root dir: {root_dir_dataset}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")


    train_dataset, val_dataset, test_dataset, n_concepts, n_classes, concept_names = get_datasets(
        dataset_name=dataset,
        root_dir=root_dir_dataset,
        img_size=img_size,
        use_soft_labels=use_soft_labels,
        concept_masks_scale=concept_masks_scale,
        attr_level=attr_level
    )

    train_loader, val_loader, test_loader = get_dataloader(
        batch_size=batch_size,
        num_workers=num_workers,
        train_dataset=train_dataset, val_dataset=val_dataset, test_dataset=test_dataset
    )

    part_names = train_dataset.dataset.parts if isinstance(train_dataset, torch.utils.data.Subset) else train_dataset.parts

    n_concepts = n_concepts_implicitly_learned if n_concepts_implicitly_learned else n_concepts
    
    model = get_cbm_wrapper(
        encoder_name=encoder_name,
        upsampler_name=upsampler_name,
        freeze_encoder=freeze_encoder,
        freeze_upsampler=freeze_upsampler,
        segmentation_module_name=segmentation_module_name,
        concept_module_name=concept_module_name,
        classification_module_name=classification_module_name,
        unified_name=unified_name,
        n_concepts=n_concepts,
        n_classes=n_classes,
        top_k_percent=top_k_percent,
        dino_ckpt_segdino=dino_ckpt_segdino,
        img_size=img_size,
    )

    model.to(device).train()

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr,
        weight_decay=weight_decay,
    )

    if checkpoint_path is not None:
        load_checkpoint(model, optimizer, checkpoint_path, device)

    config = {
        "blob_dir": blob_dir,
        "learning_rate": lr,
        "batch_size": batch_size,
        "weight_decay": weight_decay,
        "dataset": dataset,
        "epochs": epochs,
        "img_size": img_size,
        "test_id": test_id,
        "seed": 42,
        "use_soft_labels": use_soft_labels,
        "device": device,
        "architecture": {
            "encoder": encoder_name,
            "upsampler": upsampler_name,
            "segmentation_module": segmentation_module_name,
            "concept_module": concept_module_name,
            "classification_module": classification_module_name,
            "unified_model": unified_name,
            "freeze_encoder": freeze_encoder,
            "freeze_upsampler": freeze_upsampler,
        },
        "hyperparameters": {
            "affinity_num_samples": affinity_num_samples,
            "affinity_sim_threshold": affinity_sim_threshold,
            "top_k_percent": top_k_percent,
        },
        "loss_weights": {
            "lambda_concept": lambda_concept_loss,
            "lambda_concept_reg": lambda_concept_reg_loss,
            "lambda_affinity": lambda_affinity_loss,
            "lambda_mask_reg": lambda_mask_reg_loss,
            "lambda_mask": lambda_mask_loss,
            "lambda_tv": lambda_tv_loss,
            "lambda_cls": lambda_classification_loss,
        },
        "criteria": {
            "concept": concept_criterion_name,
            "concept_reg": concept_reg_criterion_name,
            "mask": mask_criterion_name,
            "mask_reg": mask_reg_criterion_name,
            "affinity": affinity_criterion_name,
            "tv": tv_criterion_name,
            "classification": classification_criterion_name,
        },
    }

    # Start a new wandb run to track this script.
    wandb.login()

    run = wandb.init(
        entity="roesch01-university-of-mannheim",
        project="master-thesis-extended",
        name=f"extended-{dataset}-{segmentation_module_name}-{concept_module_name}-{classification_module_name}",
        config=config,
        reinit="finish_previous",
        dir=blob_dir / "wandb",
    )

    concepts_pos_weights = torch.Tensor(train_loader.dataset.get_pos_weight_vector()).to(device)

    concept_criterion = get_criterion(concept_criterion_name, device=device, pos_weights=concepts_pos_weights)
    tv_criterion = get_criterion(tv_criterion_name, device=device)
    affinity_criterion = get_criterion(
        affinity_criterion_name,
        affinity_num_samples=affinity_num_samples,
        affinity_sim_threshold=affinity_sim_threshold,
        device=device,
    )
    concept_reg_criterion = get_criterion(concept_reg_criterion_name, device=device)
    classification_criterion = get_criterion(classification_criterion_name, device=device)
    mask_criterion = get_criterion(mask_criterion_name, device=device)
    mask_reg_criterion = get_criterion(mask_reg_criterion_name, device=device)

    criterion = MultiTaskLoss(
        lr=lr,
        device=device,
        tasks = [
            # --- Concept Loss ---
            Task(
                name="concepts",
                loss_fn=concept_criterion,
                weight=lambda_concept_loss,
                parameters=[
                    # Dynamische Auswahl zwischen Probs und Logits im Lambda
                    FromOutput(
                        lambda o: o.concept_module.concept_probs if concept_criterion_name == "BCELoss" else o.concept_module.concept_logits,
                        name="concept_pred"
                    ),
                    FromTarget(lambda t: t.concepts, name="concept_gt"),
                    
                    # Bedingtes Hinzufügen von Parametern funktioniert weiterhin mit Python-Logik
                    *(
                        [FromTarget(lambda t: t.concept_weights, name="weights")] 
                        if concept_criterion_name == "BCEWithLogitsCertaintiesLoss" else []
                    )
                ],
            ),

            # --- Affinity Loss ---
            Task(
                name="affinity_loss",
                loss_fn=affinity_criterion,
                weight=lambda_affinity_loss,
                parameters=[
                    FromOutput(lambda o: o.upsampler_module.features, name="img_features"), 
                    FromOutput(lambda o: o.segmentation_module.mask_logits, name="mask_logits"),
                ],
            ),

            # --- Mask Regularization (L1) ---
            Task(
                name="mask_regularization_l1",
                loss_fn=mask_reg_criterion,
                weight=lambda_mask_reg_loss,
                parameters=[
                    FromOutput(lambda o: o.segmentation_module.mask_logits, name="mask_logits"),
                ],
            ),

            # --- Concept Regularization (L1) ---
            Task(
                name="concept_regularization_l1",
                loss_fn=concept_reg_criterion,
                weight=lambda_concept_reg_loss,
                parameters=[
                    FromOutput(
                        lambda o: o.concept_module.concept_probs if concept_criterion_name == "BCELoss" else o.concept_module.concept_logits,
                        name="concept_reg_input"
                    ),
                ],
            ),

            # --- TV Loss ---
            Task(
                name="tvloss",
                loss_fn=tv_criterion,
                weight=lambda_tv_loss,
                parameters=[
                    FromOutput(lambda o: o.segmentation_module.mask_logits, name="mask_logits"),
                ],
            ),

            # --- Classification ---
            Task(
                name="classification",
                loss_fn=classification_criterion,
                weight=lambda_classification_loss,
                parameters=[
                    FromOutput(lambda o: o.classification_module.labels_logits, name="cls_logits"),
                    FromTarget(lambda t: t.labels, name="cls_labels"),
                ],
            ),

            # --- Mask Loss (Supervised) ---
            Task(
                name="mask",
                loss_fn=mask_criterion,
                weight=lambda_mask_loss,
                parameters=[
                    FromOutput(lambda o: o.segmentation_module.mask_logits, name="mask_logits"),
                    FromTarget(lambda t: t.mask_concepts, name="mask_gt"),
                ],
            ),
        ]
    )


    for epoch in range(epochs):
        train_loss, train_dice = train_one_epoch(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            device=device,
            criterion=criterion,
            dice_thr=0.5,
            epoch=epoch,
            run=run,
            image_every_n_steps=image_every_n_steps,
            calculate_dice=calculate_dice,
            calculate_fg_dice=calculate_fg_dice,
            calculate_iou=calculate_iou,
            actual_batch_size=batch_size,
            concept_names=concept_names,
            part_names=part_names
        )

        evaluate(
            model=model,
            val_loader=val_loader,
            device=device,
            dice_thr=0.5,
            run=run,
            criterion=criterion,
            calculate_dice=calculate_dice,
            calculate_fg_dice=calculate_fg_dice,
            calculate_iou=calculate_iou,
            concept_names=concept_names,
            part_names=part_names
        )

        save_checkpoint(
            model=model,
            optimizer=optimizer,
            dataset=dataset,
            epoch=epoch,
            config=config,
            run=run,
            checkpoint_dir=blob_dir / "checkpoints",
        )


if "__main__" == __name__:
    parser = cbm_extended_argument_parser()
    args = parser.parse_args()

    import os

    from huggingface_hub import login

    # Sucht nach dem Token in den Umgebungsvariablen
    token = os.getenv("HF_TOKEN")
    if token:
        login(token=token)

    main(
        affinity_num_samples=args.affinity_num_samples,
        batch_size=args.batch_size,
        checkpoint_path=args.checkpoint_path,
        upsampler_name=args.upsampler_name,
        encoder_name=args.encoder_name,
        freeze_encoder=args.freeze_encoder,
        freeze_upsampler=args.freeze_upsampler,
        dataset=args.dataset,
        weight_decay=args.weight_decay,
        num_workers=args.num_workers,
        lr=args.lr,
        affinity_sim_threshold=args.affinity_sim_threshold,
        top_k_percent=args.top_k_percent,
        epochs=args.epochs,
        segmentation_module_name=args.segmentation_module_name,
        concept_module_name=args.concept_module_name,
        classification_module_name=args.classification_module_name,
        unified_name=args.unified_name,
        img_size=args.img_size,
        lambda_concept_loss=args.lambda_concept_loss,
        lambda_concept_reg_loss=args.lambda_concept_reg_loss,
        lambda_affinity_loss=args.lambda_affinity_loss,
        lambda_mask_reg_loss=args.lambda_mask_reg_loss,
        lambda_tv_loss=args.lambda_tv_loss,
        lambda_classification_loss=args.lambda_classification_loss,
        concept_criterion_name=args.concept_criterion,
        concept_reg_criterion_name=args.concept_reg_criterion,
        affinity_criterion_name=args.affinity_criterion,
        mask_reg_criterion_name=args.mask_reg_criterion,
        tv_criterion_name=args.tv_criterion,
        classification_criterion_name=args.classification_criterion,
        calculate_dice=args.calculate_dice,
        calculate_iou=args.calculate_iou,
        calculate_fg_dice=args.calculate_fg_dice,
        image_every_n_steps=args.image_every_n_steps,
        lambda_mask_loss=args.lambda_mask_loss,
        mask_criterion_name=args.mask_criterion,
        dino_ckpt_segdino=args.dino_ckpt_segdino,
        blob_dir=get_blob_dir(),
        concept_masks_scale=args.concept_masks_scale,
        test_id=args.test_id,
        use_soft_labels=args.use_soft_labels,
        n_concepts_implicitly_learned=args.n_concepts_implicitly_learned,
        attr_level=args.attr_level
    )
