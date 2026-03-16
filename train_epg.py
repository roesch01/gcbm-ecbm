from pathlib import Path
from typing import Any, Literal

import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from tqdm import tqdm

import wandb
from architecture.classic_cbm import ClassicCBM, ResNet50_CBM
from cbm_datasets import Batch, get_dataloader, get_datasets
from utils.argparser import cbm_epg_argument_parser
from utils.attribution_methods import (
    AttributorBase,
    get_attribution_maps,
    get_attributor_by_name,
)
from utils.loss import get_criterion
from utils.others import get_blob_dir, seed_everything
from utils.visualization import save_train_figures

seed_everything(seed=42)


def train_one_epoch(
    model,
    optimizer: torch.optim.Optimizer,
    classification_criterion: nn.Module | None,
    concept_criterion: nn.Module | None,
    epg_criterion: nn.Module | None,
    run: wandb.Run,
    epoch: int,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    device,
    attributor: AttributorBase | None,
    lambda_class: float,
    lambda_concept: float,
    lambda_epg: float,
    epg_lvl: Literal["image", "concept"] | None,
    blob_dir: Path,
    accumulation_steps: int,
    list_batches_gifs: list[Batch],
    concept_names: list,
):
    """
    Trains the model for one epoch using gradient accumulation.

    Iterates through the training loader, computes the weighted sum of classification,
    concept, and EPG losses, and updates model parameters based on accumulation steps.
    Logs metrics and saves training visualizations to WandB.

    Args:
        model: The Concept Bottleneck Model to train.
        optimizer: The PyTorch optimizer.
        classification_criterion: Loss function for the final class prediction.
        concept_criterion: Loss function for concept predictions.
        epg_criterion: Loss function for Energy-Guided Pointing (EPG).
        run: The active WandB run object for logging.
        epoch: Current epoch index (0-based).
        train_loader: DataLoader containing training data.
        val_loader: DataLoader containing validation data (used for visualization).
        device: The device (CPU/GPU) to run computations on.
        attributor: method to compute attribution maps (e.g., GradCAM).
        lambda_class: Weighting factor for classification loss.
        lambda_concept: Weighting factor for concept loss.
        lambda_epg: Weighting factor for EPG loss.
        epg_lvl: Level of EPG supervision ('image' or 'concept').
        blob_dir: Directory path for saving temporary artifacts.
        accumulation_steps: Number of steps to accumulate gradients before optimizer step.
        list_batches_gifs: Pre-fetched list of batches used to generate consistent GIFs across epochs.
    """

    image_every_n_steps = len(train_loader) // accumulation_steps - 1
    model.train()

    # Initialize accumulations for logging (Gradient accumulation)
    running_class_loss = 0.0
    running_concept_loss = 0.0
    running_epg_loss = 0.0
    running_samples = 0

    optimizer.zero_grad()  # set gradients to zero

    # Total number of batches per epoch
    num_batches = len(train_loader)

    pbar = tqdm(train_loader)
    for step, batch in enumerate(pbar):
        batch: Batch
        batch = batch.to(device)

        
        B = batch.images.size(0)

        # Input x Gradients (attribution methods) requires gradients on images
        batch.images.requires_grad_(True)

        # Forard Pass
        class_logits, concept_logits, features_last_cnn_layer = model(batch.images)

        if attributor:
            attribution_maps = get_attribution_maps(
                concepts=batch.concepts,
                features_cnn_layer=features_last_cnn_layer,
                concept_logits=concept_logits,
                attributor=attributor,
                images=batch.images,
                device=device,
            )
        else:
            attribution_maps = None

        # --- 2. Loss Calculation ---

        classification_loss = torch.tensor(0.0, device=device)
        concept_loss = torch.tensor(0.0, device=device)
        epg_loss = torch.tensor(0.0, device=device)

        if classification_criterion:
            classification_loss = classification_criterion(class_logits, batch.labels)

        if concept_criterion:
            concept_loss = concept_criterion(concept_logits, batch.concepts)

        if epg_criterion and attributor:
            if epg_lvl == "image":
                epg_loss = epg_criterion(
                    attribution_maps, batch.mask_foregrounds, batch.concepts
                )
            elif epg_lvl == "concept":
                epg_loss = epg_criterion(
                    attribution_maps, batch.mask_concepts, batch.concepts
                )
            else:
                raise ValueError(f"epg_lvl has to be 'image' or 'concept', {epg_lvl} given")

        total_loss = (
            lambda_class * classification_loss
            + lambda_concept * concept_loss
            + lambda_epg * epg_loss
        )

        # --- 3. Gradient Accumulation Logic ---

        # Scale Loss for Backprop -> divide by accumulation_steps
        scaled_loss = total_loss / accumulation_steps
        scaled_loss.backward()

        # Update running stats (use .item() * B for correct averaging later)
        running_class_loss += classification_loss.item() * B
        running_concept_loss += concept_loss.item() * B
        running_epg_loss += epg_loss.item() * B
        running_samples += B

        # Updating Pbar
        pbar.set_postfix({"total_loss": total_loss.item()})

        # If we reached accumulation_steps OR if it's the very last batch of the epoch
        is_update_step = ((step + 1) % accumulation_steps == 0) or ((step + 1) == num_batches)

        if is_update_step:
            optimizer.step()
            optimizer.zero_grad()

            # Logging Logic
            global_actual_step = epoch * len(train_loader) + step
            virtual_step = global_actual_step // accumulation_steps

            # Calculate average losses
            avg_class_loss = running_class_loss / running_samples
            avg_concept_loss = running_concept_loss / running_samples
            avg_epg_loss = running_epg_loss / running_samples

            run.log(
                {
                    "train/classification_loss": avg_class_loss,
                    "train/concept_loss": avg_concept_loss,
                    "train/epg_loss": avg_epg_loss,
                    "train/lambda_class": lambda_class,
                    "train/lambda_concept": lambda_concept,
                    "train/lambda_epg": lambda_epg,
                },
                step=virtual_step,  # Use virtual step
            )

            # save images based on virtual steps
            if (virtual_step + 1) % image_every_n_steps == 0:
                save_train_figures(
                    list_batches_gifs=list_batches_gifs,
                    batch=batch,
                    step=virtual_step,
                    epoch=epoch,
                    attribution_maps=attribution_maps,
                    epg_lvl=epg_lvl,
                    concept_logits=concept_logits,
                    run=run,
                    model=model,
                    device=device,
                    attributor=attributor,
                    train_loader=train_loader,
                    val_loader=val_loader,
                    attributor_name=attributor.__class__.__name__,
                    blob_dir=blob_dir,
                    concept_names=concept_names,
                )
                model.train()

            # Reset accumulators for the next virtual step
            running_class_loss = 0.0
            running_concept_loss = 0.0
            running_epg_loss = 0.0
            running_samples = 0

    optimizer.zero_grad()  # Ensure that gradients are reset for next epoch


def evaluate(
    model: ClassicCBM,
    val_loader: torch.utils.data.DataLoader,
    device,
    run: wandb.Run,
    concept_criterion: nn.Module | None,
    classification_criterion: nn.Module | None,
    epg_criterion: nn.Module | None,
    attributor: AttributorBase | None,
):
    """
    Evaluates the model on the validation dataset.

    Computes loss, accuracy, and F1-scores (micro/macro) for both
    classification labels and concepts. Logs results to WandB.

    Args:
        model: The trained CBM model.
        val_loader: DataLoader for validation data.
        device: The device to run evaluation on.
        run: The active WandB run object.
        concept_criterion: Loss function for concepts.
        classification_criterion: Loss function for labels.
        epg_criterion: Loss function for EPG.
        attributor: Attribution method instance.

    Returns:
        tuple: A tuple containing:
            - label_accuracy (float)
            - concept_accuracy (float)
            - label_f1 (float)
            - concept_f1_micro (float)
            - concept_f1_macro (float)
    """

    model.eval()

    total_samples = 0
    correctly_classified = 0

    all_label_preds = []
    all_label_targets = []

    # --- Concept tracking ---
    tp_concepts = 0
    fp_concepts = 0
    fn_concepts = 0
    tn_concepts = 0

    # Macro-F1
    all_concept_preds = []
    all_concept_targets = []

    total_concept_loss = 0.0
    total_class_loss = 0.0
    total_epg_loss = 0.0

    pbar = tqdm(val_loader, desc="[Eval]")

    for batch in pbar:
        batch: Batch
        batch = batch.to(device)
        B = batch.images.size(0)
        total_samples += B

        batch.images.requires_grad_(True)

        # 1. Forward
        class_logits, concept_logits, features_last_cnn_layer = model(batch.images)

        # 2. Losses
        if concept_criterion:
            total_concept_loss += concept_criterion(concept_logits, batch.concepts).item() * B

        if classification_criterion:
            total_class_loss += classification_criterion(class_logits, batch.labels).item() * B

        if epg_criterion and attributor:
            attribution_maps = get_attribution_maps(
                concepts=batch.concepts,
                features_cnn_layer=features_last_cnn_layer,
                concept_logits=concept_logits,
                attributor=attributor,
                images=batch.images,
                device=device,
            )
            total_epg_loss += (
                epg_criterion(
                    attribution_maps,
                    batch.mask_foregrounds,
                    batch.concepts,
                ).item()
                * B
            )

        # 3. Concept predictions
        preds_concepts = concept_logits.sigmoid() > 0.5
        targets_concepts = batch.concepts.bool()

        tp_concepts += (preds_concepts & targets_concepts).sum().item()
        fp_concepts += (preds_concepts & ~targets_concepts).sum().item()
        fn_concepts += (~preds_concepts & targets_concepts).sum().item()
        tn_concepts += (~preds_concepts & ~targets_concepts).sum().item()

        # Gather for Macro-F1
        all_concept_preds.append(preds_concepts.cpu())
        all_concept_targets.append(targets_concepts.cpu())

        # 4. Class predictions
        preds_class = class_logits.argmax(dim=1)
        targets_class = batch.labels.argmax(dim=1)

        correctly_classified += (preds_class == targets_class).sum().item()

        all_label_preds.extend(preds_class.cpu().numpy())
        all_label_targets.extend(targets_class.cpu().numpy())

    # Calculate Metrics
    avg_concept_loss = total_concept_loss / total_samples
    avg_class_loss = total_class_loss / total_samples

    label_accuracy = correctly_classified / total_samples
    label_f1 = f1_score(all_label_targets, all_label_preds, average="weighted")

    precision_micro = tp_concepts / (tp_concepts + fp_concepts + 1e-8)
    recall_micro = tp_concepts / (tp_concepts + fn_concepts + 1e-8)
    concept_f1_micro = 2 * precision_micro * recall_micro / (precision_micro + recall_micro + 1e-8)

    concept_accuracy = (tp_concepts + tn_concepts) / (
        tp_concepts + fp_concepts + fn_concepts + tn_concepts + 1e-8
    )

    all_concept_preds = torch.cat(all_concept_preds, dim=0).numpy()
    all_concept_targets = torch.cat(all_concept_targets, dim=0).numpy()

    concept_f1_macro = f1_score(
        all_concept_targets,
        all_concept_preds,
        average="macro",
        zero_division=0,
    )

    print(
        f"[Eval] "
        f"Label-Acc: {label_accuracy:.4f} | "
        f"Label-F1: {label_f1:.4f} | "
        f"Concept-F1 (micro): {concept_f1_micro:.4f} | "
        f"Concept-F1 (macro): {concept_f1_macro:.4f}"
    )

    # --- Logging ---
    run.log(
        {
            "eval/accuracy_label": label_accuracy,
            "eval/f1_label": label_f1,
            "eval/accuracy_concept": concept_accuracy,
            "eval/f1_concept_micro": concept_f1_micro,
            "eval/f1_concept_macro": concept_f1_macro,
            "eval/loss_concept": avg_concept_loss / total_samples,
            "eval/loss_class": avg_class_loss / total_samples,
            "eval/loss_epg": total_epg_loss / total_samples,
        },
        step=run.step,
    )

    return (
        label_accuracy,
        concept_accuracy,
        label_f1,
        concept_f1_micro,
        concept_f1_macro,
    )


def save_checkpoint(
    model: ClassicCBM,
    optimizer: torch.optim.Optimizer,
    dataset: str,
    epoch: int,
    config: dict[str, Any],
    run: wandb.Run,
    checkpoint_dir: Path,
    label_accuracy: float,
    concept_accuracy: float,
    label_f1: float,
    concept_f1_micro: float,
    concept_f1_macro: float,
):
    """
    Saves the model state, optimizer state, and current metrics to a checkpoint file.

    Args:
        model: The trained model.
        optimizer: The optimizer with its current state.
        dataset: Name of the dataset (used for naming the file).
        epoch: Current epoch number.
        config: Dictionary containing run configuration/hyperparameters.
        run: WandB run object (used for Run ID).
        checkpoint_dir: Directory to save the checkpoint file.
        label_accuracy: Current label accuracy metric.
        concept_accuracy: Current concept accuracy metric.
        label_f1: Current label F1 score.
        concept_f1_micro: Current micro-averaged concept F1 score.
        concept_f1_macro: Current macro-averaged concept F1 score.
    """
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    filename = f"cbm_epg_{dataset.lower()}_run_{run.id}_epoch{epoch}.pth"
    checkpoint_path = checkpoint_dir / filename

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "config": config,
            "label_accuracy": label_accuracy,
            "concept_accuracy": concept_accuracy,
            "label_f1": label_f1,
            "concept_f1_micro": concept_f1_micro,
            "concept_f1_macro": concept_f1_macro,
        },
        checkpoint_path,
    )
    print(f"Model checkpoint saved locally at {checkpoint_path}")


def main(
    lr: float,
    weight_decay: float,
    lambda_class: float,
    lambda_concept: float,
    lambda_epg: float,
    epochs: int,
    dataset: Literal["FunnyBirds", "CUB_312", "CUB_112"],
    batch_size: int,
    target_batch_size: int,
    num_workers: int,
    attributor_name: str | None,
    epg_lvl: Literal["image", "concept"] | None,
    wandb_project: str,
    img_size: int,
    test_id: int | None,
    classification_criterion_name: str | None,
    concept_criterion_name: str | None,
    blob_dir: Path,
    concept_masks_scale: Literal["small", "medium", "large"] | None,
    root_dir_dataset: str,
    use_soft_labels: bool,
    attr_level: Literal["image", "class"]
):
    """
    Main entry point for training the CBM with EPG guidance.

    Sets up the dataset, model, optimizer, and loss functions.
    Initializes WandB and runs the training and evaluation loop.

    Args:
        lr: Learning rate.
        weight_decay: Weight decay for the optimizer.
        lambda_class: Weight for classification loss.
        lambda_concept: Weight for concept loss.
        lambda_epg: Weight for EPG loss.
        epochs: Total number of training epochs.
        dataset: Name of the dataset to use.
        batch_size: Physical batch size per GPU/step.
        num_workers: Number of DataLoader workers.
        attributor_name: Name of the attribution method (e.g., "GradCAM").
        epg_lvl: "image" or "concept" level supervision.
        wandb_project: Name of the WandB project.
        img_size: Input image resolution.
        test_id: Optional ID to tag the run.
        classification_criterion_name: Name of the loss function for class labels.
        concept_criterion_name: Name of the loss function for concepts.
        blob_dir: Base directory for storage.
        concept_masks_scale: Scale of concept masks if applicable.
        root_dir_dataset: Root directory where datasets are stored.
        use_soft_labels: Whether to use soft labels for training.
        target_batch_size: The effective batch size to achieve via gradient accumulation.
    """

    # Logic: If batch_size is 16 and target is 64, we accumulate 4 steps.
    accumulation_steps = max(1, target_batch_size // batch_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dataset, val_dataset, test_dataset, n_concepts, n_classes, concept_names = get_datasets(
        dataset_name=dataset,
        root_dir=root_dir_dataset,
        img_size=img_size,
        concept_masks_scale=concept_masks_scale,
        use_soft_labels=use_soft_labels,
        attr_level=attr_level
    )

    train_loader, val_loader, _ = get_dataloader(
        batch_size=batch_size,
        num_workers=num_workers,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=test_dataset,
    )

    # Create model
    model = ResNet50_CBM(
        n_concepts=n_concepts,
        n_classes=n_classes,
        weights="ResNet50_Weights.DEFAULT",
    )
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    config = {
        "learning_rate": lr,
        "batch_size": batch_size,
        "weight_decay": weight_decay,
        "dataset": dataset,
        "epochs": epochs,
        "hyperparameters": {
            "epg_lvl": epg_lvl,
            "lambda_epg": lambda_epg,
            "lambda_class": lambda_class,
            "lambda_concept": lambda_concept,
        },
        "test_id": test_id,
        "img_size": img_size,
        "classification_criterion_name": classification_criterion_name,
        "concept_criterion_name": concept_criterion_name,
        "attributor_name": attributor_name,
        "target_batch_size": target_batch_size,
        "accumulation_steps": accumulation_steps,
        "concept_masks_scale": concept_masks_scale,
        "use_soft_labels": use_soft_labels,
        "seed": 42,
        "optimizer": "AdamW",
    }

    classification_criterion = get_criterion(classification_criterion_name, device=device)
    concept_criterion = get_criterion(concept_criterion_name, device=device)
    epg_criterion = get_criterion("EPGLoss", device=device)

    attributor = get_attributor_by_name(
        attributor_name=attributor_name, interpolate_dims=(img_size, img_size)
    )

    # Pre-fetch validation batches for GIFs once here, instead of inside the epoch loop
    val_loader_iter = iter(val_loader)
    list_batches_gifs = []
    for _ in range(3):  # n_gif_imgs
        try:
            batch = next(val_loader_iter)
            list_batches_gifs.append(batch.to(device))
        except StopIteration:
            break

    wandb.login()

    # Start a new wandb run to track this script.
    run = wandb.init(
        entity="roesch01-university-of-mannheim",
        project=wandb_project,
        name=f"epg-{dataset}-lambdaepg={lambda_epg}-{attributor_name}-{epg_lvl}",
        config=config,
        dir=blob_dir / "wandb",
    )

    for epoch in range(epochs):
        train_one_epoch(
            model=model,
            optimizer=optimizer,
            classification_criterion=classification_criterion,
            concept_criterion=concept_criterion,
            epg_criterion=epg_criterion,
            run=run,
            epoch=epoch,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            attributor=attributor,
            lambda_class=lambda_class,
            lambda_concept=lambda_concept,
            lambda_epg=lambda_epg,
            epg_lvl=epg_lvl,
            blob_dir=blob_dir,
            accumulation_steps=accumulation_steps,
            list_batches_gifs=list_batches_gifs,
            concept_names=concept_names,
        )

        label_accuracy, concept_accuracy, label_f1, concept_f1_micro, concept_f1_macro = evaluate(
            model=model,
            val_loader=val_loader,
            device=device,
            run=run,
            concept_criterion=concept_criterion,
            classification_criterion=classification_criterion,
            epg_criterion=epg_criterion,
            attributor=attributor,
        )

        save_checkpoint(
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            label_accuracy=label_accuracy,
            concept_accuracy=concept_accuracy,
            concept_f1_micro=concept_f1_micro,
            concept_f1_macro=float(concept_f1_macro),
            label_f1=float(label_f1),
            run=run,
            config=config,
            dataset=dataset,
            checkpoint_dir=blob_dir / "checkpoints",
        )


if __name__ == "__main__":
    parser = cbm_epg_argument_parser()
    args = parser.parse_args()

    main(
        test_id=args.test_id,
        lr=args.lr,
        weight_decay=args.weight_decay,
        lambda_class=args.lambda_class,
        lambda_concept=args.lambda_concept,
        lambda_epg=args.lambda_epg,
        epochs=args.epochs,
        dataset=args.dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        attributor_name=args.attributor,
        epg_lvl=args.epg_lvl,
        wandb_project=args.wandb_project,
        img_size=args.img_size,
        classification_criterion_name=args.classification_criterion,
        concept_criterion_name=args.concept_criterion,
        blob_dir=get_blob_dir(),
        concept_masks_scale=args.concept_masks_scale,
        root_dir_dataset=args.root_dir_dataset,
        use_soft_labels=args.use_soft_labels,
        target_batch_size=args.target_batch_size,
        attr_level=args.attr_level,
    )
