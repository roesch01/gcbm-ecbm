import math
from collections.abc import Iterable
from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import torch
from captum.attr import visualization
from sklearn.decomposition import PCA

import wandb
from architecture.classic_cbm import ClassicCBM
from utils.attribution_methods import get_attribution_maps, normalize_gradcam_maps
from cbm_datasets import Batch


def _validate_and_prepare_inputs(
    title_texts: dict[str, torch.Tensor],
    concept_names: list[str],
    gt_image_normalized: torch.Tensor | None = None,
    gt_image: torch.Tensor | None = None,
    gt_masks: torch.Tensor | None = None,
    image_features: torch.Tensor | None = None,
    predicted_masks: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None, int | None]:
    """
    Validates inputs and brings them into consistent torch format:
      - gt_image_normalized: torch tensor CxHxW with C=1 or 3 (converts from numpy if needed)
      - gt_masks, predicted_masks: optional, shapes (N, H, W)
      - image_features: optional, torch or numpy (C,H,W) or (H*W, C)
      - concept_names: list[str] -> n_masks is validated or set
    Returns: (gt_image (3,H,W), gt_masks_tensor or None, predicted_masks_tensor or None, n_masks)
    Raises ValueError on inconsistencies.
    """
    # image
    assert isinstance(gt_image_normalized, torch.Tensor), (
        "gt_image_normalized muss ein torch.Tensor sein."
    )

    assert gt_image_normalized.ndim == 3, (
        "gt_image_normalized muss 3D (C,H,W) sein, got shape {tuple(gt_image_normalized.shape)}."
    )

    if gt_image_normalized is not None:
        imagenet_mean = torch.tensor([0.485, 0.456, 0.406]).view(-1, 1, 1)
        imagenet_std = torch.tensor([0.229, 0.224, 0.225]).view(-1, 1, 1)

        gt_image = gt_image_normalized * imagenet_std + imagenet_mean
        gt_image = gt_image.clamp(0.0, 1.0)
    else:
        assert isinstance(gt_image, torch.Tensor), (
            f"gt_image has to be an torch.Tensor, {type(gt_image)} given"
        )

    C, H, W = gt_image.shape

    if C not in (1, 3):
        raise ValueError(f"gt_image_normalized erwartet C=1 oder C=3, got C={C}")

    # if 1-channel -> duplicate it to 3 channels
    if C == 1:
        gt_image = gt_image.repeat(3, 1, 1)
        C = 3

    # masks
    if gt_masks is not None:
        assert isinstance(gt_masks, torch.Tensor), "gt_masks muss ein torch.Tensor sein."
        assert gt_masks.ndim == 3, (
            "gt_masks muss 3D (N,H,W) sein, got shape {tuple(gt_masks.shape)}."
        )
        assert gt_masks.shape[1:] == (H, W), (
            f"gt_masks muss Shape (N,{H},{W}) haben, got {tuple(gt_masks.shape)}."
        )
        gt_masks = gt_masks.clamp(0.0, 1.0)

    if predicted_masks is not None:
        assert isinstance(predicted_masks, torch.Tensor), (
            f"predicted_masks muss ein torch.Tensor sein. got {type(predicted_masks)}."
        )
        assert predicted_masks.ndim == 3, (
            f"predicted_masks muss 3D sein (N,H,W), got {tuple(predicted_masks.shape)}"
        )
        n_pred, Hp, Wp = predicted_masks.shape
        if (Hp, Wp) != (H, W):
            predicted_masks = torch.nn.functional.interpolate(
                predicted_masks.unsqueeze(1).float(),
                size=(H, W),
                mode="bilinear",
                align_corners=False,
            ).squeeze(1)
            # raise ValueError(
            #     f"predicted_masks hat Shape (N,{Hp},{Wp}) aber Bild hat (H,W)=({H},{W})"
            # )
        predicted_masks = predicted_masks.clamp(0.0, 1.0)

    if gt_masks is not None and predicted_masks is not None:
        assert gt_masks.shape == predicted_masks.shape, (
            "predicted_masks und gt_masks müssen das gleiche Shape haben."
        )

    # concepts / n_masks
    if concept_names is None or not isinstance(concept_names, Iterable):
        raise ValueError("concept_names muss eine Liste von Strings sein.")
    if any(not isinstance(c, str) for c in concept_names):
        raise ValueError("Jedes Element in concept_names muss ein String sein.")
    
    for key, value in title_texts.items():
        if value is not None:
            assert isinstance(value, torch.Tensor), (
                f"title_texts[{key}] muss ein torch.Tensor sein, got {type(value)}."
            )
            assert value.ndim == 1, (
                f"title_texts[{key}] muss 1D Tensor sein, got shape {tuple(value.shape)}."
            )
            assert value.shape[0] == len(concept_names), (
                f"title_texts[{key}] muss Länge {len(concept_names)} haben, got {value.shape[0]}."
            )

    n_masks = len(concept_names)

    return gt_image, gt_masks, predicted_masks, n_masks


def _build_concept_grid(
    concept_names: list[str],
    gt_concepts_activated: torch.Tensor,
    separator: str = "::",
    only_active_concepts: bool = False,
) -> tuple[dict, int, int]:
    """
    Creates a concept grid mapping concept -> (row, col).
    Row changes whenever prefix (part before '-') changes.
    Returns: mapping, rows, cols (cols = max cols in any row).
    """
    grid = {}

    if only_active_concepts:
        assert isinstance(gt_concepts_activated, torch.Tensor), (
            f"gt_concepts_activated muss ein torch.Tensor sein, wenn only_active_concepts=True. {type(gt_concepts_activated)} gegeben."
        )
        max_cols = 9

        idx = 0
        for concept_name, concept_active in zip(concept_names, gt_concepts_activated):
            if concept_active.item():
                row = idx // max_cols
                col = idx % max_cols
                grid[concept_name] = (row, col)
                idx += 1

        rows = math.ceil(len(grid) / max_cols)
        cols = min(max_cols, max(1, len(grid)))

    else:
        last_prefix = None
        row_idx = -1
        col_idx = 0
        max_cols = 0
        for concept in concept_names:
            prefix = concept.split(separator)[0]
            if prefix != last_prefix:
                row_idx += 1
                col_idx = 0
            grid[concept] = (row_idx, col_idx)
            col_idx += 1
            max_cols = max(max_cols, col_idx)
            last_prefix = prefix
        rows = row_idx + 1
        cols = max_cols if max_cols > 0 else 1

    return grid, rows, cols


def _plot_original_and_pca(
    gt_image: torch.Tensor,
    image_features,
    figsize_per_image: int = 4,
) -> plt.Figure:  # type: ignore
    """
    Plots the original image (left) and, if image_features exist: PCA to 3 components (right).
    Returns the Matplotlib figure.
    """
    ncols = 2 if image_features is not None else 1
    fig, axes = plt.subplots(
        1, ncols, figsize=(figsize_per_image * ncols, figsize_per_image), dpi=20
    )
    if ncols == 1:
        axes = [axes]

    try:
        axes[0].imshow(gt_image.permute(1, 2, 0))
    except Exception:
        axes[0].imshow(gt_image.permute(1, 2, 0).clip(0.0, 1.0))
    axes[0].set_title("Original Image")
    axes[0].axis("off")

    if image_features is not None:
        
        if image_features.ndim == 3:
            Cf, Hf, Wf = image_features.shape
            feats_flat = image_features.reshape(Cf, -1).T  # (H*W, C)
            h = Hf
            w = Wf
        elif image_features.ndim == 2:
            feats_flat = image_features  # assume (H*W, C)
            
            n_pixels = feats_flat.shape[0]
            h = w = int(math.sqrt(n_pixels))
            if h * w != n_pixels:
                # fallback: 1 x n_pixels (long strip) -> reshape to (1,n_pixels,3) später
                h = 1
                w = n_pixels
        else:
            raise ValueError("image_features muss 2D oder 3D sein (C,H,W) oder (H*W, C).")

        pca = PCA(n_components=3)
        reduced = pca.fit_transform(feats_flat)  # (H*W, 3)
        reduced_img = reduced.reshape(h, w, 3)
        reduced_img = (reduced_img - reduced_img.min()) / (
            reduced_img.max() - reduced_img.min() + 1e-8
        )
        axes[1].imshow(reduced_img)
        axes[1].set_title("PCA of Features")
        axes[1].axis("off")

    plt.tight_layout()
    return fig


def _render_masks_grid(
    gt_image: torch.Tensor,
    gt_masks: torch.Tensor | None,
    predicted_masks: torch.Tensor | None,
    concept_names: list[str],
    gt_concepts_activated: torch.Tensor,
    predicted_concepts_activated: torch.Tensor | None,
    concept_grid: dict,
    grid_rows: int,
    grid_cols: int,
    title_texts: dict[str, torch.Tensor],
    figsize_per_image: int = 4,
    overlay: bool = False,
    show_masks: bool = True,
    threshold: float | None = None,
) -> plt.Figure:  # type: ignore
    """
    Draws the mask grid. Returns a Matplotlib figure.
    """
    n_cells = grid_rows * grid_cols
    fig, axes = plt.subplots(
        grid_rows,
        grid_cols,
        figsize=(figsize_per_image * grid_cols, figsize_per_image * grid_rows),
    )

    if axes.ndim == 1:
        if grid_rows == 1:
            axes = axes[np.newaxis, :]
        else:
            axes = axes[:, np.newaxis]

    for channel, concept in enumerate(concept_names):
        if concept not in concept_grid:
            continue  # show only active concepts

        r, c = concept_grid[concept]
        ax = axes[r][c]

        gt_mask = None
        pred_mask = None
        if gt_masks is not None:
            gt_mask = (gt_masks[channel] > 0.5).bool()
        if predicted_masks is not None:
            if threshold is None:
                pred_mask = predicted_masks[channel]
            else:
                pred_mask = (predicted_masks[channel] > threshold).bool()

        # build comp image
        if gt_mask is None and pred_mask is None:
            ax.axis("off")
            continue

        if pred_mask is None:
            # nur GT anzeigen
            ax.imshow(gt_mask, cmap="viridis", interpolation="nearest", vmin=0, vmax=1)
        else:
            if overlay:
                ax.imshow(
                    gt_image.permute(1, 2, 0), interpolation="nearest", alpha=0.5, vmin=0, vmax=1
                )
            if gt_mask is None:
                comp = (pred_mask * 255).int()
                ax.imshow(
                    pred_mask,
                    cmap="viridis",
                    alpha=0.5 if overlay else 1.0,
                    interpolation="nearest",
                )
                fp = fn = tp = torch.Tensor(0)  # dummy
            else:
                if threshold is None:
                    ax.imshow(
                        pred_mask,
                        cmap="viridis",
                        interpolation="nearest",
                        alpha=0.7 if overlay else 1.0,
                        vmin=0,
                        vmax=1,
                    )
                else:
                    comp = torch.zeros((*gt_mask.shape, 3), dtype=torch.uint8)
                    tp = gt_mask & pred_mask
                    fp = ~gt_mask & pred_mask
                    fn = gt_mask & ~pred_mask

                    comp[tp] = [0, 255, 0]  # Grün
                    comp[fp] = [255, 0, 0]  # Rot
                    comp[fn] = [0, 0, 255]  # Blau

                    if overlay:
                        img_bg = (gt_image.permute(1, 2, 0) * 255).to(torch.uint8).cpu()
                        # comp ist bereits (H, W, C) uint8

                        #  (alpha blending)
                        comp = (0.5 * img_bg.float() + 0.5 * comp.float()).to(torch.uint8)

                    ax.imshow(comp, interpolation="nearest", vmin=0, vmax=255)

        title = f"{concept}\n"

        for key, value in title_texts.items():
            if value is not None:
                title += f", {key}: {value[channel].item():.2f}"

        if predicted_concepts_activated is not None:
            title += f"Act: {predicted_concepts_activated[channel].item():.2f}"

        concept_active = gt_concepts_activated[channel]
        ax.set_title(title, color="green" if concept_active else "red")
        ax.axis("off")

    total = len(concept_names)
    for i in range(total, grid_rows * grid_cols):
        r, c = divmod(i, grid_cols)
        axes[r][c].axis("off")

    plt.tight_layout()
    return fig


def show_segmentation_sample(
    gt_concepts_activated: torch.Tensor,
    concept_names: list[str],
    gt_image: torch.Tensor | None = None,
    gt_image_normalized: torch.Tensor | None = None,
    figsize_per_image: int = 4,
    overlay: bool = False,
    only_active_concepts: bool = False,
    gt_masks: torch.Tensor | None = None,
    predicted_concepts_activated: torch.Tensor | None = None,
    image_features: torch.Tensor | None = None,
    predicted_masks: torch.Tensor | None = None,
    title_texts: dict[str, torch.Tensor] = {},
    show_masks: bool = True,
    threshold: float | None = None,
    separator: str = "::",
):
    """
    Refactored / more robust version for visualization:
      - shows original image (+ optional PCA of features)
      - shows a grid of concept masks (GT vs Prediction) with TP/FP/FN color coding
    Validations are done at the beginning; clear errors are raised.
    Returns: Tuple[fig_original_pca, fig_masks] (fig_masks can be None if no masks are given)
    """

    # 1) Validation
    gt_image, gt_masks_t, pred_masks_t, n_masks = _validate_and_prepare_inputs(
        gt_image_normalized=gt_image_normalized,
        gt_image=gt_image,
        title_texts=title_texts,
        concept_names=concept_names,
        gt_masks=gt_masks,
        image_features=image_features,
        predicted_masks=predicted_masks,
    )

    concept_grid, rows_by_prefix, cols_by_prefix = _build_concept_grid(
        concept_names=concept_names,
        only_active_concepts=only_active_concepts,
        gt_concepts_activated=gt_concepts_activated,
        separator=separator,
    )

    fig_orig = _plot_original_and_pca(gt_image, image_features, figsize_per_image=figsize_per_image)
    plt.show()

    if gt_masks_t is None and pred_masks_t is None:
        return fig_orig, None

    max_cols = 9
    cols = min(max_cols, max(1, cols_by_prefix))
    rows = max(1, math.ceil(len(concept_names) / cols))
    rows = max(rows, rows_by_prefix)

    fig_masks = _render_masks_grid(
        gt_image=gt_image,
        gt_masks=gt_masks_t,
        predicted_masks=pred_masks_t,
        predicted_concepts_activated=predicted_concepts_activated,
        gt_concepts_activated=gt_concepts_activated,
        concept_names=concept_names,
        concept_grid=concept_grid,
        grid_rows=rows_by_prefix,
        grid_cols=cols_by_prefix,
        figsize_per_image=figsize_per_image,
        overlay=overlay,
        show_masks=show_masks,
        threshold=threshold,
        title_texts=title_texts,
    )

    return fig_orig, fig_masks


def save_train_figures(
    model: ClassicCBM,
    device,
    attributor,
    list_batches_gifs: list[Batch],
    batch: Batch,
    step: int,
    epoch: int,
    attribution_maps: torch.Tensor | None,  # [B, C, H, W]
    epg_lvl: Literal["image", "concept"] | None,
    concept_logits: torch.Tensor,  # [B, C]
    run: wandb.Run,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    attributor_name: str,
    blob_dir: Path,
    concept_names: list
):
    plt.close("all")

    fig_orig, fig_masks = show_segmentation_sample(
        gt_image_normalized=batch.images[0].clone().detach().cpu(),
        gt_masks=batch.mask_concepts[0].cpu(),
        gt_concepts_activated=batch.concepts[0].cpu(),
        predicted_masks=normalize_gradcam_maps(attribution_maps)[0].clone().detach().cpu()
        if attribution_maps is not None
        else None,
        predicted_concepts_activated=concept_logits[0].clone().detach().sigmoid().cpu(),
        overlay=True,
        show_masks=False,
        concept_names=concept_names,
        only_active_concepts=len(batch.concepts[0]) > 30,
        separator=":",
    )

    run.log(
        {"train/gradcam-img": wandb.Image(fig_masks)},
        step=run.step,
    )

    if not attributor_name or not epg_lvl:
        return

    model.eval()

    for i, val_batch in enumerate(list_batches_gifs):
        imagenet_mean = torch.tensor([0.485, 0.456, 0.406]).to(device).view(-1, 1, 1)
        imagenet_std = torch.tensor([0.229, 0.224, 0.225]).to(device).view(-1, 1, 1)

        gt_image_normalized = val_batch.images
        gt_image_normalized.requires_grad_(True)
        gt_image = gt_image_normalized * imagenet_std + imagenet_mean

        class_logits, concept_logits, features_last_cnn_layer = model(gt_image_normalized)

        attribution_maps = get_attribution_maps(
            concepts=val_batch.concepts,
            features_cnn_layer=features_last_cnn_layer,
            concept_logits=concept_logits,
            attributor=attributor,
            images=gt_image_normalized,
            device=device,
        )

        path = blob_dir / "attr_maps" / run.id / f"attribution_maps-val-img{i}-epoch-{epoch}.npy"

        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            attribution_maps.clone().cpu().detach(),
            path,
        )

        for concept_idx in range(len(concept_names)):
            plt.close("all")
            concept_name = concept_names[concept_idx]
            attr = attribution_maps[0, concept_idx].clone().detach().cpu().numpy()  # [256,256]
            attr = np.expand_dims(attr, axis=2)  # [256,256,1]
            original_img = gt_image[0].cpu().detach().numpy().transpose(1, 2, 0)
            if np.abs(attr).max() < 1e-8:
                continue
            fig, ax = visualization.visualize_image_attr(
                attr,
                original_img,
                method="blended_heat_map",
                sign="absolute_value",
                show_colorbar=False,
                title=f"Concept: {concept_name}, Epoch: {epoch}, Step: {step}",
            )

            run.log(
                {f"eval/img{i}-concept={concept_name}": wandb.Image(fig)},
                step=run.step,
            )
            plt.close("all")
