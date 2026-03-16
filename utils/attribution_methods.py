from abc import ABC, abstractmethod

import torch
import torch.nn.functional as F

__all__ = ["GradCamAttributor", "GradCamPlusPlusAttributor", "InputTimesGradientAttributor"]


class AttributorBase(ABC):
    """Base class for XAI attribution methods."""
    def __init__(
        self,
        interpolate_dims: tuple[int, int],
        only_positive: bool,
    ):
        super().__init__()
        self.only_positive = only_positive
        self.interpolate_dims = interpolate_dims

    @abstractmethod
    def __call__(
        self,
        features_cnn_layer: torch.Tensor,  # [B, D, h, w]
        pred: torch.Tensor,  # [B, C]
        classes: torch.Tensor,  # [B]
        images: torch.Tensor | None = None,  # [B, 3, H, W]
        **kwargs,
    ) -> torch.Tensor:  # [B, 1, H, W]
        ...

    def interpolate(
        self,
        attributions: torch.Tensor,  # [B, 1, h, w]
    ) -> torch.Tensor:  # [B, 1, H, W]
        """Upsamples attributions to the original image dimensions."""
        return F.interpolate(
            attributions, size=self.interpolate_dims, mode="bilinear", align_corners=False
        )

    def check_only_positive(
        self,
        attributions: torch.Tensor,  # [B, h, w]
    ):
        """Applies ReLU to keep only positive contributions (standard for Saliency/Grad-CAM)."""
        if self.only_positive:
            return F.relu(attributions)
        return attributions

    def apply_post_processing(
        self,
        attributions: torch.Tensor,  # # [B, 1, h, w]
    ):
        """Sequentially applies ReLU (if enabled) and interpolation."""
        attributions = self.check_only_positive(attributions)
        attributions = self.interpolate(attributions)

        return attributions


class GradCamAttributor(AttributorBase):
    def __init__(
        self,
        interpolate_dims: tuple[int, int],
        only_positive=True,
    ):
        """Standard Grad-CAM implementation."""
        super().__init__(
            only_positive=only_positive,
            interpolate_dims=interpolate_dims,
        )

    def __call__(
        self,
        features_cnn_layer: torch.Tensor,  # [B, D, h, w]
        pred: torch.Tensor,  # [B, C]
        classes: torch.Tensor,  # [B]
        images: torch.Tensor | None = None,  # [B, 3, H, W]
        **kwargs,
    ) -> torch.Tensor:  # [B, 1, H, W]
        target_outputs = torch.gather(pred, 1, classes.unsqueeze(-1))
        
        # Compute gradients w.r.t. the feature maps
        grads = torch.autograd.grad(
            torch.sum(target_outputs),
            features_cnn_layer,
            create_graph=True,
            retain_graph=True,  # instead of unbind over batch
        )[0]
        # Global Average Pooling of gradients (the 'alpha' weights)
        
        weights = grads.mean(dim=(2, 3), keepdim=True)
        prods = weights * features_cnn_layer
        attributions = F.relu(prods.sum(dim=1, keepdim=True))
        return self.apply_post_processing(attributions)


class GradCamPlusPlusAttributor(AttributorBase):
    """Grad-CAM++ implementation for improved object localization and handling multiple instances."""
    
    def __init__(self, interpolate_dims: tuple[int, int]):
        super().__init__(
            interpolate_dims=interpolate_dims,
            only_positive=True,
        )

    def __call__(
        self,
        features_cnn_layer: torch.Tensor,  # [B, D, h, w]
        pred: torch.Tensor,  # [B, C]
        classes: torch.Tensor,  # [B]
        images: torch.Tensor | None = None,  # [B, 3, H, W]
        **kwargs,
    ) -> torch.Tensor:  # [B, 1, H, W]
        
        target_outputs = torch.gather(pred, 1, classes.unsqueeze(-1))
        grads = torch.autograd.grad(
            torch.sum(target_outputs), features_cnn_layer, create_graph=True, retain_graph=True
        )[0]  # (B, C, H, W)

        
        # Grad-CAM++ requires second and third order gradients
        grad_2 = grads.pow(2)
        grad_3 = grads.pow(3)

        sum_activations = torch.sum(features_cnn_layer, dim=(2, 3), keepdim=True)

        # Calculate alpha coefficients (refer to Grad-CAM++ Eq. 19 in paper)
        eps = 1e-7
        denom = 2 * grad_2 + sum_activations * grad_3
        aij = grad_2 / (denom + eps)

        # Weights w_k^c: sum of (alphas * relu(grads))
        weights = torch.sum(aij * F.relu(grads), dim=(2, 3), keepdim=True)
        cam = torch.sum(weights * features_cnn_layer, dim=1, keepdim=True)
        attributions = F.relu(cam)

        return self.apply_post_processing(attributions)
    




class InputTimesGradientAttributor(AttributorBase):
    """Input x Gradient attribution. Computes the saliency map relative to the input pixels."""
    
    def __init__(
        self,
        interpolate_dims: tuple[int, int],
        only_positive: bool = True,
    ):
        super().__init__(
            interpolate_dims=interpolate_dims,
            only_positive=only_positive,
        )

    def __call__(
        self,
        features_cnn_layer: torch.Tensor, # not needed here but it is needed since class extends from AttributorBase
        pred: torch.Tensor,               # [B, C] Logits
        classes: torch.Tensor,            # [B] Target Classes
        images: torch.Tensor | None = None,       # [B, 3, H, W] REQUIRED
        **kwargs,
    ) -> torch.Tensor:
        if images is None:
            raise ValueError("InputTimesGradientAttributor requires the 'images' argument.")
        
        if not images.requires_grad:
             raise RuntimeError(
                 "images.requires_grad must be True BEFORE the forward pass. "
                 "Please set batch['image'].requires_grad_(True) in your training loop."
             )

        target_outputs = torch.gather(pred, 1, classes.unsqueeze(-1))
        target_score = torch.sum(target_outputs)

        # Gradient w.r.t. input image
        grads = torch.autograd.grad(
            outputs=target_score,
            inputs=images,
            create_graph=True, 
            retain_graph=True,
        )[0] # [B, 3, H, W]

        # Input * Gradient
        attribution = images * grads

        # RGB -> Single Channel (Sum across channels)
        attribution = attribution.sum(dim=1, keepdim=True)

        return self.apply_post_processing(attribution)



def get_attributor_by_name(
    attributor_name: str | None, interpolate_dims: tuple[int, int]
) -> AttributorBase | None:
    
    if attributor_name is None:
        return None
    
    if attributor_name.lower() == "GradCamAttributor".lower():
        return GradCamAttributor(
            only_positive=True,
            # binarize=False,
            interpolate_dims=interpolate_dims,
        )

    if attributor_name.lower() == "GradCamPlusPlusAttributor".lower():
        return GradCamPlusPlusAttributor(interpolate_dims=interpolate_dims)

    if attributor_name.lower() == "InputTimesGradientAttributor".lower():
        return InputTimesGradientAttributor(
            interpolate_dims=interpolate_dims,
            only_positive=True 
        )

    raise ValueError(f"Attributor {attributor_name} not found. Has to be one of {__all__}")






def get_attribution_maps(
    concepts: torch.Tensor,  # [B, C]
    features_cnn_layer: torch.Tensor,  # [B, D, h, w]
    concept_logits: torch.Tensor,  # [B, C]
    attributor: AttributorBase,
    images: torch.Tensor,  # [B, 3, H, W]
    device: str,
):
    
    """
    Computes attribution maps for all active concepts in a batch.
    
    Args:
        concepts: Binary indicators [B, C]
        features_cnn_layer: Latent features [B, D, h, w]
        concept_logits: Model output logits [B, C]
        attributor: The XAI method to use
        images: Raw input images [B, 3, H, W]
    """

    B, C = concept_logits.shape
    _, _, H, W = images.shape
    all_concept_maps: list[torch.Tensor] = []



    for c in range(C):
        
        # Optimization: Only iterate over concepts that are active at least once in the batch
        if not torch.any(concepts[:, c] > 0):
            # Kein Bild im Batch hat dieses Konzept aktiv -> Setze Map auf Null
            all_concept_maps.append(torch.zeros((B, 1, H, W), device=device))
            continue

        target_classes = torch.full((B,), c, device=device, dtype=torch.long)
        
        # Compute map for concept 'c' across the whole batch
        attr_c = attributor(
            features_cnn_layer=features_cnn_layer, 
            pred=concept_logits, 
            classes=target_classes,
            images=images
        ) # Output: [B, 1, H, W]
        
        all_concept_maps.append(attr_c)

    # Concatenate: [B, C, H, W]
    attributions = torch.cat(all_concept_maps, dim=1)
    
    # Mask out attributions for concepts that were not active for a specific sample
    concepts_active = concepts.unsqueeze(-1).unsqueeze(-1)  # [B, C, 1, 1]
    attributions = attributions * concepts_active  # [B, C, H, W]

    return attributions


def normalize_gradcam_maps(maps: torch.Tensor) -> torch.Tensor:
    """
    Normalize Grad-CAM maps to [0, 1] range for each map in the batch.
    Args:
        maps (torch.Tensor): Grad-CAM maps of shape (B, C, H, W)
    Returns:
        torch.Tensor: Normalized Grad-CAM maps of shape (B, C, H, W)
    """
    B, C, H, W = maps.shape
    maps_reshaped = maps.view(B * C, -1)
    min_vals = maps_reshaped.min(dim=1, keepdim=True)[0]
    max_vals = maps_reshaped.max(dim=1, keepdim=True)[0]
    normalized_maps = (maps_reshaped - min_vals) / (max_vals - min_vals + 1e-8)
    return normalized_maps.view(B, C, H, W)