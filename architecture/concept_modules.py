import math
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from architecture.segmentation_modules import SegmentationOutput

from . import optional_tensor_to_numpy, tensor_to_numpy

__all__ = [
    "SegMaskAvgPool",
    "SegMaskAvgPoolTrainChannelAffine",
    "SegMaskMaxPool",
    "LogitMeanTopK",
    "SALFConceptModule"
]

@dataclass
class ConceptOutputNumpy:
    concept_logits: np.ndarray
    concept_probs: np.ndarray
    scale: np.ndarray | None = None
    shift: np.ndarray | None = None


@dataclass
class ConceptOutput:
    concept_logits: torch.Tensor
    scale: torch.Tensor | None = None
    shift: torch.Tensor | None = None
    concept_probs: torch.Tensor = field(init=False)

    def __post_init__(self):
        self.concept_probs = torch.sigmoid(self.concept_logits)

    def to_numpy(self) -> ConceptOutputNumpy:
        return ConceptOutputNumpy(
            concept_logits=tensor_to_numpy(self.concept_logits),
            concept_probs=tensor_to_numpy(self.concept_probs),
            scale=optional_tensor_to_numpy(self.scale),
            shift=optional_tensor_to_numpy(self.shift),
        )

class ConceptModule(nn.Module):
    ...


def get_concept_module_by_name(name: str, **kwargs) -> ConceptModule:
    """Factory-Methode zum Abrufen von Concept-Modulen anhand ihres Namens."""
    modules = {
        "SegMaskAvgPool": SegMaskAvgPool,
        "SegMaskMaxPool": SegMaskMaxPool,
        "SegMaskAvgPoolTrainChannelAffine": SegMaskAvgPoolTrainChannelAffine,
        "LogitMeanTopK": LogitMeanTopK,
        "SALFConceptModule": SALFConceptModule,
    }
    if name not in modules:
        raise ValueError(f"Unknown Concept-Modul: {name}")
    return modules[name](**kwargs)


class SegMaskAvgPool(ConceptModule):
    """Segmentation Mask Average Pooling"""

    def __init__(self, **kwargs):
        super().__init__()

    def forward(self, segmentation_output: SegmentationOutput):
        # [B, n_concepts, H, W] -> [B, n_concepts]
        concept_logits = segmentation_output.mask_logits.mean(dim=[2, 3])
        return ConceptOutput(concept_logits)


class SegMaskMaxPool(ConceptModule):
    """Segmentation Mask Max Pooling"""

    def __init__(self, **kwargs):
        super().__init__()

    def forward(self, segmentation_output: SegmentationOutput):
        mask_logits = segmentation_output.mask_logits
        # [B, n_concepts, H, W] -> [B, n_concepts]
        concept_logits = mask_logits.amax(dim=[2, 3])
        return ConceptOutput(concept_logits=concept_logits)

class SegMaskAvgPoolTrainChannelAffine(SegMaskAvgPool):
    """Segmentation Mask Average Pooling with Channel-wise Affine Transformation"""

    def __init__(self, n_concepts: int, **kwargs):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(n_concepts))
        self.shift = nn.Parameter(torch.zeros(n_concepts))

    def forward(self, segmentation_output: SegmentationOutput):
        # 1. Pooling via Parent
        concept_output_raw = super().forward(segmentation_output)
        
        # 2. Affine Transformation
        concept_logits = concept_output_raw.concept_logits * self.scale + self.shift
        
        return ConceptOutput(
            concept_logits=concept_logits,
            scale=self.scale,
            shift=self.shift
        )


class LogitMeanTopK(ConceptModule):
    def __init__(self, top_k_percent: float, **kwargs):
        super().__init__()
        self.top_k_percent = top_k_percent

    def forward(self, segmentation_output: SegmentationOutput):
        B, C, H, W = segmentation_output.mask_logits.shape
        
        n_pixels = H * W
        k = math.ceil(n_pixels * self.top_k_percent)

        flat_logits_masks = segmentation_output.mask_logits.view(B, C, -1)

        topk_logit_values, _ = torch.topk(flat_logits_masks, k=k, dim=2)

        concept_logits = topk_logit_values.mean(dim=2)
        
        return ConceptOutput(concept_logits=concept_logits)
    

class SALFConceptModule(nn.Module):
    """
    Global softmax pooling following the SALF-CBM logic. 
    Aggregates spatial masks into a concept vector.
    """
    proj_mean: torch.Tensor
    proj_std: torch.Tensor
    def __init__(self, n_concepts: int, proj_mean=0.0, proj_std=1.0, **kwargs):
        super().__init__()
       
        self.register_buffer("proj_mean", torch.tensor(proj_mean, dtype=torch.float32))
        self.register_buffer("proj_std", torch.tensor(proj_std, dtype=torch.float32))

    def forward(self, segmentation_output: SegmentationOutput):
        # Expected shape: [B, 112, H, W] from your SegHead
        mask_logits = segmentation_output.mask_logits
        B, K, H, W = mask_logits.shape

        # 1. Flatten spatial dimensions
        # [B, K, H*W]
        flat_logits = mask_logits.view(B, K, -1)

        # 2. Compute softmax weights over the entire mask
        # This follows the SALF logic: a pixel becomes the representative of the concept.
        # High logit values receive extremely high weight.
        weights = F.softmax(flat_logits, dim=-1)  # Softmax over H*W

        # 3. Weighted sum (aggregation)
        # [B, K]
        pooled = torch.sum(flat_logits * weights, dim=-1)

        # 4. Standardization (SALF-specific)
        # Helps the downstream classifier to get more stable gradients
        proj_c = (pooled - self.proj_mean) / (self.proj_std + 1e-6)

        return ConceptOutput(concept_logits=proj_c)
