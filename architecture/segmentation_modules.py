from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt
import torch
import torch.nn as nn

from architecture.encoder_modules import EncoderOutput
from architecture.upsampler_modules import UpsamplerOutput

from . import tensor_to_numpy

__all__ = [
    "SegmentationHeadUpscaledSingle",
    "SegmentationHeadUpscaledMulti",
    "SegmentationHeadSETRPUP"
]



@dataclass
class SegmentationOutputNumpy:
    mask_logits: npt.NDArray[np.float32]
    mask_probs: np.ndarray

@dataclass
class SegmentationOutput:
    mask_logits: torch.Tensor
    mask_probs: torch.Tensor = field(init=False)

    def __post_init__(self):
        self.mask_probs = torch.sigmoid(self.mask_logits)

    def to_numpy(self) -> SegmentationOutputNumpy:
        return SegmentationOutputNumpy(
            mask_logits=tensor_to_numpy(self.mask_logits),
            mask_probs=tensor_to_numpy(self.mask_probs),
        )
    


class SegmentationModule(nn.Module):
    ...



def get_segmentation_module_by_name(name: str, **kwargs) -> SegmentationModule:
    modules = {
        "SegmentationHeadUpscaledSingle": SegmentationHeadUpscaledSingle,
        "SegmentationHeadUpscaledMulti": SegmentationHeadUpscaledMulti,
        "SegmentationHeadSETRPUP": SegmentationHeadSETRPUP
    }
    if name not in modules:
        raise ValueError(f"Unknown segmentation module: {name}. Available: {list(modules.keys())}")
    return modules[name](**kwargs)


class SegmentationHeadUpscaledMulti(SegmentationModule):
    def __init__(self, feature_dim: int, n_concepts: int, *args, **kwargs):
        
        super().__init__()
        self.net = nn.Sequential(
            # 1. Project to a stable hidden space
            nn.Conv2d(feature_dim, 512, kernel_size=1),
            nn.GroupNorm(32, 512),
            nn.GELU(),
            
            # 2. Spatial regularization
            # Dropout2d zeroes entire channels, forcing the model
            # to learn concepts across multiple feature combinations.
            nn.Dropout2d(0.3), 
            
            # 3. Feature extraction (3x3 for spatial context)
            nn.Conv2d(512, out_channels=384, kernel_size=3, padding=1),
            nn.GroupNorm(32, 384),
            nn.GELU(),
            nn.Dropout2d(0.3),

            # 4. Depth through the ResBlock (remains at 384 channels)
            ResBlock(384),
            
            # 5. Final classification into the 112 masks
            nn.Conv2d(384, n_concepts, kernel_size=1),
        )

    def forward(self, _: EncoderOutput, upsampler_outputs: UpsamplerOutput) -> SegmentationOutput:
        mask_logits = self.net(upsampler_outputs.features)
        return SegmentationOutput(
            mask_logits=mask_logits
        )


class SegmentationHeadUpscaledSingle(SegmentationModule):
    def __init__(self, feature_dim: int, n_concepts: int, *args, **kwargs):
        super().__init__()
        self.net = nn.Conv2d(feature_dim, n_concepts, kernel_size=1)

    def forward(self, _: EncoderOutput, upsampler_outputs: UpsamplerOutput) -> SegmentationOutput:
        mask_logits = self.net(upsampler_outputs.features)
        return SegmentationOutput(
            mask_logits=mask_logits
        )


class SegmentationHeadSETRPUP(SegmentationModule):
    """
    Progressive Up-Sampling (PUP) head following the SETR design.
    Gradually upsamples features to preserve fine details.
    """
    def __init__(self, feature_dim: int, n_concepts: int, inner_dim: int = 512, *args, **kwargs):
        super().__init__()
        
        # Stage 1: Reduce to 512 channels & first upsampling (x2)
        self.up1 = nn.Sequential(
            nn.Conv2d(feature_dim, inner_dim, kernel_size=3, padding=1),
            nn.GroupNorm(32, inner_dim),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        )
        
        # Stage 2: Second upsampling (x2)
        self.up2 = nn.Sequential(
            nn.Conv2d(inner_dim, inner_dim, kernel_size=3, padding=1),
            nn.GroupNorm(32, inner_dim),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        )
        
        # Stage 3: Third upsampling (x2) -> e.g., from 14x14 to 112x112
        # Additional stages can be added depending on the encoder input resolution
        self.up3 = nn.Sequential(
            nn.Conv2d(inner_dim, inner_dim, kernel_size=3, padding=1),
            nn.GroupNorm(32, inner_dim),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        )

        self.up4 = nn.Sequential(
            nn.Conv2d(inner_dim, inner_dim, kernel_size=3, padding=1),
            nn.GroupNorm(32, inner_dim),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        )

        # Final head: project to the number of concepts
        # Multilabel: each pixel can belong to multiple concepts (sigmoid applied in the loss)
        self.final_conv = nn.Conv2d(inner_dim, n_concepts, kernel_size=1)

    def forward(self, _: EncoderOutput, upsampler_outputs: UpsamplerOutput) -> SegmentationOutput:
        # Extract features from the upsampler or encoder
        x = upsampler_outputs.features 
        
        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)
        x = self.up4(x)
        
        mask_logits = self.final_conv(x)
        
        return SegmentationOutput(mask_logits=mask_logits)



class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.GroupNorm(min(32, channels), channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.GroupNorm(min(32, channels), channels)
        )
        self.relu = nn.GELU()

    def forward(self, x):
        return self.relu(x + self.net(x)) # Skip Connection
