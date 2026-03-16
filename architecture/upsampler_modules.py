from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn

from . import optional_tensor_to_numpy

__all__ = ["anyup"]


def get_upsampler_by_name(name: str) -> nn.Module:
    modules = {
        "anyup": anyup,
    }

    if name in modules:
        return modules[name]()
    else:
        raise ValueError(
            f"Upsampler module '{name}' not found. Available modules: {list(modules.keys())}"
        )
    
@dataclass
class UpsamplerOutputNumpy:
    features: np.ndarray | None = None


@dataclass
class UpsamplerOutput:
    features: torch.Tensor | None = None

    def to_numpy(self) -> UpsamplerOutputNumpy:
        return UpsamplerOutputNumpy(
            features=optional_tensor_to_numpy(self.features),
        )



def anyup() -> nn.Module:
    return torch.hub.load("wimmerth/anyup", "anyup")  # type: ignore
