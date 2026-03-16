import os
from dataclasses import dataclass

import numpy as np
import timm
import torch
import torch.nn as nn

from . import optional_tensor_to_numpy

__all__ = ["dinov3"]


def get_encoder_by_name(name: str) -> nn.Module:
    modules = {
        "dinov3": dinov3,
    }

    if name in modules:
        return modules[name]()
    else:
        raise ValueError(
            f"Encoder module '{name}' not found. Available modules: {list(modules.keys())}"
        )


def dinov3():
    workspace_dir = os.getenv('ROOT_DIR_WORKSPACE', '.')
    model_type = "base"  # 'small', 'base', or 'large'
    model = timm.create_model(
        f"vit_{model_type}_patch16_dinov3.lvd1689m",
        pretrained=True,
        features_only=True,
        out_indices=(-1,),
        cache_dir=os.path.join(workspace_dir, '.cache'),
    )
    for p in model.parameters():
        p.requires_grad = False

    # get model specific transforms (normalization, resize)
    # data_config = timm.data.resolve_model_data_config(fmodel)
    # transforms = timm.data.create_transform(**data_config, is_training=False)

    return model

@dataclass
class EncoderOutputNumpy:
    features: np.ndarray | None = None

@dataclass
class EncoderOutput:
    features: torch.Tensor | None = None

    def to_numpy(self) -> EncoderOutputNumpy:
        return EncoderOutputNumpy(
            features=optional_tensor_to_numpy(self.features),
        )

