import os
from typing import Literal

import torch
import torch.nn as nn
from dinov3.models.vision_transformer import DinoVisionTransformer

from segdino.dpt import DPT

__all__ = ["segdino_b", "segdino_s"]


def _seg_dino(dino_size: Literal["b", "s"], dino_ckpt: str, n_concepts: int):
    def get_dinov3_torchhub(dino_size: Literal["b", "s"], dino_ckpt: str) -> DinoVisionTransformer:
        if dino_size == "b":
            backbone: DinoVisionTransformer = torch.hub.load(
                "./../dinov3",
                model="dinov3_vitb16",
                source="local",
                weights=dino_ckpt,
            )
        elif dino_size == "s":
            backbone: DinoVisionTransformer = torch.hub.load(
                "./../dinov3",
                model="dinov3_vits16",
                source="local",
                weights=dino_ckpt,
            )
        else:
            raise ValueError(f"Unknown dino_size: {dino_size}")
        return backbone
    
    ROOT_DIR_WORKSPACE = os.environ['ROOT_DIR_WORKSPACE']
    dino_ckpt = os.path.join(ROOT_DIR_WORKSPACE, '.cache', dino_ckpt)
    
    backbone = get_dinov3_torchhub(dino_size=dino_size, dino_ckpt=dino_ckpt)
    model = DPT(backbone=backbone, nclass=n_concepts)
    return model


def segdino_b(dino_ckpt: str, n_concepts: int):
    return _seg_dino(dino_size="b", dino_ckpt=dino_ckpt, n_concepts=n_concepts)


def segdino_s(dino_ckpt: str, n_concepts: int):
    return _seg_dino(dino_size="s", dino_ckpt=dino_ckpt, n_concepts=n_concepts)


def get_unified_model_by_name(
    name: str, dino_ckpt_segdino: str, n_concepts: int, **kwards
) -> nn.Module:
    unified_models = {
        "segdino_b": segdino_b,
        "segdino_s": segdino_s,
    }
    if name not in unified_models:
        raise ValueError(f"Unknown unified model: {name}. Available: {list(unified_models.keys())}")
    return unified_models[name](dino_ckpt=dino_ckpt_segdino, n_concepts=n_concepts)
