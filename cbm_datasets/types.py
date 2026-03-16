from dataclasses import dataclass, fields

import torch
from numpy.typing import NDArray


@dataclass(frozen=True)
class Sample:
    image_id: str
    image_index: int
    image: NDArray
    concepts: NDArray
    labels: NDArray
    mask_foreground: NDArray
    mask_concepts: NDArray
    part_map: NDArray | None = None
    
    # Optional field (Nullable) with Default None
    concept_weights: NDArray | None = None
    concept_coords: NDArray | None = None
    concept_point_masks: NDArray | None = None



@dataclass
class Batch:
    image_ids: torch.Tensor | list  # string
    image_idxs: torch.Tensor # id (int)
    images: torch.Tensor
    labels: torch.Tensor
    # class_idxs: torch.Tensor
    # Concept data
    concepts: torch.Tensor  # Soft labels
    concept_weights: torch.Tensor | None  # Certainty
    # SAM & Point-Prompting
    concept_coords: torch.Tensor | None
    concept_point_masks: torch.Tensor | None
    mask_foregrounds: torch.Tensor
    mask_concepts: torch.Tensor

    def to(self, device: torch.device | str):
        """
        Moves all contained tensors to the specified device.
        Works dynamically for all fields of the dataclass.
        """
        res = {}
        for field in fields(self):
            value = getattr(self, field.name)
            if isinstance(value, torch.Tensor):
                res[field.name] = value.to(device)
            else:
                res[field.name] = value
        return Batch(**res)