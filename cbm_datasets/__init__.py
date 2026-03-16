import os
import random
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset, random_split

from .cub import CUB_112, CUB_312, SUB, get_transform_cub
from .funny_birds_custom import FunnyBirdsCustom, get_transform_funnybirds
from .types import Batch, Sample

__datasets__ = ["FunnyBirds", "CUB_312", "CUB_112"]


def seed_worker(worker_id: int):
    worker_seed = 42 + worker_id
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    torch.manual_seed(worker_seed)




def collate_fn(batch: list[Sample]) -> Batch:
    """
    Collects Sample dataclasses and stacks them into a Batch.
    """
    
    # Helper function: checks the first item in the batch.
    # If the field there is None, it is None for the entire batch.
    def stack_optional_field(field_name: str):
        first_val = getattr(batch[0], field_name)
        if first_val is None:
            return None
        # Stack all values of this field from the batch
        return torch.stack([torch.as_tensor(getattr(item, field_name)) for item in batch])

    return Batch(
        image_ids=[item.image_id for item in batch],
        image_idxs=torch.tensor([item.image_index for item in batch]),
        
        # Main data (always present)
        images=torch.stack([torch.as_tensor(item.image) for item in batch]),
        labels=torch.stack([torch.as_tensor(item.labels) for item in batch]),
        concepts=torch.stack([torch.as_tensor(item.concepts) for item in batch]),
        
        # Optional fields using the helper function
        concept_weights=stack_optional_field("concept_weights"),
        concept_coords=stack_optional_field("concept_coords"),
        concept_point_masks=stack_optional_field("concept_point_masks"),
        
        # Masks
        mask_foregrounds=torch.stack([torch.as_tensor(item.mask_foreground) for item in batch]),
        mask_concepts=torch.stack([torch.as_tensor(item.mask_concepts) for item in batch])
    )


def get_datasets(
    dataset_name: Literal["FunnyBirds", "CUB_312", "CUB_112", "SUB"],
    root_dir: str | Path,
    img_size: int,
    use_soft_labels: bool,
    concept_masks_scale: Literal["small", "medium", "large"] | None,
    attr_level: Literal["image", "class"],
    center_crop_size: int | None = None,
) -> tuple[Dataset, Dataset, Dataset, int, int, list] | Dataset:

    """
    Initializes and returns the training, validation, and test datasets.
    
    Args:
        dataset_name: Name of the dataset ('FunnyBirds', 'CUB_312', or 'CUB_112').
        root_dir: Root directory containing the datasets.
        img_size: Target size for image resizing.
        use_soft_labels: Whether to use soft labels for training.
        concept_masks_scale: Scale of concept masks (only for CUB datasets).
        center_crop_size: Size for center cropping (optional).

    Returns:
        A tuple containing (train_dataset, val_dataset, test_dataset).
        Note: val_dataset is None for 'FunnyBirds'.
    """
    
    if dataset_name == "FunnyBirds":

        if use_soft_labels:
            raise ValueError("FunnyBirds Dataset does not support soft labels")

        root_dir_dataset = os.path.join(root_dir, "FunnyBirds")
        transform = get_transform_funnybirds(img_size=img_size, center_crop_size=center_crop_size)
        
        train_dataset = FunnyBirdsCustom(
            root_dir=root_dir_dataset, mode="train", transform=transform, get_part_map=True
        )
        
        test_dataset = FunnyBirdsCustom(
            root_dir=root_dir_dataset, mode="test", transform=transform, get_part_map=True
        )

        
        # FunnyBirds only provides train and test dataset. Therefore, test dataset is splittet since in the train dataset some parts are removed
        total_size = len(test_dataset)
        test_size = int(0.7 * total_size)
        val_size = total_size - test_size

        val_dataset, test_dataset = random_split(test_dataset, [val_size, test_size], generator=torch.Generator().manual_seed(42))

        return train_dataset, val_dataset, test_dataset, train_dataset.n_concepts, train_dataset.n_classes, train_dataset.concepts

    elif dataset_name == "CUB_312":
        root_dir_dataset = os.path.join(root_dir, "CUB_200_2011")
        transform = get_transform_cub(img_size=img_size, center_crop_size=center_crop_size)
        train_dataset = CUB_312(
            root_dir=root_dir_dataset,
            mode="train",
            transform=transform,
            concept_masks_scale=concept_masks_scale,
            use_soft_labels=use_soft_labels,
            attr_level=attr_level,
        )
        val_dataset = CUB_312(
            root_dir=root_dir_dataset,
            mode="val",
            transform=transform,
            concept_masks_scale=concept_masks_scale,
            use_soft_labels=use_soft_labels,
            attr_level=attr_level,
        )

        test_dataset = CUB_312(
            root_dir=root_dir_dataset,
            mode="test",
            transform=transform,
            concept_masks_scale=concept_masks_scale,
            use_soft_labels=use_soft_labels,
            attr_level=attr_level,
        )

        return train_dataset, val_dataset, test_dataset, train_dataset.n_concepts, train_dataset.n_classes, train_dataset.concepts.to_list()
    

    elif dataset_name == "CUB_112":
        root_dir_dataset = os.path.join(root_dir, "CUB_200_2011")
        transform = get_transform_cub(img_size=img_size, center_crop_size=center_crop_size)

        train_dataset = CUB_112(
            root_dir=root_dir_dataset,
            mode="train",
            transform=transform,
            concept_masks_scale=concept_masks_scale,
            use_soft_labels=use_soft_labels,
            attr_level=attr_level,
        )
        val_dataset = CUB_112(
            root_dir=root_dir_dataset,
            mode="val",
            transform=transform,
            concept_masks_scale=concept_masks_scale,
            use_soft_labels=use_soft_labels,
            attr_level=attr_level,
        )
        test_dataset = CUB_112(
            root_dir=root_dir_dataset,
            mode="test",
            transform=transform,
            concept_masks_scale=concept_masks_scale,
            use_soft_labels=use_soft_labels,
            attr_level=attr_level,
        )
        return train_dataset, val_dataset, test_dataset, train_dataset.n_concepts, train_dataset.n_classes, train_dataset.concepts.to_list()
    
    elif dataset_name == "SUB":
        transform = get_transform_cub(img_size=img_size, center_crop_size=center_crop_size)
        return SUB(
            reference_dataset_name="CUB_112",
            transform=transform,
            root_dir=str(root_dir),
            mode="test"
        )
    
    raise ValueError(f"dataset has to be 'FunnyBirds' or 'CUB_200_2011'. {dataset_name} given")

    

    

def get_dataloader(
    batch_size: int,
    num_workers: int,
    train_dataset: Dataset,
    val_dataset: Dataset | Subset,
    test_dataset: Dataset | Subset,
    seed: int = 42   
):
    
    g = torch.Generator()
    g.manual_seed(seed)

    train_loader: DataLoader[Batch] = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True,
        generator=g,
        worker_init_fn=seed_worker,
        collate_fn=collate_fn
    )

    val_loader: DataLoader[Batch] = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
        collate_fn=collate_fn
    )

    test_loader: DataLoader[Batch] = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
        collate_fn=collate_fn
    )

    return train_loader, val_loader, test_loader
