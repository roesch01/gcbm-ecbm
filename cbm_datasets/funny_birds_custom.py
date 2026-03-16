import os
from typing import Literal

import albumentations as A
import numpy as np
from numpy.typing import NDArray
from PIL import Image

from .funny_birds import FunnyBirds
from .types import Sample


def get_transform_funnybirds(
    img_size: int,
    center_crop_size: int | None = None,
) -> A.Compose:
    transforms = []

    if center_crop_size is not None:
        transforms.append(
            A.CenterCrop(
                height=center_crop_size,
                width=center_crop_size,
            )
        )

    transforms.extend(
        [
            A.Resize(height=img_size, width=img_size),
            A.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
        ]
    )

    return A.Compose(
        transforms, additional_targets={"mask_concepts": "mask", "mask_foreground": "mask"}
    )


class FunnyBirdsCustom(FunnyBirds):
    parts_separator = "-"
    parts = ["beak", "eye", "foot", "tail", "wing"]
    concepts = [
        "beak-0",
        "beak-1",
        "beak-2",
        "beak-3",
        "eye-0",
        "eye-1",
        "eye-2",
        "foot-0",
        "foot-1",
        "foot-2",
        "foot-3",
        "tail-0",
        "tail-1",
        "tail-2",
        "tail-3",
        "tail-4",
        "tail-5",
        "tail-6",
        "tail-7",
        "tail-8",
        "wing-0",
        "wing-1",
        "wing-2",
        "wing-3",
        "wing-4",
        "wing-5",
    ]
    img_size = 256

    n_concepts: int = len(concepts)
    n_classes: int = 50

    def __init__(
        self,
        root_dir: str,
        mode: Literal["train", "test"],
        transform: A.Compose,
        get_part_map: bool = False,
        width: int | None = None,
        height: int | None = None,
    ):
        super().__init__(
            root_dir=root_dir, mode=mode, get_part_map=get_part_map, transform=transform
        )

        self.width = width if width is not None else self.img_size
        self.height = height if height is not None else self.img_size

    def get_pos_weight_vector(self):
        return np.ones((self.n_concepts,), dtype=np.float32)

    def __getitem__(self, idx: int):
        class_idx = self.params[idx]["class_idx"]

        img_path = os.path.join(
            self.root_dir, self.mode, str(class_idx), str(idx).zfill(6) + ".png"
        )
        image = np.array(Image.open(img_path).convert("RGB"))

        params = self.params[idx]

        part_map_path = os.path.join(
            self.root_dir, self.mode + "_part_map", str(class_idx), str(idx).zfill(6) + ".png"
        )
        part_map = np.array(Image.open(part_map_path).convert("RGB"))

        concepts_idx = super().single_params_to_part_idxs(self.params[idx])

        mask_concepts = self.create_mask_concepts(concepts_idx=concepts_idx, part_map=part_map)

        mask_foreground = self.create_mask_foreground(
            segmentation_masks=mask_concepts, part_map=part_map
        )  # H, W, 1
        labels = self.create_onehot_classes_multilabel(params=params)

        if self.transform is not None:
            alb = self.transform(
                image=image, mask_concepts=mask_concepts, mask_foreground=mask_foreground
            )  # dict

            image: NDArray = alb["image"]
            mask_concepts: NDArray = alb["mask_concepts"]
            mask_foreground: NDArray = alb["mask_foreground"]

        image = image.transpose(2, 0, 1)  # C, H, W
        mask_concepts = mask_concepts.transpose(2, 0, 1)  # C, H, W
        mask_foreground = mask_foreground.transpose(2, 0, 1)  # 1, H, W

        concept_vector = (
            mask_concepts.reshape(len(self.concepts), -1).max(axis=1).astype(np.float32)
        )  # [C,]

        sample = Sample(
            # Input & Meta
            image_id=f"{self.mode}_{class_idx}_{str(idx).zfill(6)}.png",
            image=image,  # [C, H, W]
            image_index=idx,  # [1]
            # params=params,
            # Concepts
            # "class_idx": class_idx,
            concepts=concept_vector,  # [C]
            # Classes
            labels=labels,  # [K]
            # Masks
            mask_foreground=mask_foreground,  # [1, H, W]
            mask_concepts=mask_concepts,  # [C, H, W]
            part_map=part_map,
        )

        return sample

    def create_mask_concepts(self, part_map: NDArray, concepts_idx: dict) -> NDArray:
        """
        Creates a binary segmentation mask for a specific concept/part from an RGB image.

        Args:
            part_map (NDArray): partial segmentation map of the image, HxW
            concepts_idx (dict): mapping of concept names to their indices
        """
        masks = np.zeros((self.img_size, self.img_size, len(self.concepts)), dtype=np.float32)

        for i, concept in enumerate(self.parts):
            assert concept in ["beak", "eye", "foot", "tail", "wing"], (
                f"Invalid concept: {concept}"
            )

            if concepts_idx[concept] < 0:
                continue  # Part not present, skip

            # All colors corresponding to this concept
            concept_colors = [
                color for color, part in self.colors_to_part.items() if part.startswith(concept)
            ]

            # Consider all relevant colors
            for color in concept_colors:
                color_arr = np.array(color)

                # Compare across all channels → mask (H, W)
                mask = np.all(part_map == color_arr, axis=2).astype(np.float32)

                idx = self.concepts.index(f"{concept}-{concepts_idx[concept]}")
                masks[:, :, idx] += mask

        return masks

    def create_mask_foreground(
        self, segmentation_masks: NDArray, part_map: NDArray
    ) -> NDArray:
        color_arr = np.array((170, 170, 170))

        # Compare across all channels → mask (H, W)
        body_mask = np.all(part_map == color_arr, axis=2).astype(np.uint8)
        mask = (segmentation_masks > 0).sum(axis=2) + body_mask

        mask_with_body = (mask > 0).astype(np.float32)

        return mask_with_body[..., None]

    def create_onehot_classes_multilabel(self, params) -> NDArray:
        target_classes = np.ones((50,), dtype=np.float32)

        for cls in self.classes:
            class_idx = cls["class_idx"]
            parts_specification = self.single_params_to_part_idxs(params)
            for part, part_idx in parts_specification.items():
                if part_idx == -1:
                    continue

                if part_idx != self.classes[class_idx]["parts"][part]:
                    target_classes[class_idx] = 0
                    break

        return target_classes
