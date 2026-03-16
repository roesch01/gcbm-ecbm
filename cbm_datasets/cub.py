import pickle
from pathlib import Path
from typing import Literal

import albumentations as A
import numpy as np
import pandas as pd
from datasets import load_dataset
from datasets.arrow_dataset import Dataset as ArrowDataset
from numpy.typing import NDArray
from PIL import Image
from torch.utils.data import Dataset

from .types import Sample


def get_transform_cub(
    img_size: int | None,
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

    if img_size is not None:
        transforms.append(
            A.Resize(height=img_size, width=img_size),
        )

    transforms.append(
        A.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
        ),
    )

    return A.Compose(
        transforms,
        additional_targets={"mask_foreground": "mask", "mask_concepts": "mask"},
        keypoint_params=A.KeypointParams(format="xy", remove_invisible=False),
    )


class CUB_312(Dataset):
    n_classes: int = 200
    parts_separator: str = "::"

    filtered_attributes: list[int] = []  # No filtering of the 312 concepts

    @property
    def total_raw_attributes(self) -> int:
        return 312

    @property
    def n_concepts(self) -> int:
        return self.total_raw_attributes

    # Mapping (1-indexed like in part_locs.txt)
    ATTRIBUTE_TO_PARTS = {
        "has_back_color": [1],
        "has_back_pattern": [1],
        "has_belly_color": [3],
        "has_belly_pattern": [3],
        "has_bill_color": [2],
        "has_bill_length": [2],
        "has_bill_shape": [2],
        "has_breast_color": [4],
        "has_breast_pattern": [4],
        "has_crown_color": [5],
        "has_eye_color": [7, 11],  # left eye, right eye
        "has_forehead_color": [6],
        "has_head_pattern": [5, 6, 10],  # crown, forehead, nape aggregate
        "has_leg_color": [8, 12],  # left leg, right leg
        "has_nape_color": [10],
        "has_primary_color": [9, 13],  # left wing, right wing
        "has_shape": range(1, 16),  # Global - all parts
        "has_size": range(1, 16),  # Global - all parts
        "has_color": range(1, 16),
        "has_tail_pattern": [14],
        "has_tail_shape": [14],
        "has_throat_color": [15],
        "has_under_tail_color": [14],
        "has_underparts_color": [3],  # belly
        "has_upper_tail_color": [14],
        "has_upperparts_color": [1],  # back (primary)
        "has_wing_color": [9, 13],  # left wing, right wing
        "has_wing_pattern": [9, 13],  # left wing, right wing
        "has_wing_shape": [9, 13],  # left wing, right wing
    }

    def __init__(
        self,
        root_dir: str,
        mode: Literal["train", "val", "test"],
        transform: A.Compose,
        concept_masks_scale: Literal["small", "medium", "large"] | None,
        use_soft_labels: bool,
        attr_level: Literal["image", "class"],
    ):
        self.root_dir = Path(root_dir)
        self.image_dir = self.root_dir / "images"
        self.segmentation_dir = self.root_dir / "segmentations"
        self.concept_mask_dir = self.root_dir / "concept_masks_sam2"

        self.use_soft_labels = use_soft_labels
        self.mode = mode
        self.transform = transform
        self.concept_masks_scale = concept_masks_scale
        self.attr_level = attr_level

        # --- Load Metadata ---
        self.concepts = pd.read_csv(
            self.root_dir / "attributes" / "attributes.txt",
            sep=" ",
            names=["attr_id", "attr_name"],
        ).attr_name

        # Extract base part name from attribute (e.g., "has_bill_shape" -> "bill")
        self.attr_prefixes = self.parts = self.concepts.str.split(
            self.parts_separator, expand=True
        )[0]

        # --- Load Parts Data ---
        # parts.txt: id name (e.g., 2 beak)
        parts_df = pd.DataFrame({"part_id": [], "part_name": []})

        rows = []
        with open(self.root_dir / "parts" / "parts.txt") as f:
            for line in f:
                parts = line.strip().split(maxsplit=1)  # Split only on first space
                if len(parts) == 2:
                    rows.append({"part_id": int(parts[0]), "part_name": parts[1]})

        parts_df = pd.DataFrame(rows)

        # part_locs.txt: image_id part_id x y visible
        part_locs_df = pd.read_csv(
            self.root_dir / "parts" / "part_locs.txt",
            sep=" ",
            names=["image_id", "part_id", "x", "y", "visible"],
        )

        # [Image_ID, Part_ID] -> (x, y, visible)
        self.part_locs = part_locs_df.set_index(["image_id", "part_id"])
        self.part_id_map = dict(zip(parts_df["part_name"], parts_df["part_id"]))

        images = pd.read_csv(
            self.root_dir / "images.txt", sep=" ", names=["image_id", "image_path"]
        )
        split = pd.read_csv(
            self.root_dir / "train_test_split.txt",
            sep=" ",
            names=["image_id", "is_train"],
        )
        attributes_images = pd.read_csv(
            self.root_dir / "attributes" / "image_attribute_labels.txt",
            sep=" ",
            names=["image_id", "attribute_id", "is_present", "certainty", "time"],
            on_bad_lines="skip",
        )

        image_classes = pd.read_csv(
            self.root_dir / "image_class_labels.txt",
            sep=" ",
            names=["image_id", "class_id"],
        )

        self.classes = pd.read_csv(
            self.root_dir / "classes.txt",
            sep=" ",
            names=["class_id", "class_name"],
        )

        self.data = images.merge(split, on="image_id").merge(image_classes, on="image_id")

        file_path = self.root_dir / "class_attr_data_10" / f"{mode}.pkl"
        with open(file_path, 'rb') as f:
            raw_data = pickle.load(f)

        image_ids = [item['id'] - 1 for item in raw_data]
        attributes_images_class = [item['attribute_label'] for item in raw_data]

        attributes_images_class_312 = []
        for row in attributes_images_class:
            a = np.zeros(312)
            a[self.filtered_attributes] = row
            attributes_images_class_312.append(a)

        self.data = self.data.iloc[image_ids].reset_index(drop=True)

        self.max_points_per_concept = self._compute_max_points()

        if not self.use_soft_labels:
            self.weight_matrix = None
            if self.attr_level == "image":
                self.attribute_matrix = attributes_images.pivot(
                    index="image_id", columns="attribute_id", values="is_present"
                ).loc[self.data['image_id']]
            else:
                self.attribute_matrix = pd.DataFrame(
                    attributes_images_class_312, index=self.data['image_id'], columns=self.concepts.index
                )
                
        else:
            if self.attr_level != "image":
                raise NotImplementedError("Soft-Labels are currently only implemented for image-level attributes")
            # Soft-Label Logic
            # Mapping Definition: (is_present, certainty) -> (label_value, weight)
            mapping = {
                (1, 4): (1.0, 1.0),
                (1, 3): (0.8, 0.8),
                (1, 2): (0.5, 0.2),
                (1, 1): (0.0, 0.0),
                (0, 4): (0.0, 1.0),
                (0, 3): (0.2, 0.8),
                (0, 2): (0.5, 0.2),
                (0, 1): (0.0, 0.0),
            }

            mapped_values = attributes_images.apply(
                lambda row: mapping.get((row["is_present"], row["certainty"]), (0.0, 0.0)), axis=1
            )
            attributes_images["soft_val"] = [x[0] for x in mapped_values]
            attributes_images["weight_val"] = [x[1] for x in mapped_values]

            # Pivot for Labels and Weights
            self.attribute_matrix = attributes_images.pivot(
                index="image_id", columns="attribute_id", values="soft_val"
            )
            self.weight_matrix = attributes_images.pivot(
                index="image_id", columns="attribute_id", values="weight_val"
            )

    def get_pos_weight_vector(self) -> NDArray:
        """Berechne pos_weight Vektor für BCE Loss"""
        labels: NDArray[np.float64] = self.attribute_matrix.loc[self.data['image_id']].values
        labels[np.isnan(labels)] = 0
        pos_counts: NDArray[np.float64] = labels.sum(0)
        # Number of negative examples per concept
        neg_counts = labels.shape[0] - pos_counts

        weights = neg_counts / (pos_counts + 1e-5)
        return weights.clip(max=100).astype(np.float32)

    def get_attr_group_from_concept(self, concept_id: int):
        return self.attr_prefixes.iloc[concept_id]

    def get_mask_concepts(
        self, h_orig: int, w_orig: int, image_id: int, concepts_vector: NDArray
    ):
        
        # Load SAM Concept Masks
        # If mask is not available: black.
        all_concept_masks = np.zeros(
            (h_orig, w_orig, self.total_raw_attributes), dtype=np.uint8
        )  # [H, W, C]

        if self.concept_masks_scale is None:
            return all_concept_masks

        img_mask_folder = self.concept_mask_dir / str(image_id)

        if not img_mask_folder.exists():
            print(f"Folder {img_mask_folder} does not exist. Concept Masks are zero")
            return all_concept_masks

        # We only load masks for active concepts (Performance)
        active_indices = np.where(concepts_vector > 0)[0]
        for c_idx in active_indices:
            if self.filtered_attributes and c_idx not in self.filtered_attributes:
                continue
            
            mask_file = img_mask_folder / f"{c_idx}_{self.concept_masks_scale}.png"
            if not mask_file.exists():
                # print(f"Mask {mask_file} does not exist. Concept Mask is zero")
                continue
            
            m = np.array(Image.open(mask_file).convert("L"))
            all_concept_masks[:, :, c_idx] = m
        return all_concept_masks

    def create_flat_keypoint_list(
        self, image_id: int, concepts_vector: NDArray
    ) -> tuple[dict[int, list[int]], NDArray, NDArray]:
        """Create flat Keypoint-List for Albumentations"""
        kp_idx = 0
        concept_to_kp_indices = {
            c: [] for c in range(self.total_raw_attributes)
        }
        keypoints = []

        for c_idx in range(self.total_raw_attributes):
            # If concept is not active -> skip
            if not concepts_vector[c_idx]:
                continue

            attr_group = self.get_attr_group_from_concept(c_idx)
            part_ids = self.ATTRIBUTE_TO_PARTS[attr_group]

            for part_id in part_ids:
                try:
                    # Part-Coordinates:
                    x = self.part_locs.loc[(image_id, part_id), "x"]
                    y = self.part_locs.loc[(image_id, part_id), "y"]
                    visibility = self.part_locs.loc[(image_id, part_id), "visible"]
                    if visibility != 1:  # Nur visible points
                        continue
                    keypoints.append([float(x), float(y)])  # type: ignore
                    concept_to_kp_indices[c_idx].append(kp_idx)
                    kp_idx += 1
                except KeyError:
                    print("KeyError")
                    continue

        # converting list into NumPy Array
        keypoints_array = (
            np.array(keypoints, dtype=np.float32)
            if keypoints
            else np.empty((0, 2), dtype=np.float32)
        )

        concepts_vector = np.array(
            [len(concept_to_kp_indices[c_idx]) > 0 for c_idx in range(self.total_raw_attributes)]
        )

        return concept_to_kp_indices, keypoints_array, concepts_vector

    def _compute_max_points(self) -> int:
        """Get max parts per attribut group"""
        max_points = 0
        for attr_group, part_ids in self.ATTRIBUTE_TO_PARTS.items():
            if part_ids is None:
                continue
            max_points = max(max_points, len(part_ids))
        return max(1, max_points)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        
        data = self.data.iloc[idx]
        image_id = data["image_id"]
        class_id = data["class_id"] - 1

        # Get labels and weights for concepts
        concept_vector = self.attribute_matrix.loc[image_id].values.astype(np.float32)
        concept_vector = np.nan_to_num(concept_vector, nan=0.0)

        if self.use_soft_labels:
            concept_weights = self.weight_matrix.loc[image_id].values.astype(np.float32)  # type: ignore
            concept_weights = np.nan_to_num(concept_weights, nan=0.0)
        else:
            concept_weights = np.ones_like(concept_vector)

        # Load image and masks
        image_path = self.image_dir / data["image_path"]
        image_pil = Image.open(image_path).convert("RGB")
        image_np = np.array(image_pil)
        H, W, _ = image_np.shape

        # Foreground segmentation
        mask_path = self.segmentation_dir / data["image_path"].replace(".jpg", ".png")
        if mask_path.exists():
            mask_fg = np.array(Image.open(mask_path).convert("L"))
        else:
            mask_fg = np.zeros((H, W), dtype=np.uint8)

        # SAM concept masks (loaded based on concept_vector > 0)
        mask_concepts = self.get_mask_concepts(H, W, image_id, concept_vector)

        # --- 3. Prepare keypoints & mapping ---
        # We use concept_vector only as a filter to know which parts to look for
        concept_to_kp_indices, keypoints, _ = self.create_flat_keypoint_list(
            image_id, concept_vector
        )

        # --- 4. Augmentation (Albumentations) ---
        if self.transform:
            augmented = self.transform(
                image=image_np,
                mask=mask_fg[..., None],  # [H, W, 1]
                mask_concepts=mask_concepts,  # [H, W, 312]
                keypoints=keypoints,
            )
            image = augmented["image"]  # Tensor [3, H, W]
            mask_fg = augmented["mask"]
            mask_concepts = augmented["mask_concepts"]
            transformed_kps = augmented["keypoints"]
        else:
            image = image_np
            transformed_kps = keypoints

        image = image.transpose(2, 0, 1)

        # --- 5. Post-processing (points & final labels) ---
        _, h_new, w_new = image.shape

        # keypoints_to_array now combines:
        # point visibility + original soft labels
        concept_coords, concept_point_mask = self.keypoints_to_array(
            concept_to_kp_indices, transformed_kps, h_new, w_new
        )

        # Format masks [C, H, W]
        mask_fg = (mask_fg.transpose(2, 0, 1) == 255).astype(np.float32)
        mask_concepts = (mask_concepts.transpose(2, 0, 1) == 255).astype(np.float32)

        # Class labels (one-hot)
        class_one_hot = np.eye(self.n_classes)[class_id]

        sample = Sample(
            image_id=image_id,
            image_index=idx,
            image=image,
            labels=class_one_hot,
            # class_idx=class_id,
            concepts=concept_vector,
            concept_weights=concept_weights,
            concept_coords=concept_coords,
            concept_point_masks=concept_point_mask,
            mask_foreground=mask_fg,
            mask_concepts=mask_concepts
        )
        return sample

    def keypoints_to_array(self, concept_to_kp_indices, transformed_kps, h, w):
        concept_coords = np.zeros(
            (self.total_raw_attributes, self.max_points_per_concept, 2), dtype=np.float32
        )
        concept_point_mask = np.zeros(
            (self.total_raw_attributes, self.max_points_per_concept), dtype=np.float32
        )

        for c_idx, kp_indices in concept_to_kp_indices.items():
            # check if concept moved out of the image after resizing/croping
            relevant_kps = [transformed_kps[i] for i in kp_indices]
            for p_idx, (x, y) in enumerate(relevant_kps):
                if 0 <= x < w and 0 <= y < h:
                    concept_coords[c_idx, p_idx] = [x, y]
                    concept_point_mask[c_idx, p_idx] = 1.0

        return concept_coords, concept_point_mask


# CUB_112 inherits from from CUB_312. It filters the outputs for filtered_attributes
class CUB_112(CUB_312):
    @property
    def n_concepts(self) -> int:
        return 112 

    filtered_attributes: list[int] = [
        1,
        4,
        6,
        7,
        10,
        14,
        15,
        20,
        21,
        23,
        25,
        29,
        30,
        35,
        36,
        38,
        40,
        44,
        45,
        50,
        51,
        53,
        54,
        56,
        57,
        59,
        63,
        64,
        69,
        70,
        72,
        75,
        80,
        84,
        90,
        91,
        93,
        99,
        101,
        106,
        110,
        111,
        116,
        117,
        119,
        125,
        126,
        131,
        132,
        134,
        145,
        149,
        151,
        152,
        153,
        157,
        158,
        163,
        164,
        168,
        172,
        178,
        179,
        181,
        183,
        187,
        188,
        193,
        194,
        196,
        198,
        202,
        203,
        208,
        209,
        211,
        212,
        213,
        218,
        220,
        221,
        225,
        235,
        236,
        238,
        239,
        240,
        242,
        243,
        244,
        249,
        253,
        254,
        259,
        260,
        262,
        268,
        274,
        277,
        283,
        289,
        292,
        293,
        294,
        298,
        299,
        304,
        305,
        308,
        309,
        310,
        311,
    ]

    def __init__(
        self,
        root_dir: str,
        mode: Literal["train", "val", "test"],
        transform: A.Compose,
        concept_masks_scale: Literal["small", "medium", "large"] | None,
        use_soft_labels: bool,
        attr_level: Literal["image", "class"],
    ):
        super().__init__(
            root_dir=root_dir,
            mode=mode,
            transform=transform,
            concept_masks_scale=concept_masks_scale,
            use_soft_labels=use_soft_labels,
            attr_level=attr_level,
        )
        self.concepts = self.concepts[self.filtered_attributes]

    def __getitem__(self, idx):
        sample = super().__getitem__(idx)

        f_idx = self.filtered_attributes

        sample_filtered = Sample(
            image_id=sample.image_id,
            image_index=idx,
            image=sample.image,
            labels=sample.labels,
            # class_idx=class_id,
            concepts=sample.concepts[f_idx],
            concept_weights=sample.concept_weights[f_idx] if sample.concept_weights is not None else None,
            concept_coords=sample.concept_coords[f_idx] if sample.concept_coords is not None else None,
            concept_point_masks=sample.concept_point_masks[f_idx] if sample.concept_point_masks is not None else None,
            mask_foreground=sample.mask_foreground,
            mask_concepts=sample.mask_concepts[f_idx]
        )

        return sample_filtered

    def get_pos_weight_vector(self) -> NDArray:
        return super().get_pos_weight_vector()[self.filtered_attributes]


class SUB(Dataset):
    def __init__(self, transform: A.Compose, root_dir: str, LABEL_PATH:str, reference_dataset_name:Literal['CUB_312', 'CUB_112'], mode: Literal["test"] = "test"):
        self.root_dir = root_dir
        self.transform = transform
        self.reference_dataset_name = reference_dataset_name
        
        if reference_dataset_name == "CUB_112":
            self.reference_dataset = CUB_112(root_dir=root_dir, mode='test', concept_masks_scale=None, use_soft_labels=False, attr_level="image", transform=transform)
        elif reference_dataset_name == "CUB_312":
            self.reference_dataset = CUB_312(root_dir=root_dir, mode='test', concept_masks_scale=None, use_soft_labels=False, attr_level="image", transform=transform)
        else:
            raise ValueError(f"reference_dataset_name has to be 'CUB_112' or 'CUB_312', '{reference_dataset_name}' given")
        
        dataset = load_dataset("Jessica-bader/SUB")

        if mode not in dataset:
            raise ValueError(f"Mode {mode} not available. Use mode='test'")

        self.dataset: ArrowDataset = dataset[mode]  # type: ignore

        self.attr_feature = self.dataset.features["attr_label"]
        self.bird_feature = self.dataset.features["bird_label"]

        self.LABEL_PATH = LABEL_PATH 

    def __len__(self):
        return len(self.dataset)

    def get_attr_label_name(self, attr_label: int) -> str:
        return self.attr_feature.int2str(attr_label)

    def get_bird_label_name(self, bird_label: int) -> str:
        return self.bird_feature.int2str(bird_label)

    @property
    def attributes(self):
        return self.reference_dataset.concepts
        
    def get_base_attributes(self, base_class_idx:int) -> list[int]:
        
        data = pickle.load(open(self.LABEL_PATH, 'rb'))

        for row in data:
            if row['class_label'] == base_class_idx:
                return row['attribute_label']
        raise ValueError(f'Did not find label for the class: {base_class_idx}')
    

        
    def get_bird_idx_by_name(self, bird_name: str):

        mask = self.reference_dataset.classes["class_name"].str.contains(bird_name, case=False)
        result = self.reference_dataset.classes.loc[mask, "class_id"]

        return result.iloc[0] if not result.empty else None
        
    def get_removed_attribute_index(self, base_class_idx:int, used_attr_name:str, used_attribute_names:list[str]):
        attr_type = used_attr_name.split('::')[0]
        to_check = [ua for ua in used_attribute_names if attr_type in ua]
        attr_label = self.get_base_attributes(base_class_idx)
        
        for tc in to_check:
            if attr_label[used_attribute_names.index(tc)]:
                return used_attribute_names.index(tc)
        return None

    def __getitem__(self, idx):
        sample: dict = self.dataset[idx]

        image_np = np.array(sample["image"])

        augmented = self.transform(
            image=image_np,
        )
        image: NDArray = augmented["image"].transpose(2, 0, 1)

        return {
            "image": image,
            "used_attr_label": sample["attr_label"],
            "used_attr_name": self.get_attr_label_name(sample["attr_label"]),
            "bird_label": sample["bird_label"],
            "bird_name": self.get_bird_label_name(sample["bird_label"]),
        }
