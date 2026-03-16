import argparse
import math
import os

import pandas as pd

from architecture.classification_modules import __all__ as CLASSIFIER_MODULE_CHOICES
from architecture.concept_modules import __all__ as CONCEPT_MODULE_CHOICES
from architecture.segmentation_modules import __all__ as SEGMENTATION_MODULE_CHOICES
from architecture.unified_models import __all__ as UNIFIED_MODULE_CHOICES
from architecture.upsampler_modules import __all__ as UPSAMPLER_MODULE_CHOICES
from cbm_datasets import __datasets__ as DATASET_CHOICES
from utils.attribution_methods import __all__ as ATTRIBUTOR_CHOICES
from utils.loss import __all__ as CRITERION_CHOICES


def _is_set(value) -> bool:
    """True if CSV value is set (non-NaN / not empty)."""
    if value is None:
        return False
    if isinstance(value, float) and math.isnan(value):
        return False
    if isinstance(value, str) and value.strip() == "":
        return False
    return True

# ------------------------------
# CBM Extended Argument Parser
# ------------------------------
def cbm_extended_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-id", type=int, required=False)
    parser.add_argument("--dataset", choices=DATASET_CHOICES, type=str, required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--num-workers", type=int, required=False, default=22)
    parser.add_argument("--lr", type=float, required=False, default=1e-4)
    parser.add_argument("-weight-decay", type=float, required=False, default=1e-4)
    parser.add_argument("--affinity-num-samples", type=int, required=False)
    parser.add_argument("--affinity-sim-threshold", type=float, required=False)
    parser.add_argument("--top-k-percent", type=float, required=False)
    parser.add_argument("--epochs", type=int, required=False, default=10)
    parser.add_argument("--img-size", type=int, required=False, default=256)
    parser.add_argument("--encoder-name", choices=["dinov3"], type=str, required=False)
    parser.add_argument(
        "--upsampler-name", choices=UPSAMPLER_MODULE_CHOICES, type=str, required=False
    )
    parser.add_argument(
        "--segmentation-module-name",
        choices=SEGMENTATION_MODULE_CHOICES,
        type=str,
        required=False,
    )
    parser.add_argument(
        "--concept-module-name",
        choices=CONCEPT_MODULE_CHOICES,
        type=str,
        required=True,
    )
    parser.add_argument(
        "--classification-module-name",
        choices=CLASSIFIER_MODULE_CHOICES,
        type=str,
        required=True,
    )
    parser.add_argument(
        "--unified-name",
        choices=UNIFIED_MODULE_CHOICES,
        type=str,
        required=False,
        default=None,
    )
    parser.add_argument(
        "--dino-ckpt-segdino",
        type=str,
        required=False,
    )
    parser.add_argument(
        "--freeze-encoder",
        action="store_true",
        help="Whether to freeze the encoder weights during training.",
    )
    parser.add_argument(
        "--freeze-upsampler",
        action="store_true",
        help="Whether to freeze the upsampler weights during training.",
    )
    parser.add_argument(
        "--checkpoint-path",
        type=str,
        required=False,
        default=None,
        help="Path to a checkpoint to resume training from.",
    )

    parser.add_argument(
        "--lambda-mask-loss",
        type=float,
        required=False,
    )
    parser.add_argument(
        "--lambda-concept-loss",
        type=float,
        required=False,
    )
    parser.add_argument(
        "--lambda-concept-reg-loss",
        type=float,
        required=False,
    )
    parser.add_argument(
        "--lambda-affinity-loss",
        type=float,
        required=False,
    )
    parser.add_argument(
        "--lambda-mask-reg-loss",
        type=float,
        required=False,
    )
    parser.add_argument(
        "--lambda-tv-loss",
        type=float,
        required=False,
    )
    parser.add_argument(
        "--lambda-classification-loss",
        type=float,
        required=False,
    )
    parser.add_argument(
        "--concept-criterion", type=str, required=False, choices=["BCEWithLogitsLoss", "BCELoss", "L1WithLogitsLoss", "BCEWithLogitsCertaintiesLoss"]
    )
    parser.add_argument(
        "--concept-reg-criterion", type=str, required=False, choices=CRITERION_CHOICES
    )
    parser.add_argument("--mask-criterion", type=str, required=False, choices=CRITERION_CHOICES)
    parser.add_argument("--affinity-criterion", type=str, required=False)
    parser.add_argument("--mask-reg-criterion", type=str, required=False)
    parser.add_argument("--tv-criterion", type=str, required=False)
    parser.add_argument("--classification-criterion", type=str, required=False)
    parser.add_argument(
        "--log-gradients",
        action="store_true",
        help="Whether to log gradients during training.",
    )
    parser.add_argument(
        "--calculate-dice",
        action="store_true",
    )
    parser.add_argument(
        "--calculate-fg-dice",
        action="store_true",
    )
    parser.add_argument(
        "--calculate-iou",
        action="store_true",
    )
    parser.add_argument(
        "--image-every-n-steps",
        default=50,
        type=int,
        required=False,
    )
    parser.add_argument(
        "--concept-masks-scale",
        choices=["small", "medium", "large"],
        type=str,
        required=False,
        default=None,
    )

    parser.add_argument(
        "--use-soft-labels",
        action="store_true",
        help="Whether to use soft labels for concept targets.",
    )

    parser.add_argument(
        "--n-concepts-implicitly-learned",
        type=int,
        required=False,
        default=None
    )

    parser.add_argument(
        "--attr-level",
        choices=["image", "class"],
        type=str,
        required=False,
        default="image"
    )

    return parser

# ------------------------------
# CBM Extended Argument Validator
# ------------------------------
def cbm_extended_argument_validator(args: argparse.Namespace):
    
    # ----------------------------
    # Concept loss
    # ----------------------------
    if args.concept_criterion and args.lambda_concept_loss is None:
        raise ValueError("--concept-criterion requires --lambda-concept-loss")

    if args.lambda_concept_loss is not None and not args.concept_criterion:
        raise ValueError("--lambda-concept-loss requires --concept-criterion")

    if args.lambda_mask_loss is not None and not args.mask_criterion:
        raise ValueError("--lambda-mask-loss requires --mask-criterion")

    # ----------------------------
    # Affinity loss
    # ----------------------------
    if args.affinity_criterion and args.lambda_affinity_loss is None:
        raise ValueError("--affinity-criterion requires --lambda-affinity-loss")

    if args.lambda_affinity_loss is not None and not args.affinity_criterion:
        raise ValueError("--lambda-affinity-loss requires --affinity-criterion")

    # ----------------------------
    # Regularization losses
    # ----------------------------
    if args.concept_reg_criterion and args.lambda_concept_reg_loss is None:
        raise ValueError("--concept-reg-criterion requires --lambda-concept-reg-loss")

    if args.mask_reg_criterion and args.lambda_mask_reg_loss is None:
        raise ValueError("--mask-reg-criterion requires --lambda-mask-reg-loss")

    if args.tv_criterion and args.lambda_tv_loss is None:
        raise ValueError("--tv-criterion requires --lambda-tv-loss")

    # ----------------------------
    # Classification loss
    # ----------------------------
    if args.classification_criterion and args.lambda_classification_loss is None:
        raise ValueError("--classification-criterion requires --lambda-classification-loss")

    if args.lambda_classification_loss is not None and not args.classification_criterion:
        raise ValueError("--lambda-classification-loss requires --classification-criterion")

    # ----------------------------
    # Unified module consistency
    # ----------------------------
    if args.unified_name and args.segmentation_module_name:
        raise ValueError("--unified-name cannot be combined with segmentation_module_name")

    if args.concept_module_name == "LogitMeanTopK" and not args.top_k_percent:
        raise ValueError("concept_module_name 'LogitMeanTopK' requires --top-k-percent to be set")
    
    if args.top_k_percent and args.concept_module_name != "LogitMeanTopK":
        raise ValueError("--top-k-percent requires concept_module_name 'LogitMeanTopK'")

# ------------------------------
# CBM Extended Argument Config (CSV-based)
# ------------------------------
def cbm_extended_argument_config(slurm_id:int):
    # Read CSV
    df = pd.read_csv("test_grid_experiments_extended.csv", sep=";", keep_default_na=False)

    if slurm_id >= len(df):
        raise ValueError(f"SLURM_ID {slurm_id} is larger than the number of experiments ({len(df)})")

    row = df.iloc[slurm_id]

    cmd = [
        "uv",
        "run",
        "train_cbm_extended.py",
        "--test-id" ,
        str(row.test_id),
        "--dataset",
        str(row.dataset),
        "--batch-size",
        str(row.batch_size),
        "--lr",
        str(row.lr),
        "--epochs",
        str(row.epochs),
        "--concept-module-name",
        str(row.concept_module_name),
        "--classification-module-name",
        str(row.classification_module_name),
        "--affinity-num-samples", "1000",
    ]

    # -------------------------
    # Boolean flags
    # -------------------------

    if _is_set(row.freeze_encoder):
        cmd.append(str(row.freeze_encoder))

    if _is_set(row.freeze_upsampler):
        cmd.append(str(row.freeze_upsampler))

    if _is_set(row.use_soft_labels):
        cmd.append(str(row.use_soft_labels))
        
    # -------------------------
    # Optional criteria
    # -------------------------

    if _is_set(row.n_concept_implicitely_learned):
        cmd.extend(["--n-concepts-implicitly-learned", str(row.n_concept_implicitely_learned)])

    if _is_set(row.mask_criterion):
        cmd.extend(["--mask-criterion", str(row.mask_criterion)])

    if _is_set(row.affinity_criterion):
        cmd.extend(["--affinity-criterion", str(row.affinity_criterion)])

    if _is_set(row.mask_reg_criterion):
        cmd.extend(["--mask-reg-criterion", str(row.mask_reg_criterion)])

    if _is_set(row.tv_criterion):
        cmd.extend(["--tv-criterion", str(row.tv_criterion)])

    if _is_set(row.classification_criterion):
        cmd.extend(["--classification-criterion", str(row.classification_criterion)])

    if _is_set(row.concept_criterion):
        cmd.extend(["--concept-criterion", str(row.concept_criterion)])

    if _is_set(row.lambda_concept_reg_loss):
        cmd.extend(["--lambda-concept-reg-loss", str(row.lambda_concept_reg_loss)])

    if _is_set(row.concept_reg_criterion):
        cmd.extend(["--concept-reg-criterion", str(row.concept_reg_criterion)])

    if _is_set(row.lambda_mask_loss):
        cmd.extend(["--lambda-mask-loss", str(row.lambda_mask_loss)])

    if _is_set(row.lambda_concept_loss):
        cmd.extend(["--lambda-concept-loss", str(row.lambda_concept_loss)])

    if _is_set(row.affinity_criterion):
        cmd.extend(["--affinity-criterion", str(row.affinity_criterion)])

    if _is_set(row.lambda_classification_loss):
        cmd.extend(["--lambda-classification-loss", str(row.lambda_classification_loss)])

    if _is_set(row.lambda_affinity_loss):
        cmd.extend(["--lambda-affinity-loss", str(row.lambda_affinity_loss)])


    if _is_set(row.lambda_mask_reg_loss):
        cmd.extend(["--lambda-mask-reg-loss", str(row.lambda_mask_reg_loss)])

    if _is_set(row.lambda_tv_loss):
        cmd.extend(["--lambda-tv-loss", str(row.lambda_tv_loss)])

    if _is_set(row.upsampler_name):
        cmd.extend(["--upsampler-name", str(row.upsampler_name)])

    if _is_set(row.affinity_sim_threshold):
        cmd.extend(["--affinity-sim-threshold", str(row.affinity_sim_threshold)])

    if _is_set(row.encoder_name):
        cmd.extend(["--encoder-name", str(row.encoder_name)])

    if _is_set(row.segmentation_module_name):
        cmd.extend(["--segmentation-module-name", str(row.segmentation_module_name)])

    if _is_set(row.unified_name):
        cmd.extend(["--unified-name", str(row.unified_name)])

    if _is_set(row.dino_ckpt_segdino):
        cmd.extend(["--dino-ckpt-segdino", str(row.dino_ckpt_segdino)])

    if _is_set(row.top_k_percent):
        cmd.extend(["--top-k-percent", str(row.top_k_percent)])

    if _is_set(row.attr_level):
        cmd.extend(["--attr-level", str(row.attr_level)])

    if "CUB" in row.dataset:
        cmd.extend(["--concept-masks-scale", "small"])

    # -------------------------
    # Boolean Flags for evaluation
    # -------------------------

    if str(row.calculate_dice).lower() == "true":
        cmd.extend(
            [
                "--calculate-dice",
                "--calculate-fg-dice",
                "--calculate-iou",
            ]
        )

    return cmd

# ------------------------------
# CBM EPG Argument Parser
# ------------------------------
def cbm_epg_argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-id", required=False, type=int)
    parser.add_argument(
        "--attributor", required=False, type=str, choices=ATTRIBUTOR_CHOICES, help="attributor"
    )
    parser.add_argument(
        "--epg-lvl", required=False, type=str, choices=["image", "concept"], help="epg level"
    )
    parser.add_argument(
        "--dataset", choices=DATASET_CHOICES, required=True, type=str, help="dataset"
    )
    parser.add_argument("--batch-size", required=False, type=int, default=4, help="batch size")
    parser.add_argument("--target-batch-size", required=False, type=int, default=64)
    parser.add_argument(
        "--lambda-class", required=True, type=float, help="lambda class loss"
    )
    parser.add_argument(
        "--lambda-concept", required=True, type=float, help="lambda concept loss"
    )
    parser.add_argument("--lambda-epg", required=False, type=float, help="lambda EPG loss")
    parser.add_argument("--lr", required=False, type=float, default=1e-4, help="learning rate")
    parser.add_argument("--epochs", required=False, type=int, default=10, help="number of epochs")
    parser.add_argument(
        "--weight-decay", required=False, type=float, default=1e-4, help="weight decay"
    )
    parser.add_argument(
        "--num-workers", required=False, type=int, default=30, help="number of workers"
    )
    parser.add_argument(
        "--wandb-project",
        required=False,
        default="master-thesis-playground",
        type=str,
        help="wandb project name",
    )
    parser.add_argument(
        "--img-size",
        required=False,
        default=256,
        type=int,
        help="if set, runs in debug mode with less data",
    )
    parser.add_argument(
        "--classification-criterion", required=True, choices=CRITERION_CHOICES, type=str
    )
    parser.add_argument("--concept-criterion", required=True, choices=CRITERION_CHOICES, type=str)
    parser.add_argument(
        "--concept-masks-scale", required=False, choices=["small", "medium", "large"], type=str
    )
    parser.add_argument(
        "--root-dir-dataset",
        required=False,
        type=str,
        default=os.environ["ROOT_DIR_DATASET"],
        help="root directory of datasets",
    )

    parser.add_argument(
        "--use-soft-labels",
        action="store_true",
        help="Whether to use soft labels for concept targets.",
    )

    parser.add_argument(
        "--attr-level",
        choices=["image", "class"],
        type=str,
        required=False,
        default="image"
    )

    return parser

# ------------------------------
# CBM EPG Argument Validator
# ------------------------------
def cbm_epg_argument_validator(args: argparse.Namespace):

    if args.attributor is None and args.epg_lvl is not None:
        raise ValueError("--epg-lvl requires --attributor to be set")
    
    if args.attributor is not None and args.epg_lvl is None:
        raise ValueError("--attributor requires --epg-lvl to be set")
    
    if args.lambda_concept is None and args.concept_criterion is not None:
        raise ValueError("--concept-criterion requires --lambda-concept")
    
    if args.lambda_concept is not None and args.concept_criterion is None:
        raise ValueError("--lambda-concept requires --concept-criterion")
    
    if args.lambda_class is None and args.classification_criterion is not None:
        raise ValueError("--classification-criterion requires --lambda-class")
    
    if args.lambda_class is not None and args.classification_criterion is None:
        raise ValueError("--lambda-class requires --classification-criterion")

# ------------------------------
# CBM EPG Argument Config (CSV-based)
# ------------------------------
def cbm_epg_argument_config(slurm_id:int):

    # Read CSV
    df = pd.read_csv("test_grid_experiments_epg.csv", sep=";")

    # Select row based on SLURM_ID
    if slurm_id >= len(df):
        raise ValueError(f"SLURM_ID {slurm_id} is larger than the number of experiments ({len(df)})")

    row = df.iloc[slurm_id]

    cmd = [
        "uv", "run", "train_epg.py",
        "--test-id", str(row.test_id),
        "--dataset", str(row.dataset),
        "--batch-size", str(row.batch_size),
        "--lambda-epg", str(row.lambda_epg),
        "--lambda-class", str(row.lambda_class),
        "--lambda-concept", str(row.lambda_concept),
        "--epochs", str(row.epochs),
        "--img-size", str(row.img_size),
        "--classification-criterion", str(row.classification_criterion),
        "--concept-criterion", str(row.concept_criterion),
        "--wandb-project", "epg",
        "--concept-masks-scale", "small"
    ]

    if _is_set(row.attributor):
        cmd.extend(["--attributor", str(row.attributor)])

    if _is_set(row.epg_lvl):
        cmd.extend(["--epg-lvl", str(row.epg_lvl)])

    return cmd