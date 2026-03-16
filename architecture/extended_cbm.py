from dataclasses import dataclass

import torch
import torch.nn as nn

from architecture.classification_modules import (
    ClassifcationModule,
    ClassificationOutput,
    ClassificationOutputNumpy,
    get_classification_module_by_name,
)
from architecture.concept_modules import (
    ConceptModule,
    ConceptOutput,
    ConceptOutputNumpy,
    get_concept_module_by_name,
)
from architecture.encoder_modules import EncoderOutput, EncoderOutputNumpy, get_encoder_by_name
from architecture.segmentation_modules import (
    SegmentationModule,
    SegmentationOutput,
    SegmentationOutputNumpy,
    get_segmentation_module_by_name,
)
from architecture.unified_models import get_unified_model_by_name
from architecture.upsampler_modules import (
    UpsamplerOutput,
    UpsamplerOutputNumpy,
    get_upsampler_by_name,
)


def get_cbm_wrapper(
    encoder_name: str | None,
    upsampler_name: str | None,
    freeze_encoder: bool,
    freeze_upsampler: bool,
    unified_name: str | None,
    segmentation_module_name: str | None,
    concept_module_name: str,
    classification_module_name: str,
    n_concepts: int,
    n_classes: int,
    top_k_percent: float,
    dino_ckpt_segdino: str | None,
    img_size: int,
):
    if unified_name is not None:
        if dino_ckpt_segdino is None:
            raise ValueError(
                "--dino-ckpt-segdino is None, but it has to be a path to the checkpoint"
            )
        unified_model = get_unified_model_by_name(
            name=unified_name, dino_ckpt_segdino=dino_ckpt_segdino, n_concepts=n_concepts
        )
        encoder = None
        upsampler = None
        segmentation_module = None

    else:
        if segmentation_module_name is None:
            raise ValueError("--segmentation-module-name has to be set. (Currently it is None)")

        unified_model = None
        if encoder_name:
            encoder = get_encoder_by_name(encoder_name)
        else:
            raise ValueError(f"encoder_name has to be one of ['dinov3']. '{encoder_name}' given")

        if upsampler_name is None:
            upsampler = None
        else:
            upsampler = get_upsampler_by_name(name=upsampler_name)

        if freeze_encoder:
            for p in encoder.parameters():
                p.requires_grad = False

        if freeze_upsampler and upsampler is not None:
            for p in upsampler.parameters():
                p.requires_grad = False

        segmentation_module = get_segmentation_module_by_name(
            name=segmentation_module_name, feature_dim=768, n_concepts=n_concepts, img_size=img_size
        )

    concept_extractor = get_concept_module_by_name(
        name=concept_module_name,
        top_k_percent=top_k_percent,
        n_concepts=n_concepts
    )
    classifier = get_classification_module_by_name(
        name=classification_module_name, n_concepts=n_concepts, n_classes=n_classes
    )

    model = CBMWrapper(
        # Unified model
        unified_model=unified_model,
        # Encoder
        encoder=encoder,
        encoder_name=encoder_name,
        # Upsampler
        upsampler=upsampler,
        upsampler_name=upsampler_name,
        # Segmentation Head
        segmentation_module=segmentation_module,
        # Concept module
        concept_extractor=concept_extractor,
        # Classifier module
        classification_module=classifier,
        # No grad flags
        encoder_no_grad=True,
        upsampler_no_grad=True,
    )

    return model



@dataclass
class ExtendedCBMOutputNumpy:
    encoder_module: EncoderOutputNumpy
    upsampler_module: UpsamplerOutputNumpy | None
    segmentation_module: SegmentationOutputNumpy
    concept_module: ConceptOutputNumpy
    classification_module: ClassificationOutputNumpy


@dataclass
class ExtendedCBMOutput:

    encoder_module: EncoderOutput
    upsampler_module: UpsamplerOutput | None
    segmentation_module: SegmentationOutput
    concept_module: ConceptOutput
    classification_module: ClassificationOutput

    def to_numpy(self) -> ExtendedCBMOutputNumpy:

        return ExtendedCBMOutputNumpy(
            encoder_module=self.encoder_module.to_numpy(),
            upsampler_module=None if self.upsampler_module is None else self.upsampler_module.to_numpy(),
            segmentation_module=self.segmentation_module.to_numpy(),
            concept_module=self.concept_module.to_numpy(),
            classification_module=self.classification_module.to_numpy(),
        )



class CBMWrapper(nn.Module):
    """
    Main architecture combining encoder, upsampler, segmentation, and classification.
    Supports separate encoder-decoder pairs or unified models.
    """

    def __init__(
        self,
        concept_extractor: ConceptModule,
        classification_module: ClassifcationModule,
        encoder: nn.Module | None = None,
        encoder_name: str | None = None,
        upsampler: nn.Module | None = None,
        upsampler_name: str | None = None,
        segmentation_module: SegmentationModule | None = None,
        unified_model: nn.Module | None = None,
        encoder_no_grad: bool = True,
        upsampler_no_grad: bool = True,
    ):
        super().__init__()

        assert (encoder is not None and segmentation_module is not None) or (
            unified_model is not None
        ), "Provide either encoder-decoder pair or unified_model"

        self.use_unified_model = unified_model is not None
        self.no_grad_encoder = encoder_no_grad
        self.no_grad_upsampler = upsampler_no_grad
        self.encoder_name = encoder_name
        self.upsampler_name = upsampler_name

        

        self.encoder = encoder
        self.upsampler = upsampler
        self.segmentation = segmentation_module
        self.unified = unified_model
        self.concept_extractor = concept_extractor
        self.classifier = classification_module

        

    def forward_segmentation(
        self, x: torch.Tensor, 
    ):
        # x: [B, 3, H, W]

        if self.unified:
            # segmentation_masks: [B, num_concepts, H, W], features: [B, C, h, w]
            segmentation_masks, features = self.unified(x)
            return EncoderOutput(), UpsamplerOutput(), SegmentationOutput(mask_logits=segmentation_masks)



        # Encoder pass
        if not self.encoder or not self.segmentation:
            raise ValueError("Encoder or Segmentor not available")
        if self.no_grad_encoder:
            with torch.no_grad():
                features = self.encoder(x)
        else:
            features = self.encoder(x)

        if self.encoder_name == "dinov3":
            features = features[0]  # [B, C, h, w]

        assert features.ndim == 4, f"Expected [B, C, H, W], got {features.shape}"
        
        encoder_output = EncoderOutput(features=features)

        # Upsampler pass
        if self.upsampler:
            if self.upsampler_name == "loftup":
                args = {"lr_feats": features, "img": x}
            else:
                args = (x, features)

            context = torch.no_grad() if self.no_grad_upsampler else torch.enable_grad()
            with context:
                if isinstance(args, dict):
                    features = self.upsampler(**args)
                else:
                    features = self.upsampler(*args)

        upsampler_output = UpsamplerOutput(features=features)
            
        # Generate segmentation masks: [B, num_concepts, H, W]
        segmentation_output = self.segmentation(encoder_output, upsampler_output)
        return encoder_output, upsampler_output, segmentation_output

    def forward_concept_extractor(
        self, segmentation_output: SegmentationOutput
    ):
        # Input: results dictionary containing segmentation masks
        # Output: results with "concepts.concept_logits" [B, num_concepts]
        return self.concept_extractor(segmentation_output)

    def forward_classifier(self, concept_output: ConceptOutput):
        return self.classifier(concept_output)

    def forward(self, x: torch.Tensor):
        """
        Full CBM Pipeline: Image -> Segmentation -> Concepts -> Class
        x: [B, 3, H, W]
        """
        

        encoder_output, upsampler_output, segmentation_output = self.forward_segmentation(x)
        concept_output = self.forward_concept_extractor(segmentation_output)
        classification_output = self.forward_classifier(concept_output)

        results = ExtendedCBMOutput(
            encoder_module=encoder_output,
            upsampler_module=upsampler_output,
            segmentation_module=segmentation_output,
            concept_module=concept_output,
            classification_module=classification_output
        )

        return results



def init_from_checkpoint(run_id: str, epoch:int, dataset: str, device: torch.device | str) -> CBMWrapper:
    checkpoint_path = f"/pfs/work9/workspace/scratch/ma_faroesch-master-thesis/blobs/checkpoints/cbm_anyup_dinov3_{dataset}_run_{run_id}_epoch{epoch}.pth"
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    config = checkpoint['config']

    upsampler_name = config['architecture']['upsampler']
    unified_name = config['architecture']['unified_model']
    segmentation_module_name = config['architecture']['segmentation_module']
    classification_module_name = config['architecture']['classification_module']
    n_classes = 200 if dataset == 'cub_112' else 50
    n_concepts = config['n_concepts'] if 'n_concepts' in config.keys() else 112
    top_k_percent = config['hyperparameters']['top_k_percent']
    concept_module_name = config['architecture']['concept_module']
    dino_ckpt_segdino = 'dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth'
    img_size = 256

    

    model = get_cbm_wrapper(encoder_name='dinov3', upsampler_name=upsampler_name, freeze_encoder=True,
        freeze_upsampler=True, unified_name=unified_name, 
        segmentation_module_name=segmentation_module_name, 
        classification_module_name=classification_module_name,
        n_classes=n_classes, 
        n_concepts=n_concepts, 
        top_k_percent=top_k_percent,
        concept_module_name=concept_module_name, 
        dino_ckpt_segdino=dino_ckpt_segdino, 
        img_size=img_size
        )
    
    missing, unexpected = model.load_state_dict(
        checkpoint['model_state_dict'],
        strict=True
    )

    return model.to(device)