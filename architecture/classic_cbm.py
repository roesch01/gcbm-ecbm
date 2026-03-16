import torch
import torch.nn as nn
from torchvision import models


class ClassicCBM(nn.Module):
    def __init__(self, n_concepts: int, n_classes: int, backbone: nn.Module):
        super().__init__()
        self.n_concepts = n_concepts
        self.n_classes = n_classes
        self.backbone = backbone

        # Map concept logits [B, n_concepts] -> class logits [B, n_classes]
        self.classifier = nn.Linear(n_concepts, n_classes)


class ResNet50_CBM(ClassicCBM):
    def __init__(self, n_concepts: int, n_classes: int, weights: str | None = None):
        # Backbone: ResNet50 until final conv layer
        res = models.resnet50(weights=weights)
        backbone = nn.Sequential(
            res.conv1,
            res.bn1,
            res.relu,
            res.maxpool,
            res.layer1,
            res.layer2,
            res.layer3,
            res.layer4,
        )

        super().__init__(n_concepts=n_concepts, n_classes=n_classes, backbone=backbone)
        self.backbone_out_channels = 2048

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc_concepts = nn.Linear(self.backbone_out_channels, n_concepts)

    def forward(self, x: torch.Tensor):
        # x: [B, 3, H, W]
        feats = self.backbone(x)  # [B, 2048, H/32, W/32]

        pooled = self.pool(feats).flatten(1)  # [B, 2048]
        concept_logits = self.fc_concepts(pooled)  # [B, n_concepts]
        concept_probs = torch.sigmoid(concept_logits)   # [B, n_concepts]
        class_logits = self.classifier(concept_logits)  # [B, n_classes]

        return class_logits, concept_logits, feats
    

def init_from_checkpoint(run_id: str, epoch:int, dataset: str, device: torch.device | str) -> ClassicCBM:
    
    checkpoint_path = f"/pfs/work9/workspace/scratch/ma_faroesch-master-thesis/blobs/checkpoints/cbm_epg_{dataset}_run_{run_id}_epoch{epoch}.pth"
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    n_classes = 200 if dataset == 'cub_112' else 50
    n_concepts = 112 if dataset == 'cub_112' else 26
    
    model = ResNet50_CBM(n_concepts=n_concepts, n_classes=n_classes)
    
    missing, unexpected = model.load_state_dict(
        checkpoint['model_state_dict'],
        strict=True
    )

    return model.to(device)