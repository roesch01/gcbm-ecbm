from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from segmentation_models_pytorch.losses import DiceLoss, TverskyLoss

import wandb
from architecture.extended_cbm import ExtendedCBMOutput
from cbm_datasets.types import Batch

__all__ = [
    "TVLoss",
    "EntropySharpeningLoss",
    "BCELoss",
    "BCEWithLogitsLoss",
    "DiceLoss",
    "TverskyLoss",
    "CELossFunnyBirds",
    "WSSSAffinityLoss",
    "GaussianAffinityLoss",
    "VectorizedFeatureSimilarityMaskConsistencyLoss",
    "CELossOneHot",
    "EPGLoss",
    "L1WithLogitsLoss",
    "BCEWithLogitsCertaintiesLoss",
]



def get_criterion(criterion_name: str | None, pos_weights: torch.Tensor | None = None, **kwargs) -> nn.Module | None:
    if criterion_name is None:
        return None

    if criterion_name == "TVLoss":
        return TVLoss()

    if criterion_name == "EntropySharpeningLoss":
        return EntropySharpeningLoss()

    if criterion_name == "BCELoss":
        return nn.BCELoss()

    if criterion_name == "BCEWithLogitsLoss":
        return nn.BCEWithLogitsLoss(pos_weight=pos_weights)
    
    if criterion_name == "BCEWithLogitsCertaintiesLoss":
        if pos_weights is None:
            raise ValueError("pos_weights must be provided for BCEWithLogitsCertaintiesLoss")
        return BCEWithLogitsCertaintiesLoss(pos_weight=pos_weights)

    if criterion_name == "DiceLoss":
        return DiceLoss(mode="multilabel")

    if criterion_name == "TverskyLoss":
        return TverskyLoss(mode="multilabel")

    if criterion_name == "CELossFunnyBirds":
        return CELossFunnyBirds()

    if criterion_name == "WSSSAffinityLoss":
        return WSSSAffinityLoss(num_samples=kwargs["affinity_num_samples"], pos_sim_threshold=0.95)
    
    if criterion_name == "GaussianAffinityLoss":
        return GaussianAffinityLoss(
            num_steps=kwargs.get("affinity_n_steps", 16),
            samples_per_step=kwargs.get("affinity_num_samples", 512),
            sigma=kwargs.get("affinity_sigma", 32.0),
            sim_threshold=kwargs.get("affinity_sim_threshold", 0.995),
        )
    
    if criterion_name == "VectorizedFeatureSimilarityMaskConsistencyLoss":
        return VectorizedFeatureSimilarityMaskConsistencyLoss(sim_threshold=kwargs.get("consistency_sim_threshold", 0.995))

    if criterion_name == "EPGLoss":
        return EPGLoss()

    if criterion_name == "CELossOneHot":
        return CELossOneHot()

    if criterion_name == "L1WithLogitsLoss":
        return L1WithLogitsLoss()

    raise ValueError(f"Criterion not valid. '{criterion_name}' given")


# Extractor is a function: (Outputs, Targets) -> Tensor
Extractor = Callable[['ExtendedCBMOutput', 'Batch'], torch.Tensor]

@dataclass
class Parameter:
    get_value: Extractor
    name: str = "param" # Optional

# --- Helper Function for cleaner code

def FromOutput(selector: Callable[['ExtendedCBMOutput'], Any], name: str = "out") -> Parameter:
    """Holt einen Wert aus dem Model-Output (ExtendedCBMOutput)."""
    return Parameter(get_value=lambda out, tgt: selector(out), name=name)

def FromTarget(selector: Callable[['Batch'], Any], name: str = "tgt") -> Parameter:
    """Holt einen Wert aus dem Ground-Truth Batch."""
    return Parameter(get_value=lambda out, tgt: selector(tgt), name=name)

@dataclass
class Task:
    name: str
    loss_fn: nn.Module | None
    parameters: list[Parameter]
    weight: float = 1.0

    @property
    def active(self) -> bool:
        return self.loss_fn is not None


class MultiTaskLoss(nn.Module):
    def __init__(
        self,
        lr: float,
        device: str,
        tasks: list[Task],
    ):
        
        super().__init__()
        self.lr = lr
        self.tasks = tasks
        self.device = device

    def forward(self, outputs: ExtendedCBMOutput, targets: Batch):
        total_loss = torch.tensor(0.0, device=self.device)
        loss_details = {}

        for task in self.tasks:
            if not task.active or not task.loss_fn:
                continue
            
            # 1. Gather parameters
            # We pass both outputs AND targets to each parameter.
            # The parameter itself knows what it needs (thanks to the lambda function).
            try:
                params = [param.get_value(outputs, targets) for param in task.parameters]
            except AttributeError as e:
                # Since we use lambdas, we get the exact Python errors here
                # e.g., "'NoneType' has no attribute 'mask_logits'"
                print(f"Error in Task {task.name}: {e}")
                raise e

            # 2. Compute loss
            base_loss = task.loss_fn(*params)

            # ... rest as usual ...
            weighted_loss = task.weight * base_loss
            total_loss += weighted_loss
            loss_details[f"loss_{task.name}"] = base_loss.detach().item()

        loss_details["total"] = total_loss.detach().item()
        return total_loss, loss_details


class EPGLoss(nn.Module):
    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(
        self,
        heatmaps: torch.Tensor,  # (B, C, H, W)
        masks: torch.Tensor,  # (B, C, H, W)
        concepts_active: torch.Tensor,  # (B, C) 0/1
    ) -> torch.Tensor:
        # activations within masks
        inside = (heatmaps * masks).sum(dim=(2, 3))  # (B, C)
        total = heatmaps.sum(dim=(2, 3)) + self.eps  # (B, C)
        epg_score = inside / total  # (B, C)

        # Mask only active concepts
        epg_score_active = epg_score * concepts_active  # (B, C) non-active → 0

        # number of active concepts in batch
        num_active = concepts_active.sum()

        # If no concept is active, loss is zero
        if num_active == 0:
            return heatmaps.sum() * 0

        # mean across active concepts
        mean_epg_score = epg_score_active.sum() / num_active
        return 1 - mean_epg_score  # to minimize loss


class L1WithLogitsLoss(nn.Module):
    def __init__(self):
        super().__init__()

        self.loss_fn = nn.L1Loss()

    def forward(self, pred: torch.Tensor, target: torch.Tensor | None = None):
        if target is None:
            target = torch.zeros_like(pred)
        return self.loss_fn(pred.sigmoid(), target)



class BCEWithLogitsCertaintiesLoss(nn.Module):
    """
    outputs: Logits [B, K]
    targets: Soft-Labels [B, K] (for example: 0.8)
    certainty_weights: concept_weights from dataloader [B, K]
    pos_weight_vector: Tensor with K values (6.58, 14.65, ...)
    """

    def __init__(self, pos_weight: torch.Tensor):
        super().__init__()
        self.pos_weight = pos_weight

    def forward(self, outputs: torch.Tensor, targets: torch.Tensor, certainty_weights: torch.Tensor):
        # 1. Calculate BCE with logits and Logits 
        # reduction='none' to calculate certainty
        loss = F.binary_cross_entropy_with_logits(
            outputs, 
            targets, 
            pos_weight=self.pos_weight, 
            reduction='none'
        )
        
        # 2. Certainty weighting
        weighted_loss = loss * certainty_weights
        
        # 3. Calculate normalized mean
        return weighted_loss.sum() / (certainty_weights.sum() + 1e-8)






class WSSSAffinityLoss(nn.Module):
    
    def __init__(
        self,
        num_samples: int = 2048,
        pos_sim_threshold: float = 0.95
    ):
        super().__init__()
        
        # Hyperparameter
        self.sim_threshold = pos_sim_threshold
        self.num_samples = num_samples
            

    def compute_affinity_loss(self, masks_probs, upsampler_features):
        B, K, H, W = masks_probs.shape
        S = self.num_samples
        
        # Flatten and Sample
        # (B, D, N) -> (B, D, S)
        indices = torch.stack([torch.randperm(H*W, device=masks_probs.device)[:S] for _ in range(B)])
        
        # Gather samples
        f_sample = torch.gather(upsampler_features.view(B, -1, H*W), 2, indices.unsqueeze(1).expand(-1, upsampler_features.size(1), -1))
        f_sample = F.normalize(f_sample, dim=1)
        
        m_sample = torch.gather(masks_probs.view(B, K, -1), 2, indices.unsqueeze(1).expand(-1, K, -1))

        # Batched Similarity: (B, S, D) @ (B, D, S) -> (B, S, S)
        sim_matrix = torch.bmm(f_sample.transpose(1, 2), f_sample)

        # Batched Mask Difference: (B, K, S, 1) - (B, K, 1, S) -> (B, K, S, S)
        m_diff = torch.abs(m_sample.unsqueeze(3) - m_sample.unsqueeze(2))

        # Positive Affinity
        pos_mask = (sim_matrix > self.sim_threshold).float().unsqueeze(1) # (B, 1, S, S)
        loss = (m_diff * pos_mask).sum() / (pos_mask.sum() + 1e-6)
        
        return loss

    def forward(
        self,
        masks_logits: torch.Tensor,
        upsampler_features: torch.Tensor,
    ) -> torch.Tensor:
        # 3. Feature Affinity Loss (Spatial Constraint)
        masks_probs = torch.sigmoid(masks_logits)

        affinity_loss = self.compute_affinity_loss(masks_probs, upsampler_features)

        return affinity_loss
    

class VectorizedFeatureSimilarityMaskConsistencyLoss(nn.Module):
    def __init__(self, sim_threshold:float, eps=1e-6):
        super().__init__()
        self.sim_threshold = sim_threshold
        self.eps = eps

    def forward(self, features:torch.Tensor, mask_logits:torch.Tensor):
        """
        features: (B, D, H, W)
        mask_logits: (B, K, H, W)
        returns: scalar loss
        """
        B, D, H, W = features.shape
        K = mask_logits.shape[1]
        N = H * W  # number of pixels

        # 1. Features normalization
        # (B, D, H, W) -> (B, D, N)
        features = F.normalize(features, dim=1, eps=self.eps)
        feat_flat = features.view(B, D, N)

        # 2. Flatten masks and compute probabilities
        # (B, K, H, W) -> (B, K, N)
        mask_flat = mask_logits.view(B, K, N)
        mask_prob = torch.sigmoid(mask_flat)

        # 3. Find the index of the maximum activation per mask
        # (B, K, N) -> argmax -> (B, K)
        max_idx = torch.argmax(mask_flat, dim=-1)

        # 4. Extract reference feature vectors (without loop!)
        # transpose feat_flat for gather: (B, D, N) -> (B, N, D)
        feat_trans = feat_flat.transpose(1, 2)
        # expand max_idx to include D dimension: (B, K, D)
        max_idx_expanded = max_idx.unsqueeze(-1).expand(-1, -1, D)
        # ref_vecs now contains exactly the features of the max_idx pixels: (B, K, D)
        ref_vecs = torch.gather(feat_trans, 1, max_idx_expanded)

        # 5. Compute cosine similarity (batch matrix multiplication)
        # (B, K, D) @ (B, D, N) -> (B, K, N)
        sim = torch.matmul(ref_vecs, feat_flat)

        # 6. Apply threshold
        # (B, K, N) boolean tensor
        high_sim_mask = sim > self.sim_threshold

        # 7. Extract reference mask values (sigmoid) at max_idx
        # (B, K, N) gather -> (B, K, 1)
        ref_values = torch.gather(mask_prob, 2, max_idx.unsqueeze(-1))

        # 8. Prepare MSE loss (squared difference)
        # Broadcasting automatically handles (B, K, N) - (B, K, 1)
        squared_diff = (mask_prob - ref_values) ** 2

        # Keep only the values above the threshold
        masked_se = squared_diff * high_sim_mask.float()

        # 9. Aggregate loss (analogous to total_loss / valid_count)
        # sum of errors per mask: (B, K)
        sum_se_per_map = masked_se.sum(dim=-1)
        # number of valid (high_sim) pixels per mask: (B, K)
        count_per_map = high_sim_mask.sum(dim=-1).float()

        # determine which masks have any pixels above the threshold
        valid_maps = count_per_map > 0

        # if there are no valid pixels in the whole batch
        if not valid_maps.any():
            # trick: multiply by 0.0 to preserve computational graph (gradient flow does not break)
            return features.sum() * 0.0

        # mean loss per valid mask (MSE)
        loss_per_valid_map = sum_se_per_map[valid_maps] / count_per_map[valid_maps]
        
        # average over all valid masks (equivalent to total_loss / valid_count)
        return loss_per_valid_map.mean()



class GaussianAffinityLoss(nn.Module):
    def __init__(
        self,
        num_steps: int = 16,          
        samples_per_step: int = 512, 
        sigma: float = 32.0,       
        sim_threshold: float = 0.9, 
        log_wandb: bool = False,
    ):
        super().__init__()
        self.num_steps = num_steps
        self.samples_per_step = samples_per_step
        self.sigma = sigma
        self.sim_threshold = sim_threshold
        self.log_wandb = log_wandb

    def sample_gaussian_indices(self, B, H, W, device):
        """
        Samples pixel indices from a Gaussian distribution centered around
        a randomly chosen mean for each image in the batch.

        Returns:
            flat_indices (Tensor): Shape (B, S), flattened pixel indices
                                   for each image in the batch.
        """
        # 1. Sample a random center (mean) for each image in the batch
        # Shape: (B, 1) for y- and x-coordinates
        center_y = torch.randint(0, H, (B, 1), device=device).float()
        center_x = torch.randint(0, W, (B, 1), device=device).float()

        # 2. Generate Gaussian offsets around the center
        # Shape: (B, samples_per_step)
        offset_y = torch.randn(B, self.samples_per_step, device=device) * self.sigma
        offset_x = torch.randn(B, self.samples_per_step, device=device) * self.sigma

        # 3. Compute sampled coordinates: center + offset
        sample_y = center_y + offset_y
        sample_x = center_x + offset_x

        # 4. Clamp coordinates to ensure they stay within image bounds
        sample_y = torch.clamp(sample_y, 0, H - 1).long()
        sample_x = torch.clamp(sample_x, 0, W - 1).long()

        # 5. Convert 2D coordinates to flattened indices (y * W + x)
        # Useful for gather or advanced indexing
        flat_indices = sample_y * W + sample_x
        
        return flat_indices

    def compute_step_loss(self, masks_probs, features, indices):
        """
        Computes the loss for a given set of sampled indices.
        """
        B, K, H, W = masks_probs.shape
        D = features.shape[1]

        # Flatten spatial dimensions
        # Features: (B, D, H, W) -> (B, D, N)
        # Masks:    (B, K, H, W) -> (B, K, N)
        feats_flat = features.view(B, D, -1)   # (B, D, N)
        masks_flat = masks_probs.view(B, K, -1) # (B, K, N)

        # Gather sampled features and mask probabilities using batched indexing
        # indices shape: (B, S)
        # Expand indices to match feature/mask dimensions for torch.gather
        idx_expanded_feat = indices.unsqueeze(1).expand(-1, D, -1)
        f_sample = torch.gather(feats_flat, 2, idx_expanded_feat) # (B, D, S)
        
        idx_expanded_mask = indices.unsqueeze(1).expand(-1, K, -1)
        m_sample = torch.gather(masks_flat, 2, idx_expanded_mask) # (B, K, S)

        # Normalize sampled feature vectors along the feature dimension
        f_sample = F.normalize(f_sample, dim=1)

        # Compute pairwise feature similarity matrix
        # (B, S, D) x (B, D, S) -> (B, S, S)
        sim_matrix = torch.bmm(f_sample.transpose(1, 2), f_sample)

        # Compute absolute mask probability differences
        # Shape: (B, K, S, S)
        # Uses unsqueeze for broadcasting over sample pairs
        m_diff = (m_sample.unsqueeze(3) - m_sample.unsqueeze(2)).abs()

        # --- Positive Affinity (Attraction) ---
        # Select pairs with high feature similarity
        pos_mask = (sim_matrix > self.sim_threshold).float().unsqueeze(1) # (B, 1, S, S)
        
        # Aggregate mask differences over all positive pairs and samples
        pos_term = (m_diff * pos_mask).sum() / (pos_mask.sum() + 1e-6)

        return pos_term, pos_mask.sum().item()

    def forward(self, masks_probs, upsampler_features):
        """
        Runs the sampling-and-loss computation multiple times and
        aggregates the losses across steps.
        """
        affected_pixels = 0
        # Detach features to prevent gradients from flowing into the upsampler
        features = upsampler_features.detach()
        
        B, C, H, W = masks_probs.shape
        total_loss = torch.tensor(0.0, device=masks_probs.device)

        # Loop over the number of focus shifts / sampling steps
        for step in range(self.num_steps):
            # 1. Sample new indices (new random center and Gaussian distribution)
            indices = self.sample_gaussian_indices(B, H, W, masks_probs.device)
            
            # 2. Compute loss for the current step
            step_loss, affected_pixel = self.compute_step_loss(masks_probs, features, indices)
            
            # 3. Accumulate loss and statistics
            total_loss += step_loss
            affected_pixels += affected_pixel

        # Log statistics to Weights & Biases if enabled
        if self.log_wandb and wandb.run is not None:
            wandb.log({
                "affected_pixels": affected_pixels / B,
            },
            step=wandb.run.step
            )

        # Average over steps to avoid scaling the loss with num_steps
        return total_loss / self.num_steps


class CELossFunnyBirds(nn.Module):
    def forward(self, pred_logits: torch.Tensor, target: torch.Tensor):
        """
        Computes the loss as described in the FunnyBirds paper.

        Args:
            pred_logits (Tensor): Model outputs of shape [B, 50].
            target (Tensor): Multi-hot encoded labels of shape [B, 50]
                             (1 for valid classes, 0 otherwise).
        """
        # 1. Construct a normalized target distribution
        # Ensures that each row sums to 1
        row_sums = target.sum(dim=1, keepdim=True)
        
        # Avoid division by zero in case a sample has no positive labels
        target_dist = target / row_sums.clamp(min=1e-9)

        # 2. Compute log-softmax over class logits
        log_probs = F.log_softmax(pred_logits, dim=1)  # [B, 50]

        # 3. Cross-entropy between target distribution and predicted distribution
        # Equivalent to averaging the negative log-probabilities of all valid classes per sample
        loss = -(target_dist * log_probs).sum(dim=1)  # [B]

        # Return mean loss over the batch
        return loss.mean()


class CELossOneHot(nn.Module):
    def forward(
        self,
        pred_logits: torch.Tensor,  # [B, K]
        target: torch.Tensor,  # [B, K]
    ):
        # target one-hot [B, K]
        class_idx = target.argmax(dim=1)  # [B]
        return F.cross_entropy(pred_logits, class_idx)
