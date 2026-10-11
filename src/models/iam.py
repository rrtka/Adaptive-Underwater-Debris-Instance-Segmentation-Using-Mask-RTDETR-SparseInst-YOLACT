"""
SparseInst Instance Activation Map (IAM) Module.
Official Reference: SparseInst (https://github.com/hustvl/SparseInst)

Preserves the core SparseInst IAM formulation:
1. Receives feature maps from the detector multi-scale representation.
2. Generates Instance Activation Maps (IAM) highlighting informative foreground regions.
3. Aggregates instance-aware feature vectors via spatial weighted pooling.
4. Enhances multi-scale features with foreground-guided modulation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class InstanceActivationMapModule(nn.Module):
    """
    SparseInst Instance Activation Map (IAM) Module.
    
    =============================================================================
    DOCUMENTATION OF IAM SPECIFICATION:
    - IAM Input: Multi-scale feature representation F in R^(B x C x H x W)
    - IAM Architecture:
        Conv2d(C, C, 3, padding=1) -> GroupNorm(32, C) -> ReLU ->
        Conv2d(C, N_iam, 3, padding=1) -> Spatial Softmax over (H x W)
    - IAM Output:
        1. Instance Activation Maps A in R^(B x N_iam x H x W)
        2. Instance-Aware Feature Vectors Z in R^(B x N_iam x C)
        3. Foreground-Modulated Spatial Features F_iam in R^(B x C x H x W)
    - IAM Fusion Location:
        In M2: Z is projected and fused with DETR object queries, F_iam enhances decoder memory.
        In M4: F_iam feeds into the YOLACT Protonet branch, and Z enhances instance coefficients.
    =============================================================================
    """
    def __init__(self, in_channels: int = 256, num_masks: int = 32):
        super().__init__()
        self.in_channels = in_channels
        self.num_masks = num_masks  # N_iam: number of activation maps

        # Lightweight convolutional stack for activation map prediction
        self.iam_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(num_groups=min(32, in_channels), num_channels=in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, num_masks, kernel_size=3, padding=1, bias=True)
        )

        # Instance feature refinement projection
        self.instance_proj = nn.Sequential(
            nn.Linear(in_channels, in_channels),
            nn.LayerNorm(in_channels),
            nn.ReLU(inplace=True)
        )

        # Spatial foreground modulation projection
        self.fg_gate = nn.Conv2d(num_masks, 1, kernel_size=1)

    def forward(self, feat: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            feat: Feature map F in [B, C, H, W] (e.g. from CCFM stride 8)
        Returns:
            iam_maps: Instance Activation Maps [B, N_iam, H, W]
            instance_feats: Instance-aware aggregated features [B, N_iam, C]
            modulated_feat: Foreground-modulated spatial feature [B, C, H, W]
        """
        b, c, h, w = feat.shape

        # 1. Compute raw activation logits: [B, N_iam, H, W]
        act_logits = self.iam_conv(feat)

        # 2. Compute spatial softmax over (H, W) to obtain normalized probability maps
        act_logits_flat = act_logits.view(b, self.num_masks, h * w)
        iam_maps_flat = F.softmax(act_logits_flat, dim=-1)  # Sum over H*W is 1.0 for each map
        iam_maps = iam_maps_flat.view(b, self.num_masks, h, w)

        # 3. Aggregate instance-aware features via spatial weighted pooling:
        # feat_flat: [B, C, H*W]
        # iam_maps_flat: [B, N_iam, H*W]
        # instance_feats: [B, N_iam, C] = matmul(iam_maps_flat, feat_flat.t)
        feat_flat = feat.view(b, c, h * w).transpose(1, 2)  # [B, H*W, C]
        instance_feats = torch.bmm(iam_maps_flat, feat_flat)  # [B, N_iam, C]
        instance_feats = self.instance_proj(instance_feats)

        # 4. Modulate spatial feature map with foreground saliency
        fg_saliency = torch.sigmoid(self.fg_gate(iam_maps))  # [B, 1, H, W]
        modulated_feat = feat * (1.0 + fg_saliency)          # Residual foreground enhancement

        return iam_maps, instance_feats, modulated_feat
