"""
Model 4 (Proposed Model):
Mask RT-DETR with SparseInst IAM and YOLACT Prototype Mask Branch.

Official References:
1. RT-DETR: https://github.com/lyuwenyu/RT-DETR
2. SparseInst: https://github.com/hustvl/SparseInst
3. YOLACT: https://github.com/dbolya/yolact

=============================================================================
DOCUMENTATION OF PROPOSED M4 FUSION MECHANISM:
-----------------------------------------------------------------------------
1. Feature Extraction:
   - ResNet Backbone extracts multi-scale representations C3, C4, C5.
   - Hybrid Encoder (AIFI + CCFM) performs cross-scale fusion, outputting P3, P4, P5.

2. SparseInst IAM Module:
   - Receives P3 in R^(B x C x H/8 x W/8).
   - Generates Instance Activation Maps A in R^(B x N_iam x H/8 x W/8).
   - Aggregates instance-aware feature vectors Z in R^(B x N_iam x C) via spatial weighted pooling.
   - Computes foreground-modulated spatial feature F_iam in R^(B x C x H/8 x W/8).

3. Conditioned YOLACT Prototype Generation:
   - YOLACT Protonet receives F_iam (foreground-enhanced by IAM).
   - Generates K prototype masks P in R^(B x K x 128 x 128).

4. Instance-Aware Query Decoding & Coefficients:
   - DETR object queries are initialized and modulated by IAM instance features Z.
   - Transformer decoder refines queries via cross-attention with multi-scale memory.
   - Decoder outputs feed:
     a) Classification Head -> class logits in R^(B x Q x num_classes)
     b) Bounding Box Head  -> normalized boxes in R^(B x Q x 4)
     c) Instance Coefficient Head -> C in R^(B x Q x K) with tanh activation

5. Assembly & Refinement:
   - Tensor Contraction: M_raw = C * P in R^(B x Q x 128 x 128) computed via torch.bmm.
   - Mask Refinement Module: Lightweight residual block sharpens debris instance boundaries.
   - Final instance masks: M = Sigmoid(M_refined).
=============================================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional

from .backbone import ResNetBackbone
from .hybrid_encoder import HybridEncoder
from .detr_decoder import TransformerDecoderLayer, MLP
from .iam import InstanceActivationMapModule
from .yolact_proto import YOLACTPrototypeBranch


class MaskRefinementBlock(nn.Module):
    """
    Lightweight convolutional refinement block to sharpen instance mask boundaries.
    Operates on [B * Q, 1, 128, 128] without altering memory efficiency.
    """
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
            nn.Conv2d(8, 1, kernel_size=3, padding=1, bias=True)
        )

    def forward(self, raw_masks: torch.Tensor) -> torch.Tensor:
        b, q, h, w = raw_masks.shape
        # Flatten batch and queries
        flat_masks = raw_masks.view(b * q, 1, h, w)
        residual = self.conv(flat_masks)
        refined = (flat_masks + residual).view(b, q, h, w)
        return refined


class ProposedMaskRTDETR_IAM_YOLACT(nn.Module):
    """
    M4: Proposed Architecture: Mask RT-DETR + SparseInst IAM + YOLACT Prototype Branch.
    """
    def __init__(
        self,
        backbone_name: str = "resnet50",
        pretrained_backbone: bool = False,
        hidden_dim: int = 256,
        num_classes: int = 40,
        num_queries: int = 100,
        num_decoder_layers: int = 3,
        nheads: int = 8,
        iam_num_masks: int = 32,
        num_prototypes: int = 32,
        proto_res: int = 128
    ):
        super().__init__()
        self.model_name = "M4_Proposed_RT_DETR_IAM_YOLACT"
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.num_queries = num_queries
        self.num_prototypes = num_prototypes
        self.proto_res = proto_res

        # 1. Multi-scale Backbone
        self.backbone = ResNetBackbone(name=backbone_name, pretrained=pretrained_backbone)
        in_channels = self.backbone.out_channels

        # 2. Hybrid Encoder (CCFM)
        self.hybrid_encoder = HybridEncoder(in_channels=in_channels, hidden_dim=hidden_dim, nheads=nheads)

        # 3. SparseInst IAM Module
        self.iam_module = InstanceActivationMapModule(in_channels=hidden_dim, num_masks=iam_num_masks)

        # IAM query adapter to project instance-aware features into query space
        self.iam_query_adapter = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True)
        )

        # 4. YOLACT Prototype Branch (Protonet)
        self.proto_branch = YOLACTPrototypeBranch(
            in_channels=hidden_dim,
            num_prototypes=num_prototypes,
            proto_res=proto_res
        )

        # 5. Object Queries & Decoder
        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        self.decoder_layers = nn.ModuleList([
            TransformerDecoderLayer(d_model=hidden_dim, nhead=nheads)
            for _ in range(num_decoder_layers)
        ])

        # 6. Prediction Heads
        self.class_head = nn.Linear(hidden_dim, num_classes)
        self.bbox_head = MLP(hidden_dim, hidden_dim, 4, num_layers=3)
        self.coeff_head = MLP(hidden_dim, hidden_dim, num_prototypes, num_layers=2)

        # 7. Mask Refinement Module
        self.mask_refiner = MaskRefinementBlock()

    def forward(self, images: torch.Tensor) -> Dict[str, torch.Tensor]:
        b = images.shape[0]

        # 1. Backbone multi-scale extraction
        backbone_feats = self.backbone(images)

        # 2. CCFM feature fusion -> [P3, P4, P5]
        p3, p4, p5 = self.hybrid_encoder(backbone_feats)

        # 3. SparseInst IAM module on P3
        iam_maps, iam_inst_feats, p3_modulated = self.iam_module(p3)

        # 4. Modulate Queries with IAM instance representations
        base_queries = self.query_embed.weight.unsqueeze(0).repeat(b, 1, 1)  # [B, Q, C]
        if iam_inst_feats.shape[1] != self.num_queries:
            iam_adapted = F.interpolate(
                iam_inst_feats.transpose(1, 2),
                size=self.num_queries,
                mode="linear",
                align_corners=False
            ).transpose(1, 2)
        else:
            iam_adapted = iam_inst_feats
        iam_bias = self.iam_query_adapter(iam_adapted)
        queries = base_queries + iam_bias

        # Memory with foreground-enhanced P3
        mem_p3 = p3_modulated.flatten(2).permute(0, 2, 1)
        mem_p4 = p4.flatten(2).permute(0, 2, 1)
        mem_p5 = p5.flatten(2).permute(0, 2, 1)
        memory = torch.cat([mem_p3, mem_p4, mem_p5], dim=1)

        # Decoder cross-attention layers
        for layer in self.decoder_layers:
            queries = layer(queries, memory)

        # 5. Predictions
        pred_logits = self.class_head(queries)
        pred_boxes = self.bbox_head(queries).sigmoid()

        # 6. Instance Coefficients C in R^(B x Q x K)
        coefficients = torch.tanh(self.coeff_head(queries))

        # 7. YOLACT Protonet conditioned on IAM foreground-enhanced feature p3_modulated
        prototypes = self.proto_branch.protonet(p3_modulated)

        # 8. Memory-efficient tensor contraction C * P
        k_proto = self.num_prototypes
        h_p, w_p = self.proto_res, self.proto_res
        proto_flat = prototypes.view(b, k_proto, h_p * w_p)
        raw_masks = torch.bmm(coefficients, proto_flat).view(b, self.num_queries, h_p, w_p)

        # 9. Boundary refinement
        refined_raw_masks = self.mask_refiner(raw_masks)
        pred_masks = torch.sigmoid(refined_raw_masks)

        # Crop to predicted boxes during evaluation if not in training
        if not self.training:
            pred_masks = YOLACTPrototypeBranch.crop_masks_to_boxes(pred_masks, pred_boxes)

        return {
            "pred_logits": pred_logits,
            "pred_boxes": pred_boxes,
            "pred_masks": pred_masks,
            "coefficients": coefficients,
            "prototypes": prototypes,
            "iam_maps": iam_maps,
            "iam_inst_feats": iam_inst_feats,
            "queries": queries,
        }
