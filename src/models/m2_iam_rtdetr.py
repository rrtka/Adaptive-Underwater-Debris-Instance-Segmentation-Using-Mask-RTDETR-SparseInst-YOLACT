"""
Model 2: Mask RT-DETR + SparseInst IAM.
Official References:
- RT-DETR (https://github.com/lyuwenyu/RT-DETR)
- SparseInst (https://github.com/hustvl/SparseInst)

Architecture:
Image -> Backbone -> CCFM
      -> SparseInst IAM Module:
           - Instance Activation Maps A
           - Instance-aware Pooled Features Z
           - Foreground-Modulated Spatial Features P3_iam
      -> Query Decoder (Queries modulated by IAM instance features Z)
      -> Classification Head + Bounding Box Head + IAM-Conditioned Mask Branch
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional

from .backbone import ResNetBackbone
from .hybrid_encoder import HybridEncoder
from .detr_decoder import TransformerDecoderLayer, MLP
from .iam import InstanceActivationMapModule


class MaskRTDETR_IAM(nn.Module):
    """
    M2: Mask RT-DETR with SparseInst Instance Activation Map (IAM) Module.
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
        mask_res: int = 128
    ):
        super().__init__()
        self.model_name = "M2_Mask_RT_DETR_IAM"
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.num_queries = num_queries
        self.mask_res = mask_res

        # 1. Multi-scale Backbone
        self.backbone = ResNetBackbone(name=backbone_name, pretrained=pretrained_backbone)
        in_channels = self.backbone.out_channels

        # 2. Hybrid Encoder (CCFM)
        self.hybrid_encoder = HybridEncoder(in_channels=in_channels, hidden_dim=hidden_dim, nheads=nheads)

        # 3. SparseInst IAM Module
        # IAM Input: CCFM P3 feature map [B, 256, H/8, W/8]
        # IAM Output: Activation maps [B, 32, H/8, W/8], Instance vectors [B, 32, 256], Modulated P3 [B, 256, H/8, W/8]
        self.iam_module = InstanceActivationMapModule(in_channels=hidden_dim, num_masks=iam_num_masks)

        # IAM query projector: adapts IAM instance features to object queries
        self.iam_query_adapter = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True)
        )

        # 4. Object Queries & Decoder
        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        self.decoder_layers = nn.ModuleList([
            TransformerDecoderLayer(d_model=hidden_dim, nhead=nheads)
            for _ in range(num_decoder_layers)
        ])

        # 5. Prediction Heads
        self.class_head = nn.Linear(hidden_dim, num_classes)
        self.bbox_head = MLP(hidden_dim, hidden_dim, 4, num_layers=3)

        # 6. IAM-Conditioned Mask Branch
        self.mask_proj = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1)
        self.mask_mlp = MLP(hidden_dim, hidden_dim, hidden_dim, num_layers=2)

    def forward(self, images: torch.Tensor) -> Dict[str, torch.Tensor]:
        b = images.shape[0]

        # 1. Multi-scale backbone extraction
        backbone_feats = self.backbone(images)

        # 2. Hybrid Encoder multi-scale features [P3, P4, P5]
        p3, p4, p5 = self.hybrid_encoder(backbone_feats)

        # 3. SparseInst IAM Processing on P3
        iam_maps, iam_inst_feats, p3_modulated = self.iam_module(p3)

        # 4. Modulate Queries with IAM instance representation
        base_queries = self.query_embed.weight.unsqueeze(0).repeat(b, 1, 1)  # [B, Q, C]
        
        # Broadcast/project IAM instance features across queries
        # iam_inst_feats: [B, N_iam, C] -> pool/interpolate to [B, Q, C]
        if iam_inst_feats.shape[1] != self.num_queries:
            iam_adapted = F.interpolate(iam_inst_feats.transpose(1, 2), size=self.num_queries, mode="linear", align_corners=False).transpose(1, 2)
        else:
            iam_adapted = iam_inst_feats
        iam_query_bias = self.iam_query_adapter(iam_adapted)
        queries = base_queries + iam_query_bias

        # Memory with foreground-modulated P3
        mem_p3 = p3_modulated.flatten(2).permute(0, 2, 1)
        mem_p4 = p4.flatten(2).permute(0, 2, 1)
        mem_p5 = p5.flatten(2).permute(0, 2, 1)
        memory = torch.cat([mem_p3, mem_p4, mem_p5], dim=1)

        # Decoder cross-attention
        for layer in self.decoder_layers:
            queries = layer(queries, memory)

        # 5. Classification & Bounding Box Heads
        pred_logits = self.class_head(queries)
        pred_boxes = self.bbox_head(queries).sigmoid()

        # 6. IAM-Conditioned Mask Prediction
        p3_proj = self.mask_proj(p3_modulated)
        p3_interp = F.interpolate(p3_proj, size=(self.mask_res, self.mask_res), mode="bilinear", align_corners=False)
        p3_flat = p3_interp.view(b, self.hidden_dim, -1)

        query_mask_feats = self.mask_mlp(queries)
        raw_masks = torch.bmm(query_mask_feats, p3_flat).view(b, self.num_queries, self.mask_res, self.mask_res)
        pred_masks = torch.sigmoid(raw_masks)

        return {
            "pred_logits": pred_logits,
            "pred_boxes": pred_boxes,
            "pred_masks": pred_masks,
            "iam_maps": iam_maps,
            "iam_inst_feats": iam_inst_feats,
            "queries": queries,
        }
