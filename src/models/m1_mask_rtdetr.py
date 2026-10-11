"""
Model 1: Mask RT-DETR Baseline.
Official Reference: RT-DETR (https://github.com/lyuwenyu/RT-DETR)

Architecture:
Image -> Backbone -> CCFM (Hybrid Encoder) -> Object Queries -> Transformer Decoder
      -> Classification Head
      -> Bounding Box Head
      -> Baseline Mask Head
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional

from .backbone import ResNetBackbone
from .hybrid_encoder import HybridEncoder
from .detr_decoder import TransformerDecoderLayer, MLP


class MaskRTDETRBaseline(nn.Module):
    """
    M1: Baseline Mask RT-DETR Architecture.
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
        mask_res: int = 128
    ):
        super().__init__()
        self.model_name = "M1_Mask_RT_DETR_Baseline"
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.num_queries = num_queries
        self.mask_res = mask_res

        # 1. Multi-scale Backbone
        self.backbone = ResNetBackbone(name=backbone_name, pretrained=pretrained_backbone)
        in_channels = self.backbone.out_channels

        # 2. CCFM / Hybrid Encoder
        self.hybrid_encoder = HybridEncoder(in_channels=in_channels, hidden_dim=hidden_dim, nheads=nheads)

        # 3. Object Queries & Decoder
        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        self.decoder_layers = nn.ModuleList([
            TransformerDecoderLayer(d_model=hidden_dim, nhead=nheads)
            for _ in range(num_decoder_layers)
        ])

        # 4. Prediction Heads
        self.class_head = nn.Linear(hidden_dim, num_classes)
        self.bbox_head = MLP(hidden_dim, hidden_dim, 4, num_layers=3)

        # 5. Baseline Mask Branch: Query-to-Feature Cross-Correlation
        self.mask_proj = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1)
        self.mask_mlp = MLP(hidden_dim, hidden_dim, hidden_dim, num_layers=2)

    def forward(self, images: torch.Tensor) -> Dict[str, torch.Tensor]:
        b = images.shape[0]

        # 1. Backbone features
        backbone_feats = self.backbone(images)

        # 2. Hybrid Encoder multi-scale features [P3, P4, P5]
        p3, p4, p5 = self.hybrid_encoder(backbone_feats)

        # Flatten multi-scale features as decoder memory
        # Memory is concatenation of flattened spatial features
        mem_p3 = p3.flatten(2).permute(0, 2, 1)  # [B, H3*W3, C]
        mem_p4 = p4.flatten(2).permute(0, 2, 1)  # [B, H4*W4, C]
        mem_p5 = p5.flatten(2).permute(0, 2, 1)  # [B, H5*W5, C]
        memory = torch.cat([mem_p3, mem_p4, mem_p5], dim=1)  # [B, sum(HW), C]

        # 3. Transformer Decoder
        queries = self.query_embed.weight.unsqueeze(0).repeat(b, 1, 1)  # [B, Q, C]
        for layer in self.decoder_layers:
            queries = layer(queries, memory)

        # 4. Predictions
        pred_logits = self.class_head(queries)  # [B, Q, num_classes]
        pred_boxes = self.bbox_head(queries).sigmoid()  # [B, Q, 4] normalized [cx, cy, w, h]

        # 5. Baseline Mask Generation: Dot product between query embeddings and projected P3
        p3_proj = self.mask_proj(p3)  # [B, C, H3, W3]
        p3_interp = F.interpolate(p3_proj, size=(self.mask_res, self.mask_res), mode="bilinear", align_corners=False)
        p3_flat = p3_interp.view(b, self.hidden_dim, -1)  # [B, C, mask_res*mask_res]

        query_mask_feats = self.mask_mlp(queries)  # [B, Q, C]
        raw_masks = torch.bmm(query_mask_feats, p3_flat).view(b, self.num_queries, self.mask_res, self.mask_res)
        pred_masks = torch.sigmoid(raw_masks)

        return {
            "pred_logits": pred_logits,
            "pred_boxes": pred_boxes,
            "pred_masks": pred_masks,
            "queries": queries,
            "p3": p3,
        }
