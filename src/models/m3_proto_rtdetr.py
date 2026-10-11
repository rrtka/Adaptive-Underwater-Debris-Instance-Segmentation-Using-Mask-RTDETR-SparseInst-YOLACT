"""
Model 3: Mask RT-DETR + YOLACT Prototype Mask Branch.
Official References:
- RT-DETR (https://github.com/lyuwenyu/RT-DETR)
- YOLACT (https://github.com/dbolya/yolact)

Architecture:
Image -> Backbone -> CCFM
      -> YOLACT Protonet (Generates P in R^(B x K x 128 x 128))
      -> Query Decoder -> Object Queries
           -> Classification Head
           -> Bounding Box Head
           -> Instance Mask Coefficients C in R^(B x Q x K)
      -> Prototype Assembly: M = Sigmoid(C * P) with bbox cropping
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional

from .backbone import ResNetBackbone
from .hybrid_encoder import HybridEncoder
from .detr_decoder import TransformerDecoderLayer, MLP
from .yolact_proto import YOLACTPrototypeBranch


class MaskRTDETR_YOLACTProto(nn.Module):
    """
    M3: Mask RT-DETR with YOLACT-style Prototype Mask Branch.
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
        num_prototypes: int = 32,
        proto_res: int = 128
    ):
        super().__init__()
        self.model_name = "M3_Mask_RT_DETR_YOLACT_Proto"
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

        # 3. YOLACT Prototype Mask Branch (Protonet)
        self.proto_branch = YOLACTPrototypeBranch(
            in_channels=hidden_dim,
            num_prototypes=num_prototypes,
            proto_res=proto_res
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

        # 6. Instance Mask Coefficient Head (predicts K coefficients per query)
        self.coeff_head = MLP(hidden_dim, hidden_dim, num_prototypes, num_layers=2)

    def forward(self, images: torch.Tensor) -> Dict[str, torch.Tensor]:
        b = images.shape[0]

        # 1. Multi-scale backbone extraction
        backbone_feats = self.backbone(images)

        # 2. CCFM multi-scale features [P3, P4, P5]
        p3, p4, p5 = self.hybrid_encoder(backbone_feats)

        # Memory for decoder
        mem_p3 = p3.flatten(2).permute(0, 2, 1)
        mem_p4 = p4.flatten(2).permute(0, 2, 1)
        mem_p5 = p5.flatten(2).permute(0, 2, 1)
        memory = torch.cat([mem_p3, mem_p4, mem_p5], dim=1)

        # 3. Object Queries & Decoder
        queries = self.query_embed.weight.unsqueeze(0).repeat(b, 1, 1)
        for layer in self.decoder_layers:
            queries = layer(queries, memory)

        # 4. Classification & Bounding Boxes
        pred_logits = self.class_head(queries)
        pred_boxes = self.bbox_head(queries).sigmoid()

        # 5. Instance Coefficients C in R^(B x Q x K) with tanh activation (as in YOLACT)
        coefficients = torch.tanh(self.coeff_head(queries))

        # 6. Assemble Prototype Masks via efficient tensor contraction M = Sigmoid(C * P)
        pred_masks, prototypes = self.proto_branch(
            feat=p3,
            coefficients=coefficients,
            boxes_norm=pred_boxes if not self.training else None  # crop only at inference
        )

        return {
            "pred_logits": pred_logits,
            "pred_boxes": pred_boxes,
            "pred_masks": pred_masks,
            "coefficients": coefficients,
            "prototypes": prototypes,
            "queries": queries,
        }
