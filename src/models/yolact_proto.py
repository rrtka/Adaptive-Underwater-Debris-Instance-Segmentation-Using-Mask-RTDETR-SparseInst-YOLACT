"""
YOLACT-style Prototype Mask Branch and Instance Coefficient Assembly.
Official Reference: YOLACT (https://github.com/dbolya/yolact)

Formulation:
- Prototype masks: P in R^(B x K x H_proto x W_proto)
- Instance coefficients: C in R^(B x Q x K)
- Assembled masks: M = Sigmoid(C * P) computed via memory-efficient tensor contraction.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class Protonet(nn.Module):
    """
    YOLACT Prototype Generation Network (Protonet).
    Takes high-resolution detector features (e.g. stride 8) and produces K prototype masks.
    Uses memory-efficient output resolution (e.g. 128 x 128).
    """
    def __init__(self, in_channels: int = 256, num_prototypes: int = 32, proto_res: int = 128):
        super().__init__()
        self.num_prototypes = num_prototypes
        self.proto_res = proto_res

        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, num_prototypes, kernel_size=1, bias=True)
        )

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        """
        Args:
            feat: Feature map in [B, C, H, W] (e.g. stride 8)
        Returns:
            prototypes: [B, K, proto_res, proto_res] (e.g. [B, 32, 128, 128])
        """
        # Interpolate feature map to target prototype resolution (128x128)
        feat_res = F.interpolate(feat, size=(self.proto_res, self.proto_res), mode="bilinear", align_corners=False)
        prototypes = self.net(feat_res)
        return prototypes


class YOLACTPrototypeBranch(nn.Module):
    """
    Complete YOLACT Prototype Mask Branch:
    - Generates prototypes P in R^(B x K x H_p x W_p) via Protonet.
    - Assembles instance masks via efficient tensor contraction: M = Sigmoid(C * P).
    """
    def __init__(self, in_channels: int = 256, num_prototypes: int = 32, proto_res: int = 128):
        super().__init__()
        self.num_prototypes = num_prototypes
        self.proto_res = proto_res
        self.protonet = Protonet(in_channels=in_channels, num_prototypes=num_prototypes, proto_res=proto_res)

    def forward(
        self,
        feat: torch.Tensor,
        coefficients: torch.Tensor,
        boxes_norm: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            feat: Feature map [B, C, H, W]
            coefficients: Instance coefficients [B, Q, K] (where Q = num_queries)
            boxes_norm: Optional normalized boxes [B, Q, 4] for spatial bbox cropping
        Returns:
            masks: Assembled instance masks [B, Q, proto_res, proto_res] in [0, 1]
            prototypes: Raw prototype masks [B, K, proto_res, proto_res]
        """
        b, k_proto, h_p, w_p = feat.shape[0], self.num_prototypes, self.proto_res, self.proto_res
        q = coefficients.shape[1]

        # 1. Generate prototype masks: [B, K, H_p, W_p]
        prototypes = self.protonet(feat)

        # 2. Efficient tensor contraction (C * P) without large intermediate allocations:
        # coefficients: [B, Q, K]
        # prototypes flattened: [B, K, H_p * W_p]
        # result: [B, Q, H_p * W_p]
        proto_flat = prototypes.view(b, k_proto, h_p * w_p)
        raw_masks = torch.bmm(coefficients, proto_flat).view(b, q, h_p, w_p)
        masks = torch.sigmoid(raw_masks)

        # 3. Optional bounding-box cropping (zeros out mask regions outside predicted box)
        if boxes_norm is not None:
            masks = self.crop_masks_to_boxes(masks, boxes_norm)

        return masks, prototypes

    @staticmethod
    def crop_masks_to_boxes(masks: torch.Tensor, boxes_norm: torch.Tensor) -> torch.Tensor:
        """
        Crops masks according to normalized bounding box coordinates [cx, cy, w, h].
        """
        b, q, h, w = masks.shape
        # Convert cx, cy, w, h to x1, y1, x2, y2 in [0, 1]
        cx, cy, bw, bh = boxes_norm.unbind(-1)
        x1 = ((cx - 0.5 * bw) * w).clamp(min=0, max=w)
        y1 = ((cy - 0.5 * bh) * h).clamp(min=0, max=h)
        x2 = ((cx + 0.5 * bw) * w).clamp(min=0, max=w)
        y2 = ((cy + 0.5 * bh) * h).clamp(min=0, max=h)

        # Create coordinate grids
        grid_y = torch.arange(h, device=masks.device, dtype=torch.float32).view(1, 1, h, 1)
        grid_x = torch.arange(w, device=masks.device, dtype=torch.float32).view(1, 1, 1, w)

        x1 = x1.view(b, q, 1, 1)
        y1 = y1.view(b, q, 1, 1)
        x2 = x2.view(b, q, 1, 1)
        y2 = y2.view(b, q, 1, 1)

        box_crop_mask = (grid_x >= x1) & (grid_x <= x2) & (grid_y >= y1) & (grid_y <= y2)
        return masks * box_crop_mask.float()
