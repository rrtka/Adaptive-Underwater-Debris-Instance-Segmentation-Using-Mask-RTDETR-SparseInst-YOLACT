"""
RT-DETR Hybrid Encoder / Cross-Scale Feature-Fusion Module (CCFM).
Official Reference: RT-DETR (https://github.com/lyuwenyu/RT-DETR)

Combines:
1. AIFI: Attention-based Intra-scale Feature Interaction on high-level feature C5.
2. CCFM: Cross-scale Feature Fusion Module (top-down FPN + bottom-up PAN with RepConv/CSP).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List


class ConvNormAct(nn.Module):
    """Standard Conv2d + BatchNorm + ReLU/SiLU block."""
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 1, stride: int = 1, padding: int = 0):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class RepCSPBlock(nn.Module):
    """Cross-stage partial block for efficient multi-scale feature aggregation."""
    def __init__(self, in_channels: int, out_channels: int, num_blocks: int = 2):
        super().__init__()
        mid_channels = out_channels // 2
        self.conv1 = ConvNormAct(in_channels, mid_channels, 1)
        self.conv2 = ConvNormAct(in_channels, mid_channels, 1)

        blocks = []
        for _ in range(num_blocks):
            blocks.append(nn.Sequential(
                ConvNormAct(mid_channels, mid_channels, 3, padding=1),
                ConvNormAct(mid_channels, mid_channels, 3, padding=1)
            ))
        self.blocks = nn.ModuleList(blocks)
        self.conv3 = ConvNormAct(mid_channels * 2, out_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y1 = self.conv1(x)
        y2 = self.conv2(x)
        for block in self.blocks:
            y1 = y1 + block(y1)
        return self.conv3(torch.cat([y1, y2], dim=1))


class AIFI(nn.Module):
    """
    Attention-based Intra-scale Feature Interaction for C5.
    Applies multi-head self-attention on the highest semantic level.
    """
    def __init__(self, hidden_dim: int = 256, nheads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(hidden_dim, nheads, dropout=dropout, batch_first=True)
        self.linear1 = nn.Linear(hidden_dim, hidden_dim * 4)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(hidden_dim * 4, hidden_dim)

        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, H, W]
        b, c, h, w = x.shape
        x_flat = x.flatten(2).permute(0, 2, 1)  # [B, H*W, C]

        # Multi-head attention
        attn_out, _ = self.self_attn(x_flat, x_flat, x_flat)
        x_norm = self.norm1(x_flat + attn_out)

        # FFN
        ffn_out = self.linear2(self.dropout(self.act(self.linear1(x_norm))))
        out = self.norm2(x_norm + ffn_out)

        return out.permute(0, 2, 1).view(b, c, h, w)


class HybridEncoder(nn.Module):
    """
    RT-DETR Hybrid Encoder (CCFM):
    - Input: multi-scale features from backbone (C3, C4, C5)
    - Output: enriched multi-scale features (P3, P4, P5) at strides 8, 16, 32
    """
    def __init__(
        self,
        in_channels: List[int] = [512, 1024, 2048],
        hidden_dim: int = 256,
        nheads: int = 8,
        dropout: float = 0.1
    ):
        super().__init__()
        self.hidden_dim = hidden_dim

        # Input projections
        self.input_proj = nn.ModuleList([
            ConvNormAct(in_ch, hidden_dim, 1) for in_ch in in_channels
        ])

        # AIFI on highest-level feature (C5)
        self.aifi = AIFI(hidden_dim, nheads=nheads, dropout=dropout)

        # Top-down FPN pathway
        self.fpn_conv_p4 = RepCSPBlock(hidden_dim * 2, hidden_dim, num_blocks=2)
        self.fpn_conv_p3 = RepCSPBlock(hidden_dim * 2, hidden_dim, num_blocks=2)

        # Bottom-up PAN pathway
        self.downsample_p3 = ConvNormAct(hidden_dim, hidden_dim, 3, stride=2, padding=1)
        self.pan_conv_p4 = RepCSPBlock(hidden_dim * 2, hidden_dim, num_blocks=2)

        self.downsample_p4 = ConvNormAct(hidden_dim, hidden_dim, 3, stride=2, padding=1)
        self.pan_conv_p5 = RepCSPBlock(hidden_dim * 2, hidden_dim, num_blocks=2)

    def forward(self, feats: Dict[str, torch.Tensor]) -> List[torch.Tensor]:
        c3 = self.input_proj[0](feats["c3"])  # Stride 8  [B, 256, H/8, W/8]
        c4 = self.input_proj[1](feats["c4"])  # Stride 16 [B, 256, H/16, W/16]
        c5 = self.input_proj[2](feats["c5"])  # Stride 32 [B, 256, H/32, W/32]

        # AIFI on C5
        p5_intra = self.aifi(c5)

        # Top-down FPN
        p5_up = F.interpolate(p5_intra, size=c4.shape[-2:], mode="nearest")
        p4 = self.fpn_conv_p4(torch.cat([p5_up, c4], dim=1))

        p4_up = F.interpolate(p4, size=c3.shape[-2:], mode="nearest")
        p3 = self.fpn_conv_p3(torch.cat([p4_up, c3], dim=1))

        # Bottom-up PAN
        p3_down = self.downsample_p3(p3)
        n4 = self.pan_conv_p4(torch.cat([p3_down, p4], dim=1))

        n4_down = self.downsample_p4(n4)
        n5 = self.pan_conv_p5(torch.cat([n4_down, p5_intra], dim=1))

        return [p3, n4, n5]  # Strides 8, 16, 32
