"""
Transformer Query Decoder, Hungarian Bipartite Matcher, and DETR Loss Criterion.
Official Reference: RT-DETR (https://github.com/lyuwenyu/RT-DETR)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from typing import Dict, List, Tuple, Optional


def box_cxcywh_to_xyxy(x: torch.Tensor) -> torch.Tensor:
    """Converts normalized cx, cy, w, h to x1, y1, x2, y2."""
    cx, cy, w, h = x.unbind(-1)
    b = [(cx - 0.5 * w), (cy - 0.5 * h), (cx + 0.5 * w), (cy + 0.5 * h)]
    return torch.stack(b, dim=-1)


def generalized_box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """
    Computes generalized IoU (GIoU) between boxes1 [N, 4] and boxes2 [M, 4] in (x1, y1, x2, y2).
    """
    area1 = (boxes1[:, 2] - boxes1[:, 0]).clamp(min=0) * (boxes1[:, 3] - boxes1[:, 1]).clamp(min=0)
    area2 = (boxes2[:, 2] - boxes2[:, 0]).clamp(min=0) * (boxes2[:, 3] - boxes2[:, 1]).clamp(min=0)

    lt = torch.max(boxes1[:, None, :2], boxes2[None, :, :2])
    rb = torch.min(boxes1[:, None, 2:], boxes2[None, :, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[:, :, 0] * wh[:, :, 1]

    union = area1[:, None] + area2[None, :] - inter
    iou = inter / (union + 1e-7)

    # Smallest enclosing box
    enclosing_lt = torch.min(boxes1[:, None, :2], boxes2[None, :, :2])
    enclosing_rb = torch.max(boxes1[:, None, 2:], boxes2[None, :, 2:])
    enclosing_wh = (enclosing_rb - enclosing_lt).clamp(min=0)
    enclosing_area = enclosing_wh[:, :, 0] * enclosing_wh[:, :, 1] + 1e-7

    giou = iou - (enclosing_area - union) / enclosing_area
    return giou


class MLP(nn.Module):
    """Multi-layer perceptron (used for box regression and coefficient heads)."""
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, num_layers: int):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class TransformerDecoderLayer(nn.Module):
    """Standard Transformer Decoder Layer with Self-Attention, Cross-Attention, and FFN."""
    def __init__(self, d_model: int = 256, nhead: int = 8, dim_feedforward: int = 1024, dropout: float = 0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, tgt: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        # Self-attention over queries
        q2, _ = self.self_attn(tgt, tgt, tgt)
        tgt = self.norm1(tgt + self.dropout1(q2))

        # Cross-attention with encoder feature memory
        q2, _ = self.cross_attn(tgt, memory, memory)
        tgt = self.norm2(tgt + self.dropout2(q2))

        # Feed-forward network
        ffn = self.linear2(self.dropout(F.relu(self.linear1(tgt))))
        tgt = self.norm3(tgt + self.dropout3(ffn))
        return tgt


class HungarianMatcher(nn.Module):
    """
    Computes optimal bipartite matching between predictions and targets.
    Official DETR Scipy Hungarian matcher.
    """
    def __init__(self, cost_class: float = 2.0, cost_bbox: float = 5.0, cost_giou: float = 2.0):
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou

    @torch.no_grad()
    def forward(self, outputs: Dict[str, torch.Tensor], targets: List[Dict[str, torch.Tensor]]) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        bs, num_queries = outputs["pred_logits"].shape[:2]

        out_prob = outputs["pred_logits"].flatten(0, 1).sigmoid()  # [B*Q, num_classes]
        out_bbox = outputs["pred_boxes"].flatten(0, 1)            # [B*Q, 4]

        tgt_ids = torch.cat([v["labels"] for v in targets])
        tgt_bbox = torch.cat([v["boxes"] for v in targets])

        if len(tgt_ids) == 0:
            return [(torch.empty(0, dtype=torch.int64), torch.empty(0, dtype=torch.int64)) for _ in range(bs)]

        # Classification cost: negative probability of target class
        cost_class = -out_prob[:, tgt_ids]

        # L1 cost between predicted and target boxes
        cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)

        # GIoU cost
        out_bbox_xyxy = box_cxcywh_to_xyxy(out_bbox)
        tgt_bbox_xyxy = box_cxcywh_to_xyxy(tgt_bbox)
        cost_giou = -generalized_box_iou(out_bbox_xyxy, tgt_bbox_xyxy)

        # Combined cost matrix
        c = self.cost_bbox * cost_bbox + self.cost_class * cost_class + self.cost_giou * cost_giou
        c = c.view(bs, num_queries, -1).cpu()

        sizes = [len(v["boxes"]) for v in targets]
        indices = []
        for i, c_batch in enumerate(c.split(sizes, -1)):
            if sizes[i] == 0:
                indices.append((torch.empty(0, dtype=torch.int64), torch.empty(0, dtype=torch.int64)))
            else:
                row_ind, col_ind = linear_sum_assignment(c_batch[i].numpy())
                indices.append((torch.as_tensor(row_ind, dtype=torch.int64), torch.as_tensor(col_ind, dtype=torch.int64)))

        return indices


class DETRCriterion(nn.Module):
    """
    Computes composite loss for DETR-style instance segmentation:
    - Classification loss (BCE with logits or focal loss)
    - Box L1 + GIoU loss
    - Mask BCE + Dice loss
    """
    def __init__(
        self,
        num_classes: int = 40,
        weight_cls: float = 2.0,
        weight_l1: float = 5.0,
        weight_giou: float = 2.0,
        weight_mask_bce: float = 5.0,
        weight_mask_dice: float = 5.0
    ):
        super().__init__()
        self.num_classes = num_classes
        self.matcher = HungarianMatcher(cost_class=weight_cls, cost_bbox=weight_l1, cost_giou=weight_giou)
        self.weight_cls = weight_cls
        self.weight_l1 = weight_l1
        self.weight_giou = weight_giou
        self.weight_mask_bce = weight_mask_bce
        self.weight_mask_dice = weight_mask_dice

    def forward(self, outputs: Dict[str, torch.Tensor], targets: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        indices = self.matcher(outputs, targets)
        device = outputs["pred_logits"].device

        # 1. Classification Loss (BCE with one-hot targets)
        src_logits = outputs["pred_logits"]
        target_classes = torch.full(src_logits.shape[:2], self.num_classes, dtype=torch.int64, device=device)
        for i, (src_idx, tgt_idx) in enumerate(indices):
            if len(src_idx) > 0:
                target_classes[i, src_idx] = targets[i]["labels"][tgt_idx].to(device)

        # One-hot target
        target_classes_onehot = torch.zeros(
            [src_logits.shape[0], src_logits.shape[1], self.num_classes + 1],
            dtype=torch.float32,
            device=device
        )
        target_classes_onehot.scatter_(2, target_classes.unsqueeze(-1), 1)
        target_classes_onehot = target_classes_onehot[:, :, :-1]  # Drop background class

        loss_cls = F.binary_cross_entropy_with_logits(src_logits, target_classes_onehot, reduction="mean")

        # 2. Bounding Box Losses (L1 and GIoU)
        total_matched = sum(len(src_idx) for src_idx, _ in indices)
        if total_matched == 0:
            loss_l1 = src_logits.sum() * 0.0
            loss_giou = src_logits.sum() * 0.0
            loss_mask_bce = src_logits.sum() * 0.0
            loss_mask_dice = src_logits.sum() * 0.0
        else:
            src_boxes = []
            target_boxes = []
            src_masks = []
            target_masks = []

            for i, (src_idx, tgt_idx) in enumerate(indices):
                if len(src_idx) == 0:
                    continue
                src_boxes.append(outputs["pred_boxes"][i, src_idx])
                target_boxes.append(targets[i]["boxes"][tgt_idx].to(device))

                if "pred_masks" in outputs and len(targets[i]["masks"]) > 0:
                    src_m = outputs["pred_masks"][i, src_idx]
                    tgt_m = targets[i]["masks"][tgt_idx].to(device)
                    # Resize pred masks if resolution differs
                    if src_m.shape[-2:] != tgt_m.shape[-2:]:
                        src_m = F.interpolate(src_m.unsqueeze(1), size=tgt_m.shape[-2:], mode="bilinear", align_corners=False).squeeze(1)
                    src_masks.append(src_m)
                    target_masks.append(tgt_m)

            src_boxes = torch.cat(src_boxes, dim=0)
            target_boxes = torch.cat(target_boxes, dim=0)

            loss_l1 = F.l1_loss(src_boxes, target_boxes, reduction="mean")

            src_boxes_xyxy = box_cxcywh_to_xyxy(src_boxes)
            target_boxes_xyxy = box_cxcywh_to_xyxy(target_boxes)
            giou = generalized_box_iou(src_boxes_xyxy, target_boxes_xyxy)
            loss_giou = (1.0 - torch.diag(giou)).mean()

            # 3. Mask Losses (BCE + Dice)
            if src_masks and target_masks:
                all_src_m = torch.cat(src_masks, dim=0)
                all_tgt_m = torch.cat(target_masks, dim=0)

                # Clamp values for numerical stability (run in float32 for AMP safety)
                all_src_m = all_src_m.float().clamp(1e-6, 1.0 - 1e-6)
                all_tgt_m = all_tgt_m.float()
                with torch.amp.autocast(device_type=all_src_m.device.type, enabled=False):
                    loss_mask_bce = F.binary_cross_entropy(all_src_m, all_tgt_m, reduction="mean")

                # Dice loss
                num = 2 * (all_src_m * all_tgt_m).sum(dim=[-2, -1])
                den = all_src_m.sum(dim=[-2, -1]) + all_tgt_m.sum(dim=[-2, -1]) + 1e-6
                loss_mask_dice = (1.0 - num / den).mean()
            else:
                loss_mask_bce = src_logits.sum() * 0.0
                loss_mask_dice = src_logits.sum() * 0.0

        total_loss = (
            self.weight_cls * loss_cls
            + self.weight_l1 * loss_l1
            + self.weight_giou * loss_giou
            + self.weight_mask_bce * loss_mask_bce
            + self.weight_mask_dice * loss_mask_dice
        )

        return {
            "loss": total_loss,
            "loss_cls": loss_cls,
            "loss_l1": loss_l1,
            "loss_giou": loss_giou,
            "loss_mask_bce": loss_mask_bce,
            "loss_mask_dice": loss_mask_dice,
            "loss_mask": loss_mask_bce + loss_mask_dice,
            "loss_box": loss_l1 + loss_giou,
        }
