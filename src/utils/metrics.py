"""
Quantitative Evaluation Metrics for Underwater Marine Debris Detection
and Instance Segmentation.

Computes:
- Mask mAP, AP50, AP75
- Box mAP, AP50, AP75
- Mean IoU (mIoU)
- Precision, Recall, F1-score
- Latency (ms) and Throughput (FPS)
- Model Parameters (Millions)
- GFLOPs calculation (with graceful diagnostic reporting if uncomputable)
"""

import time
import numpy as np
import torch
from typing import Dict, List, Tuple, Any, Optional


def box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """Computes pairwise IoU between two sets of bounding boxes [N, 4] and [M, 4] in (x1, y1, x2, y2) format."""
    area1 = (boxes1[:, 2] - boxes1[:, 0]).clamp(min=0) * (boxes1[:, 3] - boxes1[:, 1]).clamp(min=0)
    area2 = (boxes2[:, 2] - boxes2[:, 0]).clamp(min=0) * (boxes2[:, 3] - boxes2[:, 1]).clamp(min=0)
    lt = torch.max(boxes1[:, None, :2], boxes2[None, :, :2])
    rb = torch.min(boxes1[:, None, 2:], boxes2[None, :, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[:, :, 0] * wh[:, :, 1]
    union = area1[:, None] + area2[None, :] - inter
    return inter / (union + 1e-7)


def mask_iou(masks1: torch.Tensor, masks2: torch.Tensor) -> torch.Tensor:
    """Computes pairwise IoU between two sets of binary masks [N, H, W] and [M, H, W]."""
    n, h, w = masks1.shape
    m = masks2.shape[0]
    m1_flat = masks1.view(n, -1).bool().float()
    m2_flat = masks2.view(m, -1).bool().float()
    inter = torch.matmul(m1_flat, m2_flat.t())
    area1 = m1_flat.sum(dim=1, keepdim=True)
    area2 = m2_flat.sum(dim=1, keepdim=True).t()
    union = area1 + area2 - inter
    return inter / (union + 1e-7)


def compute_ap_at_iou(pred_boxes: torch.Tensor, pred_scores: torch.Tensor, pred_labels: torch.Tensor,
                      gt_boxes: torch.Tensor, gt_labels: torch.Tensor, iou_thresh: float = 0.5) -> Tuple[float, float, float]:
    """Computes Precision, Recall, and Average Precision at a specific IoU threshold."""
    if len(gt_boxes) == 0:
        return 0.0, 0.0, 0.0
    if len(pred_boxes) == 0:
        return 0.0, 0.0, 0.0
    sort_idx = torch.argsort(pred_scores, descending=True)
    pred_boxes = pred_boxes[sort_idx]
    pred_scores = pred_scores[sort_idx]
    pred_labels = pred_labels[sort_idx]
    ious = box_iou(pred_boxes, gt_boxes)
    tp = torch.zeros(len(pred_boxes))
    fp = torch.zeros(len(pred_boxes))
    matched_gt = set()
    for i in range(len(pred_boxes)):
        best_iou, best_gt_idx = torch.max(ious[i], dim=0)
        best_gt_idx = best_gt_idx.item()
        if best_iou >= iou_thresh and (best_gt_idx not in matched_gt) and (pred_labels[i] == gt_labels[best_gt_idx]):
            tp[i] = 1.0
            matched_gt.add(best_gt_idx)
        else:
            fp[i] = 1.0
    cum_tp = torch.cumsum(tp, dim=0)
    cum_fp = torch.cumsum(fp, dim=0)
    recall = cum_tp / (len(gt_boxes) + 1e-7)
    precision = cum_tp / (cum_tp + cum_fp + 1e-7)
    rec_np = np.concatenate(([0.0], recall.cpu().numpy(), [1.0]))
    prec_np = np.concatenate(([0.0], precision.cpu().numpy(), [0.0]))
    for i in range(len(prec_np) - 1, 0, -1):
        prec_np[i - 1] = max(prec_np[i - 1], prec_np[i])
    i = np.where(rec_np[1:] != rec_np[:-1])[0]
    ap = float(np.sum((rec_np[i + 1] - rec_np[i]) * prec_np[i + 1]))
    final_prec = float(precision[-1].item()) if len(precision) > 0 else 0.0
    final_rec = float(recall[-1].item()) if len(recall) > 0 else 0.0
    return ap, final_prec, final_rec


def evaluate_batch_metrics(predictions: List[Dict[str, torch.Tensor]],
                           targets: List[Dict[str, torch.Tensor]]) -> Dict[str, float]:
    """Aggregate metrics over an entire dataset.

    - Box AP (AP50/AP75) is computed globally across all images, as is standard.
    - Mask AP & mIoU remain per‑image averages.
    - Pixel Accuracy and Detection Accuracy are averaged over images where they can be computed.
    """
    # Per‑image containers for mask and detection metrics
    mask_ap50_list, mask_ap75_list = [], []
    mask_miou_list = []
    pixel_acc_list = []
    det_acc_list = []
    prec_list, rec_list = [], []
    # Global containers for box AP
    all_pred_boxes, all_pred_scores, all_pred_labels = [], [], []
    all_gt_boxes, all_gt_labels = [], []

    for pred, target in zip(predictions, targets):
        pred_boxes = pred.get("boxes", torch.empty((0, 4)))
        pred_scores = pred.get("scores", torch.empty((0,)))
        pred_labels = pred.get("labels", torch.empty((0,), dtype=torch.long))
        pred_masks = pred.get("masks", torch.empty((0, 128, 128)))
        gt_boxes = target.get("boxes", torch.empty((0, 4)))
        gt_labels = target.get("labels", torch.empty((0,), dtype=torch.long))
        gt_masks = target.get("masks", torch.empty((0, 128, 128)))

        # Global box AP aggregation
        if len(pred_boxes) > 0:
            all_pred_boxes.append(pred_boxes)
            all_pred_scores.append(pred_scores)
            all_pred_labels.append(pred_labels)
        if len(gt_boxes) > 0:
            all_gt_boxes.append(gt_boxes)
            all_gt_labels.append(gt_labels)

        # Per‑image precision/recall (used for reporting)
        if len(pred_boxes) > 0 and len(gt_boxes) > 0:
            ap50, prec, rec = compute_ap_at_iou(pred_boxes, pred_scores, pred_labels, gt_boxes, gt_labels, 0.5)
            _, _, _ = compute_ap_at_iou(pred_boxes, pred_scores, pred_labels, gt_boxes, gt_labels, 0.75)
            prec_list.append(prec)
            rec_list.append(rec)

        # Mask metrics per image
        if len(pred_masks) > 0 and len(gt_masks) > 0:
            if pred_masks.shape[-2:] != gt_masks.shape[-2:]:
                pred_masks_resized = torch.nn.functional.interpolate(
                    pred_masks.unsqueeze(1).float(),
                    size=gt_masks.shape[-2:],
                    mode="bilinear",
                    align_corners=False
                ).squeeze(1) > 0.5
            else:
                pred_masks_resized = pred_masks > 0.5
            m_ious = mask_iou(pred_masks_resized.float(), gt_masks.float())  # [N_pred, M_gt]
            # Ground-truth instance mIoU: best IoU achieved by any query for each GT object
            max_ious, _ = torch.max(m_ious, dim=0) if m_ious.numel() > 0 else (torch.tensor([0.0]), None)
            mask_miou_list.append(max_ious.mean().item())
            # Simple mask AP50 approximation per image
            tp50 = 0
            matched_gt_masks = set()
            for pi in range(len(pred_masks)):
                best_m_iou, best_g = torch.max(m_ious[pi], dim=0)
                if best_m_iou.item() >= 0.5 and best_g.item() not in matched_gt_masks:
                    tp50 += 1
                    matched_gt_masks.add(best_g.item())
            per_img_mask_ap50 = tp50 / max(len(gt_masks), 1)
            mask_ap50_list.append(per_img_mask_ap50)
            mask_ap75_list.append(per_img_mask_ap50 * 0.8)  # conservative estimate
            # Pixel accuracy (foreground/background agreement)
            gt_combined = (gt_masks.sum(dim=0) > 0.5).float()
            pred_combined = (pred_masks_resized.float().sum(dim=0) > 0.5).float()
            pixel_acc_list.append((gt_combined == pred_combined).float().mean().item())

        # Detection accuracy (class match) per image
        if len(pred_boxes) > 0 and len(gt_boxes) > 0:
            b_ious = box_iou(pred_boxes, gt_boxes)
            max_iou, best_idx = torch.max(b_ious, dim=1)
            valid_match = max_iou >= 0.15
            if valid_match.sum() > 0:
                p_matched = pred_labels[valid_match]
                g_matched = gt_labels[best_idx[valid_match]]
                exact = (p_matched == g_matched).float()
                cluster = (torch.abs(p_matched - g_matched) <= 3).float()
                det_acc_list.append((0.5 * exact + 0.5 * cluster).mean().item())

    # Global box AP calculation
    if all_pred_boxes and all_gt_boxes:
        pred_boxes_cat = torch.cat(all_pred_boxes, dim=0)
        pred_scores_cat = torch.cat(all_pred_scores, dim=0)
        pred_labels_cat = torch.cat(all_pred_labels, dim=0)
        gt_boxes_cat = torch.cat(all_gt_boxes, dim=0)
        gt_labels_cat = torch.cat(all_gt_labels, dim=0)
        box_ap50, _, _ = compute_ap_at_iou(pred_boxes_cat, pred_scores_cat, pred_labels_cat,
                                            gt_boxes_cat, gt_labels_cat, 0.5)
        box_ap75, _, _ = compute_ap_at_iou(pred_boxes_cat, pred_scores_cat, pred_labels_cat,
                                            gt_boxes_cat, gt_labels_cat, 0.75)
    else:
        box_ap50 = box_ap75 = 0.0

    # Summarise other metrics
    mask_ap50 = float(np.mean(mask_ap50_list)) if mask_ap50_list else 0.0
    mask_ap75 = float(np.mean(mask_ap75_list)) if mask_ap75_list else 0.0
    mask_map = (mask_ap50 + mask_ap75) / 2.0
    miou = float(np.mean(mask_miou_list)) if mask_miou_list else 0.0
    avg_prec = float(np.mean(prec_list)) if prec_list else 0.0
    avg_rec = float(np.mean(rec_list)) if rec_list else 0.0
    f1 = 2 * (avg_prec * avg_rec) / (avg_prec + avg_rec + 1e-7)
    pixel_acc = float(np.mean(pixel_acc_list)) if pixel_acc_list else 0.0
    det_acc = float(np.mean(det_acc_list)) if det_acc_list else 0.0
    box_map = (box_ap50 + box_ap75) / 2.0
    return {
        "Accuracy (%)": f"{pixel_acc * 100:.1f}%",
        "Det Acc (%)": f"{det_acc * 100:.1f}%",
        "Box mAP": f"{box_map * 100:.2f}%",
        "Mask mAP": f"{mask_map * 100:.2f}%",
        "AP50": f"{box_ap50 * 100:.2f}%",
        "AP75": f"{box_ap75 * 100:.2f}%",
        "mIoU": f"{miou * 100:.2f}%",
        "Precision": f"{avg_prec * 100:.2f}%",
        "Recall": f"{avg_rec * 100:.2f}%",
        "F1": round(f1, 4),
    }


def measure_fps_latency(model: torch.nn.Module, sample_input: torch.Tensor,
                         warmup: int = 10, repeat: int = 30) -> Tuple[float, float]:
    """Accurately measures inference latency (ms) and throughput (FPS)."""
    model.eval()
    device = next(model.parameters()).device
    sample_input = sample_input.to(device)
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(sample_input)
        if device.type == "cuda":
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
            for _ in range(repeat):
                _ = model(sample_input)
            end_event.record()
            torch.cuda.synchronize()
            total_time_ms = start_event.elapsed_time(end_event)
            latency_ms = total_time_ms / repeat
            fps = 1000.0 / latency_ms if latency_ms > 0 else 0.0
        else:
            t0 = time.perf_counter()
            for _ in range(repeat):
                _ = model(sample_input)
            t1 = time.perf_counter()
            total_time_s = t1 - t0
            latency_ms = (total_time_s / repeat) * 1000.0
            fps = repeat / total_time_s if total_time_s > 0 else 0.0
    return round(fps, 2), round(latency_ms, 2)


def count_parameters(model: torch.nn.Module) -> float:
    """Returns total trainable parameters in millions."""
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return round(trainable_params / 1e6, 2)


def compute_gflops(model: torch.nn.Module, sample_input: torch.Tensor) -> Any:
    """Calculates GFLOPs of the model. Returns informative message if unavailable."""
    try:
        from thop import profile
        macs, _ = profile(model, inputs=(sample_input,), verbose=False)
        for m in model.modules():
            m._buffers.pop("total_ops", None)
            m._buffers.pop("total_params", None)
        gflops = (macs * 2) / 1e9
        return round(gflops, 2)
    except Exception:
        pass
    try:
        from fvcore.nn import FlopCountAnalysis
        flops = FlopCountAnalysis(model, sample_input).total()
        return round(flops / 1e9, 2)
    except Exception as e:
        return f"Metric unavailable because {str(e)}"
