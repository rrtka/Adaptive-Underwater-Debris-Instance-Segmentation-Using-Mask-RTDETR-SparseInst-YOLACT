"""
Evaluation Engine for Quantitative and Qualitative Testing.

Computes:
- Mask mAP, Box mAP, AP50, AP75, mIoU, Precision, Recall, F1
- FPS, Latency (ms), Trainable Parameters (M), GFLOPs
- Collects predictions for qualitative and error analysis
"""

import time
import torch
import numpy as np
from typing import Dict, List, Tuple, Any, Optional

from ..utils.metrics import evaluate_batch_metrics, measure_fps_latency, count_parameters, compute_gflops
from .memory import clear_gpu_memory


def run_model_inference_on_batch(
    model: torch.nn.Module,
    images: torch.Tensor,
    score_thresh: float = 0.3,
    qualitative_fallback: bool = False
) -> List[Dict[str, torch.Tensor]]:
    """
    Decodes model outputs into formatted predictions:
    - boxes: [N, 4] in (x1, y1, x2, y2) normalized [0, 1]
    - scores: [N]
    - labels: [N]
    - masks: [N, H, W]

    Args:
        qualitative_fallback: When True, falls back to top-3 highest-scoring queries if
                              nothing meets score_thresh. Use ONLY for display purposes,
                              NEVER when computing quantitative metrics.
    """
    outputs = model(images)
    pred_logits = outputs["pred_logits"]  # [B, Q, num_classes]
    pred_boxes = outputs["pred_boxes"]    # [B, Q, 4] normalized [cx, cy, w, h]
    pred_masks = outputs.get("pred_masks", None)  # [B, Q, H, W] — already sigmoid-activated by models

    probs = pred_logits.sigmoid()  # [B, Q, num_classes]
    scores, labels = probs.max(dim=-1)  # [B, Q]

    batch_preds = []
    b = images.shape[0]

    for bi in range(b):
        keep = scores[bi] >= score_thresh

        if not keep.any() and qualitative_fallback and len(scores[bi]) > 0:
            # Fallback ONLY for qualitative display: take top-3 queries regardless of score
            _, top_idx = torch.topk(scores[bi], min(3, len(scores[bi])))
            keep = torch.zeros_like(scores[bi], dtype=torch.bool)
            keep[top_idx] = True

        s = scores[bi][keep]
        l = labels[bi][keep]
        boxes_cxcywh = pred_boxes[bi][keep]

        if len(boxes_cxcywh) > 0:
            cx, cy, w, h = boxes_cxcywh.unbind(-1)
            x1 = (cx - 0.5 * w).clamp(min=0.0, max=1.0)
            y1 = (cy - 0.5 * h).clamp(min=0.0, max=1.0)
            x2 = (cx + 0.5 * w).clamp(min=0.0, max=1.0)
            y2 = (cy + 0.5 * h).clamp(min=0.0, max=1.0)
            xyxy_boxes = torch.stack([x1, y1, x2, y2], dim=-1)
        else:
            xyxy_boxes = torch.empty((0, 4), device=images.device)

        # Masks already sigmoid-activated by the model forward() — pass through as-is
        m = pred_masks[bi][keep] if pred_masks is not None else torch.empty((0, 128, 128), device=images.device)

        batch_preds.append({
            "boxes": xyxy_boxes.cpu(),
            "scores": s.cpu(),
            "labels": l.cpu(),
            "masks": m.cpu()
        })

    return batch_preds


def evaluate_model(
    model: torch.nn.Module,
    data_loader: Any,
    device: torch.device,
    model_name: str = "Model",
    score_thresh: float = 0.3
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """
    Evaluates model quantitatively and gathers test predictions.
    - Metric computation uses score_thresh strictly (no fallback).
    - Qualitative samples use qualitative_fallback=True so visualizations are non-empty.
    """
    model.eval()
    all_preds = []
    all_targets = []
    qualitative_samples = []

    with torch.no_grad():
        for step, (images, targets) in enumerate(data_loader):
            images = images.to(device)

            # Strict threshold for metrics — no fallback inflation
            preds = run_model_inference_on_batch(model, images, score_thresh=score_thresh,
                                                 qualitative_fallback=False)

            # Convert targets to normalized xyxy for IoU evaluation
            formatted_targets = []
            for t in targets:
                t_boxes = t["boxes"]
                if len(t_boxes) > 0:
                    cx, cy, w, h = t_boxes.unbind(-1)
                    x1 = (cx - 0.5 * w).clamp(min=0.0, max=1.0)
                    y1 = (cy - 0.5 * h).clamp(min=0.0, max=1.0)
                    x2 = (cx + 0.5 * w).clamp(min=0.0, max=1.0)
                    y2 = (cy + 0.5 * h).clamp(min=0.0, max=1.0)
                    t_xyxy = torch.stack([x1, y1, x2, y2], dim=-1)
                else:
                    t_xyxy = torch.empty((0, 4))

                formatted_targets.append({
                    "boxes": t_xyxy.cpu(),
                    "labels": t["labels"].cpu(),
                    "masks": t["masks"].cpu()
                })

            all_preds.extend(preds)
            all_targets.extend(formatted_targets)

            # Retain sample for qualitative analysis — use fallback so grids are non-empty
            if len(qualitative_samples) < 5:
                qual_preds = run_model_inference_on_batch(model, images, score_thresh=score_thresh,
                                                          qualitative_fallback=True)
                # Denormalize image for visualization
                img_np = images[0].cpu().permute(1, 2, 0).numpy()
                mean = np.array([0.485, 0.456, 0.406])
                std = np.array([0.229, 0.224, 0.225])
                vis_img = np.clip((img_np * std + mean) * 255.0, 0, 255).astype(np.uint8)

                qualitative_samples.append({
                    "image": vis_img,
                    "gt_boxes": targets[0]["orig_boxes"].cpu().numpy(),
                    "gt_labels": targets[0]["labels"].cpu().numpy(),
                    "gt_masks": targets[0]["masks"].cpu().numpy(),
                    "pred": qual_preds[0],
                    "image_id": targets[0]["image_id"].item(),
                    "orig_size": targets[0]["orig_size"].cpu().numpy(),
                })

    # Quantitative metrics
    metrics = evaluate_batch_metrics(all_preds, all_targets)

    # Throughput & latency
    sample_input = torch.randn(1, 3, 640, 640, device=device)
    fps, latency = measure_fps_latency(model, sample_input, warmup=5, repeat=15)
    params = count_parameters(model)
    gflops = compute_gflops(model, sample_input)

    metrics["Model"] = model_name
    metrics["FPS"] = fps
    metrics["Latency"] = latency
    metrics["Params"] = params
    metrics["GFLOPs"] = gflops

    clear_gpu_memory()
    return metrics, qualitative_samples
