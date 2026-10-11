"""
Visualization utilities for Underwater Marine Debris Detection and Instance Segmentation.

Includes:
- Bounding box and colored instance mask overlay
- Side-by-side comparison grids (Original | Ground Truth | M4 Predictions)
- Error analysis visualizer (Missed debris, false positives, small debris, low contrast)
- Loss curve and metric comparison plot generation
"""

import os
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from typing import Dict, List, Optional, Tuple, Any

def safe_plt_show():
    """Renders inline if inside Jupyter notebook; avoids GUI window block in terminal."""
    if "ipykernel" in sys.modules or "IPython" in sys.modules:
        plt.show()
    else:
        plt.close()

# Distinct high-contrast color palette for marine debris instance visualization
PALETTE = [
    (255, 56, 56), (255, 157, 151), (255, 112, 31), (255, 178, 29), (207, 210, 49),
    (72, 249, 10), (146, 204, 23), (61, 219, 134), (26, 147, 65), (13, 210, 161),
    (56, 164, 255), (0, 114, 255), (84, 42, 255), (157, 36, 255), (255, 56, 203),
    (243, 89, 162), (220, 20, 60), (255, 140, 0), (50, 205, 50), (30, 144, 255)
]


def overlay_mask_and_box(
    image: np.ndarray,
    boxes: np.ndarray,
    labels: np.ndarray,
    scores: Optional[np.ndarray] = None,
    masks: Optional[np.ndarray] = None,
    category_names: Optional[Dict[int, str]] = None,
    score_thresh: float = 0.3,
    alpha: float = 0.45
) -> np.ndarray:
    """
    Overlays colored bounding boxes, labels, confidence scores, and instance masks on an image.
    
    Args:
        image: RGB image [H, W, 3] uint8 or float [0, 1].
        boxes: [N, 4] in (x1, y1, x2, y2) absolute coordinates.
        labels: [N] class indices.
        scores: [N] confidence scores (optional).
        masks: [N, H, W] binary masks or float masks [0, 1] (optional).
        category_names: Dictionary mapping class index to label string.
        score_thresh: Confidence threshold for display.
        alpha: Mask overlay transparency.
    """
    if image.dtype != np.uint8:
        if image.max() <= 1.0:
            image = (image * 255).astype(np.uint8)
        else:
            image = image.astype(np.uint8)

    h, w, _ = image.shape
    vis_img = image.copy()
    mask_overlay = np.zeros_like(vis_img)

    num_instances = len(boxes)
    if num_instances == 0:
        return vis_img

    for i in range(num_instances):
        if scores is not None and scores[i] < score_thresh:
            continue

        color = PALETTE[i % len(PALETTE)]
        label_idx = int(labels[i])
        cls_name = category_names.get(label_idx, f"Class {label_idx}") if category_names else f"Debris_{label_idx}"
        score_str = f" {scores[i]:.2f}" if scores is not None else ""
        caption = f"{cls_name}{score_str}"

        # 1. Overlay Mask
        if masks is not None and i < len(masks):
            mask = masks[i]
            if mask.shape != (h, w):
                mask = cv2.resize(mask.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)
            binary_mask = mask > 0.5
            for c in range(3):
                mask_overlay[:, :, c] = np.where(binary_mask, color[c], mask_overlay[:, :, c])

        # 2. Draw Bounding Box
        x1, y1, x2, y2 = map(int, boxes[i])
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w - 1, x2), min(h - 1, y2)
        cv2.rectangle(vis_img, (x1, y1), (x2, y2), color, 2)

        # 3. Draw Label Tag
        (txt_w, txt_h), _ = cv2.getTextSize(caption, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        tag_y1 = max(0, y1 - txt_h - 4)
        cv2.rectangle(vis_img, (x1, tag_y1), (x1 + txt_w + 4, tag_y1 + txt_h + 4), color, -1)
        cv2.putText(vis_img, caption, (x1 + 2, tag_y1 + txt_h + 1),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    # Blend mask overlay
    has_mask = (mask_overlay.sum(axis=2) > 0)
    vis_img[has_mask] = cv2.addWeighted(vis_img, 1.0 - alpha, mask_overlay, alpha, 0)[has_mask]
    return vis_img


def display_and_save_qualitative_grid(
    image: np.ndarray,
    gt_dict: Dict[str, Any],
    pred_dict: Dict[str, Any],
    sample_id: str,
    category_names: Optional[Dict[int, str]] = None,
    save_dir: str = "./results/qualitative_results",
    score_thresh: float = 0.0
) -> np.ndarray:
    """
    Renders a 3-panel comparison:
    [1. Original Image] | [2. Ground Truth (Boxes + Masks)] | [3. M4 Predictions (Boxes + Masks)]
    Displays directly in Jupyter and saves to disk.
    """
    os.makedirs(save_dir, exist_ok=True)

    gt_vis = overlay_mask_and_box(
        image=image,
        boxes=gt_dict.get("boxes", np.empty((0, 4))),
        labels=gt_dict.get("labels", np.empty((0,))),
        masks=gt_dict.get("masks", None),
        category_names=category_names,
        score_thresh=0.0
    )

    pred_vis = overlay_mask_and_box(
        image=image,
        boxes=pred_dict.get("boxes", np.empty((0, 4))),
        labels=pred_dict.get("labels", np.empty((0,))),
        scores=pred_dict.get("scores", None),
        masks=pred_dict.get("masks", None),
        category_names=category_names,
        score_thresh=score_thresh
    )

    # Plot figure with 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), dpi=120)
    axes[0].imshow(image)
    axes[0].set_title(f"[{sample_id}] Original Underwater Image", fontsize=12, fontweight="bold")
    axes[0].axis("off")

    axes[1].imshow(gt_vis)
    axes[1].set_title("Ground Truth (Boxes + Masks)", fontsize=12, fontweight="bold")
    axes[1].axis("off")

    axes[2].imshow(pred_vis)
    axes[2].set_title("M4 Proposed Model Predictions", fontsize=12, fontweight="bold")
    axes[2].axis("off")

    plt.tight_layout()
    out_path = os.path.join(save_dir, f"{sample_id}_comparison_grid.png")
    plt.savefig(out_path, bbox_inches="tight")
    safe_plt_show()
    plt.close(fig)

    return pred_vis


def visualize_error_cases(
    error_samples: List[Dict[str, Any]],
    category_names: Optional[Dict[int, str]] = None,
    save_dir: str = "./results/error_analysis"
):
    """
    Visualizes qualitative error modes:
    - Missed debris (False Negatives)
    - False positives (Hallucinations)
    - Small debris & Low-contrast debris
    """
    os.makedirs(save_dir, exist_ok=True)
    if not error_samples:
        print("[INFO] No error samples to display.")
        return

    n_samples = min(len(error_samples), 4)
    fig, axes = plt.subplots(n_samples, 2, figsize=(12, 4 * n_samples), dpi=120)
    if n_samples == 1:
        axes = np.expand_dims(axes, 0)

    for i in range(n_samples):
        item = error_samples[i]
        img = item["image"]
        gt_vis = overlay_mask_and_box(
            img, item["gt_boxes"], item["gt_labels"],
            masks=item.get("gt_masks"), category_names=category_names
        )
        pred_vis = overlay_mask_and_box(
            img, item["pred_boxes"], item["pred_labels"],
            scores=item.get("pred_scores"), masks=item.get("pred_masks"),
            category_names=category_names
        )

        error_type = item.get("error_type", "Detection Discrepancy")
        axes[i, 0].imshow(gt_vis)
        axes[i, 0].set_title(f"Sample #{i+1}: Ground Truth ({error_type})", fontsize=11)
        axes[i, 0].axis("off")

        axes[i, 1].imshow(pred_vis)
        axes[i, 1].set_title(f"Sample #{i+1}: M4 Output ({item.get('note', '')})", fontsize=11)
        axes[i, 1].axis("off")

    plt.tight_layout()
    out_path = os.path.join(save_dir, "error_analysis_grid.png")
    plt.savefig(out_path, bbox_inches="tight")
    safe_plt_show()
    plt.close(fig)


def plot_metrics_and_curves(
    training_history: Dict[str, List[Dict[str, Any]]],
    comparison_df: Any,
    save_dir: str = "./results/plots"
):
    """
    Generates all required experimental evaluation plots:
    - Training loss curves across epochs (total, cls, box, mask)
    - Model comparison bar charts (Mask mAP, AP50, AP75, mIoU, FPS, Params)
    - FPS vs Mask mAP efficiency trade-off scatter plot
    """
    os.makedirs(save_dir, exist_ok=True)

    # 1. Training Loss Curves
    fig, ax = plt.subplots(figsize=(9, 5), dpi=120)
    for model_name, hist in training_history.items():
        if not hist:
            continue
        epochs = [h["Epoch"] for h in hist]
        losses = [h["Loss"] for h in hist]
        ax.plot(epochs, losses, marker="o", label=f"{model_name} Total Loss", linewidth=2)

    ax.set_title("Training Loss Trajectory Across Epochs", fontsize=13, fontweight="bold")
    ax.set_xlabel("Epoch", fontsize=11)
    ax.set_ylabel("Loss", fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(fontsize=10)
    loss_path = os.path.join(save_dir, "training_loss_curves.png")
    plt.savefig(loss_path, bbox_inches="tight")
    safe_plt_show()
    plt.close(fig)

    # 2. Metric Comparisons Bar Chart
    if comparison_df is not None and not comparison_df.empty:
        fig, axes = plt.subplots(1, 3, figsize=(18, 5), dpi=120)

        def _to_num(series):
            return np.array([
                float(v.rstrip('%')) / 100.0 if isinstance(v, str) and v.endswith('%')
                else float(v) if isinstance(v, (int, float, np.number)) or (isinstance(v, str) and v.replace('.', '', 1).isdigit())
                else 0.0
                for v in series
            ])

        models = comparison_df["Model"].tolist()
        x = np.arange(len(models))
        width = 0.25

        # Subplot 1: Mask mAP, AP50, AP75
        axes[0].bar(x - width, _to_num(comparison_df.get("Mask mAP", [0]*len(models))), width, label="Mask mAP", color="#1f77b4")
        axes[0].bar(x, _to_num(comparison_df.get("AP50", [0]*len(models))), width, label="AP50", color="#ff7f0e")
        axes[0].bar(x + width, _to_num(comparison_df.get("AP75", [0]*len(models))), width, label="AP75", color="#2ca02c")
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(models, fontweight="bold")
        axes[0].set_title("Detection & Mask Performance (AP)", fontweight="bold")
        axes[0].set_ylabel("Score")
        axes[0].grid(axis="y", linestyle="--", alpha=0.5)
        axes[0].legend()

        # Subplot 2: mIoU, Precision, Recall
        axes[1].bar(x - width, _to_num(comparison_df.get("mIoU", [0]*len(models))), width, label="mIoU", color="#9467bd")
        axes[1].bar(x, _to_num(comparison_df.get("Precision", [0]*len(models))), width, label="Precision", color="#8c564b")
        axes[1].bar(x + width, _to_num(comparison_df.get("Recall", [0]*len(models))), width, label="Recall", color="#e377c2")
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(models, fontweight="bold")
        axes[1].set_title("Segmentation & Quality Metrics", fontweight="bold")
        axes[1].set_ylabel("Score")
        axes[1].grid(axis="y", linestyle="--", alpha=0.5)
        axes[1].legend()

        # Subplot 3: FPS vs Parameter Count
        ax_fps = axes[2]
        color_fps = "#17becf"
        ax_fps.set_xlabel("Model", fontweight="bold")
        ax_fps.set_ylabel("FPS (Throughput)", color=color_fps, fontweight="bold")
        ax_fps.bar(x - 0.15, comparison_df["FPS"], 0.3, color=color_fps, label="FPS")
        ax_fps.tick_params(axis="y", labelcolor=color_fps)
        ax_fps.set_xticks(x)
        ax_fps.set_xticklabels(models)

        # Secondary y-axis for Params
        ax_params = ax_fps.twinx()
        color_params = "#d62728"
        ax_params.set_ylabel("Parameters (M)", color=color_params, fontweight="bold")
        ax_params.plot(x + 0.15, comparison_df["Params"], color=color_params, marker="s", linewidth=2.5, label="Params (M)")
        ax_params.tick_params(axis="y", labelcolor=color_params)
        axes[2].set_title("Runtime Efficiency & Parameters", fontweight="bold")

        plt.tight_layout()
        metrics_path = os.path.join(save_dir, "model_comparison_metrics.png")
        plt.savefig(metrics_path, bbox_inches="tight")
        safe_plt_show()
        plt.close(fig)
