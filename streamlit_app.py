import os
import torch
import cv2
import numpy as np
import streamlit as st
from PIL import Image

from src.models.m4_proposed import ProposedMaskRTDETR_IAM_YOLACT
from src.utils.visualization import overlay_mask_and_box

st.set_page_config(page_title="SeaClear Marine Debris Segmentation", layout="wide")

st.title("🌊 SeaClear Marine Debris Segmentation (M4 Architecture)")
st.markdown("Real-time underwater instance segmentation powered by **Mask RT-DETR + SparseInst IAM + YOLACT Prototypes**.")

# 1. Hardware & Model Cache
@st.cache_resource
def load_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ckpt_path = os.path.join('checkpoints', 'M4_5cls', 'best.pth')
    num_classes = 5
    if not os.path.exists(ckpt_path):
        ckpt_path = os.path.join('checkpoints', 'M4', 'best.pth')
        num_classes = 40
    model = ProposedMaskRTDETR_IAM_YOLACT(num_classes=num_classes, num_queries=100)
    if os.path.exists(ckpt_path):
        sd = torch.load(ckpt_path, map_location=device)['model_state_dict']
        sd_clean = {k: v for k, v in sd.items() if 'total_ops' not in k and 'total_params' not in k}
        model.load_state_dict(sd_clean, strict=False)
    model.to(device).eval()
    return model, device, num_classes

model, device, num_classes = load_model()

CATEGORIES_5 = {
    0: 'Plastic & Packaging',
    1: 'Metal & Structural',
    2: 'Glass, Ceramic & Construction',
    3: 'Fishing Gear, Ropes & Rubber',
    4: 'Bio / Organic & Other'
}
CATEGORIES = CATEGORIES_5

# Sidebar Controls
st.sidebar.header("Inference Settings")
score_thresh = st.sidebar.slider("Confidence Threshold", 0.01, 0.90, 0.15, 0.01)
mask_thresh = st.sidebar.slider("Mask Threshold", 0.01, 0.50, 0.05, 0.01)
top_k = st.sidebar.slider("Candidate Query Fallback Count", 1, 10, 3, 1)

uploaded_file = st.file_uploader("Upload an underwater image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    pil_img = Image.open(uploaded_file).convert("RGB")
    input_img = np.array(pil_img)
    orig_h, orig_w, _ = input_img.shape

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Original Image")
        st.image(input_img, use_container_width=True)

    # Preprocessing
    resized = cv2.resize(input_img, (640, 640))
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)
    norm = ((resized / 255.0) - mean) / std
    tensor = torch.from_numpy(norm).permute(2, 0, 1).unsqueeze(0).float().to(device)

    with torch.no_grad():
        out = model(tensor)
        logits = out['pred_logits'][0]
        boxes = out['pred_boxes'][0]
        masks = out.get('pred_masks', [None])[0]

    scores, labels = logits.sigmoid().max(dim=-1)
    keep = scores >= score_thresh
    if not keep.any():
        _, top_idx = torch.topk(scores, min(top_k, len(scores)))
        keep = torch.zeros_like(scores, dtype=torch.bool)
        keep[top_idx] = True

    s = scores[keep].cpu().numpy()
    l = labels[keep].cpu().numpy()
    b = boxes[keep]

    cx, cy, bw, bh = b.unbind(-1)
    x1 = ((cx - 0.5 * bw).clamp(0, 1) * orig_w).cpu().numpy()
    y1 = ((cy - 0.5 * bh).clamp(0, 1) * orig_h).cpu().numpy()
    x2 = ((cx + 0.5 * bw).clamp(0, 1) * orig_w).cpu().numpy()
    y2 = ((cy + 0.5 * bh).clamp(0, 1) * orig_h).cpu().numpy()
    pred_boxes = np.stack([x1, y1, x2, y2], axis=-1)
    pred_masks = (masks[keep] > mask_thresh).float().cpu().numpy() if masks is not None else None

    vis_img = overlay_mask_and_box(
        image=input_img,
        boxes=pred_boxes,
        labels=l,
        scores=s,
        masks=pred_masks,
        category_names=CATEGORIES,
        score_thresh=0.0
    )

    with col2:
        st.subheader("Model Predictions (M4)")
        st.image(vis_img, use_container_width=True)

    st.subheader(f"Detections Report ({len(s)} objects identified)")
    for i, (score_val, label_idx) in enumerate(zip(s, l)):
        cls_name = CATEGORIES.get(int(label_idx), f"Class {label_idx}")
        st.write(f"- **Object {i+1}**: `{cls_name}` — Confidence: **{score_val:.2%}**")
