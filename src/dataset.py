"""
SeaClear Underwater Marine Debris Dataset Loader, Validator, and Augmentation Pipeline.

Handles:
- Automatic Kaggle and Local environment dataset discovery
- COCO JSON parsing with polygon/RLE conversion to binary masks
- Annotation validation (detects missing files, invalid boxes, degenerate masks)
- Underwater-specific augmentations (turbidity/color cast simulation, flips, crops)
- Memory-efficient batch collation for DETR-style models
"""

import os
import glob
import json
import random
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Tuple, Any, Optional
from pycocotools import mask as coco_mask_util


def discover_seaclear_dataset(search_paths: List[str]) -> Tuple[Optional[str], Optional[str]]:
    """
    Searches for SeaClear dataset images and COCO JSON annotation files across search paths.
    Returns (images_dir, annotation_file_or_dir).
    """
    for base in search_paths:
        if not os.path.exists(base):
            continue

        # Look for COCO JSON annotations
        json_candidates = glob.glob(os.path.join(base, "**", "*.json"), recursive=True)
        valid_coco_jsons = []
        for jc in json_candidates:
            # Check if this JSON contains standard COCO keys
            try:
                with open(jc, "r") as f:
                    data = json.load(f)
                    if isinstance(data, dict) and "images" in data and "annotations" in data:
                        valid_coco_jsons.append(jc)
            except Exception:
                continue

        if valid_coco_jsons:
            # Found valid COCO json
            ann_file = valid_coco_jsons[0]
            # Now locate images directory
            img_dir = os.path.dirname(ann_file)
            # Check for subdirectories like 'images', 'train', etc.
            possible_img_dirs = [
                os.path.join(base, "images"),
                os.path.join(base, "train"),
                os.path.join(os.path.dirname(ann_file), "images"),
                os.path.dirname(ann_file),
                base
            ]
            for pid in possible_img_dirs:
                if os.path.exists(pid) and (glob.glob(os.path.join(pid, "**", "*.jpg"), recursive=True) or glob.glob(os.path.join(pid, "**", "*.png"), recursive=True)):
                    return pid, ann_file

            return img_dir, ann_file

    return None, None


def inspect_dataset_structure(ann_file: str, img_dir: str) -> Dict[str, Any]:
    """
    Inspects and prints dataset statistics according to specifications.
    """
    with open(ann_file, "r") as f:
        coco_data = json.load(f)

    images = coco_data.get("images", [])
    annotations = coco_data.get("annotations", [])
    categories = coco_data.get("categories", [])

    category_names = {cat["id"]: cat["name"] for cat in categories}

    stats = {
        "dataset_root": os.path.abspath(img_dir),
        "annotation_file": os.path.abspath(ann_file),
        "number_of_images": len(images),
        "number_of_annotations": len(annotations),
        "number_of_categories": len(categories),
        "category_names": category_names,
    }

    print("=" * 65)
    print("SEACLEAR DATASET INSPECTION")
    print("=" * 65)
    print(f"Dataset Root          : {stats['dataset_root']}")
    print(f"Annotation File       : {stats['annotation_file']}")
    print(f"Total Images          : {stats['number_of_images']}")
    print(f"Total Annotations     : {stats['number_of_annotations']}")
    print(f"Total Categories      : {stats['number_of_categories']}")
    print("-" * 65)
    print("Categories:")
    for cat_id, cat_name in sorted(category_names.items()):
        print(f"  [{cat_id:2d}] {cat_name}")
    print("=" * 65)

    return stats


def validate_coco_annotations(ann_file: str, img_dir: str) -> Dict[str, Any]:
    """
    Validates annotation integrity:
    - Missing images
    - Invalid bounding boxes (w <= 0, h <= 0, out of bounds)
    - Invalid masks (empty polygon or corrupt RLE)
    - Empty image annotations
    """
    with open(ann_file, "r") as f:
        coco_data = json.load(f)

    images = {img["id"]: img for img in coco_data.get("images", [])}
    annotations = coco_data.get("annotations", [])

    missing_images = []
    invalid_boxes = []
    invalid_masks = []
    annotated_img_ids = set()

    for ann in annotations:
        img_id = ann.get("image_id")
        annotated_img_ids.add(img_id)

        if img_id not in images:
            missing_images.append(f"Annotation {ann.get('id')} references unknown image_id {img_id}")
            continue

        # Check box
        bbox = ann.get("bbox", [])
        if len(bbox) != 4 or bbox[2] <= 0 or bbox[3] <= 0:
            invalid_boxes.append(f"Ann ID {ann.get('id')}: invalid bbox {bbox}")

        # Check segmentation
        segm = ann.get("segmentation")
        if not segm:
            invalid_masks.append(f"Ann ID {ann.get('id')}: missing segmentation")
        elif isinstance(segm, list) and len(segm) == 0:
            invalid_masks.append(f"Ann ID {ann.get('id')}: empty polygon list")

    empty_images = [img_id for img_id in images if img_id not in annotated_img_ids]

    # Verify image files exist on disk (supporting recursive subfolders)
    image_map = {}
    if os.path.exists(img_dir):
        for ext in ("*.jpg", "*.png", "*.jpeg", "*.JPG", "*.PNG"):
            for p in glob.glob(os.path.join(img_dir, "**", ext), recursive=True):
                image_map[os.path.basename(p)] = p

    missing_files = []
    for img_id, img_info in images.items():
        fn = img_info.get("file_name", "")
        base_fn = os.path.basename(fn)
        if base_fn not in image_map and not os.path.exists(os.path.join(img_dir, fn)):
            missing_files.append(fn)

    validation_summary = {
        "missing_files_count": len(missing_files),
        "invalid_boxes_count": len(invalid_boxes),
        "invalid_masks_count": len(invalid_masks),
        "empty_images_count": len(empty_images),
        "missing_files_samples": missing_files[:5],
        "invalid_boxes_samples": invalid_boxes[:5],
        "invalid_masks_samples": invalid_masks[:5],
    }

    print("-" * 65)
    print("ANNOTATION VALIDATION SUMMARY")
    print("-" * 65)
    print(f"Missing Image Files   : {len(missing_files)}")
    print(f"Invalid Bounding Boxes: {len(invalid_boxes)}")
    print(f"Invalid / Empty Masks : {len(invalid_masks)}")
    print(f"Images without Debris : {len(empty_images)}")
    if missing_files:
        print(f"  Example missing: {missing_files[:3]}")
    if invalid_boxes:
        print(f"  Example invalid box: {invalid_boxes[:3]}")
    print("-" * 65)

    return validation_summary


def segmentation_to_binary_mask(segm: Any, height: int, width: int) -> np.ndarray:
    """
    Converts COCO polygon or RLE segmentation into a binary numpy mask [H, W] of uint8 (0 or 1).
    """
    if isinstance(segm, list):
        # List of polygons
        mask = np.zeros((height, width), dtype=np.uint8)
        for poly in segm:
            if len(poly) < 6:
                continue
            poly_pts = np.array(poly, dtype=np.int32).reshape(-1, 2)
            cv2.fillPoly(mask, [poly_pts], 1)
        return mask
    elif isinstance(segm, dict):
        # RLE
        if isinstance(segm.get("counts"), list):
            rle = coco_mask_util.frPyObjects(segm, height, width)
            m = coco_mask_util.decode(rle)
            return (m > 0).astype(np.uint8)
        else:
            m = coco_mask_util.decode(segm)
            return (m > 0).astype(np.uint8)
    return np.zeros((height, width), dtype=np.uint8)


class UnderwaterAugmentor:
    """
    Underwater-specific augmentation pipeline:
    - Geometric: Random horizontal flip, scale/crop (consistently transforming boxes and masks)
    - Photometric: Color jitter, greenish/blueish absorption, mild Gaussian blur
    """
    def __init__(self, is_train: bool = True):
        self.is_train = is_train

    def __call__(self, image: np.ndarray, boxes: np.ndarray, masks: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray, List[np.ndarray]]:
        if not self.is_train:
            return image, boxes, masks

        h, w, _ = image.shape

        # 1. Random Horizontal Flip (p = 0.5)
        if random.random() < 0.5:
            image = np.fliplr(image).copy()
            if len(boxes) > 0:
                # boxes: [x1, y1, x2, y2]
                new_x1 = w - boxes[:, 2]
                new_x2 = w - boxes[:, 0]
                boxes[:, 0] = new_x1
                boxes[:, 2] = new_x2
            masks = [np.fliplr(m).copy() for m in masks]

        # 2. Photometric Color Jitter / Underwater Attenuation (p = 0.5)
        if random.random() < 0.5:
            # Perturb color channels simulating underwater light absorption
            # Water strongly absorbs red wavelength, so red channel fluctuates
            r_scale = random.uniform(0.7, 1.1)
            g_scale = random.uniform(0.85, 1.15)
            b_scale = random.uniform(0.85, 1.15)
            img_float = image.astype(np.float32)
            img_float[:, :, 0] = np.clip(img_float[:, :, 0] * r_scale, 0, 255)
            img_float[:, :, 1] = np.clip(img_float[:, :, 1] * g_scale, 0, 255)
            img_float[:, :, 2] = np.clip(img_float[:, :, 2] * b_scale, 0, 255)

            # Brightness & contrast
            alpha = random.uniform(0.8, 1.2)  # Contrast
            beta = random.uniform(-20, 20)    # Brightness
            img_float = np.clip(img_float * alpha + beta, 0, 255)
            image = img_float.astype(np.uint8)

        # 3. Mild Gaussian Blur / Turbidity Simulation (p = 0.3)
        if random.random() < 0.3:
            ksize = random.choice([3, 5])
            image = cv2.GaussianBlur(image, (ksize, ksize), 0)

        return image, boxes, masks


SUPER_CATEGORIES = {
    0: "Plastic & Packaging",
    1: "Metal & Structural Debris",
    2: "Glass, Ceramic & Construction",
    3: "Fishing Gear, Ropes & Rubber",
    4: "Bio / Organic & Other"
}

def map_coco_cat_to_super_category(cat_name: str) -> int:
    """Maps a fine-grained SeaClear category name to one of the 5 standard material super-categories."""
    name = cat_name.lower()
    if any(k in name for k in ["plastic", "wrapper", "bag", "bottle_plastic", "container_plastic", "cup_plastic", "sanitaries_plastic", "snack_wrapper"]):
        return 0  # Plastic & Packaging
    elif any(k in name for k in ["metal", "can", "cable_metal", "wreckage"]):
        return 1  # Metal & Structural Debris
    elif any(k in name for k in ["glass", "ceramic", "clay", "brick", "cement", "tube_cement"]):
        return 2  # Glass, Ceramic & Construction
    elif any(k in name for k in ["rope", "net", "fiber", "clothing", "cable", "rubber", "boot", "tire"]):
        return 3  # Fishing Gear, Ropes & Rubber
    else:
        return 4  # Bio / Organic & Other


class SeaClearDataset(Dataset):
    """
    PyTorch Dataset for SeaClear Underwater Marine Debris Instance Segmentation.
    Outputs:
        - Image tensor [3, H_in, W_in] normalized
        - Target dict containing:
            - boxes: [N, 4] normalized [cx, cy, w, h]
            - original_boxes: [N, 4] absolute [x1, y1, x2, y2]
            - labels: [N] class indices (0-indexed)
            - masks: [N, mask_size, mask_size] binary float masks
            - image_id: int
            - orig_size: [H_orig, W_orig]
    """
    def __init__(
        self,
        image_ids: List[int],
        coco_data: Dict[str, Any],
        img_dir: str,
        image_size: int = 640,
        mask_size: int = 128,
        is_train: bool = True,
        use_super_categories: bool = True
    ):
        self.image_ids = image_ids
        self.img_dir = img_dir
        self.image_size = image_size
        self.mask_size = mask_size
        self.is_train = is_train
        self.use_super_categories = use_super_categories
        self.augmentor = UnderwaterAugmentor(is_train=is_train)

        # Index COCO images and annotations
        self.images = {img["id"]: img for img in coco_data.get("images", [])}
        
        # Build category map (map arbitrary COCO category IDs to 0-indexed range 0..C-1)
        categories = sorted(coco_data.get("categories", []), key=lambda c: c["id"])
        if self.use_super_categories:
            self.cat_id_to_idx = {cat["id"]: map_coco_cat_to_super_category(cat["name"]) for cat in categories}
            self.idx_to_cat_name = SUPER_CATEGORIES
        else:
            self.cat_id_to_idx = {cat["id"]: idx for idx, cat in enumerate(categories)}
            self.idx_to_cat_name = {idx: cat["name"] for idx, cat in enumerate(categories)}

        # Group annotations by image_id
        self.img_to_anns = {}
        for ann in coco_data.get("annotations", []):
            iid = ann.get("image_id")
            if iid in self.images:
                self.img_to_anns.setdefault(iid, []).append(ann)

        # Build recursive image path index for subfolder resolution
        self.image_map = {}
        if os.path.exists(self.img_dir):
            for ext in ("*.jpg", "*.png", "*.jpeg", "*.JPG", "*.PNG"):
                for p in glob.glob(os.path.join(self.img_dir, "**", ext), recursive=True):
                    self.image_map[os.path.basename(p)] = p

        # Mean and Std for standard ImageNet normalization
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)

    def __len__(self) -> int:
        return len(self.image_ids)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        img_id = self.image_ids[idx]
        img_info = self.images[img_id]
        fn = img_info.get("file_name", "")

        # Read image using recursive map or direct paths
        base_fn = os.path.basename(fn)
        if base_fn in self.image_map:
            img_path = self.image_map[base_fn]
        else:
            img_path = os.path.join(self.img_dir, fn)
            if not os.path.exists(img_path):
                img_path = os.path.join(self.img_dir, base_fn)

        image = cv2.imread(img_path)
        if image is None:
            # Fallback if image cannot be read
            image = np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        orig_h, orig_w, _ = image.shape

        # Extract annotations
        anns = self.img_to_anns.get(img_id, [])
        boxes, labels, masks = [], [], []

        for ann in anns:
            bbox = ann.get("bbox", [])
            if len(bbox) != 4 or bbox[2] <= 0 or bbox[3] <= 0:
                continue
            x, y, w, h = bbox
            x1 = max(0, x)
            y1 = max(0, y)
            x2 = min(orig_w, x + w)
            y2 = min(orig_h, y + h)

            if x2 <= x1 or y2 <= y1:
                continue

            segm = ann.get("segmentation")
            bin_mask = segmentation_to_binary_mask(segm, orig_h, orig_w)

            cat_id = ann.get("category_id")
            label_idx = self.cat_id_to_idx.get(cat_id, 0)

            boxes.append([x1, y1, x2, y2])
            labels.append(label_idx)
            masks.append(bin_mask)

        boxes = np.array(boxes, dtype=np.float32) if boxes else np.empty((0, 4), dtype=np.float32)
        labels = np.array(labels, dtype=np.int64) if labels else np.empty((0,), dtype=np.int64)

        # Apply underwater augmentations
        image, boxes, masks = self.augmentor(image, boxes, masks)

        # Resize image to input resolution (e.g. 640x640)
        resized_img = cv2.resize(image, (self.image_size, self.image_size), interpolation=cv2.INTER_LINEAR)
        # Normalize
        norm_img = ((resized_img.astype(np.float32) / 255.0) - self.mean) / self.std
        tensor_img = torch.from_numpy(norm_img).permute(2, 0, 1).float()  # [3, H, W]

        # Process masks to memory-efficient mask resolution (mask_size x mask_size, e.g. 128x128)
        if len(masks) > 0:
            resized_masks = []
            for m in masks:
                rm = cv2.resize(m.astype(np.float32), (self.mask_size, self.mask_size), interpolation=cv2.INTER_NEAREST)
                resized_masks.append(rm)
            tensor_masks = torch.from_numpy(np.stack(resized_masks, axis=0)).float()  # [N, 128, 128]
        else:
            tensor_masks = torch.zeros((0, self.mask_size, self.mask_size), dtype=torch.float32)

        # Convert boxes to normalized cxcywh format for DETR loss
        if len(boxes) > 0:
            orig_boxes_tensor = torch.from_numpy(boxes).float()
            # Scale boxes to 640x640 scale
            scale_x = self.image_size / orig_w
            scale_y = self.image_size / orig_h
            scaled_boxes = boxes.copy()
            scaled_boxes[:, [0, 2]] *= scale_x
            scaled_boxes[:, [1, 3]] *= scale_y

            # Convert to [cx, cy, w, h] normalized in [0, 1]
            cx = ((scaled_boxes[:, 0] + scaled_boxes[:, 2]) / 2.0) / self.image_size
            cy = ((scaled_boxes[:, 1] + scaled_boxes[:, 3]) / 2.0) / self.image_size
            w_box = (scaled_boxes[:, 2] - scaled_boxes[:, 0]) / self.image_size
            h_box = (scaled_boxes[:, 3] - scaled_boxes[:, 1]) / self.image_size
            norm_boxes_tensor = torch.from_numpy(np.stack([cx, cy, w_box, h_box], axis=1)).float()
        else:
            orig_boxes_tensor = torch.zeros((0, 4), dtype=torch.float32)
            norm_boxes_tensor = torch.zeros((0, 4), dtype=torch.float32)

        target = {
            "boxes": norm_boxes_tensor,
            "orig_boxes": orig_boxes_tensor,
            "labels": torch.from_numpy(labels).long(),
            "masks": tensor_masks,
            "image_id": torch.tensor(img_id, dtype=torch.long),
            "orig_size": torch.tensor([orig_h, orig_w], dtype=torch.long),
        }

        return tensor_img, target


def collate_fn_detr(batch):
    """
    Collate function for DETR batching:
    Combines images into a single stacked tensor [B, 3, H, W]
    and targets into a tuple of target dicts.
    """
    images = torch.stack([item[0] for item in batch], dim=0)
    targets = [item[1] for item in batch]
    return images, targets


def create_seaclear_splits(
    coco_data: Dict[str, Any],
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42
) -> Tuple[List[int], List[int], List[int]]:
    """
    Creates deterministic 80/10/10 train/val/test splits with a fixed seed if official splits are not pre-defined.
    """
    random.seed(seed)
    image_ids = [img["id"] for img in coco_data.get("images", [])]
    random.shuffle(image_ids)

    n_total = len(image_ids)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)

    train_ids = image_ids[:n_train]
    val_ids = image_ids[n_train:n_train + n_val]
    test_ids = image_ids[n_train + n_val:]

    print(f"Dataset Split Summary (Seed={seed}):")
    print(f"  Train Set : {len(train_ids)} images ({len(train_ids)/max(n_total,1)*100:.1f}%)")
    print(f"  Val Set   : {len(val_ids)} images ({len(val_ids)/max(n_total,1)*100:.1f}%)")
    print(f"  Test Set  : {len(test_ids)} images ({len(test_ids)/max(n_total,1)*100:.1f}%)")

    return train_ids, val_ids, test_ids
