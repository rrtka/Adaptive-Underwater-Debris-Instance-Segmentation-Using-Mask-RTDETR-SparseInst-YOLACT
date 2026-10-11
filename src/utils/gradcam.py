"""
Grad-CAM (Gradient-weighted Class Activation Mapping) for Marine Debris Detection.

Supports:
- CNN Backbones (e.g. ResNet layer4)
- RT-DETR Hybrid Encoders / CCFM feature stages
- Color heatmap generation and underwater image overlay
"""

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Union


class DebrisGradCAM:
    """
    Computes Grad-CAM heatmaps for detection models.
    Hooks into target convolutional layer to extract forward activations and backward gradients.
    """
    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.activations: Optional[torch.Tensor] = None
        self.gradients: Optional[torch.Tensor] = None

        self._forward_hook = self.target_layer.register_forward_hook(self._save_activations)
        self._backward_hook = self.target_layer.register_full_backward_hook(self._save_gradients)

    def _save_activations(self, module, input, output):
        # Store activations
        if isinstance(output, (tuple, list)):
            self.activations = output[0].detach()
        else:
            self.activations = output.detach()

    def _save_gradients(self, module, grad_input, grad_output):
        # Store gradients
        if isinstance(grad_output, (tuple, list)):
            self.gradients = grad_output[0].detach()
        else:
            self.gradients = grad_output.detach()

    def generate_cam(
        self,
        input_tensor: torch.Tensor,
        target_score: Optional[torch.Tensor] = None
    ) -> np.ndarray:
        """
        Generates normalized [0, 1] Grad-CAM heatmap for a single image.
        
        Args:
            input_tensor: [1, 3, H, W] preprocessed image tensor.
            target_score: Scalar tensor representing the target detection score to explain.
                          If None, uses the sum of top predicted logits.
        """
        self.model.eval()
        self.model.zero_grad()

        # If target_score not provided, run forward pass to obtain top prediction
        if target_score is None:
            outputs = self.model(input_tensor)
            pred_logits = outputs.get("pred_logits")  # [1, Q, num_classes]
            # Max score over queries and classes
            target_score = pred_logits.sigmoid().max()

        # Backward pass with respect to target score
        target_score.backward(retain_graph=True)

        if self.activations is None or self.gradients is None:
            raise RuntimeError("Failed to capture activations or gradients from target layer.")

        # Global average pooling of gradients: weights alpha_k
        # gradients: [1, C, H_feat, W_feat]
        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)  # [1, C, 1, 1]

        # Weighted combination of activation maps
        cam = torch.sum(weights * self.activations, dim=1, keepdim=True)  # [1, 1, H_feat, W_feat]
        cam = F.relu(cam)  # Only positive contributions

        # Normalize to [0, 1]
        cam = cam.squeeze().float().cpu().numpy()
        cam_min, cam_max = cam.min(), cam.max()
        if cam_max - cam_min > 1e-8:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = np.zeros_like(cam)

        return cam

    def remove_hooks(self):
        """Removes PyTorch forward and backward hooks."""
        self._forward_hook.remove()
        self._backward_hook.remove()


def overlay_gradcam(
    image: np.ndarray,
    cam: np.ndarray,
    alpha: float = 0.5,
    colormap: int = cv2.COLORMAP_JET
) -> np.ndarray:
    """
    Overlays Grad-CAM heatmap on RGB image.

    Args:
        image: RGB uint8 image [H, W, 3].
        cam: 2D float heatmap [H_feat, W_feat] normalized in [0, 1].
        alpha: Heatmap blend opacity.
        colormap: OpenCV colormap (default: COLORMAP_JET).
    """
    h, w = image.shape[:2]
    # Resize CAM to match original image dimensions
    cam_resized = cv2.resize(cam, (w, h), interpolation=cv2.INTER_LINEAR)
    heatmap = (cam_resized * 255.0).astype(np.uint8)
    heatmap_colored = cv2.applyColorMap(heatmap, colormap)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

    blended = (alpha * heatmap_colored + (1.0 - alpha) * image.astype(np.float32)).astype(np.uint8)
    return blended
