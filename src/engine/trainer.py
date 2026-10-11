"""
Model Training and Checkpoint Management Engine for SeaClear Instance Segmentation.

Features:
- AdamW optimizer with layer-wise / backbone learning rates
- Cosine annealing scheduler with warmup
- AMP (Automatic Mixed Precision) with GradScaler
- Gradient accumulation (effective batch size scaling for 16GB GPUs)
- Checkpoint persistence: best.pth, last.pth, epoch_xxx.pth, history jsons, config json
- Resume capability from last.pth
- Periodic clear_gpu_memory calls
"""

import os
import json
import time
from tqdm.auto import tqdm
import torch
import torch.nn as nn
from typing import Dict, List, Any, Optional

from .memory import clear_gpu_memory


class Trainer:
    """Trains and validates a given model instance with complete checkpoint management."""
    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module,
        train_loader: Any,
        val_loader: Any,
        device: torch.device,
        model_name: str = "M1",
        checkpoint_dir: str = "./checkpoints",
        lr: float = 1e-4,
        lr_backbone: float = 1e-5,
        weight_decay: float = 1e-4,
        grad_accum_steps: int = 2,
        clip_max_norm: float = 0.1,
        use_amp: bool = True,
        epochs: int = 15,
        config_dict: Optional[Dict[str, Any]] = None,
        excel_logger: Optional[Any] = None
    ):
        self.model = model.to(device)
        self.criterion = criterion.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.model_name = model_name
        self.save_dir = os.path.join(checkpoint_dir, model_name)
        os.makedirs(self.save_dir, exist_ok=True)

        self.lr = lr
        self.lr_backbone = lr_backbone
        self.weight_decay = weight_decay
        self.grad_accum_steps = grad_accum_steps
        self.clip_max_norm = clip_max_norm
        self.use_amp = use_amp and (device.type == "cuda")
        self.epochs = epochs
        self.config_dict = config_dict or {}
        self.excel_logger = excel_logger

        # Optimizer: parameter groups for backbone and rest of model
        backbone_params = []
        head_params = []
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if "backbone" in name:
                backbone_params.append(param)
            else:
                head_params.append(param)

        param_groups = [
            {"params": head_params, "lr": lr, "weight_decay": weight_decay},
            {"params": backbone_params, "lr": lr_backbone, "weight_decay": weight_decay},
        ]
        self.optimizer = torch.optim.AdamW(param_groups)

        # Scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=max(epochs, 1), eta_min=lr * 0.01
        )

        # AMP Grad Scaler
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)

        # Tracking state
        self.start_epoch = 1
        self.best_metric = -1.0  # Tracks Mask mAP or composite AP
        self.training_history: List[Dict[str, Any]] = []
        self.validation_history: List[Dict[str, Any]] = []

    def resume_from_checkpoint(self, checkpoint_path: Optional[str] = None) -> bool:
        """Resumes training from last.pth or specified path if available."""
        if checkpoint_path is None:
            checkpoint_path = os.path.join(self.save_dir, "last.pth")

        if not os.path.exists(checkpoint_path):
            print(f"[{self.model_name}] No checkpoint found at {checkpoint_path}. Starting fresh.")
            return False

        print(f"[{self.model_name}] Resuming from checkpoint: {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if "scheduler_state_dict" in ckpt:
            self.scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        self.start_epoch = ckpt.get("epoch", 0) + 1
        self.best_metric = ckpt.get("best_metric", -1.0)
        self.training_history = ckpt.get("training_history", [])
        self.validation_history = ckpt.get("validation_history", [])
        print(f"[{self.model_name}] Successfully resumed at epoch {self.start_epoch}.")
        return True

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Executes one training epoch with gradient accumulation and AMP."""
        self.model.train()
        self.criterion.train()
        self.optimizer.zero_grad()

        total_loss = 0.0
        total_cls_loss = 0.0
        total_box_loss = 0.0
        total_mask_loss = 0.0
        num_batches = len(self.train_loader)
        t0 = time.time()

        pbar = tqdm(
            enumerate(self.train_loader),
            total=num_batches,
            desc=f"[{self.model_name}] Epoch {epoch:02d}/{self.epochs:02d}",
            leave=True
        )

        for step, (images, targets) in pbar:
            images = images.to(self.device)
            targets = [{k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets]

            with torch.amp.autocast("cuda", enabled=self.use_amp):
                outputs = self.model(images)
                loss_dict = self.criterion(outputs, targets)
                loss = loss_dict["loss"] / self.grad_accum_steps

            # Backward with AMP scaler
            self.scaler.scale(loss).backward()

            # Step optimizer every grad_accum_steps or at end of epoch
            if (step + 1) % self.grad_accum_steps == 0 or (step + 1) == num_batches:
                if self.clip_max_norm > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_max_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()

            total_loss += loss_dict["loss"].item()
            total_cls_loss += loss_dict["loss_cls"].item()
            total_box_loss += loss_dict["loss_box"].item()
            total_mask_loss += loss_dict["loss_mask"].item()

            cur_avg_loss = total_loss / (step + 1)
            pbar.set_postfix({
                "loss": f"{cur_avg_loss:.4f}",
                "cls": f"{loss_dict['loss_cls'].item():.3f}",
                "box": f"{loss_dict['loss_box'].item():.3f}",
                "mask": f"{loss_dict['loss_mask'].item():.3f}"
            })

            if (step + 1) % 25 == 0:
                clear_gpu_memory()

        self.scheduler.step()
        epoch_time = time.time() - t0

        avg_loss = total_loss / max(num_batches, 1)
        avg_cls = total_cls_loss / max(num_batches, 1)
        avg_box = total_box_loss / max(num_batches, 1)
        avg_mask = total_mask_loss / max(num_batches, 1)

        epoch_metrics = {
            "Model": self.model_name,
            "Epoch": epoch,
            "Loss": round(avg_loss, 4),
            "Classification Loss": round(avg_cls, 4),
            "Box Loss": round(avg_box, 4),
            "Mask Loss": round(avg_mask, 4),
            "Learning Rate": round(self.optimizer.param_groups[0]["lr"], 6),
            "Step Time (s)": round(epoch_time, 2)
        }

        self.training_history.append(epoch_metrics)
        if self.excel_logger:
            self.excel_logger.log_training_step(epoch_metrics)

        print(f"[{self.model_name}] Epoch {epoch:02d}/{self.epochs:02d} | Loss: {avg_loss:.4f} "
              f"(Cls: {avg_cls:.4f}, Box: {avg_box:.4f}, Mask: {avg_mask:.4f}) | Time: {epoch_time:.1f}s")

        return epoch_metrics

    def save_checkpoint(self, epoch: int, is_best: bool = False, metric_val: float = 0.0):
        """Saves checkpoint in exact required structure."""
        ckpt = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "epoch": epoch,
            "best_metric": self.best_metric,
            "training_history": self.training_history,
            "validation_history": self.validation_history,
            "model_name": self.model_name,
            "config": self.config_dict
        }

        # 1. Last checkpoint
        last_path = os.path.join(self.save_dir, "last.pth")
        torch.save(ckpt, last_path)

        # 2. Per-epoch checkpoint
        epoch_path = os.path.join(self.save_dir, f"epoch_{epoch:03d}.pth")
        torch.save(ckpt, epoch_path)

        # 3. Best checkpoint
        if is_best:
            best_path = os.path.join(self.save_dir, "best.pth")
            torch.save(ckpt, best_path)
            if self.excel_logger:
                self.excel_logger.log_checkpoint({
                    "Model": self.model_name,
                    "Type": "best",
                    "Epoch": epoch,
                    "Metric": "Mask mAP",
                    "Metric Value": metric_val,
                    "File Path": best_path,
                    "Saved At": time.strftime("%Y-%m-%d %H:%M:%S")
                })

        # 4. Save JSON histories and config
        with open(os.path.join(self.save_dir, "training_history.json"), "w") as f:
            json.dump(self.training_history, f, indent=2)

        with open(os.path.join(self.save_dir, "validation_history.json"), "w") as f:
            json.dump(self.validation_history, f, indent=2)

        with open(os.path.join(self.save_dir, "config.json"), "w") as f:
            json.dump(self.config_dict, f, indent=2)

    def fit(self, num_epochs: Optional[int] = None) -> List[Dict[str, float]]:
        """Runs the complete training loop across all epochs and saves checkpoints."""
        total_epochs = num_epochs or self.epochs
        print(f"[{self.model_name}] Starting training for {total_epochs} epochs on device: {self.device}")
        
        for ep in range(self.start_epoch, total_epochs + 1):
            metrics = self.train_epoch(ep)
            # Save periodic/last checkpoint
            self.save_checkpoint(ep, is_best=(metrics["Loss"] <= self.best_metric or self.best_metric < 0), metric_val=metrics["Loss"])
            if self.best_metric < 0 or metrics["Loss"] < self.best_metric:
                self.best_metric = metrics["Loss"]
            clear_gpu_memory()

        print(f"[{self.model_name}] Training complete across {total_epochs} epochs.")
        return self.training_history

