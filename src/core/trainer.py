# src/core/trainer.py
from __future__ import annotations
from typing import Any, Dict, List, Optional

from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .utils import move_to_device
from .callbacks import Callback
from .logger import WandbLogger

class Trainer:
    """
    Generic Trainer:
    - All validation & checkpoint frequencies are based on epochs
    - Optional mixed precision (AMP)
    - Supports callbacks (EarlyStopping, ModelCheckpoint, etc.)
    """
    def __init__(
        self,
        model: torch.nn.Module,
        train_dataloader: DataLoader,
        val_dataloader: Optional[DataLoader],
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
        logger,  # python logging.Logger
        wandb_logger: Optional[WandbLogger],
        config: Dict[str, Any],
        device: torch.device,
        callbacks: Optional[List[Callback]] = None,
    ):
        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.logger = logger
        self.wandb_logger = wandb_logger
        self.config = config
        self.device = device

        trainer_cfg = config.get("trainer", {})
        self.max_epochs: int = int(trainer_cfg.get("max_epochs", 100))
        self.val_every_n_epochs: int = int(trainer_cfg.get("val_every_n_epochs", 1))
        self.ckpt_every_n_epochs: int = int(trainer_cfg.get("ckpt_every_n_epochs", 1))
        self.gradient_clip_norm = trainer_cfg.get("gradient_clip_norm", None)
        self.mixed_precision: bool = bool(trainer_cfg.get("mixed_precision", False))
        self.log_every_n_steps: int = int(trainer_cfg.get("log_every_n_steps", 50))

        self.callbacks: List[Callback] = callbacks or []
        self.current_epoch: int = 0
        self.global_step: int = 0
        self.should_stop: bool = False

        self.scaler = torch.amp.GradScaler(enabled=self.mixed_precision)

        self.model.to(self.device)

        if self.wandb_logger is not None and not self.wandb_logger.disabled:
            self.wandb_logger.watch_model(self.model)

    def fit(self) -> None:
        self.logger.info(f"Start training for {self.max_epochs} epochs.")
        for cb in self.callbacks:
            cb.on_train_start(self)

        for epoch in range(self.current_epoch, self.max_epochs):
            self.current_epoch = epoch
            self.logger.info(f"Epoch [{epoch+1}/{self.max_epochs}]")

            for cb in self.callbacks:
                cb.on_train_epoch_start(self, epoch)

            train_logs = self._train_one_epoch(epoch)

            for cb in self.callbacks:
                cb.on_train_epoch_end(self, epoch, train_logs)

            # epoch-level scheduler step
            if self.scheduler is not None:
                self.scheduler.step()

            # validation
            val_logs: Dict[str, float] = {}
            if self.val_dataloader is not None and (epoch + 1) % self.val_every_n_epochs == 0:
                val_logs = self._validate(epoch)

                for cb in self.callbacks:
                    cb.on_validation_epoch_end(self, epoch, val_logs)

            # epoch summary
            merged_logs = {**train_logs, **val_logs}
            if merged_logs:
                log_str = " | ".join(f"{k}={v:.4f}" for k, v in merged_logs.items())
                self.logger.info(f"Epoch {epoch+1} summary: {log_str}")

            # epoch-level logging to wandb
            if self.wandb_logger is not None and not self.wandb_logger.disabled and merged_logs:
                epoch_logs = {f"{k}_epoch": v for k, v in merged_logs.items()}
                epoch_logs["epoch"] = epoch + 1
                self.wandb_logger.log_metrics(epoch_logs, step=self.global_step)

            if self.should_stop:
                self.logger.info("Early stopping triggered, stop training.")
                break

        for cb in self.callbacks:
            cb.on_train_end(self)

        if self.wandb_logger is not None:
            self.wandb_logger.finish()

    def _train_one_epoch(self, epoch: int) -> Dict[str, float]:
        self.model.train()
        running_sums: Dict[str, float] = {}
        num_batches = 0

        for batch_idx, batch in enumerate(self.train_dataloader):
            batch = move_to_device(batch, self.device)

            self.optimizer.zero_grad(set_to_none=True)

            if self.mixed_precision:
                with torch.amp.autocast():
                    out = self.model.training_step(batch, batch_idx)
                    loss = out["loss"]
                self.scaler.scale(loss).backward()
                if self.gradient_clip_norm is not None:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.gradient_clip_norm,
                    )
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                out = self.model.training_step(batch, batch_idx)
                loss = out["loss"]
                loss.backward()
                if self.gradient_clip_norm is not None:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.gradient_clip_norm,
                    )
                self.optimizer.step()

            logs = out.get("log", {})
            # accumulate metrics for epoch-level averages
            for k, v in logs.items():
                value = float(v.detach().item()) if torch.is_tensor(v) else float(v)
                running_sums[k] = running_sums.get(k, 0.0) + value

            num_batches += 1
            self.global_step += 1

            # step-level logging
            if (batch_idx + 1) % self.log_every_n_steps == 0 and running_sums:
                step_logs = {k: running_sums[k] / num_batches for k in running_sums}
                step_logs["epoch"] = epoch + 1
                step_logs["step"] = self.global_step
                log_str = " | ".join(f"{k}={v:.4f}" for k, v in step_logs.items())
                self.logger.info(f"[Train] Epoch {epoch+1}, step {batch_idx+1}: {log_str}")

                if self.wandb_logger is not None and not self.wandb_logger.disabled:
                    self.wandb_logger.log_metrics(step_logs, step=self.global_step)

        epoch_logs = {k: v / max(1, num_batches) for k, v in running_sums.items()}
        return epoch_logs

    def _validate(self, epoch: int) -> Dict[str, float]:
        self.model.eval()
        running_sums: Dict[str, float] = {}
        num_batches = 0

        with torch.no_grad():
            for batch_idx, batch in enumerate(self.val_dataloader):
                batch = move_to_device(batch, self.device)
                out = self.model.validation_step(batch, batch_idx)
                logs = out.get("log", {})
                for k, v in logs.items():
                    value = float(v.detach().item()) if torch.is_tensor(v) else float(v)
                    running_sums[k] = running_sums.get(k, 0.0) + value
                num_batches += 1

        val_logs = {k: v / max(1, num_batches) for k, v in running_sums.items()}
        return val_logs
