# src/models/base_model.py
from __future__ import annotations
from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from src.core.utils import build_optimizer, build_scheduler


class BaseModel(nn.Module):
    """
    Base model that supports multi-term loss.

    - `loss_fn` is expected to return a dict with a mandatory key "loss",
      representing the total scalar loss used for backprop.
    - If `loss_fn` returns a single Tensor, it will be automatically wrapped
      into {"loss": tensor}.

    Training/validation step contract:
    ----------------------------------
    - `training_step` returns:
        {
          "loss": total_loss_tensor,
          "log": {
             "train/loss": ...,
             "train/recon": ...,
             ...
          }
        }
    - `validation_step` is analogous with "val/..." keys.
    """

    def __init__(self, loss_fn: Optional[nn.Module] = None):
        super().__init__()
        self.loss_fn = loss_fn

    def forward(self, batch: Dict[str, Any]) -> Any:
        """Subclasses implement actual forward logic."""
        raise NotImplementedError

    def compute_loss(self, outputs: Any, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """
        Returns a dict of losses. Must contain key "loss".
        """
        if self.loss_fn is None:
            raise NotImplementedError(
                "Either implement `compute_loss` in your model or "
                "pass a `loss_fn` that returns a dict with key 'loss'."
            )

        loss_out = self.loss_fn(outputs, batch)

        if isinstance(loss_out, dict):
            if "loss" not in loss_out:
                raise ValueError("Loss dict must contain key 'loss'.")
            return loss_out

        if torch.is_tensor(loss_out):
            return {"loss": loss_out}

        raise TypeError(
            "loss_fn must return either a Tensor or a dict[str, Tensor] "
            f"with key 'loss', got {type(loss_out)}."
        )

    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> Dict[str, Any]:
        outputs = self.forward(batch)
        loss_dict = self.compute_loss(outputs, batch)
        total_loss = loss_dict["loss"]

        # log all loss terms with 'train/' prefix
        logs: Dict[str, torch.Tensor] = {
            f"train/{name}": tensor.detach()
            for name, tensor in loss_dict.items()
        }

        return {"loss": total_loss, "log": logs}

    def validation_step(self, batch: Dict[str, Any], batch_idx: int) -> Dict[str, Any]:
        outputs = self.forward(batch)
        loss_dict = self.compute_loss(outputs, batch)
        total_loss = loss_dict["loss"]

        # log all loss terms with 'val/' prefix
        logs: Dict[str, torch.Tensor] = {
            f"val/{name}": tensor.detach()
            for name, tensor in loss_dict.items()
        }

        return {"loss": total_loss, "log": logs}

    def configure_optimizers(
        self,
        optimizer_cfg: Dict[str, Any],
        scheduler_cfg: Optional[Dict[str, Any]] = None,
    ):
        optimizer = build_optimizer(self.parameters(), optimizer_cfg)
        scheduler = build_scheduler(optimizer, scheduler_cfg)
        return optimizer, scheduler
