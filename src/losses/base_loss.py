# src/losses/base_loss.py
from __future__ import annotations
from typing import Any, Dict

import torch
import torch.nn as nn

class BaseLoss(nn.Module):
    """
    Base class for custom losses.

    Convention:
    ----------
    forward(outputs, batch) -> Dict[str, Tensor]

    The returned dict MUST contain a key "loss" which is the total scalar loss
    used for backpropagation. Other keys are treated as individual loss terms
    and will be logged to wandb, e.g.:

        {
            "loss": total_loss,
            "recon": recon_loss,
            "perceptual": perceptual_loss,
        }

    The trainer will:
    - backprop on "loss"
    - log all keys ("loss", "recon", "perceptual", ...)
    """

    def forward(self, outputs: Any, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        raise NotImplementedError
