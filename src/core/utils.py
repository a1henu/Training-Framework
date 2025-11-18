# src/core/utils.py
from __future__ import annotations
import os
import random
from pathlib import Path
from typing import Any, Dict, Optional, Mapping

import numpy as np
import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler

def set_seed(seed: int) -> None:
    """ Set random seed for reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def move_to_device(batch: Any, device: torch.device) -> Any:
    """ Move a batch of data to a specified device.
    """
    if torch.is_tensor(batch):
        return batch.to(device)
    if isinstance(batch, Mapping):
        return {k: move_to_device(v, device) for k, v in batch.items()}
    if isinstance(batch, (list, tuple)):
        t = [move_to_device(x, device) for x in batch]
        return type(batch)(t)
    return batch

def build_optimizer(
    params,
    cfg: Dict[str, Any],
) -> Optimizer:
    """ Build optimizer from configuration.
    """
    name = cfg.get("name", "").lower()
    lr = cfg.get("lr", 1e-3)
    weight_decay = cfg.get("weight_decay", 0.0)

    if name == "adam":
        opt_cls = torch.optim.Adam
        extra = {k: v for k, v in cfg.items() if k not in ["name", "lr", "weight_decay"]}
    elif name == "adamw":
        opt_cls = torch.optim.AdamW
        extra = {k: v for k, v in cfg.items() if k not in ["name", "lr", "weight_decay"]}
    elif name == "sgd":
        opt_cls = torch.optim.SGD
        extra = {k: v for k, v in cfg.items() if k not in ["name", "lr", "weight_decay"]}
    else:
        raise ValueError(f"Unknown optimizer: {name}")

    return opt_cls(params, lr=lr, weight_decay=weight_decay, **extra)

def build_scheduler(
    optimizer: Optimizer,
    cfg: Optional[Dict[str, Any]],
) -> Optional[_LRScheduler]:
    """ Build learning rate scheduler from configuration.
    """
    if not cfg:
        return None
    name = cfg.get("name")
    if not name:
        return None

    params = cfg.get("params", {})
    name = name.lower()

    if name == "steplr":
        return torch.optim.lr_scheduler.StepLR(optimizer, **params)
    elif name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, **params)
    else:
        # TODO: add more schedulers
        raise ValueError(f"Unknown scheduler: {name}")

def save_checkpoint(
    path: Path,
    epoch: int,
    global_step: int,
    model: torch.nn.Module,
    optimizer: Optimizer,
    scheduler: Optional[_LRScheduler],
    scaler: Optional[torch.cuda.amp.GradScaler],
    config: Dict[str, Any],
) -> None:
    """ Save training checkpoint.
        The state dict includes:
        - epoch
        - global_step
        - model state dict
        - optimizer state dict
        - scheduler state dict
        - scaler state dict
        - config
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {
        "epoch": epoch,
        "global_step": global_step,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "scaler": scaler.state_dict() if scaler is not None else None,
        "config": config,
    }
    torch.save(state, path)

def load_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: Optional[Optimizer] = None,
    scheduler: Optional[_LRScheduler] = None,
    scaler: Optional[torch.cuda.amp.GradScaler] = None,
) -> Dict[str, Any]:
    """ Load training checkpoint.
        Returns the checkpoint dictionary.
    """
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    if optimizer is not None and ckpt.get("optimizer") is not None:
        optimizer.load_state_dict(ckpt["optimizer"])
    if scheduler is not None and ckpt.get("scheduler") is not None:
        scheduler.load_state_dict(ckpt["scheduler"])
    if scaler is not None and ckpt.get("scaler") is not None:
        scaler.load_state_dict(ckpt["scaler"])
    return ckpt
