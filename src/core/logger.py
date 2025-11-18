# src/core/logger.py
from __future__ import annotations
import logging
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import wandb

def setup_logging(log_dir: Path, filename: str = "train.log") -> logging.Logger:
    log_dir.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter(
        fmt="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    # File
    fh = logging.FileHandler(log_dir / filename)
    fh.setLevel(logging.INFO)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return logger

class WandbLogger:
    def __init__(self, cfg: Dict[str, Any], output_dir: Path):
        wandb_cfg = cfg.get("wandb", {}) or {}
        project = wandb_cfg.get("project")
        mode = wandb_cfg.get("mode", "disabled")  # online/offline/disabled

        if not project or mode == "disabled":
            self._run = None
            self._disabled = True
            return

        name = wandb_cfg.get("name") or cfg.get("experiment", {}).get("name")
        entity = wandb_cfg.get("entity")
        self._run = wandb.init(
            project=project,
            entity=entity,
            name=name,
            config=cfg,
            mode=mode,
            dir=str(output_dir),
        )
        self._disabled = False

    @property
    def disabled(self) -> bool:
        return self._disabled

    def log_metrics(self, metrics: Dict[str, Any], step: Optional[int] = None) -> None:
        if self._run is None:
            return
        wandb.log(metrics, step=step)

    def watch_model(self, model) -> None:
        if self._run is None:
            return
        wandb.watch(model)

    def finish(self) -> None:
        if self._run is not None:
            self._run.finish()
            self._run = None
