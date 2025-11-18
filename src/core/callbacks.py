# src/core/callbacks.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, TYPE_CHECKING

from .utils import save_checkpoint

if TYPE_CHECKING:
    from .trainer import Trainer

class Callback:
    def on_train_start(self, trainer: Trainer) -> None:
        pass

    def on_train_end(self, trainer: Trainer) -> None:
        pass

    def on_train_epoch_start(self, trainer: Trainer, epoch: int) -> None:
        pass

    def on_train_epoch_end(self, trainer: Trainer, epoch: int, logs: Dict[str, Any]) -> None:
        pass

    def on_validation_epoch_end(self, trainer: Trainer, epoch: int, logs: Dict[str, Any]) -> None:
        pass

@dataclass
class EarlyStopping(Callback):
    monitor: str = "val/loss"
    mode: str = "min"  # "min" or "max"
    patience: int = 10
    min_delta: float = 0.0

    def __post_init__(self):
        if self.mode not in ("min", "max"):
            raise ValueError("mode must be 'min' or 'max'")
        self.best_score: Optional[float] = None
        self.wait_count: int = 0

    def _is_better(self, current: float, best: float) -> bool:
        if self.mode == "min":
            return current < best - self.min_delta
        else:
            return current > best + self.min_delta

    def on_validation_epoch_end(self, trainer: "Trainer", epoch: int, logs: Dict[str, Any]) -> None:
        if self.monitor not in logs:
            return
        current = float(logs[self.monitor])

        if self.best_score is None:
            self.best_score = current
            self.wait_count = 0
            return

        if self._is_better(current, self.best_score):
            self.best_score = current
            self.wait_count = 0
        else:
            self.wait_count += 1
            if self.wait_count >= self.patience:
                trainer.should_stop = True
                trainer.logger.info(
                    f"EarlyStopping triggered at epoch {epoch}: "
                    f"{self.monitor} did not improve for {self.patience} epochs."
                )

@dataclass
class ModelCheckpoint(Callback):
    """
    Always saves a 'best' checkpoint when the monitored metric improves.

    Additionally, if `save_every_n_epochs > 0`, it will also save
    epoch-wise checkpoints named `epoch_{k}.ckpt`.
    """
    dirpath: Path
    monitor: str = "val/loss"
    mode: str = "min"
    filename: str = "best.ckpt"
    save_every_n_epochs: int = 1  # extra periodic ckpts

    def __post_init__(self):
        if self.mode not in ("min", "max"):
            raise ValueError("mode must be 'min' or 'max'")
        self.dirpath.mkdir(parents=True, exist_ok=True)
        self.best_score: Optional[float] = None

    def _is_better(self, current: float, best: float) -> bool:
        if self.mode == "min":
            return current < best
        else:
            return current > best

    def on_validation_epoch_end(self, trainer: "Trainer", epoch: int, logs: Dict[str, Any]) -> None:
        if self.monitor not in logs:
            return

        current = float(logs[self.monitor])

        # 1) Always maintain a best checkpoint
        is_new_best = False
        if self.best_score is None or self._is_better(current, self.best_score):
            self.best_score = current
            best_path = self.dirpath / self.filename
            trainer.logger.info(
                f"[Checkpoint] New best {self.monitor}={current:.6f} at epoch {epoch+1}, "
                f"saving to {best_path}"
            )
            save_checkpoint(
                best_path,
                epoch,
                trainer.global_step,
                trainer.model,
                trainer.optimizer,
                trainer.scheduler,
                trainer.scaler,
                trainer.config,
            )
            is_new_best = True

        # 2) Optionally, also save periodic epoch checkpoints
        if self.save_every_n_epochs > 0 and ((epoch + 1) % self.save_every_n_epochs == 0):
            # Avoid double-saving the same best file with another name
            epoch_path = self.dirpath / f"epoch_{epoch+1}.ckpt"
            trainer.logger.info(
                f"[Checkpoint] Periodic save at epoch {epoch+1} to {epoch_path}"
            )
            save_checkpoint(
                epoch_path,
                epoch,
                trainer.global_step,
                trainer.model,
                trainer.optimizer,
                trainer.scheduler,
                trainer.scaler,
                trainer.config,
            )
