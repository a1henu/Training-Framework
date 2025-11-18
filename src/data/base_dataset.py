# src/data/base_dataset.py
from __future__ import annotations
from typing import Any, Dict, Optional

from abc import ABC, abstractmethod

import torch
from torch.utils.data import Dataset, DataLoader

class BaseDataset(Dataset, ABC):
    pass

class BaseDataModule(ABC):
    """ Simple data module to handle datasets and dataloaders.
        - Subclass should implement `setup` method to create datasets.
        - Provides default `train_dataloader`, `val_dataloader`, and `test_dataloader`
    """
    def __init__(self, loader_cfg: Dict[str, Any]):
        self.loader_cfg = loader_cfg
        self.train_dataset: Optional[Dataset] = None
        self.val_dataset: Optional[Dataset] = None
        self.test_dataset: Optional[Dataset] = None

    @abstractmethod
    def setup(self, stage: str = "fit") -> None:
        """ Setup datasets for different stages: 'fit', 'validate', 'test', 'predict'.
        """
        pass

    def train_dataloader(self) -> DataLoader:
        assert self.train_dataset is not None, "train_dataset is None, did you call setup('fit')?"
        return DataLoader(
            self.train_dataset,
            batch_size=self.loader_cfg.get("batch_size", 32),
            shuffle=self.loader_cfg.get("shuffle", True),
            num_workers=self.loader_cfg.get("num_workers", 4),
            pin_memory=self.loader_cfg.get("pin_memory", True),
        )

    def val_dataloader(self) -> Optional[DataLoader]:
        if self.val_dataset is None:
            return None
        return DataLoader(
            self.val_dataset,
            batch_size=self.loader_cfg.get("batch_size", 32),
            shuffle=False,
            num_workers=self.loader_cfg.get("num_workers", 4),
            pin_memory=self.loader_cfg.get("pin_memory", True),
        )

    def test_dataloader(self) -> Optional[DataLoader]:
        if self.test_dataset is None:
            return None
        return DataLoader(
            self.test_dataset,
            batch_size=self.loader_cfg.get("batch_size", 32),
            shuffle=False,
            num_workers=self.loader_cfg.get("num_workers", 4),
            pin_memory=self.loader_cfg.get("pin_memory", True),
        )
