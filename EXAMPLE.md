# Example: How to Use This Training Framework

This document shows how to derive a new project from this template and train a model with multiple loss terms, wandb logging, and best-checkpoint saving.

---

## 1. Folder Structure

You are expected to clone/copy this template and then add your own code:

```text
my_project/
├── configs/
│   └── default.yaml
├── src/
│   ├── core/
│   ├── data/
│   ├── losses/
│   ├── models/
│   └── scripts/
└── EXAMPLE.md
```

The main entry point is:

```bash
python -m src.scripts.train --config configs/default.yaml
```

You should run this command from the project root.

---

## 2. Configuration Overview

The main config file is `configs/default.yaml`. Important fields:

* `experiment.name`: experiment name (used in logs / wandb run name).
* `experiment.output_dir`: where logs and checkpoints are stored.
* `model.name`: name registered in `MODEL_REGISTRY`.
* `data.datamodule.name`: name registered in `DATAMODULE_REGISTRY`.
* `loss.name`: name registered in `LOSS_REGISTRY` (optional).
* `trainer.max_epochs`: total number of training epochs.
* `trainer.val_every_n_epochs`: how often to run validation (in epochs).
* `trainer.ckpt_every_n_epochs`: how often to save extra epoch checkpoints.
  Best checkpoint is **always** saved as `checkpoint.filename` (default: `best.ckpt`).
* `wandb`: project / entity / mode.

---

## 3. Multi-term Loss Design

The framework assumes that your loss module returns a **dict**:

```python
{
  "loss": total_loss,        # mandatory, used for backprop
  "recon": recon_loss,       # optional, will be logged
  "perceptual": perc_loss,   # optional, will be logged
  ...
}
```

During training:

* Backpropagation is performed on `loss_dict["loss"]` only.
* All terms are logged:

  * During training as `train/loss`, `train/recon`, `train/perceptual`, ...
  * During validation as `val/loss`, `val/recon`, `val/perceptual`, ...

If your `loss_fn` returns a single `Tensor`, it will be wrapped automatically into
`{"loss": tensor}`.

---

## 4. Example: DataModule

Create a new file `src/data/example_datamodule.py`:

```python
# src/data/example_datamodule.py
from pathlib import Path
from typing import Dict, Any

import torch
from torch.utils.data import Dataset

from src.data.base_dataset import BaseDataset, BaseDataModule
from src.core.registry import DATAMODULE_REGISTRY


class ToyDataset(BaseDataset):
    """
    A toy dataset that returns random inputs/targets.
    Replace this with your real dataset.
    """
    def __init__(self, length: int = 1000):
        self.length = length

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int):
        x = torch.randn(3, 32, 32)   # input
        y = torch.randn(3, 32, 32)   # target
        return {"inputs": x, "targets": y}


@DATAMODULE_REGISTRY.register("toy_datamodule")
class ToyDataModule(BaseDataModule):
    def __init__(self, loader_cfg: Dict[str, Any], train_size: int = 1000, val_size: int = 200):
        super().__init__(loader_cfg)
        self.train_size = int(train_size)
        self.val_size = int(val_size)

    def setup(self, stage: str = "fit") -> None:
        if stage in ("fit", "train"):
            self.train_dataset = ToyDataset(length=self.train_size)
            self.val_dataset = ToyDataset(length=self.val_size)
```

In `configs/default.yaml`:

```yaml
data:
  datamodule:
    name: "toy_datamodule"
    params:
      train_size: 1000
      val_size: 200
  loader:
    batch_size: 32
    num_workers: 4
    pin_memory: true
    shuffle: true
```

---

## 5. Example: Model

Create `src/models/example_convnet.py`:

```python
# src/models/example_convnet.py
import torch
import torch.nn as nn

from src.models.base_model import BaseModel
from src.core.registry import MODEL_REGISTRY


@MODEL_REGISTRY.register("toy_convnet")
class ToyConvNet(BaseModel):
    """
    A simple convolutional model example.

    Inputs:  {"inputs": Tensor[B, 3, 32, 32]}
    Targets: {"targets": Tensor[B, 3, 32, 32]}
    """

    def __init__(self, loss_fn=None, in_channels: int = 3, out_channels: int = 3, hidden_dim: int = 32):
        super().__init__(loss_fn=loss_fn)
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, out_channels, kernel_size=3, padding=1),
        )

    def forward(self, batch):
        x = batch["inputs"]
        return self.net(x)
```

In `configs/default.yaml`:

```yaml
model:
  name: "toy_convnet"
  params:
    in_channels: 3
    out_channels: 3
    hidden_dim: 32
```

---

## 6. Example: Multi-term Loss

Create `src/losses/example_loss.py`:

```python
# src/losses/example_loss.py
from typing import Any, Dict

import torch
import torch.nn.functional as F

from src.losses.base_loss import BaseLoss
from src.core.registry import LOSS_REGISTRY


@LOSS_REGISTRY.register("toy_multiterm_loss")
class ToyMultiTermLoss(BaseLoss):
    """
    Example of a multi-term loss:
      loss = alpha * L1 + beta * L2

    All terms will be logged to wandb.
    """

    def __init__(self, alpha: float = 1.0, beta: float = 0.1):
        super().__init__()
        self.alpha = float(alpha)
        self.beta = float(beta)

    def forward(self, outputs: Any, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        targets = batch["targets"]

        l1 = F.l1_loss(outputs, targets)
        l2 = F.mse_loss(outputs, targets)

        total = self.alpha * l1 + self.beta * l2

        return {
            "loss": total,
            "l1": l1,
            "l2": l2,
        }
```

In `configs/default.yaml`:

```yaml
loss:
  name: "toy_multiterm_loss"
  params:
    alpha: 1.0
    beta: 0.1
```

Now the trainer will:

* Backprop on `loss_dict["loss"]`, i.e. `alpha * L1 + beta * L2`.
* Log:

  * `train/loss`, `train/l1`, `train/l2` during training.
  * `val/loss`, `val/l1`, `val/l2` during validation.

You can directly monitor `val/loss` for early stopping and model checkpointing.

---

## 7. Checkpoints

The `ModelCheckpoint` callback always maintains a **best checkpoint** based on:

```yaml
checkpoint:
  monitor: "val/loss"
  mode: "min"
  filename: "best.ckpt"
```

* The best model is always saved at:
  `experiment.output_dir / checkpoint.filename` (default: `outputs/<exp_name>/best.ckpt`).
* Additionally, if `trainer.ckpt_every_n_epochs > 0`, the trainer will save extra checkpoints:
  `epoch_1.ckpt`, `epoch_2.ckpt`, ...

To disable extra epoch checkpoints and only keep `best.ckpt`:

```yaml
trainer:
  ckpt_every_n_epochs: 0
```

---

## 8. WandB Logging

In `configs/default.yaml`:

```yaml
wandb:
  project: "my_project"
  entity: null         # or your wandb username
  mode: "online"       # "online", "offline", "disabled"
  name: null           # default: experiment.name
```

* All loss terms are logged automatically.
* Step-level logs contain e.g. `train/loss`, `train/l1`, `train/l2`, `epoch`, `step`.
* Epoch-level logs contain averaged metrics with suffix `_epoch` (e.g. `train/loss_epoch`).

---

## 9. Running the Example

After implementing the example DataModule, model and loss:

```bash
# from project root
python -m src.scripts.train --config configs/default.yaml
```

You should see:

* Log files under `outputs/<experiment.name>/train.log`
* Best checkpoint at `outputs/<experiment.name>/best.ckpt`
* Optional epoch checkpoints `epoch_k.ckpt`
* wandb run with all loss terms plotted
