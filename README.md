# Training Framework

> This is a personal training framework for quickly reproducing.

A small, reusable training framework implemented in PyTorch.

---

## Folder Structure

```text
my_project_template/
├── configs/
│   └── template.yaml         # template config
├── src/
│   ├── core/                 # trainer, callbacks, registry, logging utils
│   ├── data/                 # BaseDataModule + your datamodules
│   ├── models/               # BaseModel + your models
│   ├── losses/               # BaseLoss + your custom losses
│   └── scripts/
│       └── train.py          # entry script
├── EXAMPLE.md                # detailed usage & toy examples
└── README.md
````

---

## Quick Start

1. **Finish your class for data/model/loss**

2. **(Optional) Login to wandb**

```bash
wandb login
```

3. **Edit config**

Modify `configs/default.yaml`:

* set `model.name` / `data.datamodule.name` / `loss.name`
* set `wandb.project`, `wandb.entity`, `wandb.mode` if you want online logging

4. **Run training**

From project root:

```bash
python -m src.scripts.train --config configs/default.yaml
```

Logs & checkpoints will be saved under:

```text
outputs/<experiment.name>/
    train.log
    best.ckpt
    epoch_*.ckpt (optional)
```

---

## How to Extend

For a new project / paper, you usually:

1. **Create a datamodule**

   * Implement `BaseDataModule` + datasets in `src/data/`
   * Register with `@DATAMODULE_REGISTRY.register("your_datamodule_name")`

2. **Create a model**

   * Implement a class inheriting `BaseModel` in `src/models/`
   * Register with `@MODEL_REGISTRY.register("your_model_name")`

3. **Create a loss (optional)**

   * Implement `BaseLoss` in `src/losses/`

   * Return a dict like:

     ```python
     {
       "loss": total_loss,   # used for backward()
       "l1": l1_loss,        # logged to wandb
       "perc": perc_loss,
     }
     ```

   * Register with `@LOSS_REGISTRY.register("your_loss_name")`

4. **Wire them in `configs/default.yaml`** and run.

---

## More Details

See **[EXAMPLE.md](./EXAMPLE.md)** for:

* A toy datamodule
* A toy convnet model
* A multi-term loss example
* Full configuration and wandb behavior

