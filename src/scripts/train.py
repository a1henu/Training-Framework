# src/scripts/train.py
from __future__ import annotations
import argparse
from pathlib import Path
from typing import Any, Dict

import yaml
import torch

from src.core.logger import setup_logging, WandbLogger
from src.core.utils import set_seed, load_checkpoint
from src.core.trainer import Trainer
from src.core.registry import MODEL_REGISTRY, DATAMODULE_REGISTRY, LOSS_REGISTRY
from src.core.callbacks import EarlyStopping, ModelCheckpoint

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to config yaml.",
    )
    return parser.parse_args()

def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg

def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    exp_cfg = cfg.get("experiment", {})
    exp_name = exp_cfg.get("name", "default_exp")
    output_dir = Path(exp_cfg.get("output_dir", f"outputs/{exp_name}"))
    output_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logging(output_dir)
    logger.info(f"Using config from {args.config}")
    logger.info(f"Output dir: {output_dir}")

    seed = cfg.get("seed", 42)
    set_seed(seed)
    logger.info(f"Set random seed to {seed}")

    device_str = cfg.get("device", "cuda")
    if device_str == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA is not available, fallback to CPU.")
        device_str = "cpu"
    device = torch.device(device_str)
    logger.info(f"Using device: {device}")

    # Build datamodule and dataloaders
    data_cfg = cfg.get("data", {})
    loader_cfg = data_cfg.get("loader", {})
    dm_cfg = data_cfg.get("datamodule", {})
    dm_name = dm_cfg.get("name")
    if dm_name is None:
        raise ValueError("data.datamodule.name is required in config.")
    dm_params = dm_cfg.get("params", {})

    DataModuleClass = DATAMODULE_REGISTRY.get(dm_name)
    datamodule = DataModuleClass(loader_cfg=loader_cfg, **dm_params)
    datamodule.setup(stage="fit")
    train_loader = datamodule.train_dataloader()
    val_loader = datamodule.val_dataloader()

    # Build loss function
    loss_cfg = cfg.get("loss", {}) or {}
    loss_name = loss_cfg.get("name", None)
    loss_fn = None
    if loss_name is not None:
        LossClass = LOSS_REGISTRY.get(loss_name)
        loss_fn = LossClass(**(loss_cfg.get("params", {}) or {}))
        logger.info(f"Use loss: {loss_name}")

    # Build model
    model_cfg = cfg.get("model", {})
    model_name = model_cfg.get("name")
    if model_name is None:
        raise ValueError("model.name is required in config.")
    model_params = model_cfg.get("params", {}) or {}
    ModelClass = MODEL_REGISTRY.get(model_name)
    model = ModelClass(loss_fn=loss_fn, **model_params)
    logger.info(f"Model: {model_name}({model_params})")

    # Build optimizer and scheduler
    optimizer_cfg = cfg.get("optimizer", {})
    scheduler_cfg = cfg.get("scheduler", {})
    optimizer, scheduler = model.configure_optimizers(optimizer_cfg, scheduler_cfg)

    # wandb logger
    wandb_logger = WandbLogger(cfg, output_dir)

    # Build callbacks
    callbacks = []

    es_cfg = cfg.get("early_stopping", {}) or {}
    if es_cfg.get("enabled", False):
        callbacks.append(
            EarlyStopping(
                monitor=es_cfg.get("monitor", "val/loss"),
                mode=es_cfg.get("mode", "min"),
                patience=int(es_cfg.get("patience", 10)),
                min_delta=float(es_cfg.get("min_delta", 0.0)),
            )
        )

    ckpt_cfg = cfg.get("checkpoint", {}) or {}
    ckpt_dir = output_dir
    callbacks.append(
        ModelCheckpoint(
            dirpath=ckpt_dir,
            monitor=ckpt_cfg.get("monitor", "val/loss"),
            mode=ckpt_cfg.get("mode", "min"),
            save_best_only=ckpt_cfg.get("save_best_only", True),
            filename=ckpt_cfg.get("filename", "best.ckpt"),
            save_every_n_epochs=int(cfg.get("trainer", {}).get("ckpt_every_n_epochs", 1)),
        )
    )

    trainer = Trainer(
        model=model,
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        logger=logger,
        wandb_logger=wandb_logger,
        config=cfg,
        device=device,
        callbacks=callbacks,
    )

    # Resume from checkpoint if specified
    resume_path = cfg.get("trainer", {}).get("resume_ckpt", None)
    if resume_path is not None:
        resume_path = Path(resume_path)
        if resume_path.is_file():
            logger.info(f"Resuming from checkpoint: {resume_path}")
            ckpt = load_checkpoint(resume_path, model, optimizer, scheduler, trainer.scaler)
            trainer.current_epoch = int(ckpt.get("epoch", 0))
            trainer.global_step = int(ckpt.get("global_step", 0))
        else:
            logger.warning(f"resume_ckpt {resume_path} not found, ignore.")

    trainer.fit()

if __name__ == "__main__":
    main()
