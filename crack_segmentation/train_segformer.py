"""FisherAdapTune fine-tuning of SegFormer for crack segmentation.

Run from the FisherAdapTune/ directory:

    python crack_segmentation/train_segformer.py \
        --config crack_segmentation/config_segformer.yaml

Requires:
  - segmentation-models-pytorch:  pip install segmentation-models-pytorch
  - monai:  pip install monai
  - Dataset with layout:
        <data_root>/images/   *.jpg / *.png
        <data_root>/masks/    *.jpg / *.png  (binary)
"""

from __future__ import annotations

import argparse
import os
import random
import sys
from datetime import datetime as dt
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent))
from scripts import EarlyStopping, FisherAdapTuneTrainer

from dataset import SegFormerSegmentationDataset


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _set_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def _build_device() -> Tuple[torch.device, str]:
    if torch.cuda.is_available():
        return torch.device("cuda"), "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps"), "mps"
    return torch.device("cpu"), "cpu"


def _disable_inplace_relu(model: nn.Module) -> None:
    for name, child in model.named_children():
        if isinstance(child, nn.ReLU) and child.inplace:
            setattr(model, name, nn.ReLU(inplace=False))
        else:
            _disable_inplace_relu(child)


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

class WeightedDiceBCELoss(nn.Module):
    def __init__(self, dice_weight: float = 0.65):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.dice_weight = dice_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = self.bce(logits, targets)
        prob = torch.sigmoid(logits)
        num = 2.0 * (prob * targets).sum()
        den = prob.sum() + targets.sum() + 1e-6
        dice = 1.0 - num / den
        return bce + self.dice_weight * dice


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def build_dataloaders(cfg: Dict):
    image_size = int(cfg.get("image_size", 256))
    train_ds = SegFormerSegmentationDataset(cfg["train_data_path"], image_size=image_size)
    val_ds   = SegFormerSegmentationDataset(cfg["validation_data_path"], image_size=image_size)
    num_workers = int(cfg.get("num_workers", 2))
    ldr_kw = dict(
        batch_size=int(cfg.get("batch_size", 4)),
        num_workers=num_workers,
        pin_memory=bool(cfg.get("pin_memory", True)),
        persistent_workers=bool(cfg.get("persistent_workers", True)) and num_workers > 0,
        prefetch_factor=int(cfg.get("prefetch_factor", 2)) if num_workers > 0 else None,
        drop_last=True,
    )
    return (
        DataLoader(train_ds, shuffle=True, **ldr_kw),
        DataLoader(val_ds, shuffle=False, drop_last=False, **{k: v for k, v in ldr_kw.items() if k != "drop_last"}),
    )


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def build_segformer(cfg: Dict, device: torch.device, model_name_override: Optional[str] = None) -> nn.Module:
    try:
        import segmentation_models_pytorch as smp
    except ImportError as exc:
        raise RuntimeError(
            "segmentation-models-pytorch not found.  "
            "Install it with:  pip install segmentation-models-pytorch"
        ) from exc

    model_name = model_name_override or str(cfg.get("segformer_model_name", "mit_b0"))
    encoder_weights = cfg.get("encoder_weights", "imagenet")
    num_classes = int(cfg.get("num_classes", 1))

    model = smp.Segformer(model_name, encoder_weights=encoder_weights, classes=num_classes)
    model.to(device)
    _disable_inplace_relu(model)
    for param in model.parameters():
        param.requires_grad = True

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"SegFormer ({model_name})  total={total:,}  trainable={trainable:,}")
    return model


# ---------------------------------------------------------------------------
# Forward functions
# ---------------------------------------------------------------------------

def make_train_step(loss_fn: nn.Module, device: torch.device, devtype: str):
    def train_step(model, batch):
        xb, yb = batch
        xb = xb.to(device, non_blocking=(devtype == "cuda"))
        yb = yb.to(device, non_blocking=(devtype == "cuda"))
        return loss_fn(model(xb), yb)
    return train_step


def make_val_step(loss_fn: nn.Module, device: torch.device, devtype: str):
    def val_step(model, batch):
        xb, yb = batch
        xb = xb.to(device, non_blocking=(devtype == "cuda"))
        yb = yb.to(device, non_blocking=(devtype == "cuda"))
        with torch.inference_mode():
            out = model(xb)
        loss_val = loss_fn(out, yb).item()
        preds = (torch.sigmoid(out) >= 0.5).long()
        gt    = (yb >= 0.5).long()
        tp = int((preds * gt).sum())
        fp = int((preds * (1 - gt)).sum())
        fn = int(((1 - preds) * gt).sum())
        tn = int(((1 - preds) * (1 - gt)).sum())
        total = tp + fp + fn + tn
        return {
            "loss":     loss_val,
            "accuracy": (tp + tn) / max(1, total),
            "f1":       (2 * tp) / max(1, 2 * tp + fp + fn),
        }
    return val_step


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="FisherAdapTune — SegFormer crack segmentation")
    p.add_argument("--config", default="crack_segmentation/config_segformer.yaml")
    p.add_argument("--segformer-model-name", default=None, help="Override segformer_model_name in config.")
    p.add_argument("--slurm", default=None)
    p.add_argument("--random-seed", type=int, default=None)
    p.add_argument("--disable-wandb", action="store_true")
    p.add_argument("--wandb-project", default=None)
    p.add_argument("--wandb-group", default=None)
    p.add_argument("--wandb-name", default=None)
    p.add_argument("--wandb-mode", default=None)
    p.add_argument("--fisher-ema-interval", type=int, default=None)
    p.add_argument("--freeze-interval", type=int, default=None)
    p.add_argument("--prev-js-ema-decay", type=float, default=None)
    p.add_argument("--fisher-ema-decay", type=float, default=None)
    p.add_argument("--stage2-js-variance-lambda", type=float, default=None)
    p.add_argument("--chunk-selection-metric", choices=("total_variation", "mean_js"), default=None)
    p.add_argument("--fisher-slice-mode", choices=("row", "column"), default=None)
    p.add_argument("--fisher-slice-blocks", type=int, default=None)
    p.add_argument("--fisher-row-blocks", type=int, default=None, help="Alias for --fisher-slice-blocks.")
    p.add_argument("--js-distance-mode", choices=("log", "raw"), default=None)
    p.add_argument("--val-interval", type=int, default=None)
    p.add_argument("--save-file-name", default=None)
    p.add_argument("--disable-fisher-plots", action="store_true")
    return p.parse_args()


def main() -> None:
    os.environ.setdefault("MPLBACKEND", "Agg")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

    args = parse_args()
    with open(args.config, "r", encoding="utf-8") as f:
        cfg: Dict[str, Any] = yaml.safe_load(f) or {}

    if args.slurm:
        for key in ("train_data_path", "validation_data_path"):
            cfg[key] = cfg.get(key, "").replace("$TMPDIR/src", args.slurm)

    def _pick(arg_val, cfg_key, default):
        return arg_val if arg_val is not None else cfg.get(cfg_key, default)

    seed = int(_pick(args.random_seed, "seed", 0))
    num_epochs = int(cfg.get("num_epochs", 50))
    lr = float(cfg.get("learning_rate", 1e-4))
    weight_decay = float(cfg.get("weight_decay", 5e-5))
    lambda_value = float(cfg.get("lambda_value", 0.65))
    fisher_gammas = list(cfg.get("adafisher_gamma", [0.8, 0.8]))
    fisher_tcov = int(cfg.get("adafisher_tcov", 1))
    prev_js_ema_decay = float(_pick(args.prev_js_ema_decay, "prev_js_ema_decay", 0.9))
    fisher_ema_decay = float(_pick(args.fisher_ema_decay, "fisher_ema_decay", prev_js_ema_decay))
    fisher_ema_interval = max(1, int(_pick(args.fisher_ema_interval, "fisher_ema_interval", 300)))
    freeze_interval = max(1, int(_pick(args.freeze_interval, "freeze_interval", 3000)))
    fisher_slice_mode = str(_pick(args.fisher_slice_mode, "fisher_slice_mode", "row"))
    fisher_slice_blocks_arg = args.fisher_slice_blocks if args.fisher_slice_blocks is not None else args.fisher_row_blocks
    fisher_slice_blocks = max(1, int(_pick(fisher_slice_blocks_arg, "fisher_slice_blocks", 4)))
    js_distance_mode = str(_pick(args.js_distance_mode, "js_distance_mode", "log"))
    chunk_selection_metric = str(_pick(args.chunk_selection_metric, "chunk_selection_metric", "total_variation"))
    js_variance_lambda = float(_pick(args.stage2_js_variance_lambda, "stage2_js_variance_lambda", 1.0))
    val_interval_raw = _pick(args.val_interval, "val_interval", None)
    val_interval = max(1, int(val_interval_raw)) if val_interval_raw is not None else None
    save_file_name = args.save_file_name or cfg.get("save_file_name", "segformer_fisher_adapt_tune")

    _set_seed(seed)
    device, devtype = _build_device()
    print(f"Device: {device}  seed: {seed}")

    train_loader, val_loader = build_dataloaders(cfg)
    model = build_segformer(cfg, device, model_name_override=args.segformer_model_name)

    loss_fn = WeightedDiceBCELoss(dice_weight=lambda_value)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    early_stopping = EarlyStopping(patience=int(cfg.get("early_stopping_patience", 5)), verbose=True)

    wandb_run = None
    if not args.disable_wandb:
        try:
            import wandb
            timestamp = dt.now().strftime("%Y%m%d-%H%M%S")
            model_name = args.segformer_model_name or cfg.get("segformer_model_name", "mit_b0")
            wandb_run = wandb.init(
                project=args.wandb_project or cfg.get("wandb_project", "FisherAdapTune-SegFormer"),
                name=args.wandb_name or cfg.get("wandb_name", f"segformer_{model_name}_{timestamp}"),
                group=args.wandb_group or cfg.get("wandb_group"),
                mode=args.wandb_mode or cfg.get("wandb_mode"),
                config={k: cfg[k] for k in cfg if not k.endswith("_path")},
            )
        except ImportError:
            pass

    trainer = FisherAdapTuneTrainer(
        model=model,
        optimizer=optimizer,
        train_loader=train_loader,
        train_step_fn=make_train_step(loss_fn, device, devtype),
        val_loader=val_loader,
        val_step_fn=make_val_step(loss_fn, device, devtype),
        scheduler=scheduler,
        early_stopping=early_stopping,
        num_epochs=num_epochs,
        weight_decay=weight_decay,
        val_interval=val_interval,
        checkpoint_interval=int(cfg.get("checkpoint_interval", 5)),
        output_dir=cfg.get("output_dir", "."),
        save_file_name=save_file_name,
        fisher_gammas=fisher_gammas,
        fisher_tcov=fisher_tcov,
        fisher_ema_decay=fisher_ema_decay,
        prev_js_ema_decay=prev_js_ema_decay,
        fisher_ema_interval=fisher_ema_interval,
        freeze_interval=freeze_interval,
        fisher_slice_mode=fisher_slice_mode,
        fisher_slice_blocks=fisher_slice_blocks,
        js_distance_mode=js_distance_mode,
        chunk_selection_metric=chunk_selection_metric,
        js_variance_lambda=js_variance_lambda,
        wandb_run=wandb_run,
        disable_fisher_plots=args.disable_fisher_plots,
        device=device,
    )
    trainer.fit()

    if wandb_run is not None:
        import wandb as _wandb
        _wandb.finish()


if __name__ == "__main__":
    main()
