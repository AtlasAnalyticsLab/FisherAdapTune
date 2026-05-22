"""FisherAdapTune fine-tuning of SAM2 for crack segmentation.

Run from the FisherAdapTune/ directory:

    python crack_segmentation/train_sam2.py \
        --config crack_segmentation/config_sam2.yaml

Requires:
  - SAM2 package:  https://github.com/facebookresearch/segment-anything-2
  - A SAM2 checkpoint (.pt) matching the model_cfg
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

from dataset import SAM2SegmentationDataset


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


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

class DiceBCELoss(nn.Module):
    def __init__(self, gamma: float = 0.65):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = self.bce(logits, targets)
        prob = torch.sigmoid(logits)
        num = 2.0 * (prob * targets).sum()
        den = prob.sum() + targets.sum() + 1e-6
        dice = 1.0 - num / den
        return bce + self.gamma * dice


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def build_dataloaders(cfg: Dict, device: torch.device):
    ds_kwargs = dict(
        image_transform=None,   # uses sam2_default_transforms() internally
        mask_transform=None,
    )
    train_ds = SAM2SegmentationDataset(cfg["train_data_path"], **ds_kwargs)
    val_ds   = SAM2SegmentationDataset(cfg["validation_data_path"], **ds_kwargs)
    ldr_kw = dict(
        batch_size=int(cfg.get("batch_size", 2)),
        num_workers=int(cfg.get("num_workers", 8)),
        pin_memory=bool(cfg.get("pin_memory", True)),
        persistent_workers=bool(cfg.get("persistent_workers", True)) and int(cfg.get("num_workers", 8)) > 0,
        prefetch_factor=int(cfg.get("prefetch_factor", 2)) if int(cfg.get("num_workers", 8)) > 0 else None,
    )
    return (
        DataLoader(train_ds, shuffle=True, **ldr_kw),
        DataLoader(val_ds, shuffle=False, **ldr_kw),
    )


# ---------------------------------------------------------------------------
# Forward functions
# ---------------------------------------------------------------------------

def make_train_step(predictor, loss_fn, device: torch.device, devtype: str):
    def train_step(model, batch):
        image_batch, mask_batch, bbox_batch = batch
        masks = mask_batch.to(device, non_blocking=(devtype == "cuda"))
        if masks.sum() == 0:
            return torch.tensor(0.0, device=device, requires_grad=True)

        images = [img.permute(1, 2, 0).cpu().numpy() for img in image_batch]
        bboxes = [bb if bb.ndim > 1 else bb[None, None, :] for bb in bbox_batch]

        predictor.set_image_batch(images)
        _, _, _, unnorm_box = predictor._prep_prompts(
            point_coords=None, point_labels=None,
            box=torch.stack(bboxes).to(device, non_blocking=(devtype == "cuda")),
            mask_logits=None, normalize_coords=False,
        )
        sparse_emb, dense_emb = predictor.model.sam_prompt_encoder(
            points=None,
            boxes=unnorm_box.to(device, non_blocking=(devtype == "cuda")),
            masks=None,
        )
        high_res = [
            feat[-1].unsqueeze(0).to(device, non_blocking=(devtype == "cuda"))
            for feat in predictor._features["high_res_feats"]
        ]
        low_res_masks, _, _, _ = predictor.model.sam_mask_decoder(
            image_embeddings=predictor._features["image_embed"],
            image_pe=predictor.model.sam_prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_emb,
            dense_prompt_embeddings=dense_emb,
            multimask_output=False,
            repeat_image=False,
            high_res_features=high_res,
        )
        prd_masks = predictor._transforms.postprocess_masks(low_res_masks, predictor._orig_hw[-1])
        return loss_fn(prd_masks, masks)

    return train_step


def make_val_step(predictor, loss_fn, device: torch.device, devtype: str):
    def val_step(model, batch):
        image_batch, mask_batch, bbox_batch = batch
        masks = mask_batch.to(device, non_blocking=(devtype == "cuda"))
        if masks.sum() == 0:
            return {}

        images = [img.permute(1, 2, 0).cpu().numpy() for img in image_batch]
        bboxes = [bb if bb.ndim > 1 else bb[None, None, :] for bb in bbox_batch]

        with torch.inference_mode():
            predictor.set_image_batch(images)
            _, _, _, unnorm_box = predictor._prep_prompts(
                point_coords=None, point_labels=None,
                box=torch.stack(bboxes).to(device, non_blocking=(devtype == "cuda")),
                mask_logits=None, normalize_coords=False,
            )
            sparse_emb, dense_emb = predictor.model.sam_prompt_encoder(
                points=None,
                boxes=unnorm_box.to(device, non_blocking=(devtype == "cuda")),
                masks=None,
            )
            high_res = [
                feat[-1].unsqueeze(0).to(device, non_blocking=(devtype == "cuda"))
                for feat in predictor._features["high_res_feats"]
            ]
            low_res_masks, _, _, _ = predictor.model.sam_mask_decoder(
                image_embeddings=predictor._features["image_embed"],
                image_pe=predictor.model.sam_prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_emb,
                dense_prompt_embeddings=dense_emb,
                multimask_output=False,
                repeat_image=False,
                high_res_features=high_res,
            )
            prd_masks = predictor._transforms.postprocess_masks(low_res_masks, predictor._orig_hw[-1])

        loss_val = loss_fn(prd_masks, masks).item()
        pred_bin = prd_masks > 0.0
        gt_bin   = masks > 0.5
        tp = int((pred_bin & gt_bin).sum())
        fp = int((pred_bin & ~gt_bin).sum())
        fn = int((~pred_bin & gt_bin).sum())
        tn = int((~pred_bin & ~gt_bin).sum())
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
    p = argparse.ArgumentParser(description="FisherAdapTune — SAM2 crack segmentation")
    p.add_argument("--config", default="crack_segmentation/config_sam2.yaml")
    p.add_argument("--train-data-path", default=None)
    p.add_argument("--val-data-path", default=None)
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

    if args.train_data_path:
        cfg["train_data_path"] = args.train_data_path
    if args.val_data_path:
        cfg["validation_data_path"] = args.val_data_path

    if args.slurm:
        for key in ("train_data_path", "validation_data_path"):
            cfg[key] = cfg.get(key, "").replace("$TMPDIR/src", args.slurm)

    def _pick(arg_val, cfg_key, default):
        return arg_val if arg_val is not None else cfg.get(cfg_key, default)

    seed = int(_pick(args.random_seed, "seed", 0))
    num_epochs = int(cfg.get("num_epochs", 5))
    lr = float(cfg.get("learning_rate", 1e-6))
    weight_decay = float(cfg.get("weight_decay", 5e-5))
    lambda_value = float(cfg.get("lambda_value", 0.65))
    fisher_gammas = list(cfg.get("adafisher_gamma", [0.8, 0.8]))
    fisher_tcov = int(cfg.get("adafisher_tcov", 1))
    prev_js_ema_decay = float(_pick(args.prev_js_ema_decay, "prev_js_ema_decay", 0.9))
    fisher_ema_decay = float(_pick(args.fisher_ema_decay, "fisher_ema_decay", prev_js_ema_decay))
    fisher_ema_interval = max(1, int(_pick(args.fisher_ema_interval, "fisher_ema_interval", 300)))
    freeze_interval = max(1, int(_pick(args.freeze_interval, "freeze_interval", 3000)))
    fisher_slice_mode = str(_pick(args.fisher_slice_mode, "fisher_slice_mode", "row"))
    fisher_slice_blocks = max(1, int(_pick(args.fisher_slice_blocks, "fisher_slice_blocks", 4)))
    js_distance_mode = str(_pick(args.js_distance_mode, "js_distance_mode", "log"))
    chunk_selection_metric = str(_pick(args.chunk_selection_metric, "chunk_selection_metric", "total_variation"))
    js_variance_lambda = float(_pick(args.stage2_js_variance_lambda, "stage2_js_variance_lambda", 1.0))
    val_interval_raw = _pick(args.val_interval, "val_interval", None)
    val_interval = max(1, int(val_interval_raw)) if val_interval_raw is not None else None
    save_file_name = args.save_file_name or cfg.get("save_file_name", "sam2_fisher_adapt_tune")

    _set_seed(seed)
    device, devtype = _build_device()
    print(f"Device: {device}  seed: {seed}")

    train_loader, val_loader = build_dataloaders(cfg, device)

    try:
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor
    except ImportError as exc:
        raise RuntimeError(
            "SAM2 package not found. "
            "Install it from https://github.com/facebookresearch/segment-anything-2"
        ) from exc

    sam2_model = build_sam2(cfg["model_cfg"], cfg["sam2_checkpoint_path"], device=device)
    predictor = SAM2ImagePredictor(sam2_model)
    predictor.model.train()

    loss_fn = DiceBCELoss(gamma=lambda_value)
    optimizer = torch.optim.AdamW(predictor.model.parameters(), lr=lr, weight_decay=0.0)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    early_stopping = EarlyStopping(patience=int(cfg.get("early_stopping_patience", 3)), verbose=True)

    wandb_run = None
    if not args.disable_wandb:
        try:
            import wandb
            timestamp = dt.now().strftime("%Y%m%d-%H%M%S")
            wandb_run = wandb.init(
                project=args.wandb_project or cfg.get("wandb_project", "FisherAdapTune-SAM2"),
                name=args.wandb_name or cfg.get("wandb_name", f"sam2_{timestamp}"),
                group=args.wandb_group or cfg.get("wandb_group"),
                mode=args.wandb_mode or cfg.get("wandb_mode"),
                config={k: cfg[k] for k in cfg if not k.endswith("_path")},
            )
        except ImportError:
            pass

    trainer = FisherAdapTuneTrainer(
        model=predictor.model,
        optimizer=optimizer,
        train_loader=train_loader,
        train_step_fn=make_train_step(predictor, loss_fn, device, devtype),
        val_loader=val_loader,
        val_step_fn=make_val_step(predictor, loss_fn, device, devtype),
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
