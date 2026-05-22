"""Minimal FisherAdapTune example — image classification with synthetic data.

No real dataset required. Generates random images and labels so you can verify
the full training loop works end-to-end before plugging in your own data.

Run (zero setup):
    python examples/minimal_image_classifier.py

With a real ImageFolder dataset instead:
    python examples/minimal_image_classifier.py \
        --train-dir /path/to/data/train \
        --val-dir   /path/to/data/val

Optional flags:
    --num-epochs 5
    --batch-size 16
    --lr 1e-4
    --output-dir ./checkpoints
    --disable-wandb
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# -- Allow running from repo root without installing the package --------------
sys.path.insert(0, str(Path(__file__).parent.parent))
from scripts import EarlyStopping, FisherAdapTuneTrainer

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

NUM_CLASSES = 4
IMAGE_SIZE = 64  # smaller for fast synthetic runs; use 224 for real ImageFolder data
SYNTH_TRAIN_N = 128
SYNTH_VAL_N = 32


def build_synthetic_loaders(batch_size: int):
    """Random tensors shaped like ImageNet images — no files needed."""

    def _make(n):
        images = torch.randn(n, 3, IMAGE_SIZE, IMAGE_SIZE)
        labels = torch.randint(0, NUM_CLASSES, (n,))
        return DataLoader(TensorDataset(images, labels), batch_size=batch_size, shuffle=True)

    return _make(SYNTH_TRAIN_N), _make(SYNTH_VAL_N), NUM_CLASSES


def build_folder_loaders(train_dir: str, val_dir: str, batch_size: int, num_workers: int = 4):
    """Real ImageFolder dataset. Layout: <split>/<class_name>/*.jpg"""
    try:
        from torchvision import datasets, transforms
    except ImportError as e:
        raise ImportError(
            "torchvision is required for folder datasets: pip install torchvision"
        ) from e
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    train_tf = transforms.Compose(
        [
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )
    val_tf = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )
    train_ds = datasets.ImageFolder(train_dir, transform=train_tf)
    val_ds = datasets.ImageFolder(val_dir, transform=val_tf)
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True
    )
    return train_loader, val_loader, len(train_ds.classes)


# ---------------------------------------------------------------------------
# Step functions
# ---------------------------------------------------------------------------


def make_train_step(loss_fn, device):
    def train_step(model, batch):
        images, labels = batch
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        return loss_fn(logits, labels)

    return train_step


def make_val_step(loss_fn, device):
    def val_step(model, batch):
        images, labels = batch
        images, labels = images.to(device), labels.to(device)
        with torch.inference_mode():
            logits = model(images)
        loss = loss_fn(logits, labels).item()
        preds = logits.argmax(dim=1)
        acc = (preds == labels).float().mean().item()
        return {"loss": loss, "accuracy": acc}

    return val_step


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args():
    p = argparse.ArgumentParser(description="FisherAdapTune minimal example")
    p.add_argument(
        "--train-dir",
        default=None,
        help="Path to training image folder (omit to use synthetic data)",
    )
    p.add_argument(
        "--val-dir",
        default=None,
        help="Path to validation image folder (omit to use synthetic data)",
    )
    p.add_argument("--num-epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=5e-5)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--output-dir", default="./checkpoints")
    p.add_argument("--disable-wandb", action="store_true")
    # Fisher / freeze knobs (sensible defaults for a small classifier)
    p.add_argument("--fisher-ema-interval", type=int, default=50)
    p.add_argument("--freeze-interval", type=int, default=200)
    p.add_argument("--fisher-slice-blocks", type=int, default=4)
    p.add_argument("--js-variance-lambda", type=float, default=1.0)
    return p.parse_args()


def main():
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    if args.train_dir and args.val_dir:
        train_loader, val_loader, num_classes = build_folder_loaders(
            args.train_dir, args.val_dir, args.batch_size, args.num_workers
        )
    else:
        print("No data paths provided — using synthetic random data.")
        train_loader, val_loader, num_classes = build_synthetic_loaders(args.batch_size)

    # Simple CNN — no inplace ops, no residual skip connections.
    # Models with inplace operations (e.g. ResNet's relu_ and out += identity)
    # conflict with FisherAdapTune's backward hooks and must be avoided or patched.
    model = nn.Sequential(
        nn.Conv2d(3, 32, kernel_size=3, padding=1),
        nn.BatchNorm2d(32),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Conv2d(32, 64, kernel_size=3, padding=1),
        nn.BatchNorm2d(64),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Conv2d(64, 128, kernel_size=3, padding=1),
        nn.BatchNorm2d(128),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d(4),
        nn.Flatten(),
        nn.Linear(128 * 4 * 4, 256),
        nn.ReLU(),
        nn.Linear(256, num_classes),
    )
    model = model.to(device)

    loss_fn = nn.CrossEntropyLoss()
    # NOTE: set weight_decay=0.0 in the optimizer — FisherAdapTune applies it
    #       manually to masked parameters only.
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_epochs)
    early_stopping = EarlyStopping(patience=3, verbose=True)

    wandb_run = None
    if not args.disable_wandb:
        try:
            import wandb

            wandb_run = wandb.init(project="FisherAdapTune-example", config=vars(args))
        except ImportError:
            print("wandb not installed — skipping logging.")

    trainer = FisherAdapTuneTrainer(
        model=model,
        optimizer=optimizer,
        train_loader=train_loader,
        train_step_fn=make_train_step(loss_fn, device),
        val_loader=val_loader,
        val_step_fn=make_val_step(loss_fn, device),
        scheduler=scheduler,
        early_stopping=early_stopping,
        num_epochs=args.num_epochs,
        weight_decay=args.weight_decay,
        output_dir=args.output_dir,
        save_file_name="simplecnn_fisher",
        # Fisher / freeze hyperparameters
        fisher_gammas=[0.8, 0.8],
        fisher_ema_decay=0.9,
        prev_js_ema_decay=0.9,
        fisher_ema_interval=args.fisher_ema_interval,
        freeze_interval=args.freeze_interval,
        fisher_slice_mode="row",
        fisher_slice_blocks=args.fisher_slice_blocks,
        js_distance_mode="log",
        chunk_selection_metric="total_variation",
        js_variance_lambda=args.js_variance_lambda,
        wandb_run=wandb_run,
        device=device,
    )

    trainer.fit()

    if wandb_run is not None:
        import wandb as _wandb

        _wandb.finish()


if __name__ == "__main__":
    main()
