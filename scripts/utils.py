"""Utility classes for FisherAdapTune: early stopping, checkpointing, and JS history plotting."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn


class EarlyStopping:
    def __init__(self, patience: int = 5, delta: float = 0.0, verbose: bool = False):
        self.patience = patience
        self.delta = delta
        self.verbose = verbose
        self.best_loss = float("inf")
        self.epochs_without_improvement = 0
        self.should_stop = False

    def __call__(self, val_loss: float) -> None:
        if val_loss < self.best_loss - self.delta:
            self.best_loss = val_loss
            self.epochs_without_improvement = 0
        else:
            self.epochs_without_improvement += 1
            if self.verbose:
                print(
                    f"[EarlyStopping] No improvement for "
                    f"{self.epochs_without_improvement}/{self.patience} epochs."
                )
        if self.epochs_without_improvement >= self.patience:
            self.should_stop = True


def save_checkpoint(
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    epoch: Optional[int] = None,
    file_name: str = "checkpoint",
    output_dir: str = "saved_models",
    extra: Optional[dict] = None,
) -> Path:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    payload: dict = {"model_state_dict": model.state_dict()}
    if optimizer is not None:
        payload["optimizer_state_dict"] = optimizer.state_dict()
    if epoch is not None:
        payload["epoch"] = int(epoch)
    if extra:
        payload["extra"] = extra
    path = out / f"{file_name}.pth"
    torch.save(payload, path)
    print(f"[Checkpoint] Saved to {path}")
    return path


def plot_js_history(
    chunk_history: Dict[str, List[Tuple[int, float]]],
    out_path: Path,
    step: int,
    chunk_to_class: Optional[Dict[str, str]] = None,
    log_to_wandb: bool = False,
    wandb_key: str = "fisher/js_grid",
) -> None:
    """Plot per-chunk JS-distance history as a grid, one subplot per module class.

    Requires matplotlib. Silently skips if not installed or no history to plot.
    """
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    if not chunk_history:
        return

    chunk_to_class = chunk_to_class or {}
    classes = sorted({chunk_to_class.get(k, "Other") for k in chunk_history})
    if not classes:
        classes = ["All"]

    n_cols = min(3, len(classes))
    n_rows = (len(classes) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 4 * n_rows), squeeze=False)
    fig.suptitle(f"JS-distance (step {step})", fontsize=12)

    for ax_idx, cls in enumerate(classes):
        ax = axes[ax_idx // n_cols][ax_idx % n_cols]
        keys = [k for k in chunk_history if chunk_to_class.get(k, "Other") == cls]
        for key in keys:
            history = chunk_history[key]
            if not history:
                continue
            steps, values = zip(*history)
            ax.plot(steps, values, linewidth=0.8, alpha=0.6)
        ax.set_title(cls, fontsize=9)
        ax.set_xlabel("step")
        ax.set_ylabel("JS distance")

    # Hide unused axes
    for ax_idx in range(len(classes), n_rows * n_cols):
        axes[ax_idx // n_cols][ax_idx % n_cols].set_visible(False)

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=100, bbox_inches="tight")
    plt.close(fig)

    if log_to_wandb:
        try:
            import wandb

            wandb.log({wandb_key: wandb.Image(str(out_path))}, step=step)
        except Exception:
            pass
