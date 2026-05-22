"""FisherAdapTune trainer — model-agnostic Fisher-guided iterative chunk freeze.

Usage
-----
Define two callables:

    def train_step(model, batch):
        # run forward pass, compute loss
        return loss  # scalar Tensor with grad_fn

    def val_step(model, batch):
        # run forward pass under inference_mode, compute metrics
        return {"metric_name": value, ...}

Then:

    trainer = FisherAdapTuneTrainer(
        model=model,
        optimizer=optimizer,
        train_loader=train_loader,
        train_step_fn=train_step,
        ...
    )
    trainer.fit(num_epochs=5)
"""

from __future__ import annotations

import os
from datetime import datetime as dt
from pathlib import Path
from typing import Callable, Dict, List, Optional, Set, Tuple

import numpy as np
import torch
import torch.nn as nn

from .adafisher import AdaFisher
from .fisher_core import (
    apply_chunk_masks,
    apply_weight_decay_to_masked,
    build_all_chunk_specs,
    collect_chunk_fisher_arrays,
    collect_fisher_tensors,
    freeze_zero_fisher_chunks,
    index_param_to_module,
    js_distance_log,
    js_distance_raw,
    prune_chunk_state,
    register_mask_hooks,
    update_chunk_js_history,
)
from .utils import EarlyStopping, plot_js_history, save_checkpoint


class FisherAdapTuneTrainer:
    """Fisher-guided Adaptive Fine-Tuning trainer.

    Parameters
    ----------
    model : nn.Module
        The model to fine-tune. Must have at least one of Linear / Conv2d /
        BatchNorm2d / LayerNorm layers for Fisher tracking.
    optimizer : torch.optim.Optimizer
        Outer optimizer (e.g. AdamW).  Weight decay should be set to 0 here
        because FisherAdapTune applies it manually to masked entries only.
    train_loader : iterable
        Yields batches passed directly to ``train_step_fn``.
    train_step_fn : callable(model, batch) -> Tensor
        Computes and returns a scalar loss.  Do NOT call ``.backward()`` or
        ``optimizer.step()`` inside — the trainer handles those.
    val_loader : iterable, optional
        Yields batches passed to ``val_step_fn``.
    val_step_fn : callable(model, batch) -> dict, optional
        Returns a dict of metric name → scalar value.  The trainer averages
        each metric across all batches and logs the result.
    scheduler : LR scheduler, optional
        Called once per epoch via ``scheduler.step()``.
    early_stopping : EarlyStopping, optional
        Checked after each epoch.  If None, no early stopping is applied.
    num_epochs : int
        Number of training epochs.
    weight_decay : float
        Applied manually to masked gradient entries (decoupled from optimizer).
    val_interval : int, optional
        Run validation every N training steps (in addition to end-of-epoch).
    checkpoint_interval : int
        Save a checkpoint every N epochs (0 = only at the end).
    output_dir : str
        Base directory for checkpoints and Fisher visualisations.
    save_file_name : str
        Base name for checkpoint files.

    Fisher / freeze hyperparameters
    --------------------------------
    fisher_gammas : list[float, float]
        EMA coefficients [1-α, α] for AdaFisher activation/gradient stats.
    fisher_tcov : int
        AdaFisher stats update interval (steps).
    fisher_ema_decay : float
        EMA decay for Fisher tensors between consecutive steps.
    prev_js_ema_decay : float
        EMA decay applied to the JS-distance signal.
    fisher_ema_interval : int
        Steps between Fisher / JS-distance updates.
    freeze_interval : int
        Steps between chunk-freeze decisions.
    fisher_slice_mode : {"row", "column"}
        Axis used to partition weight tensors into chunks.
    fisher_slice_blocks : int
        Number of contiguous chunks per tensor along the slice axis.
    js_distance_mode : {"log", "raw"}
        Whether to use log-normalised or raw Fisher histograms for JS distance.
    chunk_selection_metric : {"total_variation", "mean_js"}
        Statistic used to rank chunks at each freeze step.
    js_variance_lambda : float
        Threshold = mean(score) + λ × std(score).  Larger λ keeps more chunks.

    Wandb / logging
    ---------------
    wandb_run : wandb.sdk.wandb_run.Run, optional
        Active wandb run.  If None, wandb logging is skipped.
    disable_fisher_plots : bool
        Skip saving per-step JS-distance grid plots (reduces I/O overhead).
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        train_loader,
        train_step_fn: Callable,
        # Validation
        val_loader=None,
        val_step_fn: Optional[Callable] = None,
        # LR schedule / stopping
        scheduler=None,
        early_stopping: Optional[EarlyStopping] = None,
        # Training
        num_epochs: int = 1,
        weight_decay: float = 5e-5,
        val_interval: Optional[int] = None,
        checkpoint_interval: int = 5,
        output_dir: str = ".",
        save_file_name: str = "fisher_adapt_tune_checkpoint",
        # Fisher hyperparams
        fisher_gammas: List[float] = None,
        fisher_tcov: int = 1,
        fisher_ema_decay: float = 0.9,
        prev_js_ema_decay: float = 0.9,
        fisher_ema_interval: int = 1,
        freeze_interval: int = 300,
        fisher_slice_mode: str = "row",
        fisher_slice_blocks: int = 4,
        js_distance_mode: str = "log",
        chunk_selection_metric: str = "total_variation",
        js_variance_lambda: float = 1.0,
        # Logging
        wandb_run=None,
        disable_fisher_plots: bool = False,
        device: Optional[torch.device] = None,
    ):
        self.model = model
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.train_step_fn = train_step_fn
        self.val_loader = val_loader
        self.val_step_fn = val_step_fn
        self.scheduler = scheduler
        self.early_stopping = early_stopping
        self.num_epochs = num_epochs
        self.weight_decay = weight_decay
        self.val_interval = val_interval
        self.checkpoint_interval = checkpoint_interval
        self.output_dir = Path(output_dir)
        self.save_file_name = save_file_name

        if fisher_gammas is None:
            fisher_gammas = [0.8, 0.8]
        self.fisher_gammas = fisher_gammas
        self.fisher_tcov = fisher_tcov
        self.fisher_ema_decay = fisher_ema_decay
        self.prev_js_ema_decay = prev_js_ema_decay
        self.fisher_ema_interval = max(1, fisher_ema_interval)
        self.freeze_interval = max(1, freeze_interval)
        self.fisher_slice_mode = fisher_slice_mode
        self.fisher_slice_blocks = max(1, fisher_slice_blocks)

        if js_distance_mode not in ("log", "raw"):
            raise ValueError(f"js_distance_mode must be 'log' or 'raw', got {js_distance_mode!r}")
        self.js_distance_fn = js_distance_log if js_distance_mode == "log" else js_distance_raw

        if chunk_selection_metric not in ("total_variation", "mean_js"):
            raise ValueError(
                f"chunk_selection_metric must be 'total_variation' or 'mean_js', "
                f"got {chunk_selection_metric!r}"
            )
        self.chunk_selection_metric = chunk_selection_metric
        self.js_variance_lambda = js_variance_lambda

        self.wandb_run = wandb_run
        self.disable_fisher_plots = disable_fisher_plots

        if device is None:
            if torch.cuda.is_available():
                device = torch.device("cuda")
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device = torch.device("mps")
            else:
                device = torch.device("cpu")
        self.device = device

        timestamp = dt.now().strftime("%Y%m%d-%H%M%S")
        self.fisher_dir = self.output_dir / "fisher_js_plots" / f"run_{timestamp}"

        self._total_steps = 0
        self._setup_fisher()

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def _setup_fisher(self) -> None:
        self.fisher_opt = AdaFisher(
            self.model,
            TCov=self.fisher_tcov,
            gammas=self.fisher_gammas,
            Lambda=0,
        )
        param_to_modinfo = index_param_to_module(self.model)
        self._chunk_specs_by_param, self._chunk_to_class = build_all_chunk_specs(
            self.model,
            param_to_modinfo=param_to_modinfo,
            slice_blocks=self.fisher_slice_blocks,
            slice_mode=self.fisher_slice_mode,
        )
        self._param_train_masks: Dict[str, torch.Tensor] = {
            name: torch.ones_like(param, dtype=torch.bool)
            for name, param in self.model.named_parameters()
            if name in self._chunk_specs_by_param
        }
        register_mask_hooks(self.model, self._param_train_masks)

        self._trainable_chunks: Set[str] = {
            spec.key
            for specs in self._chunk_specs_by_param.values()
            for spec in specs
        }
        self._chunk_fisher_ema: Dict[str, np.ndarray] = {}
        self._chunk_js_ema: Dict[str, float] = {}
        self._chunk_js_history: Dict[str, List[Tuple[int, float]]] = {}
        self._chunk_total_variation: Dict[str, float] = {}
        self._chunk_js_sum: Dict[str, float] = {}
        self._chunk_js_count: Dict[str, int] = {}
        self._zero_fisher_frozen = False

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def _run_validation(self) -> Dict[str, float]:
        if self.val_loader is None or self.val_step_fn is None:
            return {}
        self.model.eval()
        accumulator: Dict[str, float] = {}
        counts: Dict[str, int] = {}
        for batch in self.val_loader:
            metrics = self.val_step_fn(self.model, batch)
            for k, v in metrics.items():
                accumulator[k] = accumulator.get(k, 0.0) + float(v)
                counts[k] = counts.get(k, 0) + 1
        self.model.train()
        return {k: accumulator[k] / max(1, counts[k]) for k in accumulator}

    # ------------------------------------------------------------------
    # Freeze helpers
    # ------------------------------------------------------------------

    def _apply_initial_zero_fisher_freeze(self, ftensors) -> None:
        frozen = freeze_zero_fisher_chunks(
            ftensors=ftensors,
            chunk_specs_by_param=self._chunk_specs_by_param,
        )
        self._trainable_chunks -= frozen
        names, chunks, elems = apply_chunk_masks(
            model=self.model,
            optimizer=self.optimizer,
            chunk_specs_by_param=self._chunk_specs_by_param,
            selected_chunks=self._trainable_chunks,
            param_train_masks=self._param_train_masks,
        )
        print(f"[Freeze] Step 0 zero-Fisher freeze: tensors={len(names)}, chunks={chunks}, params={elems:,}")
        self._zero_fisher_frozen = True

    def _apply_variance_freeze(self) -> None:
        if not self._trainable_chunks:
            return
        if self.chunk_selection_metric == "total_variation":
            scores = {k: self._chunk_total_variation.get(k, 0.0) for k in self._trainable_chunks}
        else:
            scores = {
                k: self._chunk_js_sum.get(k, 0.0) / max(1, self._chunk_js_count.get(k, 0))
                for k in self._trainable_chunks
            }
        if not scores:
            return

        score_arr = np.array(list(scores.values()), dtype=np.float64)
        mean_s = float(np.mean(score_arr))
        std_s = float(np.std(score_arr))
        threshold = mean_s + self.js_variance_lambda * std_s

        next_chunks = {k for k, s in scores.items() if s >= threshold}
        if not next_chunks:
            next_chunks = {max(scores, key=lambda k: scores[k])}

        names, chunk_count, elem_count = apply_chunk_masks(
            model=self.model,
            optimizer=self.optimizer,
            chunk_specs_by_param=self._chunk_specs_by_param,
            selected_chunks=next_chunks,
            param_train_masks=self._param_train_masks,
        )
        frozen_count = len(self._trainable_chunks - next_chunks)
        self._trainable_chunks = next_chunks

        prune_chunk_state(
            chunk_fisher_ema=self._chunk_fisher_ema,
            chunk_js_distance_ema=self._chunk_js_ema,
            chunk_js_history=self._chunk_js_history,
            chunk_total_variation=self._chunk_total_variation,
            chunk_js_sum=self._chunk_js_sum,
            chunk_js_count=self._chunk_js_count,
            allowed_chunks=self._trainable_chunks,
        )

        metric_label = "variation" if self.chunk_selection_metric == "total_variation" else "mean_js"
        print(
            f"[Freeze] Step {self._total_steps}: {metric_label} "
            f"mean={mean_s:.4f} std={std_s:.4f} λ={self.js_variance_lambda:.3f} "
            f"threshold={threshold:.4f} | frozen={frozen_count} remaining={len(next_chunks)}"
        )
        print(
            f"[Freeze] Step {self._total_steps}: "
            f"trainable tensors={len(names)} chunks={chunk_count} params={elem_count:,}"
        )

        if self.wandb_run is not None:
            try:
                import wandb
                wandb.log(
                    {
                        "freeze/score_mean": mean_s,
                        "freeze/score_std": std_s,
                        "freeze/score_threshold": threshold,
                        "freeze/trainable_tensors": len(names),
                        "freeze/trainable_chunks": chunk_count,
                        "freeze/trainable_params": elem_count,
                        "freeze/frozen_chunks": frozen_count,
                    },
                    step=self._total_steps,
                )
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Per-step Fisher / JS update
    # ------------------------------------------------------------------

    def _update_fisher_js(self, ftensors) -> Dict[str, float]:
        chunk_arrays = collect_chunk_fisher_arrays(
            ftensors=ftensors,
            chunk_specs_by_param=self._chunk_specs_by_param,
            allowed_chunks=self._trainable_chunks,
        )
        return update_chunk_js_history(
            chunk_arrays=chunk_arrays,
            chunk_fisher_ema=self._chunk_fisher_ema,
            chunk_js_distance_ema=self._chunk_js_ema,
            chunk_js_history=self._chunk_js_history,
            chunk_total_variation=self._chunk_total_variation,
            chunk_js_sum=self._chunk_js_sum,
            chunk_js_count=self._chunk_js_count,
            step=self._total_steps,
            fisher_decay=self.fisher_ema_decay,
            js_decay=self.prev_js_ema_decay,
            js_distance_fn=self.js_distance_fn,
        )

    def _maybe_plot_js(self) -> None:
        if self.disable_fisher_plots:
            return
        out_path = (
            self.fisher_dir
            / f"step_{self._total_steps:07d}"
            / "js_distance.png"
        )
        plot_js_history(
            chunk_history=self._chunk_js_history,
            out_path=out_path,
            step=self._total_steps,
            chunk_to_class=self._chunk_to_class,
            log_to_wandb=self.wandb_run is not None,
            wandb_key="fisher/chunk_js_grid",
        )

    # ------------------------------------------------------------------
    # Main training loop
    # ------------------------------------------------------------------

    def fit(self, num_epochs: Optional[int] = None) -> None:
        """Run the FisherAdapTune training loop.

        Parameters
        ----------
        num_epochs : int, optional
            Overrides the value set in ``__init__`` if provided.
        """
        n_epochs = num_epochs if num_epochs is not None else self.num_epochs
        self.model.train()

        # Baseline validation
        if self.val_loader is not None and self.val_step_fn is not None:
            print("[Val] Baseline validation before training...")
            val_metrics = self._run_validation()
            self._log_val_metrics(val_metrics, step=0)
            print("[Val] Baseline: " + "  ".join(f"{k}={v:.4f}" for k, v in val_metrics.items()))

        for epoch in range(n_epochs):
            self.model.train()
            epoch_loss = 0.0
            processed_batches = 0

            for batch in self.train_loader:
                self.optimizer.zero_grad()

                loss = self.train_step_fn(self.model, batch)
                loss.backward()

                ftensors = collect_fisher_tensors(
                    model=self.model,
                    adafisher_opt=self.fisher_opt,
                    out_dtype=torch.float32,
                )

                # Initial zero-Fisher freeze (once, at step 0)
                if self._total_steps == 0 and not self._zero_fisher_frozen:
                    self._apply_initial_zero_fisher_freeze(ftensors)

                should_update_ema = (
                    self._total_steps == 0
                    or self._total_steps % self.fisher_ema_interval == 0
                )
                if should_update_ema:
                    js_latest = self._update_fisher_js(ftensors)
                else:
                    js_latest = {}

                should_plot = not self.disable_fisher_plots and (
                    self._total_steps == 0
                    or self._total_steps % self.fisher_ema_interval == 0
                )
                if should_plot:
                    self._maybe_plot_js()

                if self._total_steps > 0 and self._total_steps % self.freeze_interval == 0:
                    self._apply_variance_freeze()

                apply_weight_decay_to_masked(
                    model=self.model,
                    param_train_masks=self._param_train_masks,
                    lr=float(self.optimizer.param_groups[0]["lr"]),
                    weight_decay=self.weight_decay,
                )
                self.optimizer.step()

                self._total_steps += 1
                processed_batches += 1
                epoch_loss += float(loss.item())

                # Step-level validation
                if (
                    self.val_interval is not None
                    and self._total_steps % self.val_interval == 0
                    and self.val_loader is not None
                    and self.val_step_fn is not None
                ):
                    step_val = self._run_validation()
                    print(
                        f"[Val] Step {self._total_steps}: "
                        + "  ".join(f"{k}={v:.4f}" for k, v in step_val.items())
                    )
                    self._log_val_metrics(step_val, step=self._total_steps)
                    self.model.train()

                # Step-level wandb log
                if self.wandb_run is not None:
                    payload: Dict = {
                        "loss/batch": float(loss.item()),
                        "lr": float(self.optimizer.param_groups[0]["lr"]),
                        "epoch": epoch + 1,
                        "trainable/chunks": len(self._trainable_chunks),
                        "trainable/params": int(
                            sum(m.sum().item() for m in self._param_train_masks.values())
                        ),
                    }
                    if js_latest:
                        js_vals = list(js_latest.values())
                        payload["fisher/js_mean"] = float(np.mean(js_vals))
                        payload["fisher/js_max"] = float(np.max(js_vals))
                    try:
                        import wandb
                        wandb.log(payload, step=self._total_steps)
                    except Exception:
                        pass

            # End-of-epoch
            if self.scheduler is not None:
                self.scheduler.step()

            avg_loss = epoch_loss / max(1, processed_batches)

            # End-of-epoch validation
            val_metrics: Dict[str, float] = {}
            if self.val_loader is not None and self.val_step_fn is not None:
                val_metrics = self._run_validation()

            print(
                f"Epoch {epoch + 1}/{n_epochs}: train_loss={avg_loss:.4f}  "
                + "  ".join(f"{k}={v:.4f}" for k, v in val_metrics.items())
            )

            if self.wandb_run is not None:
                epoch_payload = {"loss/train": avg_loss, "epoch": epoch + 1}
                epoch_payload.update({f"val/{k}": v for k, v in val_metrics.items()})
                try:
                    import wandb
                    wandb.log(epoch_payload, step=self._total_steps)
                except Exception:
                    pass

            # Early stopping
            if self.early_stopping is not None and val_metrics:
                monitor = val_metrics.get("loss", next(iter(val_metrics.values())))
                self.early_stopping(monitor)
                if self.early_stopping.should_stop:
                    print("[EarlyStopping] Triggered — stopping training.")
                    break

            # Periodic checkpoint
            if (
                self.checkpoint_interval > 0
                and (epoch + 1) % self.checkpoint_interval == 0
            ):
                save_checkpoint(
                    model=self.model,
                    optimizer=self.optimizer,
                    epoch=epoch + 1,
                    file_name=f"{self.save_file_name}_epoch{epoch + 1}",
                    output_dir=str(self.output_dir / "checkpoints"),
                )

        # Final checkpoint
        save_checkpoint(
            model=self.model,
            optimizer=self.optimizer,
            epoch=n_epochs,
            file_name=self.save_file_name,
            output_dir=str(self.output_dir / "checkpoints"),
        )

    # ------------------------------------------------------------------
    # Wandb helpers
    # ------------------------------------------------------------------

    def _log_val_metrics(self, metrics: Dict[str, float], step: int) -> None:
        if self.wandb_run is None or not metrics:
            return
        try:
            import wandb
            wandb.log({f"val/{k}": v for k, v in metrics.items()}, step=step)
        except Exception:
            pass
