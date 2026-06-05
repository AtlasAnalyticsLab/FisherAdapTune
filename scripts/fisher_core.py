"""Core Fisher utilities for FisherAdapTune.

Contains:
  - Fisher tensor collection from AdaFisher hooks
  - JS-distance functions for Fisher distribution drift
  - Chunk-spec building and Fisher-guided iterative freeze logic
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Set, Tuple

import numpy as np
import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# JS-distance constants
# ---------------------------------------------------------------------------
_JS_HIST_BINS = 64
_JS_EPS = 1e-12
_JS_PERCENTILES = (1.0, 99.0)


# ---------------------------------------------------------------------------
# Parameter → module indexing
# ---------------------------------------------------------------------------


@torch.no_grad()
def index_param_to_module(model: nn.Module) -> Dict[str, Tuple[str, str]]:
    """Map param name → (module_path, module_class_name)."""
    id2name = {id(p): n for n, p in model.named_parameters()}
    owned: Dict[str, Tuple[str, str]] = {}
    for mpath, mod in model.named_modules():
        for pname, p in mod.named_parameters(recurse=False):
            full = f"{mpath}.{pname}" if mpath else pname
            if id(p) in id2name and id2name[id(p)] == full:
                owned[full] = (mpath, mod.__class__.__name__)
    return owned


def layer_to_class_map(param_to_modinfo: Dict[str, Tuple[str, str]]) -> Dict[str, str]:
    """Map layer path → class name for visualisation grouping."""
    mapping: Dict[str, str] = {}
    for _param, (layer_path, cls_name) in param_to_modinfo.items():
        if not layer_path:
            continue
        mapping.setdefault(layer_path, cls_name)
        lower = layer_path.lower()
        if ".attn.qkv" in lower:
            for label in ("q", "k", "v"):
                mapping.setdefault(f"{layer_path}.{label}", cls_name)
    return mapping


# ---------------------------------------------------------------------------
# Fisher tensor collection
# ---------------------------------------------------------------------------


@torch.no_grad()
def collect_fisher_tensors(
    model: nn.Module,
    adafisher_opt,
    out_dtype: torch.dtype = torch.float32,
) -> Dict[str, torch.Tensor]:
    """Collect per-parameter Fisher approximation tensors from AdaFisher hooks.

    Returns a dict mapping parameter name → Fisher tensor (same shape as param).
    Parameters not covered by AdaFisher get zero-filled tensors.
    """
    device = next(model.parameters()).device
    name_to_ft: Dict[str, torch.Tensor] = {}

    def _module_has_grad(m: nn.Module) -> bool:
        return any(p.requires_grad and p.grad is not None for p in m.parameters(recurse=False))

    for m in adafisher_opt.modules:
        if not _module_has_grad(m):
            continue
        try:
            Ft = adafisher_opt._get_F_tilde(m)
        except Exception:
            Ft = None

        ft_w, ft_b = None, None
        if isinstance(Ft, list):
            if len(Ft) >= 1:
                ft_w = Ft[0]
            if len(Ft) >= 2:
                ft_b = Ft[1]
        elif isinstance(Ft, torch.Tensor):
            ft_w = Ft

        if hasattr(m, "weight"):
            p_w = m.weight
            if p_w is not None and p_w.requires_grad:
                if ft_w is None:
                    ft_w = torch.zeros_like(p_w, device=device, dtype=out_dtype)
                else:
                    ft_w = torch.nan_to_num(ft_w.to(device=device, dtype=out_dtype)).abs()
                for n, p2 in model.named_parameters():
                    if p2 is p_w:
                        name_to_ft[n] = ft_w
                        break

        if hasattr(m, "bias"):
            p_b = m.bias
            if p_b is not None and p_b.requires_grad:
                if ft_b is None:
                    ft_b = torch.zeros_like(p_b, device=device, dtype=out_dtype)
                else:
                    ft_b = torch.nan_to_num(ft_b.to(device=device, dtype=out_dtype)).abs()
                for n, p2 in model.named_parameters():
                    if p2 is p_b:
                        name_to_ft[n] = ft_b
                        break

    for n, p in model.named_parameters():
        if p.requires_grad and n not in name_to_ft:
            name_to_ft[n] = torch.zeros_like(p, device=device, dtype=out_dtype)

    return name_to_ft


# ---------------------------------------------------------------------------
# JS-distance helpers
# ---------------------------------------------------------------------------


def _layer_log_range(
    arrays: Tuple[np.ndarray, ...],
    eps: float,
    percentile_range: Tuple[float, float],
) -> Tuple[float, float]:
    collected: List[float] = []
    for arr in arrays:
        if arr.size == 0:
            continue
        log_vals = np.log10(np.abs(arr.astype(np.float64)) + eps)
        finite = log_vals[np.isfinite(log_vals)]
        if finite.size > 0:
            collected.extend(finite.tolist())
    if not collected:
        return (math.log10(eps), math.log10(eps) + 1.0)
    vals = np.asarray(collected, dtype=np.float64)
    low, high = np.percentile(vals, percentile_range)
    if not np.isfinite(low):
        low = float(np.nanmin(vals))
    if not np.isfinite(high):
        high = float(np.nanmax(vals))
    if high <= low:
        high = low + 1e-3
    return float(low), float(high)


def _build_histogram_log(values: np.ndarray, bins: np.ndarray, eps: float) -> np.ndarray:
    if values.size == 0:
        log_vals = np.array([bins[0]], dtype=np.float64)
    else:
        log_vals = np.log10(np.abs(values.astype(np.float64)) + eps)
    hist, _ = np.histogram(log_vals, bins=bins)
    hist = hist.astype(np.float64) + eps
    s = float(hist.sum())
    if s <= 0 or not np.isfinite(s):
        return np.full(hist.shape, 1.0 / len(hist), dtype=np.float64)
    return hist / s


def _build_histogram_raw(values: np.ndarray, bins: np.ndarray, eps: float) -> np.ndarray:
    hist_vals = (
        np.abs(values.astype(np.float64))
        if values.size > 0
        else np.array([bins[0]], dtype=np.float64)
    )
    hist, _ = np.histogram(hist_vals, bins=bins)
    hist = hist.astype(np.float64) + eps
    s = float(hist.sum())
    if s <= 0 or not np.isfinite(s):
        return np.full(hist.shape, 1.0 / len(hist), dtype=np.float64)
    return hist / s


def _js_div_sqrt(p: np.ndarray, q: np.ndarray, eps: float = _JS_EPS) -> float:
    p = np.clip(np.asarray(p, dtype=np.float64), eps, None)
    q = np.clip(np.asarray(q, dtype=np.float64), eps, None)
    p /= float(p.sum())
    q /= float(q.sum())
    m = 0.5 * (p + q)
    kl_pm = float(np.sum(p * np.log(p / m)))
    kl_qm = float(np.sum(q * np.log(q / m)))
    return float(np.sqrt(max(0.0, 0.5 * (kl_pm + kl_qm))))


def js_distance_log(ref: np.ndarray, cur: np.ndarray) -> Optional[float]:
    """JS distance using log-space Fisher histograms (default, scale-invariant)."""
    if ref.size == 0 or cur.size == 0:
        return None
    log_min, log_max = _layer_log_range((ref, cur), eps=_JS_EPS, percentile_range=_JS_PERCENTILES)
    bins = np.linspace(log_min, log_max, _JS_HIST_BINS + 1, dtype=np.float64)
    return _js_div_sqrt(
        _build_histogram_log(ref, bins, _JS_EPS),
        _build_histogram_log(cur, bins, _JS_EPS),
    )


def js_distance_raw(ref: np.ndarray, cur: np.ndarray) -> Optional[float]:
    """JS distance using raw-magnitude Fisher histograms."""
    if ref.size == 0 or cur.size == 0:
        return None
    ref_v = np.abs(ref.astype(np.float64))
    cur_v = np.abs(cur.astype(np.float64))
    combined = np.concatenate((ref_v, cur_v))
    finite = combined[np.isfinite(combined)]
    if finite.size == 0:
        return None
    low = float(np.min(finite))
    high = float(np.max(finite))
    if high <= low:
        high = low + 1e-12
    bins = np.linspace(low, high, _JS_HIST_BINS + 1, dtype=np.float64)
    return _js_div_sqrt(
        _build_histogram_raw(ref_v, bins, _JS_EPS),
        _build_histogram_raw(cur_v, bins, _JS_EPS),
    )


# ---------------------------------------------------------------------------
# Chunk specifications
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ChunkSpec:
    key: str
    param_name: str
    layer_key: str
    cls_name: str
    index: Tuple


def _param_kind(name: str) -> str:
    if name.endswith(".weight"):
        return "weight"
    if name.endswith(".bias"):
        return "bias"
    return name.rsplit(".", 1)[-1]


def _axis_block_slices(axis_size: int, num_blocks: int) -> List[slice]:
    if axis_size <= 0:
        return []
    num_blocks = max(1, min(int(num_blocks), int(axis_size)))
    boundaries = np.linspace(0, axis_size, num_blocks + 1, dtype=np.int64)
    slices: List[slice] = []
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        if int(end) > int(start):
            slices.append(slice(int(start), int(end)))
    return slices


def _build_chunk_specs_for_param(
    param_name: str,
    param: torch.Tensor,
    layer_path: str,
    cls_name: str,
    slice_blocks: int,
    slice_mode: str,
) -> List[ChunkSpec]:
    if param.ndim == 0 or param.numel() == 0:
        return []

    kind = _param_kind(param_name)
    specs: List[ChunkSpec] = []

    if slice_mode == "row":
        rows = int(param.shape[0])
        if rows <= 0:
            return []
    elif slice_mode == "column":
        if param.ndim < 2:
            layer_key = f"{layer_path}.{kind}" if layer_path else f"{param_name}.{kind}"
            return [
                ChunkSpec(
                    key=f"{layer_key}.full",
                    param_name=param_name,
                    layer_key=layer_path,
                    cls_name=cls_name,
                    index=(slice(None),),
                )
            ]
        cols = int(param.shape[1])
        if cols <= 0:
            return []
    else:
        raise ValueError(f"Unsupported slice_mode: {slice_mode!r}. Choose 'row' or 'column'.")

    # Special handling: split QKV weight rows into Q, K, V sub-chunks
    if slice_mode == "row" and layer_path.endswith(".attn.qkv"):
        rows = int(param.shape[0])
        if rows % 3 == 0:
            split_rows = rows // 3
            for label_idx, label in enumerate(("q", "k", "v")):
                split_start = label_idx * split_rows
                layer_key = f"{layer_path}.{label}.{kind}"
                for block_idx, row_slice in enumerate(_axis_block_slices(split_rows, slice_blocks)):
                    index = (
                        slice(split_start + row_slice.start, split_start + row_slice.stop),
                    ) + tuple(slice(None) for _ in range(1, param.ndim))
                    specs.append(
                        ChunkSpec(
                            key=f"{layer_key}.block{block_idx}",
                            param_name=param_name,
                            layer_key=f"{layer_path}.{label}",
                            cls_name=cls_name,
                            index=index,
                        )
                    )
            return specs

    layer_key = f"{layer_path}.{kind}" if layer_path else f"{param_name}.{kind}"
    axis_size = int(param.shape[0]) if slice_mode == "row" else int(param.shape[1])
    for block_idx, axis_slice in enumerate(_axis_block_slices(axis_size, slice_blocks)):
        if slice_mode == "row":
            index = (axis_slice,) + tuple(slice(None) for _ in range(1, param.ndim))
            suffix = f"rowblock{block_idx}"
        else:
            index = (slice(None), axis_slice) + tuple(slice(None) for _ in range(2, param.ndim))
            suffix = f"colblock{block_idx}"
        specs.append(
            ChunkSpec(
                key=f"{layer_key}.{suffix}",
                param_name=param_name,
                layer_key=layer_path,
                cls_name=cls_name,
                index=index,
            )
        )
    return specs


def build_all_chunk_specs(
    model: nn.Module,
    param_to_modinfo: Dict[str, Tuple[str, str]],
    slice_blocks: int,
    slice_mode: str,
) -> Tuple[Dict[str, List[ChunkSpec]], Dict[str, str]]:
    """Build ChunkSpec objects for every trainable parameter in the model."""
    specs_by_param: Dict[str, List[ChunkSpec]] = {}
    chunk_to_class: Dict[str, str] = {}
    ltoc = layer_to_class_map(param_to_modinfo)
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        layer_path, cls_name = param_to_modinfo.get(name, ("", ""))
        if not layer_path:
            layer_path = name.rsplit(".", 1)[0]
        if not cls_name:
            cls_name = ltoc.get(layer_path, "")
        specs = _build_chunk_specs_for_param(
            name, param, layer_path, cls_name, slice_blocks, slice_mode
        )
        if specs:
            specs_by_param[name] = specs
            for spec in specs:
                chunk_to_class[spec.key] = spec.cls_name
    return specs_by_param, chunk_to_class


# ---------------------------------------------------------------------------
# Chunk Fisher array collection
# ---------------------------------------------------------------------------


def collect_chunk_fisher_arrays(
    ftensors: Dict[str, torch.Tensor],
    chunk_specs_by_param: Dict[str, List[ChunkSpec]],
    allowed_chunks: Optional[Set[str]] = None,
) -> Dict[str, np.ndarray]:
    chunk_arrays: Dict[str, np.ndarray] = {}
    for param_name, specs in chunk_specs_by_param.items():
        ftensor = ftensors.get(param_name)
        if ftensor is None or ftensor.numel() == 0:
            continue
        ftensor = torch.nan_to_num(ftensor.detach(), nan=0.0, posinf=0.0, neginf=0.0).abs()
        for spec in specs:
            if allowed_chunks is not None and spec.key not in allowed_chunks:
                continue
            values = ftensor[spec.index].reshape(-1)
            if values.numel() == 0:
                continue
            chunk_arrays[spec.key] = values.float().cpu().numpy().copy()
    return chunk_arrays


# ---------------------------------------------------------------------------
# JS history tracking
# ---------------------------------------------------------------------------


def update_chunk_js_history(
    chunk_arrays: Dict[str, np.ndarray],
    chunk_fisher_ema: Dict[str, np.ndarray],
    chunk_js_distance_ema: Dict[str, float],
    chunk_js_history: Dict[str, List[Tuple[int, float]]],
    chunk_total_variation: Dict[str, float],
    chunk_js_sum: Dict[str, float],
    chunk_js_count: Dict[str, int],
    step: int,
    fisher_decay: float,
    js_decay: float,
    js_distance_fn: Callable,
) -> Dict[str, float]:
    js_latest: Dict[str, float] = {}
    for chunk_key, values in chunk_arrays.items():
        if values.size == 0:
            continue
        prev_values = chunk_fisher_ema.get(chunk_key)
        prev_js_ema = chunk_js_distance_ema.get(chunk_key)
        if prev_values is None or prev_values.size == 0 or prev_js_ema is None:
            chunk_fisher_ema[chunk_key] = values.copy()
            chunk_js_distance_ema[chunk_key] = 0.0
            chunk_js_history.setdefault(chunk_key, []).append((step, 0.0))
            chunk_total_variation.setdefault(chunk_key, 0.0)
            chunk_js_sum.setdefault(chunk_key, 0.0)
            chunk_js_count.setdefault(chunk_key, 1)
            js_latest[chunk_key] = 0.0
            continue
        js = js_distance_fn(prev_values, values)
        chunk_fisher_ema[chunk_key] = fisher_decay * prev_values + (1.0 - fisher_decay) * values
        if js is None:
            continue
        curr_js_ema = js_decay * prev_js_ema + (1.0 - js_decay) * float(js)
        chunk_js_distance_ema[chunk_key] = curr_js_ema
        js_latest[chunk_key] = curr_js_ema
        history = chunk_js_history.setdefault(chunk_key, [])
        if history:
            chunk_total_variation[chunk_key] = chunk_total_variation.get(chunk_key, 0.0) + abs(
                curr_js_ema - history[-1][1]
            )
        else:
            chunk_total_variation.setdefault(chunk_key, 0.0)
        history.append((step, curr_js_ema))
        chunk_js_sum[chunk_key] = chunk_js_sum.get(chunk_key, 0.0) + curr_js_ema
        chunk_js_count[chunk_key] = chunk_js_count.get(chunk_key, 0) + 1
    return js_latest


def prune_chunk_state(
    chunk_fisher_ema: Dict[str, np.ndarray],
    chunk_js_distance_ema: Dict[str, float],
    chunk_js_history: Dict[str, List[Tuple[int, float]]],
    chunk_total_variation: Dict[str, float],
    chunk_js_sum: Dict[str, float],
    chunk_js_count: Dict[str, int],
    allowed_chunks: Optional[Set[str]],
) -> None:
    if allowed_chunks is None:
        return
    for state in (
        chunk_fisher_ema,
        chunk_js_distance_ema,
        chunk_js_history,
        chunk_total_variation,
        chunk_js_sum,
        chunk_js_count,
    ):
        for key in list(state.keys()):
            if key not in allowed_chunks:
                del state[key]


# ---------------------------------------------------------------------------
# Gradient masking
# ---------------------------------------------------------------------------


def _zero_optimizer_state(
    optimizer: torch.optim.Optimizer,
    param: nn.Parameter,
    mask: torch.Tensor,
) -> None:
    state = optimizer.state.get(param)
    if not state:
        return
    for value in state.values():
        if torch.is_tensor(value) and value.shape == param.shape:
            value.masked_fill_(mask, 0)


def apply_weight_decay_to_masked(
    model: nn.Module,
    param_train_masks: Dict[str, torch.Tensor],
    lr: float,
    weight_decay: float,
) -> None:
    if weight_decay <= 0.0:
        return
    factor = 1.0 - lr * weight_decay
    with torch.no_grad():
        for name, param in model.named_parameters():
            mask = param_train_masks.get(name)
            if mask is None or not param.requires_grad:
                continue
            if bool(mask.any()):
                param.data[mask] *= factor


def apply_chunk_masks(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    chunk_specs_by_param: Dict[str, List[ChunkSpec]],
    selected_chunks: Set[str],
    param_train_masks: Dict[str, torch.Tensor],
) -> Tuple[List[str], int, int]:
    active_names: List[str] = []
    active_chunk_count = 0
    active_elements = 0

    with torch.no_grad():
        for name, param in model.named_parameters():
            specs = chunk_specs_by_param.get(name, [])
            if not specs:
                continue
            new_mask = torch.zeros_like(param, dtype=torch.bool)
            for spec in specs:
                if spec.key in selected_chunks:
                    new_mask[spec.index] = True
            old_mask = param_train_masks[name]
            newly_frozen = old_mask & (~new_mask)
            if bool(newly_frozen.any()):
                _zero_optimizer_state(optimizer, param, newly_frozen)
            param_train_masks[name] = new_mask
            if not bool(new_mask.any()):
                continue
            if param.grad is not None:
                param.grad.mul_(new_mask)
            active_names.append(name)
            active_chunk_count += sum(1 for spec in specs if spec.key in selected_chunks)
            active_elements += int(new_mask.sum().item())

    return active_names, active_chunk_count, active_elements


def freeze_zero_fisher_chunks(
    ftensors: Dict[str, torch.Tensor],
    chunk_specs_by_param: Dict[str, List[ChunkSpec]],
) -> Set[str]:
    """Identify and return chunk keys whose Fisher tensor is entirely zero."""
    frozen: Set[str] = set()
    for param_name, specs in chunk_specs_by_param.items():
        ftensor = ftensors.get(param_name)
        if ftensor is None or ftensor.numel() == 0:
            continue
        ftensor = torch.nan_to_num(ftensor.detach(), nan=0.0, posinf=0.0, neginf=0.0).abs()
        for spec in specs:
            values = ftensor[spec.index]
            if values.numel() > 0 and torch.count_nonzero(values).item() == 0:
                frozen.add(spec.key)
    if frozen:
        print(f"[Freeze] Zero-Fisher chunks found and frozen: {len(frozen)}")
    return frozen


def register_mask_hooks(
    model: nn.Module,
    param_train_masks: Dict[str, torch.Tensor],
) -> None:
    """Register backward hooks that multiply gradients by the current trainability mask."""
    for name, param in model.named_parameters():
        if name not in param_train_masks:
            continue
        if not param.requires_grad:
            continue

        def _make_hook(pname: str):
            def _mask_grad(grad: torch.Tensor) -> torch.Tensor:
                return grad * param_train_masks[pname].to(device=grad.device, dtype=grad.dtype)

            return _mask_grad

        param.register_hook(_make_hook(name))
