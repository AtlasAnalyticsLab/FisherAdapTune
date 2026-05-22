"""AdaFisher optimizer for computing diagonal Fisher Information Matrix approximations.

Based on: "AdaFisher: Adaptive Second Order Optimization via Fisher Information"
https://github.com/AtlasAnalyticsLab/AdaFisher

Used here solely for Fisher statistic collection during FisherAdapTune training.
"""

from __future__ import annotations

from math import prod
from typing import Callable, Dict, List, Optional, Tuple

import torch.distributed as dist
from torch import Tensor, einsum, inf, is_grad_enabled, kron, no_grad, ones_like, zeros_like
from torch.nn import BatchNorm2d, Conv2d, LayerNorm, Linear, Module, Parameter
from torch.nn.functional import pad
from torch.optim import Optimizer


def _smart_detect_inf(tensor: Tensor) -> Tensor:
    result = tensor.clone()
    result[tensor == inf] = 1.0
    result[tensor == -inf] = 0.0
    return result


def _update_running_avg(new: Tensor, current: Tensor, gammas: list) -> None:
    current *= 1 - gammas[0]
    current += new * gammas[1]


def _extract_patches(
    x: Tensor,
    kernel_size: Tuple[int, int],
    stride: Tuple[int, int],
    padding: Tuple[int, int],
    groups: int,
) -> Tensor:
    if padding[0] + padding[1] > 0:
        x = pad(x, (padding[1], padding[1], padding[0], padding[0]))
    batch_size, in_channels, height, width = x.size()
    x = x.view(batch_size, groups, in_channels // groups, height, width)
    x = x.unfold(3, kernel_size[0], stride[0])
    x = x.unfold(4, kernel_size[1], stride[1])
    x = x.permute(0, 1, 3, 4, 2, 5, 6).contiguous()
    x = x.view(batch_size, groups, -1, in_channels // groups * kernel_size[0] * kernel_size[1])
    x = x.view(batch_size, -1, x.size(2), x.size(3))
    return x


class _ComputeHBarD:
    @classmethod
    def __call__(cls, h: Tensor, layer: Module) -> Tensor:
        if isinstance(layer, Linear):
            return cls._linear(h, layer)
        if isinstance(layer, Conv2d):
            return cls._conv2d(h, layer)
        if isinstance(layer, BatchNorm2d):
            return cls._batchnorm2d(h, layer)
        if isinstance(layer, LayerNorm):
            return cls._layernorm(h, layer)
        raise NotImplementedError(f"Unsupported layer type: {type(layer)}")

    @staticmethod
    def _conv2d(h: Tensor, layer: Conv2d) -> Tensor:
        from torch import cat

        batch_size = h.size(0)
        h = _extract_patches(h, layer.kernel_size, layer.stride, layer.padding, layer.groups)
        spatial_size = h.size(2) * h.size(3)
        h = h.reshape(-1, h.size(-1))
        if layer.bias is not None:
            h_bar = cat([h, h.new(h.size(0), 1).fill_(1)], 1)
            return einsum("ij,ij->j", h_bar, h_bar) / (batch_size * spatial_size)
        return einsum("ij,ij->j", h, h) / batch_size

    @staticmethod
    def _linear(h: Tensor, layer: Linear) -> Tensor:
        from torch import cat

        if len(h.shape) > 2:
            h = h.reshape(-1, h.shape[-1])
        batch_size = h.size(0)
        if layer.bias is not None:
            h_bar = cat([h, h.new(h.size(0), 1).fill_(1)], 1)
            return einsum("ij,ij->j", h_bar, h_bar) / batch_size
        return einsum("ij,ij->j", h, h) / batch_size

    @staticmethod
    def _batchnorm2d(h: Tensor, layer: BatchNorm2d) -> Tensor:
        from torch import cat
        from torch import sum as tsum

        batch_size, spatial_size = h.size(0), h.size(2) * h.size(3)
        sum_h = tsum(h, dim=(0, 2, 3)).unsqueeze(1) / (spatial_size**2)
        h_bar = cat([sum_h, sum_h.new(sum_h.size(0), 1).fill_(1)], 1)
        return einsum("ij,ij->j", h_bar, h_bar) / (batch_size**2)

    @staticmethod
    def _layernorm(h: Tensor, layer: LayerNorm) -> Tensor:
        from torch import cat
        from torch import sum as tsum

        dim_to_reduce = [d for d in range(h.ndim) if d != 1]
        batch_size = h.shape[0]
        dim_norm = prod([h.shape[dim] for dim in dim_to_reduce if dim != 0])
        sum_h = tsum(h, dim=dim_to_reduce).unsqueeze(1) / (dim_norm**2)
        h_bar = cat([sum_h, sum_h.new(sum_h.size(0), 1).fill_(1)], 1)
        return einsum("ij,ij->j", h_bar, h_bar) / (batch_size**2)


class _ComputeSD:
    @classmethod
    def __call__(cls, s: Tensor, layer: Module) -> Tensor:
        if isinstance(layer, Conv2d):
            return cls._conv2d(s, layer)
        if isinstance(layer, Linear):
            return cls._linear(s, layer)
        if isinstance(layer, BatchNorm2d):
            return cls._batchnorm2d(s, layer)
        if isinstance(layer, LayerNorm):
            return cls._layernorm(s, layer)
        raise NotImplementedError(f"Unsupported layer type: {type(layer)}")

    @staticmethod
    def _conv2d(s: Tensor, layer: Conv2d) -> Tensor:
        batch_size = s.shape[0]
        spatial_size = s.size(2) * s.size(3)
        s = s.transpose(1, 2).transpose(2, 3).reshape(-1, s.size(1))
        return einsum("ij,ij->j", s, s) / (batch_size * spatial_size)

    @staticmethod
    def _linear(s: Tensor, layer: Linear) -> Tensor:
        if len(s.shape) > 2:
            s = s.reshape(-1, s.shape[-1])
        batch_size = s.size(0)
        return einsum("ij,ij->j", s, s) / batch_size

    @staticmethod
    def _batchnorm2d(s: Tensor, layer: BatchNorm2d) -> Tensor:
        from torch import sum as tsum

        batch_size = s.size(0)
        sum_s = tsum(s, dim=(0, 2, 3))
        return einsum("i,i->i", sum_s, sum_s) / batch_size

    @staticmethod
    def _layernorm(s: Tensor, layer: LayerNorm) -> Tensor:
        from torch import sum as tsum

        batch_size = s.size(0)
        sum_s = tsum(s, dim=tuple(range(s.ndim - 1)))
        return einsum("i,i->i", sum_s, sum_s) / batch_size


class AdaFisherBackbone(Optimizer):
    SUPPORTED_MODULES: Tuple[str, ...] = ("Linear", "Conv2d", "BatchNorm2d", "LayerNorm")

    def __init__(
        self,
        model: Module,
        lr: float = 1e-3,
        beta: float = 0.9,
        Lambda: float = 1e-3,
        gammas: List[float] = None,
        TCov: int = 100,
        weight_decay: float = 0.0,
        dist_training: bool = False,
    ):
        if gammas is None:
            gammas = [0.92, 0.008]
        defaults = dict(lr=lr, beta=beta, weight_decay=weight_decay)
        self.gammas = gammas
        self.Lambda = Lambda
        self.model = model
        self.TCov = TCov
        self.dist_training = dist_training
        self.steps = 0
        self.H_bar_D: Dict[Module, Tensor] = {}
        self.S_D: Dict[Module, Tensor] = {}
        self.modules: List[Module] = []
        self._compute_H = _ComputeHBarD()
        self._compute_S = _ComputeSD()
        self._prepare_model()
        super().__init__(model.parameters(), defaults)

    def _save_input(self, module: Module, input, output) -> None:
        if is_grad_enabled() and self.steps % self.TCov == 0:
            H_i = self._compute_H(input[0].data, module)
            if self.steps == 0:
                self.H_bar_D[module] = H_i.new(H_i.size(0)).fill_(1)
            _update_running_avg(H_i, self.H_bar_D[module], self.gammas)

    def _save_grad_output(self, module: Module, grad_input, grad_output) -> None:
        if self.steps % self.TCov == 0:
            S_i = self._compute_S(grad_output[0].data, module)
            if self.steps == 0:
                self.S_D[module] = S_i.new(S_i.size(0)).fill_(1)
            _update_running_avg(S_i, self.S_D[module], self.gammas)

    def _prepare_model(self) -> None:
        for module in self.model.modules():
            if module.__class__.__name__ in self.SUPPORTED_MODULES:
                self.modules.append(module)
                module.register_forward_hook(self._save_input)
                module.register_full_backward_hook(self._save_grad_output)

    def _get_F_tilde(self, module: Module):
        if self.dist_training:
            dist.all_reduce(self.H_bar_D[module], op=dist.ReduceOp.SUM)
            dist.all_reduce(self.S_D[module], op=dist.ReduceOp.SUM)
            self.H_bar_D[module] /= dist.get_world_size()
            self.S_D[module] /= dist.get_world_size()
        F_tilde = (
            kron(self.H_bar_D[module].unsqueeze(1), self.S_D[module].unsqueeze(0)).t() + self.Lambda
        )
        if module.bias is not None:
            F_tilde = [F_tilde[:, :-1], F_tilde[:, -1:]]
            F_tilde[0] = F_tilde[0].view(*module.weight.grad.data.size())
            F_tilde[1] = F_tilde[1].view(*module.bias.grad.data.size())
            return F_tilde
        return F_tilde.reshape(module.weight.grad.data.size())

    def _check_dim(self, param: List[Parameter], idx_module: int, idx_param: int) -> bool:
        p = param[idx_param]
        m = self.modules[idx_module]
        return p.data.size() == m.weight.data.size() or (
            m.bias is not None and p.data.size() == m.bias.data.size()
        )


class AdaFisher(AdaFisherBackbone):
    """AdaFisher optimizer — used by FisherAdapTune to collect diagonal FIM approximations."""

    def __init__(
        self,
        model: Module,
        lr: float = 1e-3,
        beta: float = 0.9,
        Lambda: float = 1e-3,
        gammas: List[float] = None,
        TCov: int = 100,
        weight_decay: float = 0.0,
        dist_training: bool = False,
    ):
        super().__init__(
            model,
            lr=lr,
            beta=beta,
            Lambda=Lambda,
            gammas=gammas or [0.92, 0.008],
            TCov=TCov,
            weight_decay=weight_decay,
            dist_training=dist_training,
        )

    @no_grad()
    def _step(self, hparams: Dict, param: Parameter, F_tilde: Tensor) -> None:
        grad = param.grad
        state = self.state[param]
        if len(state) == 0:
            state["step"] = 0
            state["exp_avg"] = zeros_like(param)
        exp_avg = state["exp_avg"]
        beta = hparams["beta"]
        state["step"] += 1
        bias_correction = 1 - beta ** state["step"]
        if hparams["weight_decay"] != 0:
            grad = grad.add(param, alpha=hparams["weight_decay"])
        exp_avg.mul_(beta).add_(grad, alpha=1 - beta)
        step_size = hparams["lr"] / bias_correction
        param.addcdiv_(exp_avg, F_tilde, value=-step_size)

    @no_grad()
    def step(self, closure: Optional[Callable] = None) -> None:
        if closure is not None:
            raise NotImplementedError("Closure not supported.")
        for group in self.param_groups:
            idx_param, idx_module, buffer_count = 0, 0, 0
            params = group["params"]
            hparams = {k: group[k] for k in ("lr", "beta", "weight_decay")}
            for _ in range(len(self.modules)):
                if params[idx_param].grad is None:
                    idx_param += 1
                    if params[idx_param].ndim > 1:
                        idx_module += 1
                    else:
                        buffer_count += 1
                    if buffer_count == 2:
                        idx_module += 1
                        buffer_count = 0
                    continue
                m = self.modules[idx_module]
                if self._check_dim(params, idx_module, idx_param):
                    F_tilde = self._get_F_tilde(m)
                    idx_module += 1
                else:
                    F_tilde = ones_like(params[idx_param])
                if isinstance(F_tilde, list):
                    for ft in F_tilde:
                        self._step(hparams, params[idx_param], ft)
                        idx_param += 1
                else:
                    self._step(hparams, params[idx_param], F_tilde)
                    idx_param += 1
        self.steps += 1
