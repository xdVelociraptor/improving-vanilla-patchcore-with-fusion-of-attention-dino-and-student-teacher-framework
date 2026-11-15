"""StableAdamW and WarmCosineScheduler, ported verbatim in behaviour from Dinomaly
(github.com/guojiajeremy/Dinomaly @ 1f252be, optimizers/StableAdamW.py and utils.py; Apache-2.0).
"""
from __future__ import annotations

import math

import numpy as np
import torch
from torch.optim.lr_scheduler import _LRScheduler
from torch.optim.optimizer import Optimizer


class StableAdamW(Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-2, amsgrad=False,
                 clip_threshold: float = 1.0):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, amsgrad=amsgrad,
                        clip_threshold=clip_threshold)
        super().__init__(params, defaults)

    @staticmethod
    def _rms(t: torch.Tensor) -> torch.Tensor:
        return t.norm(2) / (t.numel() ** 0.5)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                p.data.mul_(1 - group["lr"] * group["weight_decay"])
                grad = p.grad
                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p)
                    state["exp_avg_sq"] = torch.zeros_like(p)
                    if group["amsgrad"]:
                        state["max_exp_avg_sq"] = torch.zeros_like(p)
                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                beta1, beta2 = group["betas"]
                state["step"] += 1
                bc1 = 1 - beta1 ** state["step"]
                bc2 = 1 - beta2 ** state["step"]
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                if group["amsgrad"]:
                    torch.max(state["max_exp_avg_sq"], exp_avg_sq, out=state["max_exp_avg_sq"])
                    denom = (state["max_exp_avg_sq"].sqrt() / math.sqrt(bc2)).add_(group["eps"])
                else:
                    denom = (exp_avg_sq.sqrt() / math.sqrt(bc2)).add_(group["eps"])
                lr_scale = grad / denom
                lr_scale = max(1.0, float(self._rms(lr_scale)) / group["clip_threshold"])
                step_size = group["lr"] / bc1 / lr_scale
                p.data.addcdiv_(exp_avg, denom, value=-step_size)
        return loss


class WarmCosineScheduler(_LRScheduler):
    def __init__(self, optimizer, base_value, final_value, total_iters, warmup_iters=0, start_warmup_value=0):
        self.final_value = final_value
        self.total_iters = total_iters
        warmup = np.linspace(start_warmup_value, base_value, warmup_iters)
        iters = np.arange(total_iters - warmup_iters)
        cos = final_value + 0.5 * (base_value - final_value) * (1 + np.cos(np.pi * iters / len(iters)))
        self.schedule = np.concatenate((warmup, cos))
        super().__init__(optimizer)

    def get_lr(self):
        if self.last_epoch >= self.total_iters:
            return [self.final_value for _ in self.base_lrs]
        return [self.schedule[self.last_epoch] for _ in self.base_lrs]
