"""LR schedules (multiplicative factor on each param group's peak LR)."""

from __future__ import annotations

import math

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR


def build_scheduler(
    optimizer: Optimizer,
    name: str,
    total_steps: int,
    warmup_steps: int,
    hold_ratio: float = 0.4,
    final_lr_scale: float = 0.05,
) -> LambdaLR:
    warmup_steps = max(0, min(warmup_steps, total_steps))
    decay_total = max(1, total_steps - warmup_steps)

    def warm(step: int) -> float:
        return (step + 1) / max(1, warmup_steps)

    if name == "constant":
        def fn(step):
            return warm(step) if step < warmup_steps else 1.0
    elif name == "linear":
        def fn(step):
            if step < warmup_steps:
                return warm(step)
            return max(0.0, 1.0 - (step - warmup_steps) / decay_total)
    elif name == "cosine":
        def fn(step):
            if step < warmup_steps:
                return warm(step)
            t = min(1.0, (step - warmup_steps) / decay_total)
            return 0.5 * (1.0 + math.cos(math.pi * t))
    elif name == "tri_stage":
        hold = int(decay_total * hold_ratio)
        decay = max(1, decay_total - hold)
        log_final = math.log(final_lr_scale)

        def fn(step):
            if step < warmup_steps:
                return warm(step)
            if step < warmup_steps + hold:
                return 1.0
            t = min(1.0, (step - warmup_steps - hold) / decay)
            return math.exp(log_final * t)
    else:
        raise ValueError(f"Unknown scheduler {name}")

    return LambdaLR(optimizer, fn)
