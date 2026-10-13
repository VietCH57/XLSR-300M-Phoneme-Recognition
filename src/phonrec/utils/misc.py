from __future__ import annotations

import os
import random
import time

import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def autocast_dtype(precision: str):
    return {"bf16": torch.bfloat16, "fp16": torch.float16}.get(precision)


def autocast_ctx(device: torch.device, precision: str):
    dtype = autocast_dtype(precision)
    enabled = dtype is not None and device.type == "cuda"
    return torch.autocast(device_type=device.type, dtype=dtype or torch.float32, enabled=enabled)


def limit_gpu_memory(fraction: float, device: torch.device) -> None:
    """Cap this process' share of the GPU memory (0 = no cap). Exceeding it raises OOM."""
    if fraction and device.type == "cuda":
        torch.cuda.set_per_process_memory_fraction(fraction, device)


class DutyCycle:
    """Throttle GPU usage to ``pct`` % of wall time by sleeping after each busy period."""

    def __init__(self, pct: int, device: torch.device):
        self.enabled = pct < 100
        self.ratio = (100.0 / pct - 1.0) if self.enabled else 0.0
        self.device = device
        self.busy = 0.0
        self._t0 = None

    def start(self) -> None:
        if self.enabled:
            self._t0 = time.time()

    def stop(self) -> None:
        if self.enabled and self._t0 is not None:
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
            self.busy += time.time() - self._t0
            self._t0 = None

    def throttle(self) -> None:
        if self.enabled and self.busy > 0:
            time.sleep(self.busy * self.ratio)
            self.busy = 0.0


def gpu_mem_gb(device: torch.device) -> float:
    """Peak allocated memory since the last call (GiB); resets the peak counter."""
    if device.type != "cuda":
        return 0.0
    peak = torch.cuda.max_memory_allocated(device) / 2**30
    torch.cuda.reset_peak_memory_stats(device)
    return peak


def count_parameters(model: torch.nn.Module) -> tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def format_seconds(s: float) -> str:
    s = int(s)
    return f"{s // 3600:d}h{(s % 3600) // 60:02d}m{s % 60:02d}s"
