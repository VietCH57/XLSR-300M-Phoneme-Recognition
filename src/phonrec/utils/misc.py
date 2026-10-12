from __future__ import annotations

import os
import random

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


def count_parameters(model: torch.nn.Module) -> tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def format_seconds(s: float) -> str:
    s = int(s)
    return f"{s // 3600:d}h{(s % 3600) // 60:02d}m{s % 60:02d}s"
