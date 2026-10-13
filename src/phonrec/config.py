"""Experiment configuration.

All hyperparameters live in a single flat dataclass (`Config`). Values are resolved in
this order (later wins):

    1. dataclass defaults below
    2. YAML file given by ``--config`` (may be flat or grouped into sections)
    3. command-line flags, e.g. ``--lr 3e-5 --domains l2a el_easy``

Every field is automatically exposed as a CLI flag with the same name.
"""

from __future__ import annotations

import argparse
import dataclasses
import typing
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import yaml

DATA_ROOT = "/home/mdd/lamnh/MDD-2026/data/stage2_combined/server"


@dataclass
class Config:
    # ------------------------------------------------------------------ data
    train_csv: str = f"{DATA_ROOT}/fullnative_el2a_train_ctc.csv"
    dev_csv: str = f"{DATA_ROOT}/fullnative_el2a_dev_ctc.csv"
    vocab_path: str = "configs/vocab.json"
    # Keep only these domains in BOTH train and dev (None = all domains).
    # Choices: native_en native_vi l2a el_easy
    domains: Optional[List[str]] = None
    # Rewrite audio path prefixes: OLD=NEW (repeatable)
    path_map: List[str] = field(default_factory=list)
    # Durations cache and (optional) pre-resampled 16 kHz audio produced by prepare_audio.py
    cache_dir: str = "cache"
    sample_rate: int = 16000
    min_audio_seconds: float = 0.3
    # Training utterances longer than this are dropped (dev/test are never filtered)
    max_audio_seconds: float = 30.0
    blank_token: str = "[PAD]"
    unk_token: str = "[UNK]"
    # Literal [UNK] labels in training transcripts: keep (as a class) | drop_token | drop_utt
    train_unk_policy: str = "keep"

    # ----------------------------------------------------------------- model
    pretrained_name: str = "facebook/wav2vec2-xls-r-300m"
    freeze_feature_encoder: bool = True
    # Train only the linear head for the first N optimizer steps
    freeze_transformer_steps: int = 0
    gradient_checkpointing: bool = True
    attention_dropout: float = 0.1
    hidden_dropout: float = 0.1
    activation_dropout: float = 0.0
    feat_proj_dropout: float = 0.0
    final_dropout: float = 0.1
    layerdrop: float = 0.05
    # SpecAugment (applied inside wav2vec2 during training)
    mask_time_prob: float = 0.05
    mask_time_length: int = 10
    mask_feature_prob: float = 0.0
    mask_feature_length: int = 64
    ctc_reduction: str = "mean"  # mean | sum

    # ---------------------------------------------------------- optimization
    output_dir: str = "runs/default"
    seed: int = 42
    max_steps: int = 20000  # optimizer steps
    max_epochs: int = 1000  # safety cap; training normally stops at max_steps
    max_batch_seconds: float = 200.0  # padded audio seconds per batch
    max_batch_size: int = 64
    grad_accum_steps: int = 1
    lr: float = 5e-5  # XLS-R backbone
    head_lr: float = 5e-4  # linear head
    weight_decay: float = 0.0
    adam_beta1: float = 0.9
    adam_beta2: float = 0.98
    adam_eps: float = 1e-8
    scheduler: str = "linear"  # linear | cosine | tri_stage | constant
    warmup_steps: int = 1000
    tri_stage_hold_ratio: float = 0.4  # fraction of post-warmup steps held at peak LR
    final_lr_scale: float = 0.05  # tri_stage: final LR = peak * final_lr_scale
    max_grad_norm: float = 1.0
    precision: str = "bf16"  # bf16 | fp16 | fp32
    allow_tf32: bool = True
    # Oversampling per domain, e.g. --domain_weights l2a=3 el_easy=4 (default weight 1)
    domain_weights: List[str] = field(default_factory=list)
    num_workers: int = 6
    bucket_size_multiplier: int = 100
    # Shared-GPU etiquette: cap this process' VRAM (fraction of the card, 0 = no cap) and
    # GPU time (percentage of wall time spent computing during training, 100 = no throttling)
    gpu_memory_fraction: float = 0.0
    gpu_duty_pct: int = 100

    # ------------------------------------------------------ eval / checkpoint
    log_every: int = 50
    eval_every: int = 1000
    eval_max_batch_seconds: float = 400.0
    select_metric: str = "per/ALL"  # per/ALL | per/native_en | per/native_vi | per/l2a | per/el_easy | loss
    save_top_k: int = 3
    early_stopping_patience: int = 0  # number of evals without improvement; 0 = disabled
    resume_from: Optional[str] = None

    # ------------------------------------------------------------------------
    def validate(self) -> None:
        from phonrec.data.manifest import DOMAINS

        if self.domains:
            bad = [d for d in self.domains if d not in DOMAINS]
            if bad:
                raise ValueError(f"Unknown domains {bad}; choose from {DOMAINS}")
        if self.precision not in ("bf16", "fp16", "fp32"):
            raise ValueError(f"precision must be bf16|fp16|fp32, got {self.precision}")
        if self.scheduler not in ("linear", "cosine", "tri_stage", "constant"):
            raise ValueError(f"Unknown scheduler {self.scheduler}")
        if self.train_unk_policy not in ("keep", "drop_token", "drop_utt"):
            raise ValueError("train_unk_policy must be keep|drop_token|drop_utt")
        if self.ctc_reduction not in ("mean", "sum"):
            raise ValueError("ctc_reduction must be mean|sum")
        if not 0.0 <= self.gpu_memory_fraction <= 1.0:
            raise ValueError("gpu_memory_fraction must be in [0, 1]")
        if not 1 <= self.gpu_duty_pct <= 100:
            raise ValueError("gpu_duty_pct must be in [1, 100]")
        if self.grad_accum_steps < 1:
            raise ValueError("grad_accum_steps must be >= 1")
        if not (self.select_metric == "loss" or self.select_metric.startswith("per/")):
            raise ValueError("select_metric must be 'loss' or 'per/<domain|ALL>'")
        for item in self.domain_weights:
            if "=" not in item:
                raise ValueError(f"domain_weights entries must look like name=weight, got {item}")
        for item in self.path_map:
            if "=" not in item:
                raise ValueError(f"path_map entries must look like OLD=NEW, got {item}")

    def domain_weight_dict(self) -> Dict[str, float]:
        out = {}
        for item in self.domain_weights:
            k, v = item.split("=", 1)
            out[k.strip()] = float(v)
        return out

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)

    def save(self, path: str | Path) -> None:
        with open(path, "w") as f:
            yaml.safe_dump(self.to_dict(), f, sort_keys=False, allow_unicode=True)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Config":
        flat = _flatten(d)
        names = {f.name for f in dataclasses.fields(cls)}
        unknown = set(flat) - names
        if unknown:
            raise ValueError(f"Unknown config keys: {sorted(unknown)}")
        return cls(**flat)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Config":
        with open(path) as f:
            return cls.from_dict(yaml.safe_load(f) or {})


# ---------------------------------------------------------------------- helpers
def _flatten(d: Dict[str, Any]) -> Dict[str, Any]:
    """Allow YAML grouped into sections (data:, model:, train:, ...)."""
    names = {f.name for f in dataclasses.fields(Config)}
    out: Dict[str, Any] = {}
    for k, v in d.items():
        if isinstance(v, dict) and k not in names:
            out.update(_flatten(v))
        else:
            out[k] = v
    return out


def _str2bool(v: str) -> bool:
    if isinstance(v, bool):
        return v
    s = v.lower()
    if s in ("1", "true", "yes", "y", "on"):
        return True
    if s in ("0", "false", "no", "n", "off"):
        return False
    raise argparse.ArgumentTypeError(f"Expected a boolean, got {v}")


def _optional(cast):
    def parse(v: str):
        return None if v.lower() in ("none", "null") else cast(v)

    return parse


def _add_field_argument(parser: argparse.ArgumentParser, f: dataclasses.Field, hint) -> None:
    origin = typing.get_origin(hint)
    args = typing.get_args(hint)
    optional = origin is typing.Union and type(None) in args
    if optional:
        hint = next(a for a in args if a is not type(None))
        origin = typing.get_origin(hint)
        args = typing.get_args(hint)

    kw: Dict[str, Any] = {"default": argparse.SUPPRESS, "dest": f.name}
    if origin in (list, List):
        kw["nargs"] = "*"
        kw["type"] = args[0] if args else str
        kw["metavar"] = f.name.upper()
    elif hint is bool:
        kw["type"] = _str2bool
        kw["metavar"] = "BOOL"
    else:
        kw["type"] = _optional(hint) if optional else hint
    parser.add_argument(f"--{f.name}", **kw)


def build_arg_parser(description: str = "") -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=description, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--config", type=str, default=None, help="YAML config file")
    hints = typing.get_type_hints(Config)
    for f in dataclasses.fields(Config):
        _add_field_argument(parser, f, hints[f.name])
    return parser


def parse_config(argv: Optional[Sequence[str]] = None, description: str = "") -> Config:
    parser = build_arg_parser(description)
    ns = vars(parser.parse_args(argv))
    config_path = ns.pop("config", None)
    base: Dict[str, Any] = {}
    if config_path:
        with open(config_path) as f:
            base = _flatten(yaml.safe_load(f) or {})
    base.update(ns)  # CLI overrides YAML
    cfg = Config.from_dict(base)
    cfg.validate()
    return cfg
