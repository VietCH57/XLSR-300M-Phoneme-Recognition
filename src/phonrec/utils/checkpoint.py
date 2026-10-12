"""Checkpoint layout inside ``output_dir``::

    checkpoints/step_XXXXXX/   top-k by the selection metric (model only)
    best/                      copy of the best checkpoint  -> use this for inference
    last/                      latest model + optimizer/scheduler state -> for --resume_from

Every checkpoint directory is self-contained for inference:
    model.pt, encoder_config.json, head_config.json, vocab.json, config.yaml, metrics.json
"""

from __future__ import annotations

import dataclasses
import json
import logging
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

logger = logging.getLogger(__name__)

TRAINER_STATE_FILE = "trainer_state.pt"


def _write_ckpt(model, ckpt_dir: Path, cfg, tokenizer, metrics: Optional[Dict[str, Any]], step: int) -> None:
    tmp = ckpt_dir.with_name(ckpt_dir.name + ".tmp")
    if tmp.exists():
        shutil.rmtree(tmp)
    model.save(tmp)
    tokenizer.save(tmp / "vocab.json")
    dataclasses.replace(cfg, resume_from=None).save(tmp / "config.yaml")
    with open(tmp / "metrics.json", "w") as f:
        json.dump({"step": step, **(metrics or {})}, f, indent=2)
    if ckpt_dir.exists():
        shutil.rmtree(ckpt_dir)
    tmp.rename(ckpt_dir)


class CheckpointManager:
    def __init__(self, output_dir: str | Path, metric: str, top_k: int = 3, mode: str = "min"):
        self.root = Path(output_dir)
        self.ckpt_root = self.root / "checkpoints"
        self.ckpt_root.mkdir(parents=True, exist_ok=True)
        self.metric = metric
        self.top_k = top_k
        self.sign = 1.0 if mode == "min" else -1.0
        self.saved: List[Dict[str, Any]] = []  # [{"step", "score", "path"}]
        self.best_score: Optional[float] = None
        self.best_step: Optional[int] = None

    # ------------------------------------------------------------ eval ckpts
    def is_better(self, score: float) -> bool:
        return self.best_score is None or self.sign * score < self.sign * self.best_score

    def save_eval(self, step: int, model, cfg, tokenizer, metrics: Dict[str, Any]) -> bool:
        """Save top-k / best checkpoints. Returns True if this is a new best."""
        score = float(metrics[self.metric])
        improved = self.is_better(score)

        path = None
        if self.top_k > 0:
            worst = max(self.saved, key=lambda r: self.sign * r["score"]) if self.saved else None
            if len(self.saved) < self.top_k or self.sign * score < self.sign * worst["score"]:
                path = self.ckpt_root / f"step_{step:07d}"
                _write_ckpt(model, path, cfg, tokenizer, metrics, step)
                self.saved.append({"step": step, "score": score, "path": str(path)})
                if len(self.saved) > self.top_k:
                    self.saved.sort(key=lambda r: self.sign * r["score"])
                    for r in self.saved[self.top_k:]:
                        shutil.rmtree(r["path"], ignore_errors=True)
                    self.saved = self.saved[: self.top_k]

        if improved:
            self.best_score, self.best_step = score, step
            best = self.root / "best"
            if path is not None:
                if best.exists():
                    shutil.rmtree(best)
                shutil.copytree(path, best)
            else:
                _write_ckpt(model, best, cfg, tokenizer, metrics, step)
            logger.info("New best %s = %.4f at step %d -> %s", self.metric, score, step, best)
        return improved

    # ------------------------------------------------------------ last ckpt
    def save_last(self, step: int, model, cfg, tokenizer, trainer_state: Dict[str, Any]) -> None:
        last = self.root / "last"
        _write_ckpt(model, last, cfg, tokenizer, None, step)
        trainer_state = dict(trainer_state)
        trainer_state["ckpt_manager"] = self.state_dict()
        torch.save(trainer_state, last / TRAINER_STATE_FILE)

    def state_dict(self) -> Dict[str, Any]:
        return {"saved": self.saved, "best_score": self.best_score, "best_step": self.best_step}

    def load_state_dict(self, d: Dict[str, Any]) -> None:
        self.saved = [r for r in d.get("saved", []) if Path(r["path"]).exists()]
        self.best_score = d.get("best_score")
        self.best_step = d.get("best_step")


def load_trainer_state(ckpt_dir: str | Path) -> Dict[str, Any]:
    return torch.load(Path(ckpt_dir) / TRAINER_STATE_FILE, map_location="cpu", weights_only=False)
