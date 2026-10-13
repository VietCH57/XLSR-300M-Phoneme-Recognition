#!/usr/bin/env python
"""Fine-tune XLS-R-300M + CTC for phoneme recognition.

Examples
--------
    python scripts/train.py --config configs/default.yaml --output_dir runs/exp1
    python scripts/train.py --config configs/default.yaml --domains native_en l2a el_easy \
        --lr 3e-5 --max_steps 30000 --domain_weights l2a=2 el_easy=4
    python scripts/train.py --config runs/exp1/config.yaml --resume_from runs/exp1/last
"""

from __future__ import annotations

import dataclasses
import logging
import math
import shlex
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from phonrec.config import Config, parse_config  # noqa: E402
from phonrec.data import (  # noqa: E402
    CTCCollator,
    DynamicBatchSampler,
    PhonemeTokenizer,
    build_dataset,
    make_sample_weights,
)
from phonrec.engine import evaluate, make_eval_loader  # noqa: E402
from phonrec.models import XLSRPhonemeCTC  # noqa: E402
from phonrec.utils.misc import DutyCycle, gpu_mem_gb, limit_gpu_memory  # noqa: E402
from phonrec.utils import (  # noqa: E402
    CheckpointManager,
    autocast_ctx,
    build_scheduler,
    count_parameters,
    format_seconds,
    get_device,
    load_trainer_state,
    set_seed,
    setup_logging,
)

logger = logging.getLogger("train")


def build_optimizer(model: XLSRPhonemeCTC, cfg: Config) -> torch.optim.Optimizer:
    def is_no_decay(name: str) -> bool:
        return name.endswith(".bias") or "norm" in name or "masked_spec_embed" in name

    groups = {k: [] for k in ("bb_decay", "bb_nodecay", "head_decay", "head_nodecay")}
    for name, p in model.named_parameters():
        if not p.requires_grad:  # e.g. frozen CNN feature encoder
            continue
        part = "head" if name.startswith("head.") else "bb"
        groups[f"{part}_{'nodecay' if is_no_decay(name) else 'decay'}"].append(p)

    param_groups = [
        {"params": groups["bb_decay"], "lr": cfg.lr, "weight_decay": cfg.weight_decay},
        {"params": groups["bb_nodecay"], "lr": cfg.lr, "weight_decay": 0.0},
        {"params": groups["head_decay"], "lr": cfg.head_lr, "weight_decay": cfg.weight_decay},
        {"params": groups["head_nodecay"], "lr": cfg.head_lr, "weight_decay": 0.0},
    ]
    param_groups = [g for g in param_groups if g["params"]]
    return torch.optim.AdamW(param_groups, betas=(cfg.adam_beta1, cfg.adam_beta2), eps=cfg.adam_eps)


def run_eval(model, loader, tokenizer, device, cfg, step: int) -> dict:
    res = evaluate(model, loader, tokenizer, device, precision=cfg.precision)
    logger.info("=== Eval @ step %d | dev loss %.4f | %.0fs\n%s",
                step, res["loss"] or float("nan"), res["seconds"], res["meter"].table())
    return res["metrics"]


def main(argv=None) -> None:
    cfg = parse_config(argv, description=__doc__)
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(out_dir / "train.log")
    logger.info("Command: %s", " ".join(shlex.quote(a) for a in sys.argv))
    # resume_from is a one-off action: keep it out of the saved config so the file can be reused
    dataclasses.replace(cfg, resume_from=None).save(out_dir / "config.yaml")
    logger.info("Config:\n%s", "\n".join(f"  {k}: {v}" for k, v in cfg.to_dict().items()))

    set_seed(cfg.seed)
    device = get_device()
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = cfg.allow_tf32
        torch.backends.cudnn.allow_tf32 = cfg.allow_tf32
        logger.info("Device: %s (%s)", device, torch.cuda.get_device_name(0))
        limit_gpu_memory(cfg.gpu_memory_fraction, device)
        if cfg.gpu_memory_fraction:
            total_gb = torch.cuda.get_device_properties(device).total_memory / 2**30
            logger.info("GPU memory capped at %.0f%% (%.1f GiB of %.1f GiB)",
                        100 * cfg.gpu_memory_fraction, cfg.gpu_memory_fraction * total_gb, total_gb)
        if cfg.gpu_duty_pct < 100:
            logger.info("GPU duty cycle limited to %d%% during training", cfg.gpu_duty_pct)
    else:
        logger.warning("CUDA not available - training on CPU")
    if cfg.precision == "bf16" and device.type == "cuda" and not torch.cuda.is_bf16_supported():
        raise RuntimeError("bf16 not supported on this GPU, use --precision fp16")

    # ------------------------------------------------------------------ data
    tokenizer = PhonemeTokenizer.from_file(cfg.vocab_path, blank_token=cfg.blank_token, unk_token=cfg.unk_token)
    logger.info("Vocab: %d tokens, blank=%s(%d), unk=%s(%d)", len(tokenizer), tokenizer.blank_token,
                tokenizer.blank_id, tokenizer.unk_token, tokenizer.unk_id)
    common = dict(sample_rate=cfg.sample_rate, cache_dir=cfg.cache_dir, path_map=cfg.path_map,
                  domains=cfg.domains or None, num_workers=max(cfg.num_workers, 8))
    train_ds = build_dataset(cfg.train_csv, tokenizer, split="train",
                             min_audio_seconds=cfg.min_audio_seconds,
                             max_audio_seconds=cfg.max_audio_seconds,
                             unk_policy=cfg.train_unk_policy, **common)
    dev_ds = build_dataset(cfg.dev_csv, tokenizer, split="dev", **common)
    if len(train_ds) == 0 or len(dev_ds) == 0:
        raise RuntimeError("Empty train or dev set after filtering")

    weights = make_sample_weights(train_ds.domains, cfg.domain_weight_dict())
    if weights is not None:
        logger.info("Domain oversampling weights: %s", cfg.domain_weight_dict())
    train_sampler = DynamicBatchSampler(
        train_ds.durations, cfg.max_batch_seconds, cfg.max_batch_size, shuffle=True,
        seed=cfg.seed, weights=weights, bucket_size_multiplier=cfg.bucket_size_multiplier,
    )
    train_loader = DataLoader(
        train_ds, batch_sampler=train_sampler, collate_fn=CTCCollator(),
        num_workers=cfg.num_workers, pin_memory=device.type == "cuda",
        persistent_workers=cfg.num_workers > 0,
        prefetch_factor=4 if cfg.num_workers > 0 else None,
    )
    dev_loader = make_eval_loader(dev_ds, cfg.eval_max_batch_seconds, cfg.max_batch_size, cfg.num_workers)
    steps_per_epoch = math.ceil(len(train_sampler) / cfg.grad_accum_steps)
    logger.info("Train batches/epoch: %d (≈%d optimizer steps); dev batches: %d",
                len(train_sampler), steps_per_epoch, len(dev_loader))

    # ----------------------------------------------------------------- model
    model = XLSRPhonemeCTC.from_pretrained_backbone(cfg, len(tokenizer), tokenizer.blank_id).to(device)
    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg.scheduler, cfg.max_steps, cfg.warmup_steps,
                                cfg.tri_stage_hold_ratio, cfg.final_lr_scale)
    scaler = torch.amp.GradScaler(device.type, enabled=(cfg.precision == "fp16" and device.type == "cuda"))
    ckpt = CheckpointManager(out_dir, cfg.select_metric, cfg.save_top_k,
                             mode="min")  # PER and loss: lower is better

    step, epoch, patience = 0, 0, 0
    if cfg.resume_from:
        state = load_trainer_state(cfg.resume_from)
        model.load_state_dict(torch.load(Path(cfg.resume_from) / "model.pt", map_location="cpu",
                                         weights_only=True))
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        if "scaler" in state:
            scaler.load_state_dict(state["scaler"])
        ckpt.load_state_dict(state["ckpt_manager"])
        step, epoch, patience = state["step"], state["epoch"] + 1, state.get("patience", 0)
        logger.info("Resumed from %s at step %d (next epoch %d)", cfg.resume_from, step, epoch)

    backbone_frozen = step < cfg.freeze_transformer_steps
    model.set_backbone_trainable(not backbone_frozen)
    if backbone_frozen:
        logger.info("Transformer frozen until step %d (training head only)", cfg.freeze_transformer_steps)
    total, trainable = count_parameters(model)
    logger.info("Parameters: %.1fM total, %.1fM trainable", total / 1e6, trainable / 1e6)

    def trainer_state() -> dict:
        return {"optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                "scaler": scaler.state_dict(), "step": step, "epoch": epoch, "patience": patience}

    # ----------------------------------------------------------------- train
    model.train()
    optimizer.zero_grad(set_to_none=True)
    micro = 0
    win = {"loss": 0.0, "n": 0, "audio_sec": 0.0, "skipped": 0, "t": time.time()}
    t_start = time.time()
    last_eval_step = -1
    stop = False
    duty = DutyCycle(cfg.gpu_duty_pct, device)

    while step < cfg.max_steps and epoch < cfg.max_epochs and not stop:
        train_sampler.set_epoch(epoch)
        for batch in train_loader:
            duty.start()
            inputs = batch["input_values"].to(device, non_blocking=True)
            mask = batch["attention_mask"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)
            lab_lens = batch["label_lengths"].to(device, non_blocking=True)
            try:
                with autocast_ctx(device, cfg.precision):
                    loss = model(inputs, mask, labels=labels, label_lengths=lab_lens)["loss"]
                if not torch.isfinite(loss):
                    raise FloatingPointError("non-finite loss")
                scaler.scale(loss / cfg.grad_accum_steps).backward()
            except (torch.cuda.OutOfMemoryError, FloatingPointError) as e:
                logger.warning("Skipping batch (%s): bs=%d, max_len=%.1fs", type(e).__name__,
                               inputs.size(0), inputs.size(1) / cfg.sample_rate)
                optimizer.zero_grad(set_to_none=True)
                micro = 0
                win["skipped"] += 1
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                duty.stop()
                continue

            micro += 1
            win["loss"] += float(loss.detach())
            win["n"] += 1
            win["audio_sec"] += float(mask.sum()) / cfg.sample_rate
            if micro < cfg.grad_accum_steps:
                duty.stop()
                continue
            micro = 0

            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], cfg.max_grad_norm)
            if not scaler.is_enabled() and not torch.isfinite(grad_norm):
                logger.warning("Non-finite gradient norm at step %d, skipping update", step)
                optimizer.zero_grad(set_to_none=True)
                win["skipped"] += 1
                duty.stop()
                continue
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            step += 1
            duty.stop()
            duty.throttle()

            if backbone_frozen and step >= cfg.freeze_transformer_steps:
                backbone_frozen = False
                model.set_backbone_trainable(True)
                logger.info("Step %d: unfreezing transformer", step)

            if step % cfg.log_every == 0:
                dt = time.time() - win["t"]
                lrs = scheduler.get_last_lr()
                logger.info(
                    "step %d/%d | ep %d | loss %.4f | grad %.2f | lr %.2e/%.2e | %.0f audio-s/s | "
                    "peak mem %.1f GiB | skipped %d | elapsed %s",
                    step, cfg.max_steps, epoch, win["loss"] / max(win["n"], 1), float(grad_norm),
                    lrs[0], lrs[-1], win["audio_sec"] / max(dt, 1e-6), gpu_mem_gb(device), win["skipped"],
                    format_seconds(time.time() - t_start),
                )
                win = {"loss": 0.0, "n": 0, "audio_sec": 0.0, "skipped": 0, "t": time.time()}

            if step % cfg.eval_every == 0 or step >= cfg.max_steps:
                metrics = run_eval(model, dev_loader, tokenizer, device, cfg, step)
                last_eval_step = step
                improved = ckpt.save_eval(step, model, cfg, tokenizer, metrics)
                patience = 0 if improved else patience + 1
                ckpt.save_last(step, model, cfg, tokenizer, trainer_state())
                if cfg.early_stopping_patience and patience >= cfg.early_stopping_patience:
                    logger.info("Early stopping: no improvement in %s for %d evals",
                                cfg.select_metric, patience)
                    stop = True
            if stop or step >= cfg.max_steps:
                break
        epoch += 1

    if last_eval_step != step:
        metrics = run_eval(model, dev_loader, tokenizer, device, cfg, step)
        ckpt.save_eval(step, model, cfg, tokenizer, metrics)
        ckpt.save_last(step, model, cfg, tokenizer, trainer_state())

    logger.info("Done in %s. Best %s = %s at step %s -> %s", format_seconds(time.time() - t_start),
                cfg.select_metric, ckpt.best_score, ckpt.best_step, out_dir / "best")


if __name__ == "__main__":
    main()
