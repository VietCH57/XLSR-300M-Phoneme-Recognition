#!/usr/bin/env python
"""Run a trained checkpoint on one or more CSV manifests and report PER per domain.

No domain filtering is applied here: every readable utterance in each CSV is decoded.

Examples
--------
    python scripts/infer.py --ckpt runs/exp1/best \
        --csv /home/mdd/lamnh/MDD-2026/data/stage2_combined/server/fullnative_el2a_test_ctc.csv

    # several sets at once (CSVs without a `domain` column get it inferred from the path)
    python scripts/infer.py --ckpt runs/exp1/best --csv test.csv l2a_v_dev_ctc.csv \
        --output_dir runs/exp1/infer_best
"""

from __future__ import annotations

import argparse
import json
import logging
import shlex
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd  # noqa: E402

from phonrec.config import DATA_ROOT  # noqa: E402
from phonrec.data import PhonemeTokenizer, build_dataset  # noqa: E402
from phonrec.engine import evaluate, make_eval_loader  # noqa: E402
from phonrec.models import XLSRPhonemeCTC  # noqa: E402
from phonrec.utils import get_device, setup_logging  # noqa: E402
from phonrec.utils.misc import limit_gpu_memory  # noqa: E402

logger = logging.getLogger("infer")


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", required=True, help="checkpoint dir (e.g. runs/exp1/best)")
    p.add_argument("--csv", nargs="+", default=[f"{DATA_ROOT}/fullnative_el2a_test_ctc.csv"])
    p.add_argument("--output_dir", default=None, help="default: <ckpt>/infer")
    p.add_argument("--path_map", nargs="*", default=[], help="OLD=NEW audio path prefix rewrites")
    p.add_argument("--cache_dir", default="cache")
    p.add_argument("--max_batch_seconds", type=float, default=400.0)
    p.add_argument("--max_batch_size", type=int, default=64)
    p.add_argument("--num_workers", type=int, default=6)
    p.add_argument("--precision", default="bf16", choices=["bf16", "fp16", "fp32"])
    p.add_argument("--gpu_memory_fraction", type=float, default=0.0, help="VRAM cap (0 = none)")
    return p.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    ckpt = Path(args.ckpt)
    out_root = Path(args.output_dir) if args.output_dir else ckpt / "infer"
    out_root.mkdir(parents=True, exist_ok=True)
    setup_logging(out_root / "infer.log")
    logger.info("Command: %s", " ".join(shlex.quote(a) for a in sys.argv))

    device = get_device()
    limit_gpu_memory(args.gpu_memory_fraction, device)
    with open(ckpt / "head_config.json") as f:
        blank_id = json.load(f)["blank_id"]
    vocab = json.load(open(ckpt / "vocab.json"))
    id2tok = {i: t for t, i in vocab.items()}
    unk = "[UNK]" if "[UNK]" in vocab else next(t for t in vocab if t != id2tok[blank_id])
    tokenizer = PhonemeTokenizer(vocab, blank_token=id2tok[blank_id], unk_token=unk)

    model = XLSRPhonemeCTC.from_checkpoint(ckpt).to(device).eval()
    meta = json.load(open(ckpt / "metrics.json")) if (ckpt / "metrics.json").exists() else {}
    logger.info("Loaded %s (step %s)", ckpt, meta.get("step"))

    summary = {}
    for csv in args.csv:
        has_ref = "text" in pd.read_csv(csv, nrows=0).columns
        ds = build_dataset(csv, tokenizer, split="test", cache_dir=args.cache_dir, path_map=args.path_map,
                           domains=None, require_text=has_ref, num_workers=max(args.num_workers, 8))
        if len(ds) == 0:
            logger.warning("%s: nothing to decode", csv)
            continue
        loader = make_eval_loader(ds, args.max_batch_seconds, args.max_batch_size, args.num_workers)
        res = evaluate(model, loader, tokenizer, device, precision=args.precision, compute_loss=has_ref,
                       has_reference=has_ref, keep_predictions=True)

        name = Path(csv).stem
        out_dir = out_root / name
        out_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(res["predictions"]).rename(columns={"dele": "del"}).to_csv(
            out_dir / "predictions.csv", index=False)
        report = {"csv": str(csv), "checkpoint": str(ckpt), "checkpoint_step": meta.get("step"),
                  "num_utts": len(ds), "loss": res["loss"]}
        if has_ref:
            report["per"] = res["meter"].results()
            logger.info("=== %s | loss %.4f | %.0fs\n%s", name, res["loss"], res["seconds"],
                        res["meter"].table())
            summary[name] = report["per"]["ALL"]["per"]
        else:
            logger.info("=== %s | %d utterances decoded (no reference, PER skipped)", name, len(ds))
        with open(out_dir / "report.json", "w") as f:
            json.dump(report, f, indent=2)
        logger.info("Wrote %s/{predictions.csv,report.json}", out_dir)

    if len(summary) > 1:
        logger.info("Summary (PER %% ALL): %s", {k: round(v, 2) for k, v in summary.items()})


if __name__ == "__main__":
    main()
