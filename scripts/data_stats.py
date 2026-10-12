#!/usr/bin/env python
"""Dataset statistics per domain: utterances, hours, duration / label-length percentiles,
accent-marked phoneme share, missing files, vocabulary coverage. Useful for choosing
``--max_audio_seconds`` and ``--max_batch_seconds``.

    python scripts/data_stats.py --csv \
        /home/mdd/lamnh/MDD-2026/data/stage2_combined/server/fullnative_el2a_{train,dev,test}_ctc.csv
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from phonrec.data.audio import DurationCache  # noqa: E402
from phonrec.data.manifest import DOMAINS, load_manifest  # noqa: E402
from phonrec.utils import setup_logging  # noqa: E402


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--csv", nargs="+", required=True)
    p.add_argument("--vocab", default="configs/vocab.json")
    p.add_argument("--cache_dir", default="cache")
    p.add_argument("--path_map", nargs="*", default=[])
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--no_audio", action="store_true", help="skip reading audio headers")
    args = p.parse_args(argv)
    setup_logging()
    pd.set_option("display.width", 200)

    vocab = json.load(open(args.vocab))
    counters = {}
    for csv in args.csv:
        df = load_manifest(csv, path_map=args.path_map)
        df["phones"] = df["text"].str.split().map(len)
        df["accented"] = df["text"].str.count(r"\*")
        print(f"\n===== {csv}")
        if not args.no_audio:
            df["duration"] = DurationCache(args.cache_dir).get(df["audio"].tolist(), args.num_workers)
            missing = df["duration"].isna()
            print(f"missing/unreadable audio: {int(missing.sum())}")
            for pth in df.loc[missing, "audio"].head(10):
                print("   ", pth)
        g = df.groupby("domain")
        stats = pd.DataFrame({"utts": g.size(), "phones_mean": g["phones"].mean(),
                              "phones_max": g["phones"].max(),
                              "accented_%": 100 * g["accented"].sum() / g["phones"].sum()})
        if "duration" in df:
            stats["hours"] = g["duration"].sum() / 3600
            for q in (50, 95, 99, 100):
                stats[f"dur_p{q}"] = g["duration"].apply(lambda x, q=q: np.nanpercentile(x, q))
            for thr in (20, 30, 40):
                stats[f">{thr}s"] = g["duration"].apply(lambda x, t=thr: int((x > t).sum()))
        order = [d for d in DOMAINS if d in stats.index] + [d for d in stats.index if d not in DOMAINS]
        print(stats.reindex(order).round(2).to_string())
        c = collections.Counter(t for s in df["text"] for t in s.split())
        counters[Path(csv).stem] = c
        oov = sorted(t for t in c if t not in vocab)
        if oov:
            print("OOV tokens:", oov)

    print("\n===== Vocabulary coverage (count per file)")
    rows = {tok: {k: c.get(tok, 0) for k, c in counters.items()} for tok in vocab}
    cov = pd.DataFrame(rows).T
    print("tokens never seen:", [t for t in cov.index if cov.loc[t].sum() == 0 and not t.startswith("[")])
    print("rare tokens (<20 in first file):")
    first = cov.columns[0]
    print(cov[(cov[first] < 20)].to_string())


if __name__ == "__main__":
    main()
