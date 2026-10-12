#!/usr/bin/env python
"""(Optional) decode every audio file once and store a 16 kHz mono FLAC copy in
``<cache_dir>/audio16k``. Training/inference automatically use the cached copy when present,
which removes mp3 decoding + resampling from the data-loading hot path (8 vCPU server).
Also fills ``<cache_dir>/durations.json``.

    python scripts/prepare_audio.py --csv \
        /home/mdd/lamnh/MDD-2026/data/stage2_combined/server/fullnative_el2a_{train,dev,test}_ctc.csv
"""

from __future__ import annotations

import argparse
import logging
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from phonrec.data.audio import DurationCache, cached_audio_path, load_audio  # noqa: E402
from phonrec.data.manifest import load_manifest  # noqa: E402
from phonrec.utils import setup_logging  # noqa: E402

logger = logging.getLogger("prepare_audio")


def _convert(job):
    src, dst, sr = job
    import soundfile as sf

    try:
        if not Path(dst).exists():
            wav = load_audio(src, sr)
            Path(dst).parent.mkdir(parents=True, exist_ok=True)
            tmp = dst + ".tmp.flac"
            sf.write(tmp, wav, sr, format="FLAC", subtype="PCM_16")
            Path(tmp).rename(dst)
            return src, len(wav) / sr, None
        return src, sf.info(dst).frames / sr, None
    except Exception as e:  # noqa: BLE001
        return src, None, str(e)


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--csv", nargs="+", required=True)
    p.add_argument("--cache_dir", default="cache")
    p.add_argument("--path_map", nargs="*", default=[])
    p.add_argument("--sample_rate", type=int, default=16000)
    p.add_argument("--num_workers", type=int, default=8)
    args = p.parse_args(argv)
    setup_logging(Path(args.cache_dir) / "prepare_audio.log")

    paths = set()
    for csv in args.csv:
        df = load_manifest(csv, path_map=args.path_map, require_text=False)
        paths.update(df["audio"])
    paths = sorted(paths)
    logger.info("%d unique audio files", len(paths))

    cache = DurationCache(args.cache_dir)
    jobs = [(pth, str(cached_audio_path(args.cache_dir, pth)), args.sample_rate) for pth in paths]
    failed = 0
    with ProcessPoolExecutor(max_workers=args.num_workers) as ex:
        for i, (src, dur, err) in enumerate(ex.map(_convert, jobs, chunksize=32), 1):
            if err is not None:
                failed += 1
                logger.warning("FAILED %s: %s", src, err)
            else:
                cache.data[src] = dur
            if i % 2000 == 0:
                logger.info("%d / %d done", i, len(jobs))
                cache.save()
    cache.save()
    logger.info("Finished: %d converted/cached, %d failed. Cache: %s", len(jobs) - failed, failed,
                Path(args.cache_dir).resolve())


if __name__ == "__main__":
    main()
