"""Torch dataset + builder that loads a manifest, caches durations and filters bad rows."""

from __future__ import annotations

import logging
from typing import Dict, Optional, Sequence

import numpy as np
import pandas as pd
from torch.utils.data import Dataset

from .audio import DurationCache, load_audio_cached
from .manifest import DOMAINS, load_manifest
from .vocab import PhonemeTokenizer

logger = logging.getLogger(__name__)

# wav2vec2 / XLS-R convolutional feature encoder
CONV_KERNEL = (10, 3, 3, 3, 3, 2, 2)
CONV_STRIDE = (5, 2, 2, 2, 2, 2, 2)


def num_output_frames(num_samples: int, kernels=CONV_KERNEL, strides=CONV_STRIDE) -> int:
    n = num_samples
    for k, s in zip(kernels, strides):
        n = (n - k) // s + 1
    return max(n, 0)


def ctc_min_frames(ids: Sequence[int]) -> int:
    """Minimum number of frames CTC needs: labels + one blank between each repeated pair."""
    repeats = sum(1 for a, b in zip(ids[:-1], ids[1:]) if a == b)
    return len(ids) + repeats


class PhonemeDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        sample_rate: int = 16000,
        cache_dir: Optional[str] = None,
        normalize: bool = True,
    ):
        self.df = df.reset_index(drop=True)
        self.audio = self.df["audio"].tolist()
        self.ids = self.df["ids"].tolist()
        self.durations = self.df["duration"].to_numpy(dtype=np.float64)
        self.domains = self.df["domain"].tolist()
        self.sample_rate = sample_rate
        self.cache_dir = cache_dir
        self.normalize = normalize

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, i: int) -> Dict:
        wav = load_audio_cached(self.audio[i], self.sample_rate, self.cache_dir)
        if self.normalize:  # XLS-R expects zero-mean / unit-variance input
            wav = (wav - wav.mean()) / np.sqrt(wav.var() + 1e-7)
        return {"input_values": wav, "labels": self.ids[i], "index": i}

    def summary(self) -> pd.DataFrame:
        g = self.df.groupby("domain")
        s = pd.DataFrame(
            {
                "utts": g.size(),
                "hours": g["duration"].sum() / 3600,
                "max_sec": g["duration"].max(),
                "phones": g["ids"].apply(lambda x: sum(map(len, x))),
            }
        )
        order = [d for d in DOMAINS if d in s.index] + sorted(d for d in s.index if d not in DOMAINS)
        s = s.reindex(order)
        s.loc["ALL"] = [len(self.df), self.df["duration"].sum() / 3600,
                        self.df["duration"].max(), sum(map(len, self.ids))]
        return s.astype({"utts": int, "phones": int})


def build_dataset(
    csv_path: str,
    tokenizer: PhonemeTokenizer,
    *,
    split: str,
    sample_rate: int = 16000,
    cache_dir: str = "cache",
    path_map: Sequence[str] = (),
    domains: Optional[Sequence[str]] = None,
    min_audio_seconds: float = 0.0,
    max_audio_seconds: Optional[float] = None,
    require_text: bool = True,
    num_workers: int = 8,
    unk_policy: str = "keep",
) -> PhonemeDataset:
    """Build a dataset.

    ``split='train'`` drops utterances outside [min, max] seconds and those too short for
    their CTC target. Eval splits keep everything that can be read.
    """
    df = load_manifest(csv_path, path_map=path_map, domains=domains, require_text=require_text)
    df["ids"] = df["text"].map(tokenizer.encode)

    # Some L2-ARCTIC transcripts contain a literal [UNK] (unintelligible segment).
    has_unk = df["ids"].map(lambda x: tokenizer.unk_id in x)
    if has_unk.any():
        logger.info("%s: %d utterances contain %s (policy for train: %s)", csv_path,
                    int(has_unk.sum()), tokenizer.unk_token, unk_policy if split == "train" else "keep")
        if split == "train" and unk_policy == "drop_token":
            df["ids"] = df["ids"].map(lambda x: [i for i in x if i != tokenizer.unk_id])
        elif split == "train" and unk_policy == "drop_utt":
            df = df[~has_unk].copy()

    durations = DurationCache(cache_dir).get(df["audio"].tolist(), num_workers=num_workers)
    df["duration"] = durations
    unreadable = df["duration"].isna()
    if unreadable.any():
        logger.warning("%s: %d unreadable/missing audio files dropped", csv_path, int(unreadable.sum()))
        for p in df.loc[unreadable, "audio"].head(20):
            logger.warning("    missing: %s", p)
        df = df[~unreadable].copy()

    if split == "train":
        n0 = len(df)
        keep = df["duration"] >= min_audio_seconds
        if max_audio_seconds:
            keep &= df["duration"] <= max_audio_seconds
        if (~keep).any():
            logger.info(
                "%s: dropped %d / %d utterances outside [%.1f, %s] s (by domain: %s)",
                csv_path, int((~keep).sum()), n0, min_audio_seconds, max_audio_seconds,
                df.loc[~keep, "domain"].value_counts().to_dict(),
            )
        df = df[keep].copy()

        frames = (df["duration"] * sample_rate).astype(int).map(num_output_frames)
        feasible = frames >= df["ids"].map(ctc_min_frames)
        if (~feasible).any():
            logger.warning("%s: dropped %d utterances too short for their CTC target",
                           csv_path, int((~feasible).sum()))
        df = df[feasible].copy()

    ds = PhonemeDataset(df, sample_rate=sample_rate, cache_dir=cache_dir)
    logger.info("%s split from %s:\n%s", split, csv_path, ds.summary().round(2).to_string())
    return ds
