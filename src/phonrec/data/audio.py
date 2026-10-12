"""Audio I/O: wav / flac / mp3 / NIST SPHERE (TIMIT) -> mono float32 @ target sample rate.

Also provides a duration cache (header reads only) and the location of the optional
pre-resampled 16 kHz copies written by ``scripts/prepare_audio.py``.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------ loading
def _read_soundfile(path: str):
    import soundfile as sf

    wav, sr = sf.read(path, dtype="float32", always_2d=True)
    return wav.mean(axis=1), sr


def _read_torchaudio(path: str):
    import torchaudio

    wav, sr = torchaudio.load(path)
    return wav.mean(dim=0).numpy().astype(np.float32), sr


def _read_librosa(path: str):
    import librosa

    wav, sr = librosa.load(path, sr=None, mono=True)
    return wav.astype(np.float32), sr


_READERS = (_read_soundfile, _read_torchaudio, _read_librosa)


def resample(wav: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    if orig_sr == target_sr:
        return wav
    try:
        import torch
        import torchaudio.functional as AF

        return AF.resample(torch.from_numpy(wav), orig_sr, target_sr).numpy()
    except ImportError:
        from math import gcd

        from scipy.signal import resample_poly

        g = gcd(orig_sr, target_sr)
        return resample_poly(wav, target_sr // g, orig_sr // g).astype(np.float32)


def load_audio(path: str, target_sr: int = 16000) -> np.ndarray:
    """Return mono float32 waveform at ``target_sr``. Tries several backends."""
    errors = []
    for reader in _READERS:
        try:
            wav, sr = reader(path)
            break
        except Exception as e:  # noqa: BLE001 - fall through to the next backend
            errors.append(f"{reader.__name__}: {e}")
    else:
        raise RuntimeError(f"Could not read {path}:\n  " + "\n  ".join(errors))
    return np.ascontiguousarray(resample(wav, sr, target_sr), dtype=np.float32)


# ------------------------------------------------------------- audio cache
def cached_audio_path(cache_dir: str | Path, path: str) -> Path:
    h = hashlib.sha1(path.encode("utf-8")).hexdigest()
    return Path(cache_dir) / "audio16k" / h[:2] / f"{h}.flac"


def load_audio_cached(path: str, target_sr: int, cache_dir: Optional[str]) -> np.ndarray:
    if cache_dir:
        cp = cached_audio_path(cache_dir, path)
        if cp.exists():
            return load_audio(str(cp), target_sr)
    return load_audio(path, target_sr)


# --------------------------------------------------------------- durations
def audio_duration(path: str) -> Optional[float]:
    """Duration in seconds from the file header (falls back to decoding)."""
    try:
        import soundfile as sf

        info = sf.info(path)
        if info.frames > 0:
            return info.frames / info.samplerate
    except Exception:  # noqa: BLE001
        pass
    try:
        import torchaudio

        info = torchaudio.info(path)
        if info.num_frames > 0:
            return info.num_frames / info.sample_rate
    except Exception:  # noqa: BLE001
        pass
    try:
        return len(load_audio(path, 16000)) / 16000
    except Exception:  # noqa: BLE001
        return None


class DurationCache:
    """JSON cache ``{audio_path: seconds}`` shared by all runs in ``cache_dir``."""

    def __init__(self, cache_dir: str | Path):
        self.path = Path(cache_dir) / "durations.json"
        self.data: Dict[str, float] = {}
        if self.path.exists():
            with open(self.path) as f:
                self.data = json.load(f)

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        with open(tmp, "w") as f:
            json.dump(self.data, f)
        os.replace(tmp, self.path)

    def get(self, paths: Sequence[str], num_workers: int = 8) -> List[Optional[float]]:
        missing = sorted({p for p in paths if p not in self.data})
        if missing:
            logger.info("Reading durations of %d files (cached afterwards in %s)", len(missing), self.path)
            with ThreadPoolExecutor(max_workers=max(1, num_workers)) as ex:
                for p, d in zip(missing, ex.map(audio_duration, missing, chunksize=64)):
                    if d is not None:
                        self.data[p] = d
            self.save()
        return [self.data.get(p) for p in paths]
