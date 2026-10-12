"""Duration-based dynamic batching with optional per-domain oversampling."""

from __future__ import annotations

from typing import Dict, Iterator, List, Optional, Sequence

import numpy as np
from torch.utils.data import Sampler


class DynamicBatchSampler(Sampler[List[int]]):
    """Groups utterances of similar length so that ``max_dur_in_batch * batch_size``
    (the padded audio seconds) stays below ``max_batch_seconds``.

    Training (``shuffle=True``): every epoch the (optionally oversampled) indices are
    shuffled, split into buckets of ``max_batch_size * bucket_size_multiplier``, sorted by
    duration inside each bucket, batched, and the batches are shuffled.

    Evaluation (``shuffle=False``): deterministic, longest-first batches.

    ``weights``: per-utterance sampling weight; weight w means the utterance appears
    floor(w) times per epoch plus once more with probability frac(w).
    """

    def __init__(
        self,
        durations: Sequence[float],
        max_batch_seconds: float,
        max_batch_size: int = 64,
        shuffle: bool = True,
        seed: int = 0,
        weights: Optional[Sequence[float]] = None,
        bucket_size_multiplier: int = 100,
    ):
        self.durations = np.asarray(durations, dtype=np.float64)
        self.max_batch_seconds = float(max_batch_seconds)
        self.max_batch_size = int(max_batch_size)
        self.shuffle = shuffle
        self.seed = seed
        self.weights = None if weights is None else np.asarray(weights, dtype=np.float64)
        self.bucket = max(1, self.max_batch_size * bucket_size_multiplier)
        self.epoch = 0
        self._batches = self._make_batches()

    def set_epoch(self, epoch: int) -> None:
        if epoch != self.epoch:
            self.epoch = epoch
            self._batches = self._make_batches()

    def _make_batches(self) -> List[List[int]]:
        rng = np.random.default_rng(self.seed + self.epoch)
        idx = np.arange(len(self.durations))
        if self.weights is not None and self.shuffle:
            reps = np.floor(self.weights).astype(int)
            reps += rng.random(len(idx)) < (self.weights - np.floor(self.weights))
            idx = np.repeat(idx, reps)

        if self.shuffle:
            rng.shuffle(idx)
            chunks = [idx[i:i + self.bucket] for i in range(0, len(idx), self.bucket)]
        else:
            chunks = [idx]

        batches: List[List[int]] = []
        for chunk in chunks:
            order = chunk[np.argsort(-self.durations[chunk], kind="stable")]
            cur: List[int] = []
            cur_max = 0.0
            for i in order.tolist():
                d = self.durations[i]
                new_max = max(cur_max, d)
                if cur and (new_max * (len(cur) + 1) > self.max_batch_seconds
                            or len(cur) >= self.max_batch_size):
                    batches.append(cur)
                    cur, new_max = [], d
                cur.append(i)
                cur_max = new_max
            if cur:
                batches.append(cur)

        if self.shuffle:
            perm = rng.permutation(len(batches))
            batches = [batches[i] for i in perm]
        return batches

    def __iter__(self) -> Iterator[List[int]]:
        return iter(self._batches)

    def __len__(self) -> int:
        return len(self._batches)


def make_sample_weights(domains: Sequence[str], domain_weights: Dict[str, float]) -> Optional[np.ndarray]:
    if not domain_weights:
        return None
    return np.array([domain_weights.get(d, 1.0) for d in domains], dtype=np.float64)
