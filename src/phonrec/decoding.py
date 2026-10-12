"""CTC decoding."""

from __future__ import annotations

from typing import List

import torch


@torch.no_grad()
def ctc_greedy_decode(log_probs: torch.Tensor, lengths: torch.Tensor, blank_id: int) -> List[List[int]]:
    """Best-path decoding: argmax per frame, merge repeats, drop blanks.

    log_probs: (B, T, V); lengths: (B,) valid frames per utterance.
    """
    best = log_probs.argmax(dim=-1).cpu()
    lengths = lengths.cpu().tolist()
    out: List[List[int]] = []
    for b, n in enumerate(lengths):
        seq = best[b, :n]
        if n == 0:
            out.append([])
            continue
        keep = torch.ones_like(seq, dtype=torch.bool)
        keep[1:] = seq[1:] != seq[:-1]
        seq = seq[keep]
        out.append(seq[seq != blank_id].tolist())
    return out
