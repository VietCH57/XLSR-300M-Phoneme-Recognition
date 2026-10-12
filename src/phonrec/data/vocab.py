"""Phoneme tokenizer built on the ARPAbet (+ accent-marked '*') vocabulary."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

logger = logging.getLogger(__name__)


class PhonemeTokenizer:
    """Maps space-separated phoneme strings to ids and back.

    Tokens ending with '*' (e.g. ``aa*``) are accented / L1-shifted realisations and are
    distinct classes from their plain counterparts. The blank token is used as the CTC blank.
    """

    def __init__(self, vocab: Dict[str, int], blank_token: str = "[PAD]", unk_token: str = "[UNK]"):
        ids = sorted(vocab.values())
        if ids != list(range(len(ids))):
            raise ValueError("Vocabulary ids must be contiguous and start at 0")
        if blank_token not in vocab:
            raise ValueError(f"Blank token {blank_token!r} not in vocabulary")
        if unk_token not in vocab:
            raise ValueError(f"Unknown token {unk_token!r} not in vocabulary")
        self.token_to_id = dict(vocab)
        self.id_to_token = {i: t for t, i in vocab.items()}
        self.blank_token = blank_token
        self.unk_token = unk_token
        self.blank_id = vocab[blank_token]
        self.unk_id = vocab[unk_token]
        self._oov_seen: set[str] = set()

    # ------------------------------------------------------------------ io
    @classmethod
    def from_file(cls, path: str | Path, **kw) -> "PhonemeTokenizer":
        with open(path) as f:
            return cls(json.load(f), **kw)

    def save(self, path: str | Path) -> None:
        with open(path, "w") as f:
            json.dump(self.token_to_id, f, indent=2, ensure_ascii=False)

    # ------------------------------------------------------------- helpers
    def __len__(self) -> int:
        return len(self.token_to_id)

    @property
    def special_ids(self) -> set[int]:
        return {self.blank_id, self.unk_id}

    @staticmethod
    def split(text: str) -> List[str]:
        # str.split() also absorbs the double spaces present in the CSVs
        return text.split()

    def encode(self, text: str | Sequence[str]) -> List[int]:
        tokens = self.split(text) if isinstance(text, str) else list(text)
        out = []
        for t in tokens:
            i = self.token_to_id.get(t)
            if i is None:
                if t not in self._oov_seen:
                    logger.warning("OOV phoneme %r mapped to %s", t, self.unk_token)
                    self._oov_seen.add(t)
                i = self.unk_id
            out.append(i)
        return out

    def decode(self, ids: Iterable[int], skip_blank: bool = True) -> List[str]:
        return [
            self.id_to_token[int(i)]
            for i in ids
            if not (skip_blank and int(i) == self.blank_id)
        ]
