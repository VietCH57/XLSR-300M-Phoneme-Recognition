"""Phone Error Rate (PER) with substitution / deletion / insertion breakdown, per domain.

    PER = (S + D + I) / N,   N = number of reference phonemes (micro average over utterances)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

from phonrec.data.manifest import DOMAINS

try:
    from rapidfuzz.distance import Levenshtein as _RF
except ImportError:  # pragma: no cover
    _RF = None


@dataclass
class EditCounts:
    sub: int = 0
    dele: int = 0
    ins: int = 0
    ref_len: int = 0
    utts: int = 0

    @property
    def errors(self) -> int:
        return self.sub + self.dele + self.ins

    @property
    def per(self) -> float:
        return 100.0 * self.errors / self.ref_len if self.ref_len else 0.0

    def __iadd__(self, o: "EditCounts") -> "EditCounts":
        self.sub += o.sub
        self.dele += o.dele
        self.ins += o.ins
        self.ref_len += o.ref_len
        self.utts += o.utts
        return self

    def to_dict(self) -> Dict[str, float]:
        return {"utts": self.utts, "ref_phones": self.ref_len, "per": self.per,
                "sub": self.sub, "del": self.dele, "ins": self.ins}


def _align_python(ref: Sequence[str], hyp: Sequence[str]) -> Tuple[int, int, int]:
    """Levenshtein DP keeping (cost, S, D, I); ties prefer substitutions."""
    prev = [(j, 0, 0, j) for j in range(len(hyp) + 1)]
    for i in range(1, len(ref) + 1):
        cur = [(i, 0, i, 0)]
        for j in range(1, len(hyp) + 1):
            if ref[i - 1] == hyp[j - 1]:
                best = prev[j - 1]
            else:
                c, s, d, n = prev[j - 1]
                best = (c + 1, s + 1, d, n)
            c, s, d, n = prev[j]
            if c + 1 < best[0]:
                best = (c + 1, s, d + 1, n)
            c, s, d, n = cur[j - 1]
            if c + 1 < best[0]:
                best = (c + 1, s, d, n + 1)
            cur.append(best)
        prev = cur
    _, s, d, n = prev[-1]
    return s, d, n


def edit_counts(ref: Sequence[str], hyp: Sequence[str]) -> EditCounts:
    if _RF is not None:
        s = d = n = 0
        for op in _RF.editops(list(ref), list(hyp)):
            tag = op.tag
            if tag == "replace":
                s += 1
            elif tag == "delete":
                d += 1
            else:
                n += 1
    else:
        s, d, n = _align_python(ref, hyp)
    return EditCounts(sub=s, dele=d, ins=n, ref_len=len(ref), utts=1)


def utterance_per(ref: Sequence[str], hyp: Sequence[str]) -> float:
    return edit_counts(ref, hyp).per


class PERMeter:
    """Accumulates edit counts per domain and overall."""

    def __init__(self):
        self.by_domain: Dict[str, EditCounts] = {}

    def update(self, ref: Sequence[str], hyp: Sequence[str], domain: str = "unknown") -> EditCounts:
        c = edit_counts(ref, hyp)
        self.by_domain.setdefault(domain, EditCounts())
        self.by_domain[domain] += c
        return c

    def total(self) -> EditCounts:
        t = EditCounts()
        for c in self.by_domain.values():
            t += c
        return t

    def ordered_domains(self) -> List[str]:
        known = [d for d in DOMAINS if d in self.by_domain]
        return known + sorted(d for d in self.by_domain if d not in DOMAINS)

    def results(self) -> Dict[str, Dict[str, float]]:
        out = {d: self.by_domain[d].to_dict() for d in self.ordered_domains()}
        out["ALL"] = self.total().to_dict()
        return out

    def flat_metrics(self) -> Dict[str, float]:
        """{'per/native_en': .., ..., 'per/ALL': ..}"""
        return {f"per/{k}": v["per"] for k, v in self.results().items()}

    def table(self) -> str:
        rows = self.results()
        header = f"{'domain':<12}{'#utt':>7}{'#phones':>10}{'PER(%)':>9}{'S':>8}{'D':>8}{'I':>8}"
        lines = [header, "-" * len(header)]
        for name, r in rows.items():
            if name == "ALL":
                lines.append("-" * len(header))
            lines.append(f"{name:<12}{r['utts']:>7}{r['ref_phones']:>10}{r['per']:>9.2f}"
                         f"{r['sub']:>8}{r['del']:>8}{r['ins']:>8}")
        return "\n".join(lines)


def corpus_per(refs: Iterable[Sequence[str]], hyps: Iterable[Sequence[str]]) -> float:
    m = PERMeter()
    for r, h in zip(refs, hyps):
        m.update(r, h)
    return m.total().per
