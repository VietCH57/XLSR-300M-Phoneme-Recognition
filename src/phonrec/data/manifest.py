"""Reading and cleaning the CSV manifests (columns: audio, text[, domain])."""

from __future__ import annotations

import logging
from typing import Iterable, List, Optional, Sequence, Tuple

import pandas as pd

logger = logging.getLogger(__name__)

# The combined dataset is the union of these sub-datasets (column `domain`):
#   native_en : TIMIT, CommonVoice EN, FLEURS en_us        (native English)
#   native_vi : CommonVoice VI, FLEURS vi_vn, balanced_vi   (native Vietnamese, ARPAbet-mapped)
#   l2a       : L2-ARCTIC                                   (L2 English)
#   el_easy   : eLearningMDD (easy, chunked)                (Vietnamese learners' English)
DOMAINS: Tuple[str, ...] = ("native_en", "native_vi", "l2a", "el_easy")

# Used only when a CSV has no `domain` column (e.g. the older per-corpus test files).
DOMAIN_RULES: Tuple[Tuple[str, str], ...] = (
    ("l2arctic", "l2a"),
    ("elearningmdd", "el_easy"),
    ("timit", "native_en"),
    ("commonvoice_en", "native_en"),
    ("fleurs/en_us", "native_en"),
    ("commonvoice_vi", "native_vi"),
    ("fleurs/vi_vn", "native_vi"),
    ("balanced_vi", "native_vi"),
)


def infer_domain(path: str) -> str:
    p = path.lower().replace("\\", "/")
    for key, dom in DOMAIN_RULES:
        if key in p:
            return dom
    return "unknown"


def parse_path_map(items: Iterable[str]) -> List[Tuple[str, str]]:
    out = []
    for item in items:
        old, new = item.split("=", 1)
        out.append((old, new))
    return out


def apply_path_map(path: str, mapping: Sequence[Tuple[str, str]]) -> str:
    for old, new in mapping:
        if path.startswith(old):
            return new + path[len(old):]
    return path


def load_manifest(
    csv_path: str,
    path_map: Sequence[str] = (),
    domains: Optional[Sequence[str]] = None,
    require_text: bool = True,
) -> pd.DataFrame:
    """Load a manifest and return a clean DataFrame with columns
    ``audio, text, domain, row`` (``row`` = line index in the original CSV).

    - rows with empty transcripts are dropped (with a warning) when ``require_text``
    - ``domain`` is inferred from the path if the column is missing
    - ``domains`` keeps only the given sub-datasets (used for training only)
    """
    # keep_default_na=False: never turn phoneme strings into NaN; only empty cells are missing
    df = pd.read_csv(csv_path, keep_default_na=False, na_values=[""], dtype=str)
    if "audio" not in df.columns:
        raise ValueError(f"{csv_path}: missing 'audio' column")
    df["row"] = range(len(df))
    has_text = "text" in df.columns
    if not has_text:
        if require_text:
            raise ValueError(f"{csv_path}: missing 'text' column")
        df["text"] = ""
    df["text"] = df["text"].fillna("").map(lambda s: " ".join(str(s).split()))

    mapping = parse_path_map(path_map)
    if mapping:
        df["audio"] = df["audio"].map(lambda p: apply_path_map(p, mapping))

    if "domain" not in df.columns or df["domain"].isna().all():
        df["domain"] = df["audio"].map(infer_domain)
        logger.info("%s: no 'domain' column, inferred from paths", csv_path)
    df["domain"] = df["domain"].fillna("unknown")

    if require_text and has_text:
        empty = df["text"] == ""
        if empty.any():
            logger.warning(
                "%s: dropping %d rows with empty transcripts (by domain: %s)",
                csv_path, int(empty.sum()), df.loc[empty, "domain"].value_counts().to_dict(),
            )
            for p in df.loc[empty, "audio"].head(20):
                logger.warning("    empty transcript: %s", p)
            df = df[~empty]

    if domains:
        before = len(df)
        df = df[df["domain"].isin(domains)]
        logger.info("%s: kept domains %s -> %d / %d rows", csv_path, list(domains), len(df), before)

    return df.reset_index(drop=True)[["audio", "text", "domain", "row"]]
