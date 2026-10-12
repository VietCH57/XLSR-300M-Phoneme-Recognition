from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import soundfile as sf

from phonrec.data import (
    CTCCollator,
    DynamicBatchSampler,
    PhonemeTokenizer,
    build_dataset,
    infer_domain,
    load_manifest,
    make_sample_weights,
    num_output_frames,
)

ROOT = Path(__file__).resolve().parents[1]
VOCAB = ROOT / "configs" / "vocab.json"


@pytest.fixture
def tok():
    return PhonemeTokenizer.from_file(VOCAB)


def test_tokenizer_roundtrip(tok):
    assert len(tok) == 79 and tok.blank_id == 78 and tok.unk_id == 77
    ids = tok.encode("aa  aa* b   err")  # double spaces are ignored
    assert tok.decode(ids) == ["aa", "aa*", "b", "err"]
    assert tok.encode("xx") == [tok.unk_id]
    assert tok.decode([tok.blank_id, 0]) == ["aa"]


def test_infer_domain():
    assert infer_domain("/d/l2arctic_release_v5.0/THV/wav/a.wav") == "l2a"
    assert infer_domain("/d/eLearningMDD/x.wav") == "el_easy"
    assert infer_domain("/d/timit/data/TEST/DR1/X/SA1.WAV") == "native_en"
    assert infer_domain("/d/fleurs/vi_vn/1.wav") == "native_vi"
    assert infer_domain("/d/other/1.wav") == "unknown"


def test_manifest(tmp_path):
    csv = tmp_path / "m.csv"
    pd.DataFrame({
        "audio": ["/old/a.wav", "/old/b.wav", "/old/c.wav", "/x/d.wav"],
        "text": ["aa b", "", "n a", "iy  t"],  # "n a" must not become NaN
        "domain": ["l2a", "el_easy", "native_vi", "native_en"],
    }).to_csv(csv, index=False)
    df = load_manifest(str(csv), path_map=["/old=/new"])
    assert len(df) == 3 and df["row"].tolist() == [0, 2, 3]
    assert df["audio"].tolist() == ["/new/a.wav", "/new/c.wav", "/x/d.wav"]
    assert df["text"].tolist() == ["aa b", "n a", "iy t"]
    df = load_manifest(str(csv), domains=["l2a", "native_en"])
    assert sorted(df["domain"]) == ["l2a", "native_en"]

    pd.DataFrame({"audio": ["/a/l2arctic/x.wav"], "text": ["aa"]}).to_csv(csv, index=False)
    assert load_manifest(str(csv))["domain"].tolist() == ["l2a"]


def test_sampler_limits_and_coverage():
    rng = np.random.default_rng(0)
    dur = rng.uniform(1, 15, size=500)
    s = DynamicBatchSampler(dur, max_batch_seconds=60, max_batch_size=8, shuffle=True, seed=1)
    seen = []
    for b in s:
        assert len(b) <= 8
        assert len(b) == 1 or max(dur[b]) * len(b) <= 60 + 1e-9
        seen += b
    assert sorted(seen) == list(range(500))
    first = list(s)
    s.set_epoch(1)
    assert list(s) != first  # reshuffled
    ev = DynamicBatchSampler(dur, 60, 8, shuffle=False)
    assert sorted(i for b in ev for i in b) == list(range(500))


def test_sampler_weights():
    dur = np.ones(100)
    domains = ["a"] * 50 + ["b"] * 50
    w = make_sample_weights(domains, {"b": 3})
    s = DynamicBatchSampler(dur, 10, 10, shuffle=True, weights=w)
    idx = [i for b in s for i in b]
    assert sum(i >= 50 for i in idx) == 150 and sum(i < 50 for i in idx) == 50
    assert make_sample_weights(domains, {}) is None


def test_collator():
    items = [
        {"input_values": np.ones(100, np.float32), "labels": [1, 2, 3], "index": 0},
        {"input_values": np.ones(50, np.float32), "labels": [4], "index": 1},
    ]
    b = CTCCollator()(items)
    assert b["input_values"].shape == (2, 100)
    assert b["attention_mask"].sum(1).tolist() == [100, 50]
    assert b["labels"].tolist() == [[1, 2, 3], [4, -100, -100]]
    assert b["label_lengths"].tolist() == [3, 1]


def test_num_output_frames():
    assert num_output_frames(16000) == 49  # wav2vec2: ~50 frames per second


def test_build_dataset_filters(tmp_path, tok):
    sr = 16000
    files = []
    for name, sec in [("a", 1.0), ("b", 5.0), ("c", 0.1)]:
        p = tmp_path / f"{name}.wav"
        sf.write(p, np.random.randn(int(sec * sr)).astype(np.float32) * 0.1, sr)
        files.append(str(p))
    csv = tmp_path / "m.csv"
    long_text = " ".join(["aa"] * 30)  # 30 repeats need 59 frames > 49 frames of 1 s
    pd.DataFrame({"audio": files + [str(tmp_path / "missing.wav")],
                  "text": [long_text, "b iy", "t", "aa"],
                  "domain": ["l2a"] * 4}).to_csv(csv, index=False)
    ds = build_dataset(str(csv), tok, split="train", cache_dir=str(tmp_path / "cache"),
                       min_audio_seconds=0.3, max_audio_seconds=30)
    assert ds.audio == [files[1]]  # a: CTC-infeasible, c: too short, missing: unreadable
    item = ds[0]
    assert item["input_values"].shape == (5 * sr,)
    assert abs(float(item["input_values"].mean())) < 1e-3
    dev = build_dataset(str(csv), tok, split="dev", cache_dir=str(tmp_path / "cache"))
    assert len(dev) == 3  # eval keeps everything readable
