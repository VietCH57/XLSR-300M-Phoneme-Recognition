import random

import pytest

from phonrec.config import Config, parse_config
from phonrec.metrics import PERMeter, edit_counts
from phonrec.metrics.per import _align_python


def test_edit_counts_basic():
    ref = "aa b k d".split()
    hyp = "aa p k d iy".split()  # 1 sub, 1 ins
    c = edit_counts(ref, hyp)
    assert (c.sub, c.dele, c.ins, c.ref_len) == (1, 0, 1, 4)
    assert c.per == pytest.approx(50.0)
    c = edit_counts(ref, [])
    assert (c.sub, c.dele, c.ins) == (0, 4, 0)
    assert edit_counts(["aa"], ["aa*"]).sub == 1  # accent-marked phones are distinct classes


def test_rapidfuzz_matches_python_dp():
    rng = random.Random(0)
    alphabet = ["aa", "aa*", "b", "k", "iy", "t"]
    for _ in range(300):
        ref = [rng.choice(alphabet) for _ in range(rng.randint(0, 12))]
        hyp = [rng.choice(alphabet) for _ in range(rng.randint(0, 12))]
        c = edit_counts(ref, hyp)
        assert c.errors == sum(_align_python(ref, hyp))


def test_meter_per_domain():
    m = PERMeter()
    m.update(["a", "b"], ["a", "b"], "native_en")
    m.update(["a", "b"], ["a"], "l2a")
    m.update(["a", "b", "c", "d"], ["x", "b", "c", "d"], "l2a")
    r = m.results()
    assert list(r) == ["native_en", "l2a", "ALL"]  # canonical domain order, ALL last
    assert r["l2a"]["per"] == pytest.approx(100 * 2 / 6)
    assert r["ALL"]["per"] == pytest.approx(100 * 2 / 8)
    assert m.flat_metrics()["per/native_en"] == 0.0
    assert "ALL" in m.table()


def test_config_yaml_and_cli(tmp_path):
    y = tmp_path / "c.yaml"
    y.write_text("train:\n  lr: 1.0e-4\n  output_dir: runs/x\nmodel:\n  layerdrop: 0.2\n")
    cfg = parse_config(["--config", str(y), "--lr", "3e-5", "--domains", "l2a", "el_easy",
                        "--gradient_checkpointing", "false", "--resume_from", "none",
                        "--domain_weights", "l2a=2.5"])
    assert cfg.lr == 3e-5 and cfg.output_dir == "runs/x" and cfg.layerdrop == 0.2
    assert cfg.domains == ["l2a", "el_easy"] and cfg.gradient_checkpointing is False
    assert cfg.resume_from is None and cfg.domain_weight_dict() == {"l2a": 2.5}
    cfg.save(tmp_path / "saved.yaml")
    assert Config.from_yaml(tmp_path / "saved.yaml") == cfg
    with pytest.raises(ValueError):
        parse_config(["--domains", "klingon"])
    with pytest.raises(ValueError):
        Config.from_dict({"not_a_key": 1})
