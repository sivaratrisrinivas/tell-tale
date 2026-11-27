import json

import numpy as np
import pandas as pd
import pytest

from telltale.config import heldout_families, load_models, training_families
from telltale.mirrors.client import RouterClient
from telltale.mirrors.filters import check_mirror
from telltale.mirrors.generate import assign_families
from telltale.mirrors.ledger import BudgetExceeded, Ledger, ResponseCache, Usage
from telltale.mirrors.prompts import mirror_prompt
from telltale.text import compose, ngram_overlap, normalize_body, normalize_title, split_generated, uses_template
from telltale.train.calibrate import fit_temperature, p_machine, threshold_for_fpr
from tests.helpers import FakeBackend


def test_normalizer_same_for_both_classes():
    human = "Title: Fix crash\r\n"
    assert normalize_title(human) == "Fix crash"
    assert normalize_title("**Title:** Fix crash") == "Fix crash"
    assert normalize_title("# Fix crash") == "Fix crash"
    body = "Line one   \r\n\r\n\r\n\r\n<!-- please fill the template -->Line two\r\n"
    assert normalize_body(body) == "Line one\n\nLine two"


def test_split_generated_handles_labels_and_fences():
    t, b = split_generated("```\nTitle: Add thing\nBody:\nIt adds the thing.\n```")
    assert (t, b) == ("Add thing", "It adds the thing.")
    t, b = split_generated("\n\nFix bug\n\nDetails here")
    assert (t, b) == ("Fix bug", "Details here")


def test_template_detection():
    assert uses_template("### What version?\n1.2\n### What did you do?\nstuff")
    assert not uses_template("just a plain issue body")


def test_mirror_filters():
    src = " ".join(f"w{i}" for i in range(100))
    assert check_mirror("", "body", src) == "empty"
    assert check_mirror("T", "I'm sorry, I can't help with that " + "x " * 60, src) == "refusal"
    assert check_mirror("T", "short body", src) == "length"
    assert check_mirror("T", src, src) == "overlap"
    assert check_mirror("T", " ".join(f"z{i}" for i in range(90)), src) is None
    assert ngram_overlap(src, src) == 1.0


def test_prompts_contain_template_hint():
    p = mirror_prompt(persona="A", kind="issue", repo="golang/go", topic="T", n_words=80, headings=["What version?"])
    assert "Use this repository's template with these sections: What version?" in p
    p = mirror_prompt(persona="B", kind="pr", repo="golang/go", topic="T", n_words=80, headings=[])
    assert "made this change" in p and "template" not in p


def test_ledger_cap_and_persistence(tmp_path):
    led = Ledger(tmp_path / "l.jsonl", cap_usd=0.001)
    c = led.charge(model="m", provider="p", usage=Usage(1000, 1000), price_in=0.5, price_out=0.5, purpose="t")
    assert c == pytest.approx(0.001)
    with pytest.raises(BudgetExceeded):
        led.check()
    led2 = Ledger(tmp_path / "l.jsonl", cap_usd=1.0)   # reloads spend from disk
    assert led2.spent == pytest.approx(0.001) and led2.calls == 1


def test_client_caches_and_backs_off(tmp_path, monkeypatch):
    monkeypatch.setattr("time.sleep", lambda s: None)
    models = load_models()
    be = FakeBackend(fail_first=2)
    cl = RouterClient(be, Ledger(tmp_path / "l.jsonl", 5.0), ResponseCache(tmp_path / "c"))
    r1 = cl.chat(models["topic"], "Reply with ok", max_tokens=10, purpose="t")
    r2 = cl.chat(models["topic"], "Reply with ok", max_tokens=10, purpose="t")
    assert not r1.cached and r2.cached and be.calls == 3      # 2 retries + 1 success, then cache hit
    assert cl.ledger.calls == 1 and r1.cost_usd > 0
    assert be.models[0] == "openai/gpt-oss-20b:coreweave"


def test_client_refuses_over_budget(tmp_path):
    models = load_models()
    cl = RouterClient(FakeBackend(), Ledger(tmp_path / "l.jsonl", 0.0), ResponseCache(tmp_path / "c"))
    with pytest.raises(BudgetExceeded):
        cl.chat(models["kimi"], "hello", max_tokens=900)


def test_roster_licenses_and_heldout():
    m = load_models()
    fams = training_families(m)
    assert "mistral" not in fams and heldout_families(m) == ["mistral"]
    assert "llama" not in fams           # disabled: naming clause in Llama license
    assert len(fams) == 10
    assert all(m[f].license_ok for f in fams)


def test_assign_families_respects_quota():
    ids = [str(i) for i in range(1000)]
    a = assign_families(ids, ["a", "b", "kimi"], {"kimi": 30}, seed=1)
    counts = pd.Series(list(a.values())).value_counts()
    assert counts["kimi"] == 30 and len(a) == 1000
    assert a == assign_families(ids, ["a", "b", "kimi"], {"kimi": 30}, seed=1)  # deterministic


def test_calibration_and_threshold():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 2000)
    logits = np.stack([np.zeros(2000), 10.0 * ((y * 2 - 1) * 1.0 + rng.normal(0, 1, 2000))], 1)
    T = fit_temperature(logits, y)
    assert T > 1.0                                   # over-confident logits get softened
    p = p_machine(logits, T)
    thr = threshold_for_fpr(p[y == 0], 0.01)
    assert (p[y == 0] >= thr).mean() <= 0.01


def test_compose():
    assert compose("Title: X", "a\r\nb") == "X\n\na\nb"


def test_v0_plan_uses_configured_mistral_count():
    from telltale.config import load_yaml
    from telltale.mirrors.generate import plan_v0

    cfg = load_yaml("data.yaml")
    assert cfg["sizes"]["mistral_mirrors"] == 300
    human = pd.DataFrame({"id": [f"r#{i}" for i in range(2000)],
                          "split": ["test"] * 1000 + ["train"] * 1000})
    plan = plan_v0(human, load_models(thinking=None), cfg)
    assert len(plan["heldout"]) == 300 and set(plan["heldout"].values()) == {"mistral"}
    assert len(plan["seen"]) == 2000


def test_fit_batch_size_halves_until_it_fits(monkeypatch):
    import telltale.train.train as T

    fits = {16: False, 8: False, 4: True}
    monkeypatch.setattr(T, "_step_fits", lambda m, bs, *a: fits[bs])
    monkeypatch.setattr(T.torch.cuda, "max_memory_allocated", lambda: 0)
    assert T.fit_batch_size(None, 16, 2, 1024, 100, "cuda", True) == (4, 8)   # effective batch stays 32
    assert T.fit_batch_size(None, 16, 2, 1024, 100, "cpu", False) == (16, 2)


def test_length_grouped_batches_cover_everything_once():
    from telltale.train.train import LengthGroupedBatches

    lengths = [int(x) for x in np.random.default_rng(0).integers(10, 5000, 1003)]
    s = LengthGroupedBatches(lengths, 8, seed=1)
    batches = list(s)
    flat = sorted(i for b in batches for i in b)
    assert flat == list(range(1003)) and len(batches) == len(s)
    spread = np.mean([max(lengths[i] for i in b) - min(lengths[i] for i in b) for b in batches])
    assert spread < 500          # similar lengths grouped together -> little padding
    assert list(s) != batches    # reshuffled each epoch


def test_polished_train_rows_only_from_train_split():
    from telltale.train.dataset import build_table, guard_no_test_leak

    human = pd.DataFrame({"id": ["a", "b", "c"], "split": ["train", "polish", "val"], "title": ["T"] * 3,
                          "body": ["x y z"] * 3, "repo": ["r"] * 3, "kind": ["pr"] * 3})
    pol = pd.DataFrame({"source_id": ["a", "b", "c"], "title": ["T"] * 3, "body": ["x y z."] * 3,
                        "repo": ["r"] * 3, "kind": ["pr"] * 3, "family": ["qwen"] * 3})
    t = build_table(human, pol.iloc[:0].assign(reject=None), ["qwen"], polished_train=pol)
    p = t[t["origin"] == "polished_human"]
    assert list(p["source_id"]) == ["a"] and (p["label"] == 0).all() and (p["split"] == "train").all()
    guard_no_test_leak(t)


def test_operating_points_stricter_threshold_lowers_fpr():
    from telltale.eval.report import operating_points

    rng = np.random.default_rng(0)
    def df(n_h, n_m):
        return pd.DataFrame({"label": [0] * n_h + [1] * n_m,
                             "p": np.r_[rng.beta(1, 20, n_h), rng.beta(20, 1, n_m)]})
    op = operating_points(df(1000, 1000), df(1000, 1000), None, df(500, 0))
    assert list(op["val target FPR"]) == [0.01, 0.0075, 0.005]
    assert op["threshold"].is_monotonic_increasing and op["T2 TPR"].is_monotonic_decreasing
