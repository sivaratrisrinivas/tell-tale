"""End-to-end smoke test on CPU: extract -> mirrors (fake LLM) -> train tiny model -> eval
-> hard-negative round -> ONNX int8 -> Action (mocked GitHub). No network, no spend."""
import json
from pathlib import Path

import pandas as pd
import pytest

from telltale.config import load_models, load_yaml, training_families
from telltale.data.extract import bot_regex, clean_items
from telltale.data.dedupe import dedupe
from telltale.data.realworld import find_candidates
from telltale.data.splits import counts_report, make_splits
from telltale.mirrors.client import RouterClient
from telltale.mirrors.generate import MirrorGenerator, estimate_cost, pilot_sample, plan_v0, review_pairs_md
from telltale.mirrors.ledger import Ledger, ResponseCache
from telltale.text import compose
from telltale.train.dataset import build_table, guard_no_test_leak
from tests.helpers import REPOS, FakeBackend, make_issue_tables, make_tiny_base


@pytest.fixture(scope="module")
def ws(tmp_path_factory):
    root = tmp_path_factory.mktemp("ws")
    make_issue_tables(root / "src", per_repo=260)
    return root


def test_pipeline(ws, monkeypatch):
    cfg = load_yaml("data.yaml")
    bots = bot_regex(cfg["bot_patterns"])
    frames = []
    for repo in REPOS:
        raw = pd.read_parquet(ws / "src" / "data" / "issues" / repo / "0.parquet")
        items = clean_items(raw, repo, cutoff=cfg["cutoff"], min_words=30, bots=bots)
        assert (pd.to_datetime(items["created_at"]) < pd.Timestamp("2022-01-01", tz="UTC")).all()
        assert not items["author"].str.contains(r"\[bot\]|robot-").any()
        assert (items["n_words"] >= 30).all()
        frames.append(items)
    df = dedupe(pd.concat(frames, ignore_index=True))
    assert not df["body"].duplicated().any()

    sp = make_splits(df, heldout_repo="vuejs/core", core=600, pool=150, heldout_n=60, polish=30, seed=1)
    assert set(sp.loc[sp["repo"] == "vuejs/core", "split"]) == {"heldout_repo"}
    assert set(sp["split"]) == {"train", "val", "test", "pool", "heldout_repo", "polish"}
    assert sp["id"].is_unique
    print(counts_report(sp))

    # mirrors with the fake backend
    models = load_models()
    budget = load_yaml("budget.yaml")
    be = FakeBackend()
    client = RouterClient(be, Ledger(ws / "ledger.jsonl", 18.0), ResponseCache(ws / "cache"))
    gen = MirrorGenerator(client, models, budget)
    data_cfg = dict(cfg, sizes=dict(cfg["sizes"], mistral_mirrors=40))
    plan = plan_v0(sp, models, data_cfg)
    est = estimate_cost(plan["seen"], models, budget)
    assert 0 < est["total"] < 18
    seen = gen.run(sp, plan["seen"], ws / "m" / "v0.jsonl")
    held = gen.run(sp, plan["heldout"], ws / "m" / "held.jsonl")
    assert set(held["family"]) == {"mistral"} and set(held["split"]) == {"test"}
    assert {"model", "provider", "persona", "temperature", "prompt_tokens", "completion_tokens", "cost_usd"} <= set(seen.columns)
    assert seen["reject"].isna().mean() > 0.5
    assert set(seen["persona"]) == {"A", "B"}
    calls = be.calls
    gen.run(sp, plan["seen"], ws / "m" / "v0.jsonl")               # rerun: nothing new
    assert be.calls == calls
    pilot = pilot_sample(sp, models, n=24)
    assert len(set(pilot.values())) == 11                           # 10 training families + mistral
    assert "Pair 1" in review_pairs_md(sp, seen, n=3)

    mirrors = pd.concat([seen, held], ignore_index=True)
    fams = training_families(models)
    table = build_table(sp, mirrors, fams)
    guard_no_test_leak(table)
    assert set(table.loc[table["family_name"] == "mistral", "split"]) == {"test"}
    assert (table.loc[table["label"] == 0, "family"] == -1).all()

    # tiny model training
    base = make_tiny_base(ws / "base", table["text"].tolist())
    tcfg = dict(load_yaml("train.yaml"), per_device_batch_size=16, grad_accum=1, epochs=3, max_length=256,
                lr=2e-3, warmup_ratio=0.1, fp16=False)
    from telltale.train.train import train

    meta = train(table, fams, tcfg, ws / "model", base_model=str(base), log_every=1000)
    assert meta["val_auroc"] > 0.9, meta["val_auroc"]
    model_dir = ws / "model" / "best"

    # eval
    from telltale.eval.report import evaluate, save
    from telltale.inference import TorchScorer

    polished = gen.polish(sp[sp["split"] == "polish"], ["gpt-oss", "gemma"], ws / "m" / "polished.jsonl")
    rw = find_candidates(local_dir=str(ws / "src"), out_csv=str(ws / "t6.csv"))
    assert len(rw) > 0
    rw["verdict"] = ["slop" if i % 2 else "human" for i in range(len(rw))]
    scorer = TorchScorer(model_dir)
    res = evaluate(scorer, table, fams, polished, rw)
    for k in ("T1", "T2", "T3", "T4", "T5", "T6", "T7"):
        assert k in res["sets"], k
    report = save(res, ws / "reports" / "v0.md", title="v0 smoke")
    print(report.read_text())

    # hard negatives round 1
    from telltale.train.hard_negatives import mine, save_round

    mined = mine(scorer, sp, k=20)
    assert len(mined) == 20 and set(mined["split"]) == {"pool"}
    save_round(mined, sp[sp["split"] == "pool"], 1, ws / "hn")
    hn_mirrors = gen.run(sp, {i: fams[n % len(fams)] for n, i in enumerate(mined["id"])}, ws / "m" / "hn1.jsonl")
    table2 = build_table(sp, pd.concat([mirrors, hn_mirrors]), fams, set(mined["id"]))
    guard_no_test_leak(table2)
    assert (table2["origin"] == "hard_negative").sum() == 20
    assert (table2.loc[table2["origin"].str.startswith("hard_negative"), "split"] == "train").all()

    # ONNX int8
    from telltale.export.onnx_export import export, latency, parity
    from telltale.inference import OnnxScorer

    onnx_dir = export(model_dir, ws / "onnx")
    texts = table[table["split"] == "test"]["text"].head(20).tolist()
    par = parity(model_dir, onnx_dir, texts)
    print(par)
    assert par["flag_agreement"] >= 0.8
    lat = latency(onnx_dir)
    assert lat["load_s"] + lat["score_s_median"] < 30

    # Action, mocked GitHub
    import action.entrypoint as ep

    calls = []

    class FakeGH:
        def get_config(self, path):
            calls.append(("GET", path)); return {"min_words": 30, "exempt_users": ["trusted"]}
        def ensure_label(self, name): calls.append(("POST", "labels"))
        def add_label(self, n, name): calls.append(("POST", f"issues/{n}/labels"))
        def remove_label(self, n, name): calls.append(("DELETE", f"issues/{n}/labels"))
        def find_comment(self, n): return {"id": 1} if any(c[0] == "POST" and "comments" in c[1] for c in calls) else None
        def upsert_comment(self, n, body):
            assert ep.MARKER in body and "maintainer should decide" in body.lower() or "no longer flagged" in body
            existing = self.find_comment(n)
            calls.append(("PATCH" if existing else "POST", f"issues/{n}/comments")); return "updated" if existing else "created"

    gh = ep.GitHub.__new__(ep.GitHub)
    gh = FakeGH()
    gh.load = None
    machine = table[(table["label"] == 1) & (table["split"] == "test")]["text"].iloc[0]
    title, body = machine.split("\n\n", 1)
    pr = {"number": 7, "title": title, "body": body, "user": {"login": "rando", "type": "User"},
          "author_association": "NONE", "labels": []}
    ev = {"action": "opened", "pull_request": pr}
    r = ep.run(ev, "pull_request_target", gh, lambda: OnnxScorer(onnx_dir), "Srini410/telltale")
    assert r["status"] == "scored"
    if r["flagged"]:
        assert r["actions"] == ["label", "comment_created"]
        r2 = ep.run({"action": "edited", "pull_request": pr}, "pull_request_target", gh, lambda: OnnxScorer(onnx_dir), "x/y")
        assert "comment_updated" in r2["actions"]
        assert sum(1 for c in calls if c == ("POST", "issues/7/comments")) == 1   # one comment, then edits
    forbidden = [c for c in calls if any(w in c[1] for w in ("state", "lock", "reviews", "merge"))]
    assert not forbidden
    # skips
    for user, assoc, why in [({"login": "dependabot[bot]", "type": "Bot"}, "NONE", "bot"),
                             ({"login": "maint", "type": "User"}, "MEMBER", "member"),
                             ({"login": "trusted", "type": "User"}, "NONE", "exempt")]:
        rr = ep.run({"action": "opened", "issue": dict(pr, user=user, author_association=assoc)}, "issues", gh,
                    lambda: OnnxScorer(onnx_dir), "x/y")
        assert rr["status"] == "skipped" and why in rr["reason"], rr
    short = ep.run({"action": "opened", "issue": dict(pr, body="too short")}, "issues", gh, lambda: OnnxScorer(onnx_dir), "x/y")
    assert short["status"] == "skipped"
    closed = ep.run({"action": "closed", "issue": pr}, "issues", gh, lambda: OnnxScorer(onnx_dir), "x/y")
    assert closed["status"] == "skipped"
