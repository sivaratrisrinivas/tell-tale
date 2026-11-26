"""`telltale` command line. Each subcommand maps to a milestone in the spec's build order."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

RUNS = Path("runs")
HUMAN = RUNS / "data" / "human.parquet"
MIRRORS_V0 = RUNS / "mirrors" / "v0.jsonl"
MIRRORS_HELD = RUNS / "mirrors" / "heldout.jsonl"
MIRRORS_PILOT = RUNS / "mirrors" / "pilot.jsonl"
POLISHED = RUNS / "mirrors" / "polished.jsonl"
TABLE = RUNS / "table.parquet"


def _hn_mirror_paths() -> list[Path]:
    return sorted((RUNS / "mirrors").glob("hardneg_round*.jsonl"))


def _families():
    from .config import load_models, training_families

    return training_families(load_models())


# -- M0 ---------------------------------------------------------------------------------
def cmd_test_call(a):
    from .mirrors.generate import MirrorGenerator

    g = MirrorGenerator.from_configs()
    r = g.client.chat(g.models["topic"], "Reply with the single word: ok", max_tokens=a.max_tokens,
                      purpose="m0_test")
    print(json.dumps({"text": r.text, "usage": r.usage.__dict__, "cost_usd": r.cost_usd, "cached": r.cached}))
    print(f"ledger: ${g.client.ledger.spent:.6f} spent of ${g.client.ledger.cap:.2f} cap")


def cmd_ledger(a):
    from .config import load_yaml
    from .mirrors.ledger import Ledger

    b = load_yaml("budget.yaml")
    print(json.dumps(Ledger(b["ledger_path"], b["cap_usd"]).summary(), indent=2))


# -- M1 ---------------------------------------------------------------------------------
def cmd_extract(a):
    from .config import load_yaml
    from .data.extract import extract
    from .data.splits import counts_report, save_splits, make_splits

    cfg = load_yaml("data.yaml")
    df = extract(local_dir=a.local_dir, limit_per_repo=a.limit_per_repo)
    s = cfg["sizes"]
    sp = make_splits(df, heldout_repo=cfg["heldout_repo"], core=s["core"], pool=s["pool"],
                     heldout_n=s["heldout_repo"], polish=s["polished"], seed=cfg["seed"])
    save_splits(sp)
    print(counts_report(sp))


def cmd_realworld(a):
    from .data.realworld import find_candidates

    df = find_candidates(local_dir=a.local_dir, since=a.since)
    print(f"{len(df)} candidates -> runs/data/t6_candidates.csv. Hand-check them: set verdict to slop/human/unsure.")


# -- M2 / M3 ----------------------------------------------------------------------------
def _load_human():
    import pandas as pd

    return pd.read_parquet(HUMAN)


def cmd_estimate(a):
    from .config import load_models, load_yaml
    from .mirrors.generate import estimate_cost, plan_v0

    models, budget, data = load_models(), load_yaml("budget.yaml"), load_yaml("data.yaml")
    plan = plan_v0(_load_human(), models, data)
    seen = estimate_cost(plan["seen"], models, budget)
    held = estimate_cost(plan["heldout"], models, budget, n_topics=0)
    print(json.dumps({"seen": seen, "heldout": held, "total_usd": round(seen["total"] + held["total"], 4),
                      "cap_usd": budget["cap_usd"]}, indent=2))


def cmd_pilot(a):
    import pandas as pd

    from .mirrors.generate import MirrorGenerator, pilot_sample, review_pairs_md

    human = _load_human()
    g = MirrorGenerator.from_configs()
    assign = pilot_sample(human, g.models, n=a.n)
    m = g.run(human, assign, MIRRORS_PILOT)
    (RUNS / "mirrors" / "pilot_review.md").write_text(review_pairs_md(human, m, n=20))
    ok = m[m["reject"].isna()]
    by = m.groupby("family").agg(n=("source_id", "size"), rejected=("reject", lambda s: s.notna().sum()),
                                 cost=("cost_usd", "sum"), out_tok=("completion_tokens", "mean"),
                                 reasoning_tok=("reasoning_tokens", "mean"))
    print(by.to_markdown())
    per = m["cost_usd"].sum() / max(1, len(m))
    print(f"\npilot: {len(ok)}/{len(m)} accepted; mirror cost ${m['cost_usd'].sum():.4f}; "
          f"ledger total ${g.client.ledger.spent:.4f}")
    print(f"extrapolated 10,000 mirrors at pilot mix: ~${per * 10000:.2f} (+ topics). "
          f"Review runs/mirrors/pilot_review.md before running `telltale mirrors v0`.")


def cmd_v0(a):
    from .config import load_yaml
    from .mirrors.generate import MirrorGenerator, plan_v0

    human = _load_human()
    g = MirrorGenerator.from_configs()
    plan = plan_v0(human, g.models, load_yaml("data.yaml"))
    g.run(human, plan["seen"], MIRRORS_V0)
    g.run(human, plan["heldout"], MIRRORS_HELD)
    print(f"ledger total ${g.client.ledger.spent:.4f}")


def cmd_polish(a):
    from .config import load_yaml
    from .mirrors.generate import MirrorGenerator

    human = _load_human()
    g = MirrorGenerator.from_configs()
    items = human[human["split"] == "polish"]
    g.polish(items, load_yaml("models.yaml")["polishers"], POLISHED)
    print(f"polished {len(items)} -> {POLISHED}; ledger total ${g.client.ledger.spent:.4f}")


def cmd_hardneg_mirrors(a):
    import pandas as pd

    from .config import load_models, training_families
    from .mirrors.generate import MirrorGenerator, assign_families

    human = _load_human()
    ids = pd.read_csv(RUNS / "hard_neg" / f"round{a.round}_ids.csv")["id"].tolist()
    g = MirrorGenerator.from_configs()
    fams = [f for f in training_families(g.models) if g.models[f].quota is None]  # skip quota'd Kimi here
    assign = assign_families(ids, fams, {}, seed=100 + a.round)
    g.run(human, assign, RUNS / "mirrors" / f"hardneg_round{a.round}.jsonl")
    print(f"ledger total ${g.client.ledger.spent:.4f}")


# -- dataset ----------------------------------------------------------------------------
def cmd_build(a):
    from .train.dataset import build_table, guard_no_test_leak, load_mirrors
    from .train.hard_negatives import all_mined_ids

    human = _load_human()
    mirrors = load_mirrors([MIRRORS_V0, MIRRORS_HELD, *_hn_mirror_paths()])
    hn = all_mined_ids(RUNS / "hard_neg") if a.with_hard_negatives else set()
    table = build_table(human, mirrors, _families(), hn)
    guard_no_test_leak(table)
    table.to_parquet(TABLE, index=False)
    print(table.groupby(["split", "label"]).size().unstack(fill_value=0).to_markdown())


def cmd_push_dataset(a):
    from .data.card import push_dataset

    push_dataset(a.repo_id, private=not a.public)


# -- M4 / M6 ----------------------------------------------------------------------------
def cmd_train(a):
    import pandas as pd

    from .config import load_yaml
    from .train.train import train

    cfg = load_yaml(a.config)
    for k in ("epochs", "per_device_batch_size", "grad_accum", "max_length"):
        v = getattr(a, k, None)
        if v:
            cfg[k] = v
    train(pd.read_parquet(a.table), _families(), cfg, a.out, base_model=a.base_model, max_steps=a.max_steps)


def cmd_mine(a):
    from .inference import load_scorer
    from .train.hard_negatives import all_mined_ids, mine, save_round

    human = _load_human()
    scorer = load_scorer(a.model)
    mined = mine(scorer, human, k=a.k, exclude=all_mined_ids(RUNS / "hard_neg"))
    save_round(mined, human[human["split"] == "pool"], a.round)
    print(f"round {a.round}: mined {len(mined)}; min p_machine {mined['p_machine'].min():.3f}")


# -- M5 / M7 ----------------------------------------------------------------------------
def cmd_eval(a):
    import pandas as pd

    from .eval.report import evaluate, save
    from .inference import load_scorer

    scorer = load_scorer(a.model)
    table = pd.read_parquet(a.table)
    polished = pd.read_json(POLISHED, lines=True) if POLISHED.exists() else None
    rw = pd.read_csv(a.realworld) if a.realworld and Path(a.realworld).exists() else None
    lat = None
    if a.latency_onnx:
        from .export.onnx_export import latency

        lat = latency(a.latency_onnx)
    res = evaluate(scorer, table, scorer.meta["families"], polished, rw)
    p = save(res, a.out, title=a.title, latency=lat)
    print(p.read_text())


# -- M8 ---------------------------------------------------------------------------------
def cmd_export(a):
    from .export.onnx_export import export, latency, parity

    out = export(a.model, a.out)
    print(json.dumps(latency(out), indent=2))
    print(json.dumps(parity(a.model, out, ["Fix typo in README", "This PR adds comprehensive support for "
                                           "the new feature, improving robustness and maintainability."]), indent=2))


def cmd_push_model(a):
    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(a.repo_id, private=not a.public, exist_ok=True)
    api.upload_folder(repo_id=a.repo_id, folder_path=a.onnx_dir, path_in_repo="onnx")
    if a.torch_dir:
        api.upload_folder(repo_id=a.repo_id, folder_path=a.torch_dir, path_in_repo=".")
    if a.card:
        api.upload_file(path_or_fileobj=a.card, path_in_repo="README.md", repo_id=a.repo_id)
    print(f"pushed to https://huggingface.co/{a.repo_id} ({'public' if a.public else 'private'})")


def cmd_score(a):
    from .inference import load_scorer

    body = Path(a.body_file).read_text() if a.body_file else sys.stdin.read()
    print(json.dumps(load_scorer(a.model).score(a.title, body).to_dict(), indent=2))


def main(argv=None):
    p = argparse.ArgumentParser("telltale")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("test-call", help="M0: one cheap call that logs its cost"); s.add_argument("--max-tokens", type=int, default=64); s.set_defaults(f=cmd_test_call)
    s = sub.add_parser("ledger", help="spend summary"); s.set_defaults(f=cmd_ledger)

    d = sub.add_parser("data").add_subparsers(dest="sub", required=True)
    s = d.add_parser("extract", help="M1"); s.add_argument("--local-dir"); s.add_argument("--limit-per-repo", type=int); s.set_defaults(f=cmd_extract)
    s = d.add_parser("realworld", help="T6 candidates"); s.add_argument("--local-dir"); s.add_argument("--since", default="2025-01-01"); s.set_defaults(f=cmd_realworld)
    s = d.add_parser("build", help="assemble the labeled table"); s.add_argument("--with-hard-negatives", action="store_true"); s.set_defaults(f=cmd_build)
    s = d.add_parser("push", help="push dataset to the Hub (private by default)"); s.add_argument("repo_id"); s.add_argument("--public", action="store_true"); s.set_defaults(f=cmd_push_dataset)

    m = sub.add_parser("mirrors").add_subparsers(dest="sub", required=True)
    s = m.add_parser("estimate"); s.set_defaults(f=cmd_estimate)
    s = m.add_parser("pilot", help="M2"); s.add_argument("--n", type=int, default=200); s.set_defaults(f=cmd_pilot)
    s = m.add_parser("v0", help="M3"); s.set_defaults(f=cmd_v0)
    s = m.add_parser("polish", help="T5 set"); s.set_defaults(f=cmd_polish)
    s = m.add_parser("hardneg", help="M6 mirrors for a mined round"); s.add_argument("--round", type=int, required=True); s.set_defaults(f=cmd_hardneg_mirrors)

    s = sub.add_parser("train", help="M4/M6")
    s.add_argument("--table", default=str(TABLE)); s.add_argument("--out", default="runs/model/v0")
    s.add_argument("--config", default="train.yaml"); s.add_argument("--base-model"); s.add_argument("--max-steps", type=int)
    s.add_argument("--epochs", type=int); s.add_argument("--per-device-batch-size", type=int); s.add_argument("--grad-accum", type=int); s.add_argument("--max-length", type=int)
    s.set_defaults(f=cmd_train)
    s = sub.add_parser("mine", help="M6 hard negatives"); s.add_argument("--model", required=True); s.add_argument("--round", type=int, required=True); s.add_argument("--k", type=int, default=3000); s.set_defaults(f=cmd_mine)

    s = sub.add_parser("eval", help="M5/M7")
    s.add_argument("--model", required=True); s.add_argument("--table", default=str(TABLE)); s.add_argument("--out", default="runs/reports/eval.md")
    s.add_argument("--realworld", default="runs/data/t6_candidates.csv"); s.add_argument("--latency-onnx"); s.add_argument("--title", default="Telltale evaluation")
    s.set_defaults(f=cmd_eval)

    s = sub.add_parser("export", help="M8"); s.add_argument("--model", required=True); s.add_argument("--out", default="runs/onnx"); s.set_defaults(f=cmd_export)
    s = sub.add_parser("push-model"); s.add_argument("repo_id"); s.add_argument("--onnx-dir", default="runs/onnx"); s.add_argument("--torch-dir"); s.add_argument("--card", default="MODEL_CARD.md"); s.add_argument("--public", action="store_true"); s.set_defaults(f=cmd_push_model)
    s = sub.add_parser("score"); s.add_argument("--model", required=True); s.add_argument("--title", default=""); s.add_argument("--body-file"); s.set_defaults(f=cmd_score)

    a = p.parse_args(argv)
    a.f(a)


if __name__ == "__main__":
    main()
