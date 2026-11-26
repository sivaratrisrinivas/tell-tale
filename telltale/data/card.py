"""Dataset card + push to the Hub (private by default)."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

CARD = """---
license: odc-by
language: [en]
pretty_name: Telltale human/mirror GitHub text
task_categories: [text-classification]
tags: [ai-text-detection, github, pull-requests, issues]
configs:
- config_name: human
  data_files: human.parquet
- config_name: mirrors
  data_files: mirrors.parquet
- config_name: polished
  data_files: polished.parquet
---

# Telltale dataset

Training data for **Telltale**, a detector for fully machine-written GitHub PR and issue descriptions.

## Contents

- `human`: GitHub issues and PRs created **before 2022-01-01** (human by construction, pre-ChatGPT) from
  golang/go, python/cpython, rust-lang/rust, swiftlang/swift, mdn/content and vuejs/core. Bots removed,
  bodies under 30 words dropped, exact and MinHash near-duplicates removed, issue templates kept as filled in.
  Columns: `id, repo, kind, created_at, title, body, labels, used_template, headings, n_words, split`.
- `mirrors`: LLM-written "mirrors". For each human item a cheap model wrote a 1-3 sentence topic; a rotating
  model family then wrote a new PR/issue on that topic (persona A: a person who asked a chatbot; persona B: an
  autonomous coding agent). Each row records `family, model, provider, persona, temperature, token counts, cost`.
  Mirrors inherit their source item's `split`.
- `polished`: held-out human items run through "Fix grammar and spelling only. Change nothing else."
  Used only for evaluation (Telltale must NOT flag these).

## Splits

`train/val/test` 80/10/10 stratified by repo and kind; `heldout_repo` = vuejs/core (unseen-repo test);
`pool` = hard-negative mining pool; `polish` = AI-polished test sources. Mistral mirrors are an unseen-model
test and never appear in training.

{counts}

## Sources and licenses

- Human text: [open-index/open-github-issues](https://hf.co/datasets/open-index/open-github-issues), ODC-By 1.0.
  Attribution: data from GitHub's public API via open-index. GitHub's Terms of Service apply to the original data.
- Mirrors: generated through Hugging Face Inference Providers. Each generator's license was checked for
  training-on-outputs; see `configs/models.yaml` in the Telltale repo.

## Intended use

Training and evaluating detectors that help maintainers triage. Not for punishing contributors.
"""


def push_dataset(repo_id: str, private: bool = True, runs: str | Path = "runs") -> None:
    from huggingface_hub import HfApi

    runs = Path(runs)
    stage = runs / "hub_dataset"
    stage.mkdir(parents=True, exist_ok=True)
    human = pd.read_parquet(runs / "data" / "human.parquet")
    human.drop(columns=["author"], errors="ignore").to_parquet(stage / "human.parquet", index=False)
    frames = [pd.read_json(p, lines=True) for p in sorted((runs / "mirrors").glob("*.jsonl"))
              if p.name not in ("polished.jsonl", "pilot.jsonl") and p.stat().st_size]
    if frames:
        pd.concat(frames, ignore_index=True).to_parquet(stage / "mirrors.parquet", index=False)
    pol = runs / "mirrors" / "polished.jsonl"
    if pol.exists():
        pd.read_json(pol, lines=True).to_parquet(stage / "polished.parquet", index=False)
    # raw pipeline artifacts, so a Kaggle notebook can rebuild runs/ exactly
    import shutil
    raw = stage / "raw"
    for rel in ["data/human.parquet", "data/counts.md", "data/t6_candidates.csv", "table.parquet"]:
        if (runs / rel).exists():
            (raw / rel).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(runs / rel, raw / rel)
    for sub in ("mirrors", "hard_neg"):
        for f in sorted((runs / sub).glob("*")) if (runs / sub).exists() else []:
            if f.is_file() and f.suffix in (".jsonl", ".csv", ".md"):
                (raw / sub).mkdir(parents=True, exist_ok=True)
                shutil.copy2(f, raw / sub / f.name)
    counts = (runs / "data" / "counts.md").read_text() if (runs / "data" / "counts.md").exists() else ""
    (stage / "README.md").write_text(CARD.replace("{counts}", counts))
    api = HfApi()
    api.create_repo(repo_id, repo_type="dataset", private=private, exist_ok=True)
    api.upload_folder(repo_id=repo_id, repo_type="dataset", folder_path=str(stage))
    print(f"pushed https://huggingface.co/datasets/{repo_id} ({'private' if private else 'public'})")
