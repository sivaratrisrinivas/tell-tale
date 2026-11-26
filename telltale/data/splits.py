"""Splits: core 80/10/10 stratified by repo+kind, a hard-negative pool, the
held-out repo (vuejs/core), and a held-out set for the AI-polished test.

Every human item gets exactly one `split` value:
  train | val | test        core set, each item gets one mirror
  pool                      hard-negative pool (scored, mirrored only if mined)
  heldout_repo              vuejs/core, T4
  polish                    T5 source items (never trained on)
Mirrors inherit their source item's split, so a topic never crosses splits.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..text import length_bucket


def _stratified(df: pd.DataFrame, fracs: tuple[float, float, float], rng: np.random.Generator) -> pd.Series:
    split = pd.Series("train", index=df.index)
    for _, g in df.groupby(["repo", "kind"]):
        idx = rng.permutation(g.index.values)
        n = len(idx)
        n_val = int(round(n * fracs[1]))
        n_test = int(round(n * fracs[2]))
        split.loc[idx[:n_test]] = "test"
        split.loc[idx[n_test:n_test + n_val]] = "val"
    return split


def make_splits(df: pd.DataFrame, *, heldout_repo: str, core: int, pool: int, heldout_n: int, polish: int,
                seed: int = 13) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = df.copy()
    df["split"] = "unused"

    hr = df.index[df["repo"] == heldout_repo].values
    hr = rng.permutation(hr)[:heldout_n]
    df.loc[hr, "split"] = "heldout_repo"

    rest = rng.permutation(df.index[df["repo"] != heldout_repo].values)
    core_idx = rest[:core]
    polish_idx = rest[core:core + polish]
    pool_idx = rest[core + polish:core + polish + pool]
    df.loc[core_idx, "split"] = _stratified(df.loc[core_idx], (0.8, 0.1, 0.1), rng)
    df.loc[polish_idx, "split"] = "polish"
    df.loc[pool_idx, "split"] = "pool"
    return df[df["split"] != "unused"].reset_index(drop=True)


def counts_report(df: pd.DataFrame) -> str:
    d = df.assign(length=df["n_words"].map(length_bucket))
    parts = ["## Counts by split", d["split"].value_counts().to_frame("n").to_markdown(),
             "", "## By split x repo", pd.crosstab(d["repo"], d["split"]).to_markdown(),
             "", "## By split x kind", pd.crosstab(d["kind"], d["split"]).to_markdown(),
             "", "## By split x length (words)", pd.crosstab(d["length"], d["split"]).to_markdown(),
             "", f"Template use: {d['used_template'].mean():.1%} of items"]
    return "\n".join(parts)


def save_splits(df: pd.DataFrame, out_dir: str | Path = "runs/data") -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    p = out / "human.parquet"
    df.to_parquet(p, index=False)
    (out / "counts.md").write_text(counts_report(df))
    return p
