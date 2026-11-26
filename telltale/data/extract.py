"""M1: extract pre-2022 human issues and PRs from open-index/open-github-issues.

The dataset's `issues` table holds both issues and PRs (is_pull_request), with
title and body, so it is the only table needed for the human set. The
`pull_requests` table only adds merge/diff stats, which Telltale ignores
(description text only).
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from ..config import load_yaml
from ..text import normalize_body, normalize_title, template_headings, uses_template, word_count
from .dedupe import dedupe

ISSUE_COLS = ["number", "is_pull_request", "title", "body", "author", "created_at", "labels"]


def repo_file(table: str, repo: str) -> str:
    return f"data/{table}/{repo}/0.parquet"


def download_table(table: str, repo: str, source: str, cache_dir: str | None = None) -> Path:
    from huggingface_hub import hf_hub_download

    return Path(hf_hub_download(source, repo_file(table, repo), repo_type="dataset", cache_dir=cache_dir))


def bot_regex(patterns: Iterable[str]) -> re.Pattern:
    return re.compile("|".join(f"(?:{p})" for p in patterns), re.I)


def _labels(v) -> list[str]:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return []
    if isinstance(v, (list, tuple, np.ndarray)):
        return [str(x) for x in v]
    try:
        out = json.loads(v)
        return [str(x) for x in out] if isinstance(out, list) else []
    except (TypeError, ValueError):
        return []


def clean_items(df: pd.DataFrame, repo: str, *, cutoff: str, min_words: int, bots: re.Pattern,
                before: bool = True) -> pd.DataFrame:
    """Apply the human-set rules to one repo's raw `issues` rows.

    before=True keeps items created before `cutoff` (human set); before=False keeps
    items created on/after it (used for the 2025+ real-world candidates).
    """
    df = df.copy()
    df["created_at"] = pd.to_datetime(df["created_at"], utc=True, errors="coerce")
    cut = pd.Timestamp(cutoff, tz="UTC")
    df = df[df["created_at"] < cut] if before else df[df["created_at"] >= cut]
    author = df["author"].fillna("").astype(str)
    df = df[(author != "") & ~author.str.contains(bots)]
    df["title"] = df["title"].map(normalize_title)
    df["body"] = df["body"].map(normalize_body)
    df["n_words"] = df["body"].map(word_count)
    df = df[df["n_words"] >= min_words]
    out = pd.DataFrame({
        "id": repo + "#" + df["number"].astype(str),
        "repo": repo,
        "kind": np.where(df["is_pull_request"].astype(bool), "pr", "issue"),
        "created_at": df["created_at"].dt.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "title": df["title"],
        "body": df["body"],
        "labels": df["labels"].map(lambda v: json.dumps(_labels(v))),
        "used_template": df["body"].map(uses_template),
        "headings": df["body"].map(lambda b: json.dumps(template_headings(b))),
        "n_words": df["n_words"].astype(int),
        "author": author.loc[df.index],
    })
    return out.reset_index(drop=True)


def extract(cfg_path: str = "data.yaml", out_dir: str = "runs/data", local_dir: str | None = None,
            limit_per_repo: int | None = None) -> pd.DataFrame:
    """Download (or read from `local_dir`) each repo's issues table and build the human set."""
    cfg = load_yaml(cfg_path)
    bots = bot_regex(cfg["bot_patterns"])
    frames = []
    for repo in cfg["repos"]:
        if local_dir:
            path = Path(local_dir) / repo_file("issues", repo)
        else:
            path = download_table("issues", repo, cfg["source_dataset"])
        raw = pd.read_parquet(path, columns=ISSUE_COLS)
        if limit_per_repo:
            raw = raw.head(limit_per_repo)
        items = clean_items(raw, repo, cutoff=cfg["cutoff"], min_words=cfg["min_body_words"], bots=bots)
        print(f"{repo}: {len(raw):>7} raw -> {len(items):>7} after date/bot/length filters")
        frames.append(items)
    df = pd.concat(frames, ignore_index=True)
    mh = cfg["minhash"]
    df = dedupe(df, num_perm=mh["num_perm"], threshold=mh["threshold"], shingle=mh["shingle"])
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out / "human_all.parquet", index=False)
    return df
