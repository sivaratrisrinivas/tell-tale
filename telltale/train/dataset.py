"""Assemble the labeled training table from human items + mirrors.

Columns: text, label (0 human / 1 machine), family (int id or -1), family_name,
split, source_id, repo, kind, n_words, origin.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from ..text import compose, word_count

COLUMNS = ["text", "label", "family", "family_name", "split", "source_id", "repo", "kind", "n_words", "origin"]


def human_rows(human: pd.DataFrame, splits: list[str], origin: str = "human") -> pd.DataFrame:
    h = human[human["split"].isin(splits)]
    text = [compose(t, b) for t, b in zip(h["title"], h["body"])]
    return pd.DataFrame({"text": text, "label": 0, "family": -1, "family_name": "", "split": h["split"].values,
                         "source_id": h["id"].values, "repo": h["repo"].values, "kind": h["kind"].values,
                         "n_words": [word_count(t) for t in text], "origin": origin})


def mirror_rows(mirrors: pd.DataFrame, human: pd.DataFrame, families: list[str], origin: str = "mirror") -> pd.DataFrame:
    m = mirrors[mirrors["reject"].isna()].copy()
    split_of = human.set_index("id")["split"]
    m["split"] = m["source_id"].map(split_of).fillna(m.get("split", ""))
    fam_id = {f: i for i, f in enumerate(families)}
    text = [compose(t, b) for t, b in zip(m["title"], m["body"])]
    return pd.DataFrame({"text": text, "label": 1, "family": [fam_id.get(f, -1) for f in m["family"]],
                         "family_name": m["family"].values, "split": m["split"].values,
                         "source_id": m["source_id"].values, "repo": m["repo"].values, "kind": m["kind"].values,
                         "n_words": [word_count(t) for t in text], "origin": origin})


def load_mirrors(paths: list[str | Path]) -> pd.DataFrame:
    frames = [pd.read_json(p, lines=True) for p in paths if Path(p).exists() and Path(p).stat().st_size > 0]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=["source_id", "family", "reject"])


def build_table(human: pd.DataFrame, mirrors: pd.DataFrame, families: list[str],
                hard_negative_ids: set[str] | None = None) -> pd.DataFrame:
    """train/val/test/heldout_repo rows from the core set; mined pool items (and their mirrors) join `train`."""
    h = human.copy()
    hn = set(hard_negative_ids or ())
    if hn:
        mined = h["id"].isin(hn) & (h["split"] == "pool")
        h.loc[mined, "split"] = "train"
    keep = ["train", "val", "test", "heldout_repo"]
    hr = human_rows(h, keep)
    mr = mirror_rows(mirrors, h, families)
    mr = mr[mr["split"].isin(keep)]
    if hn:
        hr.loc[hr["source_id"].isin(hn), "origin"] = "hard_negative"
        mr.loc[mr["source_id"].isin(hn), "origin"] = "hard_negative_mirror"
    return pd.concat([hr, mr], ignore_index=True)[COLUMNS]


def guard_no_test_leak(table: pd.DataFrame) -> None:
    """Spec rule: never train on any test set."""
    train_src = set(table.loc[table["split"] == "train", "source_id"])
    for s in ("val", "test", "heldout_repo", "polish"):
        leak = train_src & set(table.loc[table["split"] == s, "source_id"])
        if leak:
            raise AssertionError(f"{len(leak)} source items appear in both train and {s}")
    bad = table[(table["split"] == "train") & (table["family_name"] == "mistral")]
    if len(bad):
        raise AssertionError("held-out family (mistral) found in training rows")
