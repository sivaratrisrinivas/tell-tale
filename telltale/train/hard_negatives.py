"""M6: hard-negative mining.

1. score the human pool with the current model
2. take the top-k by p_machine (false positives by construction: all pre-2022)
3. mirror them across all families, add both to train, retrain from base
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from ..text import compose, length_bucket

_CODE = re.compile(r"```")
_LIST = re.compile(r"^\s*(?:[-*]|\d+\.)\s", re.M)


def fp_features(df: pd.DataFrame) -> pd.DataFrame:
    text = [compose(t, b) for t, b in zip(df["title"], df["body"])]
    return pd.DataFrame({
        "repo": df["repo"].values,
        "kind": df["kind"].values,
        "length": df["n_words"].map(length_bucket).values,
        "used_template": df["used_template"].values,
        "has_code_fence": [bool(_CODE.search(t)) for t in text],
        "list_heavy": [len(_LIST.findall(t)) >= 4 for t in text],
        "non_ascii_ratio": [sum(ord(c) > 127 for c in t) / max(1, len(t)) for t in text],
        "headings": [t.count("\n#") for t in text],
    })


def analyze(mined: pd.DataFrame, pool: pd.DataFrame) -> str:
    """Markdown comparing mined false positives to the pool baseline (goes into the model card)."""
    fm, fp = fp_features(mined), fp_features(pool)
    lines = ["| feature | mined FPs | pool baseline |", "|---|---|---|"]
    for col in ("used_template", "has_code_fence", "list_heavy"):
        lines.append(f"| {col} | {fm[col].mean():.1%} | {fp[col].mean():.1%} |")
    lines.append(f"| mean non-ASCII ratio | {fm['non_ascii_ratio'].mean():.4f} | {fp['non_ascii_ratio'].mean():.4f} |")
    for col in ("length", "repo", "kind"):
        a = fm[col].value_counts(normalize=True)
        b = fp[col].value_counts(normalize=True)
        for k in sorted(set(a.index) | set(b.index)):
            lines.append(f"| {col}={k} | {a.get(k, 0):.1%} | {b.get(k, 0):.1%} |")
    return "\n".join(lines)


def mine(scorer, human: pd.DataFrame, k: int = 3000, exclude: set[str] | None = None,
         batch: int = 256) -> pd.DataFrame:
    pool = human[(human["split"] == "pool") & ~human["id"].isin(exclude or set())].reset_index(drop=True)
    scores = []
    texts = [compose(t, b) for t, b in zip(pool["title"], pool["body"])]
    for i in range(0, len(texts), batch):
        scores += [s.p_machine for s in scorer.score_texts(texts[i:i + batch])]
    pool = pool.assign(p_machine=np.array(scores))
    return pool.sort_values("p_machine", ascending=False).head(k)


def save_round(mined: pd.DataFrame, pool: pd.DataFrame, round_no: int, out_dir: str | Path = "runs/hard_neg") -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    mined[["id", "p_machine"]].to_csv(out / f"round{round_no}_ids.csv", index=False)
    (out / f"round{round_no}_analysis.md").write_text(
        f"# Hard-negative round {round_no}\n\nTop {len(mined)} pool items by p_machine "
        f"(min {mined['p_machine'].min():.3f}, median {mined['p_machine'].median():.3f}).\n\n"
        + analyze(mined, pool) + "\n")
    return out


def all_mined_ids(out_dir: str | Path = "runs/hard_neg") -> set[str]:
    ids: set[str] = set()
    for p in sorted(Path(out_dir).glob("round*_ids.csv")):
        ids |= set(pd.read_csv(p)["id"])
    return ids


def log_fpr(round_no: int, fpr: float, path: str | Path = "runs/hard_neg/fpr_by_round.jsonl") -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps({"round": round_no, "val_fpr": fpr}) + "\n")
