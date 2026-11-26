"""Exact + near-duplicate removal on bodies with MinHash LSH."""
from __future__ import annotations

import hashlib

import pandas as pd


def _shingles(text: str, k: int) -> set[bytes]:
    toks = text.lower().split()
    if len(toks) < k:
        return {" ".join(toks).encode()}
    return {" ".join(toks[i:i + k]).encode() for i in range(len(toks) - k + 1)}


def dedupe(df: pd.DataFrame, *, num_perm: int = 128, threshold: float = 0.8, shingle: int = 5,
           col: str = "body", verbose: bool = True) -> pd.DataFrame:
    from datasketch import MinHash, MinHashLSH

    n0 = len(df)
    h = df[col].map(lambda s: hashlib.sha1(" ".join(s.lower().split()).encode()).hexdigest())
    df = df.loc[~h.duplicated()].reset_index(drop=True)
    n1 = len(df)

    lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
    keep = []
    for i, text in enumerate(df[col]):
        m = MinHash(num_perm=num_perm)
        m.update_batch(list(_shingles(text, shingle)))
        if lsh.query(m):
            continue
        lsh.insert(str(i), m)
        keep.append(i)
    out = df.iloc[keep].reset_index(drop=True)
    if verbose:
        print(f"dedupe: {n0} -> {n1} after exact -> {len(out)} after MinHash (t={threshold})")
    return out
