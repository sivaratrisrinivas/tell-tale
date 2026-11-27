"""M5/M7: evaluation harness for T1-T7 and a markdown report."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, roc_auc_score

from ..text import compose, length_bucket, word_count

TARGETS = {
    "T1": ("FPR", "<= 1% (stretch 0.1%)", lambda r: r["fpr"] <= 0.01),
    "T2": ("TPR", ">= 90%", lambda r: r["tpr"] >= 0.90),
    "T3": ("TPR", ">= 80%", lambda r: r["tpr"] >= 0.80),
    "T4": ("FPR / TPR", "<= 2% / >= 85%", lambda r: r["fpr"] <= 0.02 and r["tpr"] >= 0.85),
    "T5": ("flagged", "<= 5%", lambda r: r["flag_rate"] <= 0.05),
}


@dataclass
class Scored:
    df: pd.DataFrame  # text,label,repo,kind,n_words,family_name + p, flag, fam_pred


def score_table(scorer, df: pd.DataFrame, threshold: float | None = None, batch: int = 64) -> pd.DataFrame:
    out = []
    texts = df["text"].tolist()
    for i in range(0, len(texts), batch):
        out += scorer.score_texts(texts[i:i + batch], threshold)
    fams = scorer.meta["families"]
    res = df.copy().reset_index(drop=True)
    res["p"] = [s.p_machine for s in out]
    res["flag"] = [s.flagged for s in out]
    res["fam_pred"] = [max(s.family_probs, key=s.family_probs.get) if s.family_probs else "" for s in out]
    res["length"] = res["n_words"].map(length_bucket)
    res.attrs["threshold"] = out[0].threshold if out else None
    res.attrs["families"] = fams
    return res


def rates(d: pd.DataFrame) -> dict:
    h, m = d[d["label"] == 0], d[d["label"] == 1]
    r = {"n_human": len(h), "n_machine": len(m),
         "fpr": float(h["flag"].mean()) if len(h) else float("nan"),
         "tpr": float(m["flag"].mean()) if len(m) else float("nan")}
    r["auroc"] = float(roc_auc_score(d["label"], d["p"])) if len(h) and len(m) else float("nan")
    return r


def breakdown(d: pd.DataFrame, by: str) -> pd.DataFrame:
    rows = []
    for k, g in d.groupby(by):
        rows.append({by: k, **rates(g)})
    return pd.DataFrame(rows)


OPERATING_FPRS = (0.01, 0.0075, 0.005)


def _thr_for_fpr(p_human: np.ndarray, target: float) -> float:
    """Same rule as train.calibrate.threshold_for_fpr (kept here so eval needs no torch)."""
    s = np.sort(np.asarray(p_human))[::-1]
    if not len(s):
        return 0.5
    k = int(np.floor(target * len(s)))
    return float(s[-1]) if k >= len(s) else float(np.nextafter(s[k], 1.0))


def operating_points(val: pd.DataFrame, t12: pd.DataFrame, s3: pd.DataFrame | None, s5: pd.DataFrame | None,
                     fprs=OPERATING_FPRS) -> pd.DataFrame:
    """The T1/T2/T3/T5 trade-off for thresholds picked on validation at several target FPRs."""
    rows = []
    for f in fprs:
        thr = _thr_for_fpr(val.loc[val["label"] == 0, "p"].to_numpy(), f)
        r = {"val target FPR": f, "threshold": thr,
             "T1 FPR": float((t12.loc[t12["label"] == 0, "p"] >= thr).mean()),
             "T2 TPR": float((t12.loc[t12["label"] == 1, "p"] >= thr).mean())}
        if s3 is not None:
            r["T3 TPR"] = float((s3["p"] >= thr).mean())
        if s5 is not None:
            r["T5 flagged"] = float((s5["p"] >= thr).mean())
        rows.append(r)
    return pd.DataFrame(rows)


def _pct(x: float) -> str:
    return "n/a" if x != x else f"{x:.2%}"


def evaluate(scorer, table: pd.DataFrame, families: list[str], polished: pd.DataFrame | None = None,
             realworld: pd.DataFrame | None = None, threshold: float | None = None) -> dict:
    """table: output of train.dataset.build_table (+ mistral mirrors in test split)."""
    test = table[table["split"] == "test"]
    t12 = score_table(scorer, test[(test["label"] == 0) | test["family_name"].isin(families)], threshold)
    res: dict = {"threshold": t12.attrs["threshold"], "sets": {}}
    r = rates(t12)
    res["sets"]["T1"] = {"fpr": r["fpr"], "n": r["n_human"]}
    res["sets"]["T2"] = {"tpr": r["tpr"], "n": r["n_machine"]}
    res["auroc_T1_T2"] = r["auroc"]
    res["by_repo"] = breakdown(t12, "repo")
    res["by_kind"] = breakdown(t12, "kind")
    res["by_length"] = breakdown(t12, "length")

    s3 = s5 = None
    t3 = test[(test["label"] == 1) & ~test["family_name"].isin(families) & (test["family_name"] != "")]
    if len(t3):
        s3 = score_table(scorer, t3, threshold)
        res["sets"]["T3"] = {"tpr": float(s3["flag"].mean()), "n": len(s3),
                             "families": sorted(t3["family_name"].unique())}

    t4 = table[table["split"] == "heldout_repo"]
    if len(t4):
        s4 = score_table(scorer, t4, threshold)
        r4 = rates(s4)
        res["sets"]["T4"] = {"fpr": r4["fpr"], "tpr": r4["tpr"], "auroc": r4["auroc"], "n": len(s4)}

    if polished is not None and len(polished):
        p = pd.DataFrame({"text": [compose(t, b) for t, b in zip(polished["title"], polished["body"])],
                          "label": 0, "repo": polished["repo"], "kind": polished["kind"]})
        p["n_words"] = p["text"].map(word_count)
        s5 = score_table(scorer, p, threshold)
        res["sets"]["T5"] = {"flag_rate": float(s5["flag"].mean()), "n": len(s5)}

    if realworld is not None and len(realworld):
        rw = realworld[realworld["verdict"].isin(["slop", "human"])]
        if len(rw):
            d = pd.DataFrame({"text": [compose(t, b) for t, b in zip(rw["title"], rw["body"])],
                              "label": (rw["verdict"] == "slop").astype(int).values,
                              "repo": rw["repo"].values, "kind": rw["kind"].values})
            d["n_words"] = d["text"].map(word_count)
            s6 = score_table(scorer, d, threshold)
            tp = int(((s6["flag"]) & (s6["label"] == 1)).sum())
            fp = int(((s6["flag"]) & (s6["label"] == 0)).sum())
            fn = int(((~s6["flag"]) & (s6["label"] == 1)).sum())
            res["sets"]["T6"] = {"precision": tp / (tp + fp) if tp + fp else float("nan"),
                                 "recall": tp / (tp + fn) if tp + fn else float("nan"),
                                 "n": len(s6), "n_slop": int(s6["label"].sum())}

    val = table[table["split"] == "val"]
    if len(val) and val["label"].nunique() == 2:
        res["operating_points"] = operating_points(score_table(scorer, val), t12, s3, s5)

    mach = t12[t12["label"] == 1]
    if len(mach):
        acc = float((mach["fam_pred"] == mach["family_name"]).mean())
        cm = confusion_matrix(mach["family_name"], mach["fam_pred"], labels=families)
        res["sets"]["T7"] = {"top1": acc, "n": len(mach)}
        res["confusion"] = pd.DataFrame(cm, index=families, columns=families)
    return res


def acceptance(res: dict) -> dict:
    s = res["sets"]
    ok = {}
    for k in ("T1", "T2", "T5"):
        ok[k] = bool(k in s and TARGETS[k][2](s[k]))
    return ok


def to_markdown(res: dict, title: str = "Telltale evaluation", latency: dict | None = None) -> str:
    s = res["sets"]
    L = [f"# {title}", "", f"Threshold: `{res['threshold']:.4f}` · AUROC (T1+T2): **{res['auroc_T1_T2']:.4f}**", "",
         "| Set | What | Result | n | v1 target | Met |", "|---|---|---|---|---|---|"]

    def row(k, what, val, n):
        tgt = TARGETS.get(k, (None, "report only", None))
        met = "" if tgt[2] is None or k not in s else ("yes" if tgt[2](s[k]) else "**no**")
        L.append(f"| {k} | {what} | {val} | {n} | {tgt[1]} | {met} |")

    if "T1" in s: row("T1", "held-out human, same repos: FPR", _pct(s["T1"]["fpr"]), s["T1"]["n"])
    if "T2" in s: row("T2", "held-out mirrors, seen families: TPR", _pct(s["T2"]["tpr"]), s["T2"]["n"])
    if "T3" in s: row("T3", f"unseen family ({', '.join(s['T3']['families'])}): TPR", _pct(s["T3"]["tpr"]), s["T3"]["n"])
    if "T4" in s: row("T4", "unseen repo (vuejs/core): FPR / TPR", f"{_pct(s['T4']['fpr'])} / {_pct(s['T4']['tpr'])}", s["T4"]["n"])
    if "T5" in s: row("T5", "AI-polished human: flagged", _pct(s["T5"]["flag_rate"]), s["T5"]["n"])
    if "T6" in s: row("T6", "real-world, hand-checked: precision / recall",
                      f"{_pct(s['T6']['precision'])} / {_pct(s['T6']['recall'])}", s["T6"]["n"])
    if "T7" in s: row("T7", "fingerprint top-1 accuracy", _pct(s["T7"]["top1"]), s["T7"]["n"])
    acc = acceptance(res)
    L += ["", "**v1 acceptance (T1, T2, T5):** " + ", ".join(f"{k} {'pass' if v else 'FAIL'}" for k, v in acc.items())]
    if latency:
        ok = latency.get("action_total_s", latency["load_s"] + latency["score_s_median"]) < 30
        L += [f"**Latency:** load {latency['load_s']}s, score {latency['score_s_median']}s "
              f"({latency['model_mb']} MB {latency.get('precision', 'int8')}) — {'pass' if ok else 'FAIL'} vs 30 s budget"]
    if "operating_points" in res:
        op = res["operating_points"].copy()
        for c in op.columns:
            op[c] = op[c].map(lambda x: f"{x:.4f}") if c == "threshold" else op[c].map(_pct)
        L += ["", "## Trade-off: thresholds picked on validation at other target FPRs", "",
              "The first row is the threshold shipped above (if it was fit at 1%). Lower rows are stricter.", "",
              op.to_markdown(index=False)]
    for name in ("by_repo", "by_kind", "by_length"):
        df = res[name].copy()
        for c in ("fpr", "tpr", "auroc"):
            df[c] = df[c].map(_pct if c != "auroc" else (lambda x: "n/a" if x != x else f"{x:.4f}"))
        L += ["", f"## Breakdown {name.replace('_', ' ')}", "", df.to_markdown(index=False)]
    if "confusion" in res:
        L += ["", "## T7 fingerprint confusion matrix (rows = true family)", "", res["confusion"].to_markdown()]
    return "\n".join(L) + "\n"


def save(res: dict, out: str | Path, title: str = "Telltale evaluation", latency: dict | None = None) -> Path:
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(to_markdown(res, title, latency))
    js = {k: v for k, v in res.items() if not isinstance(v, pd.DataFrame)}
    out.with_suffix(".json").write_text(json.dumps(js, indent=2, default=float))
    return out
