"""Mirror generation: topic extraction, persona prompts, filters, family rotation.

Every mirror records model, provider, persona, temperature, token counts and cost.
"""
from __future__ import annotations

import json
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from ..config import ModelSpec, heldout_families, load_models, load_yaml, training_families
from ..text import compose, split_generated, stable_hash
from .client import OpenAIBackend, RouterClient
from .filters import check_mirror
from .ledger import BudgetExceeded, Ledger, ResponseCache
from .prompts import POLISH_PROMPT, mirror_prompt, topic_prompt


@dataclass
class MirrorRow:
    source_id: str
    repo: str
    kind: str
    split: str
    family: str
    model: str
    provider: str
    persona: str
    temperature: float
    topic: str
    title: str
    body: str
    prompt_tokens: int
    completion_tokens: int
    reasoning_tokens: int
    cost_usd: float
    reject: str | None
    attempts: int


def assign_families(ids: list[str], families: list[str], quotas: dict[str, int | None], seed: int) -> dict[str, str]:
    """Round-robin assignment with per-family quotas (e.g. Kimi: 300).

    Deterministic for a given seed and id list, so reruns hit the cache.
    """
    rng = random.Random(seed)
    order = list(ids)
    rng.shuffle(order)
    used = {f: 0 for f in families}
    out: dict[str, str] = {}
    i = 0
    for sid in order:
        for _ in range(len(families)):
            f = families[i % len(families)]
            i += 1
            q = quotas.get(f)
            if q is None or used[f] < q:
                used[f] += 1
                out[sid] = f
                break
        else:
            raise ValueError("all families exhausted their quotas")
    return out


def _round_words(n: int) -> int:
    return max(30, int(round(n / 10.0)) * 10)


class MirrorGenerator:
    def __init__(self, client: RouterClient, models: dict[str, ModelSpec], budget: dict[str, Any], seed: int = 13):
        self.client = client
        self.models = models
        self.budget = budget
        self.seed = seed

    @classmethod
    def from_configs(cls, models_cfg: str = "models.yaml", budget_cfg: str = "budget.yaml",
                     backend=None, seed: int = 13) -> "MirrorGenerator":
        models = load_models(models_cfg)
        budget = load_yaml(budget_cfg)
        ledger = Ledger(budget["ledger_path"], budget["cap_usd"])
        cache = ResponseCache(budget["cache_dir"])
        backend = backend or OpenAIBackend(bill_to=budget.get("bill_to"))
        return cls(RouterClient(backend, ledger, cache, budget.get("max_retries", 6)), models, budget, seed)

    # -- single item -----------------------------------------------------------
    def topic(self, item: dict) -> str:
        r = self.client.chat(self.models["topic"], topic_prompt(kind=item["kind"], repo=item["repo"],
                                                               title=item["title"], body=item["body"]),
                             max_tokens=self.budget["max_tokens"]["topic"], temperature=0.2, purpose="topic",
                             meta={"source_id": item["id"]})
        return r.text.strip()

    def mirror(self, item: dict, family: str, max_attempts: int = 2) -> MirrorRow:
        spec = self.models[family]
        rng = random.Random(stable_hash(self.seed, item["id"], family))
        persona = "A" if rng.random() < 0.5 else "B"
        topic = self.topic(item)
        headings = json.loads(item.get("headings") or "[]") if item.get("used_template") else []
        source_text = compose(item["title"], item["body"])
        n_words = _round_words(int(item.get("n_words") or len(item["body"].split())))
        tot = {"p": 0, "c": 0, "r": 0, "cost": 0.0}
        title = body = ""
        reject: str | None = "empty"
        temp = 0.0
        attempt = 0
        for attempt in range(1, max_attempts + 1):
            temp = round(rng.uniform(0.6, 1.0), 3)
            prompt = mirror_prompt(persona=persona, kind=item["kind"], repo=item["repo"], topic=topic,
                                   n_words=n_words, headings=headings)
            r = self.client.chat(spec, prompt, max_tokens=self.budget["max_tokens"]["mirror"], temperature=temp,
                                 purpose="mirror", meta={"source_id": item["id"], "family": family,
                                                         "persona": persona, "temperature": temp})
            tot["p"] += r.usage.prompt_tokens
            tot["c"] += r.usage.completion_tokens
            tot["r"] += r.usage.reasoning_tokens
            tot["cost"] += r.cost_usd
            title, body = split_generated(r.text)
            reject = check_mirror(title, body, source_text)
            if reject is None:
                break
        return MirrorRow(item["id"], item["repo"], item["kind"], item.get("split", ""), family, spec.model,
                         spec.provider, persona, temp, topic, title, body, tot["p"], tot["c"], tot["r"],
                         round(tot["cost"], 8), reject, attempt)

    # -- batch -------------------------------------------------------------------
    def run(self, items: pd.DataFrame, assignment: dict[str, str], out_path: str | Path,
            concurrency: int | None = None) -> pd.DataFrame:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        done: set[tuple[str, str]] = set()
        if out_path.exists():
            prev = pd.read_json(out_path, lines=True)
            done = set(zip(prev["source_id"], prev["family"]))
        todo = [r for r in items.to_dict("records") if r["id"] in assignment
                and (r["id"], assignment[r["id"]]) not in done]
        print(f"mirrors: {len(todo)} to generate ({len(done)} already in {out_path.name}); "
              f"spent so far ${self.client.ledger.spent:.4f} of ${self.client.ledger.cap:.2f}")
        n = 0
        with ThreadPoolExecutor(max_workers=concurrency or self.budget.get("concurrency", 8)) as ex, \
                open(out_path, "a") as f:
            futs = {ex.submit(self.mirror, r, assignment[r["id"]]): r["id"] for r in todo}
            try:
                for fut in as_completed(futs):
                    try:
                        row = fut.result()
                    except BudgetExceeded as exc:
                        print(f"HALT: {exc}")
                        for other in futs:
                            other.cancel()
                        break
                    except Exception as exc:  # noqa: BLE001 — log and keep going
                        print(f"error on {futs[fut]}: {type(exc).__name__}: {exc}")
                        continue
                    f.write(json.dumps(asdict(row)) + "\n")
                    f.flush()
                    n += 1
                    if n % 100 == 0:
                        print(f"  {n}/{len(todo)}  spent ${self.client.ledger.spent:.4f}")
            finally:
                ex.shutdown(cancel_futures=True)
        return pd.read_json(out_path, lines=True) if out_path.exists() else pd.DataFrame()

    def polish(self, items: pd.DataFrame, polishers: Iterable[str], out_path: str | Path) -> pd.DataFrame:
        """T5: "Fix grammar and spelling only. Change nothing else." on 2-3 models."""
        rows = []
        polishers = list(polishers)
        for i, r in enumerate(items.to_dict("records")):
            fam = polishers[i % len(polishers)]
            text = compose(r["title"], r["body"])
            res = self.client.chat(self.models[fam], POLISH_PROMPT.format(text=text),
                                   max_tokens=self.budget["max_tokens"]["polish"], temperature=0.0,
                                   purpose="polish", meta={"source_id": r["id"], "family": fam})
            t, b = split_generated(res.text)
            rows.append({"source_id": r["id"], "repo": r["repo"], "kind": r["kind"], "family": fam,
                         "title": t, "body": b, "cost_usd": res.cost_usd})
        df = pd.DataFrame(rows)
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        df.to_json(out_path, orient="records", lines=True)
        return df


# -- planning helpers -------------------------------------------------------------
def plan_v0(human: pd.DataFrame, models: dict[str, ModelSpec], data_cfg: dict, seed: int = 13) -> dict[str, dict[str, str]]:
    """Return {"seen": {source_id: family}, "heldout": {source_id: family}}.

    seen: every train/val/test/heldout_repo item gets one mirror from a training family.
    heldout: `mistral_mirrors` test items additionally get a Mistral mirror (T3).
    """
    fams = training_families(models)
    quotas = {k: models[k].quota for k in fams}
    ids = human.loc[human["split"].isin(["train", "val", "test", "heldout_repo"]), "id"].tolist()
    seen = assign_families(ids, fams, quotas, seed)
    held = {}
    hf = heldout_families(models)
    if hf:
        test_ids = human.loc[human["split"] == "test", "id"].tolist()
        rng = random.Random(seed + 1)
        rng.shuffle(test_ids)
        held = {sid: hf[0] for sid in test_ids[: data_cfg["sizes"]["mistral_mirrors"]]}
    return {"seen": seen, "heldout": held}


def estimate_cost(assignment: dict[str, str], models: dict[str, ModelSpec], budget: dict, n_topics: int | None = None) -> dict:
    e = budget["estimate"]
    n_topics = len(assignment) if n_topics is None else n_topics
    t = models["topic"]
    topic_cost = n_topics * (e["topic_in"] * t.price_in + e["topic_out"] * t.price_out) / 1e6
    by: dict[str, float] = {}
    for fam in assignment.values():
        m = models[fam]
        by[fam] = by.get(fam, 0.0) + (e["mirror_in"] * m.price_in + e["mirror_out"] * m.price_out) / 1e6
    counts = pd.Series(list(assignment.values())).value_counts().to_dict() if assignment else {}
    return {"topics": round(topic_cost, 4), "mirrors": {k: {"n": counts[k], "usd": round(v, 4)} for k, v in by.items()},
            "total": round(topic_cost + sum(by.values()), 4)}


def pilot_sample(human: pd.DataFrame, models: dict[str, ModelSpec], n: int = 200, seed: int = 13) -> dict[str, str]:
    """M2: n mirrors spread evenly across ALL families (training + held-out)."""
    fams = training_families(models) + heldout_families(models)
    pool = human[human["split"] == "train"].sample(n=min(n, (human["split"] == "train").sum()), random_state=seed)
    return {sid: fams[i % len(fams)] for i, sid in enumerate(pool["id"])}


def review_pairs_md(human: pd.DataFrame, mirrors: pd.DataFrame, n: int = 20, seed: int = 13) -> str:
    """Markdown with n human/mirror pairs for the owner's M2 review."""
    ok = mirrors[mirrors["reject"].isna()]
    sample = ok.sample(n=min(n, len(ok)), random_state=seed)
    h = human.set_index("id")
    out = ["# M2 pilot: human vs mirror pairs\n"]
    for i, r in enumerate(sample.itertuples(), 1):
        src = h.loc[r.source_id]
        out += [f"## Pair {i}: {r.source_id} ({r.kind}) — {r.family}, persona {r.persona}, T={r.temperature}",
                f"**Topic:** {r.topic}\n", "### Human (pre-2022)", "```markdown",
                compose(src["title"], src["body"])[:3000], "```", "### Mirror", "```markdown",
                compose(r.title, r.body)[:3000], "```", ""]
    return "\n".join(out)
