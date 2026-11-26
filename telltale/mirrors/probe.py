"""`telltale mirrors probe`: find, per family, a setting that actually turns reasoning off.

Why: the M2 pilot showed 8 of 11 families still spending ~800-900 hidden reasoning tokens
per call even with `chat_template_kwargs` set. With max_tokens 900 the reasoning ate the
whole budget, the visible answer came back empty or cut off, the mirror was rejected, and
the retry paid again. Providers on the HF router each honour a different switch, so instead
of guessing we try the known switches with one short prompt and keep the cheapest that works.

Cost: ~8 variants x 10-11 families x <=400 output tokens. About $0.05-0.10 in total,
most of it Kimi. Every call goes through the ledger and the cache like any other.
"""
from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import Any

import yaml

from ..config import THINKING_FILE, ModelSpec
from .client import RouterClient

PROBE_PROMPT = (
    "Write a short GitHub issue for the golang/go repository about a typo in the net/http documentation. "
    "Make it about 60 words. Output the title on the first line, then the body. Nothing else."
)

# Order matters: on a tie, the earlier (more standard) switch wins.
VARIANTS: list[tuple[str, dict[str, Any]]] = [
    ("none", {}),
    ("reasoning_effort=none", {"reasoning_effort": "none"}),
    ("chat_template_kwargs.enable_thinking=false", {"extra_body": {"chat_template_kwargs": {"enable_thinking": False}}}),
    ("chat_template_kwargs.thinking=false", {"extra_body": {"chat_template_kwargs": {"thinking": False}}}),
    ("thinking.type=disabled", {"extra_body": {"thinking": {"type": "disabled"}}}),
    ("reasoning.enabled=false", {"extra_body": {"reasoning": {"enabled": False}}}),
    ("reasoning_effort=low", {"reasoning_effort": "low"}),
    ("prompt /no_think", {"prompt_suffix": "\n\n/no_think"}),
]

GOOD_REASONING = 50          # tokens: counts as "thinking is off"
MAX_TOKENS = 400


def probe_family(client: RouterClient, spec: ModelSpec, max_tokens: int = MAX_TOKENS) -> list[dict]:
    rows = []
    for name, params in VARIANTS:
        rec: dict[str, Any] = {"variant": name, "params": params}
        try:
            r = client.chat(spec, PROBE_PROMPT, max_tokens=max_tokens, temperature=0.7, purpose="probe",
                            meta={"family": spec.key, "variant": name}, thinking=params)
            rec.update(ok=bool(r.text.strip()) and r.finish_reason != "length",
                       reasoning=r.usage.reasoning_tokens, completion=r.usage.completion_tokens,
                       cost=r.cost_usd, finish=r.finish_reason, error=None)
        except Exception as exc:  # noqa: BLE001 — a 400 just means "this provider rejects that switch"
            if type(exc).__name__ == "BudgetExceeded":
                raise
            rec.update(ok=False, reasoning=None, completion=None, cost=0.0, finish=None,
                       error=f"{type(exc).__name__}: {str(exc)[:120]}")
        rows.append(rec)
        if rec["ok"] and (rec["reasoning"] or 0) <= GOOD_REASONING:
            break  # found a working switch; no need to try the rest
    return rows


def choose(rows: list[dict]) -> dict:
    ok = [r for r in rows if r["ok"]]
    answered = [r for r in rows if r.get("error") is None and r.get("reasoning") is not None]
    if not ok and not answered:
        # every switch errored: keep the models.yaml default
        return {"params": None, "reasoning_tokens": None, "status": "no_variant_worked"}
    pool = ok or answered          # if nothing fit in the probe budget, still keep the least-reasoning switch
    best = min(pool, key=lambda r: (r["reasoning"] or 0))
    status = "off" if ok and (best["reasoning"] or 0) <= GOOD_REASONING else "still_reasoning"
    return {"params": best["params"], "variant": best["variant"], "reasoning_tokens": best["reasoning"],
            "completion_tokens": best["completion"], "status": status,
            "truncated_in_probe": not ok}


def run_probe(client: RouterClient, models: dict[str, ModelSpec], families: list[str],
              out: str | Path = THINKING_FILE, base_mirror_tokens: int = 900) -> dict:
    result: dict[str, Any] = {"checked_on": dt.date.today().isoformat(), "families": {}}
    if Path(out).exists():
        result["families"] = (yaml.safe_load(Path(out).read_text()) or {}).get("families", {}) or {}
    for fam in families:
        rows = probe_family(client, models[fam])
        pick = choose(rows)
        if pick["status"] == "still_reasoning":
            # reasoning can't be switched off: give the answer room so we pay once instead of twice
            r = pick["reasoning_tokens"] or 0
            r = 2 * r if pick.get("truncated_in_probe") else 1.5 * r   # truncated: r is only a lower bound
            pick["mirror_max_tokens"] = int(min(4000, base_mirror_tokens + r))
        pick["tried"] = [{k: r[k] for k in ("variant", "ok", "reasoning", "error")} for r in rows]
        result["families"][fam] = pick
        print(f"{fam:10s} {pick['status']:18s} {pick.get('variant', '-'):45s} reasoning={pick.get('reasoning_tokens')}")
    Path(out).write_text("# Written by `telltale mirrors probe`. Per-family thinking-off settings; edit by hand if needed.\n"
                         + yaml.safe_dump(result, sort_keys=False))
    return result
