"""Cost ledger with a hard cap, and an on-disk response cache.

Every paid call goes through `Ledger.charge()` after it returns, and through
`Ledger.check()` before it is sent. Once spend passes the cap, `check()`
raises BudgetExceeded and nothing else is sent. Cached responses are free and
never touch the ledger.
"""
from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..text import stable_hash


class BudgetExceeded(RuntimeError):
    pass


@dataclass
class Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    reasoning_tokens: int = 0


class Ledger:
    def __init__(self, path: str | Path, cap_usd: float):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.cap = float(cap_usd)
        self._lock = threading.Lock()
        self._spent = 0.0
        self._calls = 0
        if self.path.exists():
            with open(self.path) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        rec = json.loads(line)
                        self._spent += float(rec.get("cost_usd", 0.0))
                        self._calls += 1

    @property
    def spent(self) -> float:
        with self._lock:
            return self._spent

    @property
    def calls(self) -> int:
        with self._lock:
            return self._calls

    def remaining(self) -> float:
        return self.cap - self.spent

    def check(self, worst_case_usd: float = 0.0) -> None:
        """Refuse a call if spend already passed the cap, or this call could push it over."""
        with self._lock:
            if self._spent >= self.cap or self._spent + worst_case_usd > self.cap:
                raise BudgetExceeded(
                    f"Budget cap ${self.cap:.2f} reached (spent ${self._spent:.4f}, "
                    f"next call worst case ${worst_case_usd:.4f}). Ask the owner before raising it."
                )

    @staticmethod
    def cost(usage: Usage, price_in: float, price_out: float) -> float:
        # reasoning tokens are billed as output; most providers already include
        # them in completion_tokens, so we do NOT add them twice.
        return (usage.prompt_tokens * price_in + usage.completion_tokens * price_out) / 1e6

    def charge(self, *, model: str, provider: str, usage: Usage, price_in: float, price_out: float,
               purpose: str, meta: dict[str, Any] | None = None) -> float:
        c = self.cost(usage, price_in, price_out)
        rec = {
            "ts": time.time(),
            "model": model,
            "provider": provider,
            "purpose": purpose,
            "prompt_tokens": usage.prompt_tokens,
            "completion_tokens": usage.completion_tokens,
            "reasoning_tokens": usage.reasoning_tokens,
            "cost_usd": round(c, 8),
            **(meta or {}),
        }
        with self._lock:
            self._spent += c
            self._calls += 1
            with open(self.path, "a") as f:
                f.write(json.dumps(rec) + "\n")
        return c

    def summary(self) -> dict[str, Any]:
        by: dict[str, dict[str, float]] = {}
        if self.path.exists():
            for line in open(self.path):
                if not line.strip():
                    continue
                r = json.loads(line)
                key = f"{r['purpose']}:{r['model']}"
                d = by.setdefault(key, {"calls": 0, "cost_usd": 0.0, "prompt_tokens": 0,
                                        "completion_tokens": 0, "reasoning_tokens": 0})
                d["calls"] += 1
                d["cost_usd"] += r["cost_usd"]
                for k in ("prompt_tokens", "completion_tokens", "reasoning_tokens"):
                    d[k] += r.get(k, 0)
        return {"cap_usd": self.cap, "spent_usd": round(self.spent, 6), "calls": self.calls, "by_model": by}


class ResponseCache:
    """Disk cache keyed by hash(model, messages, params). Reruns are free."""

    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def key(route: str, messages: list[dict], params: dict[str, Any]) -> str:
        return stable_hash(route, json.dumps(messages, sort_keys=True), json.dumps(params, sort_keys=True))

    def _path(self, k: str) -> Path:
        return self.root / k[:2] / f"{k}.json"

    def get(self, k: str) -> dict[str, Any] | None:
        p = self._path(k)
        if p.exists():
            try:
                return json.loads(p.read_text())
            except json.JSONDecodeError:
                return None
        return None

    def put(self, k: str, value: dict[str, Any]) -> None:
        p = self._path(k)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(".tmp")
        tmp.write_text(json.dumps(value))
        tmp.replace(p)
