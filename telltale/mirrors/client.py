"""HF Inference Providers router client (OpenAI-compatible) with cache, ledger and backoff."""
from __future__ import annotations

import os
import random
import re
import time
from dataclasses import dataclass
from typing import Any, Protocol

from ..config import ModelSpec
from .ledger import BudgetExceeded, Ledger, ResponseCache, Usage

ROUTER_URL = "https://router.huggingface.co/v1"


class ChatBackend(Protocol):
    def create(self, *, model: str, messages: list[dict], **params: Any) -> Any: ...


class OpenAIBackend:
    def __init__(self, api_key: str | None = None, bill_to: str | None = None, base_url: str = ROUTER_URL):
        from openai import OpenAI

        key = api_key or os.environ.get("HF_TOKEN")
        if not key:
            raise RuntimeError("HF_TOKEN is not set. Create a token with 'Inference Providers' permission.")
        headers = {"X-HF-Bill-To": bill_to} if bill_to else None
        self._client = OpenAI(base_url=base_url, api_key=key, default_headers=headers, max_retries=0)

    def create(self, *, model: str, messages: list[dict], **params: Any) -> Any:
        return self._client.chat.completions.create(model=model, messages=messages, **params)


@dataclass
class ChatResult:
    text: str
    usage: Usage
    cost_usd: float
    cached: bool
    finish_reason: str | None = None


def _retryable(exc: Exception) -> bool:
    status = getattr(exc, "status_code", None) or getattr(getattr(exc, "response", None), "status_code", None)
    if status is not None:
        return status == 429 or status >= 500
    name = type(exc).__name__
    return name in {"APIConnectionError", "APITimeoutError", "Timeout", "ConnectionError"}


_THINK = re.compile(r"<think(?:ing)?>.*?(?:</think(?:ing)?>|$)", re.S | re.I)


def strip_thinking(text: str) -> str:
    """Some providers return reasoning inline as <think>...</think>; it must never reach a mirror."""
    return _THINK.sub("", text or "").strip()


def _reasoning_text(message: Any) -> str:
    for attr in ("reasoning_content", "reasoning"):
        v = getattr(message, attr, None)
        if isinstance(v, str) and v:
            return v
    return ""


def _usage_from(resp: Any) -> Usage:
    u = getattr(resp, "usage", None)
    if u is None:
        return Usage()
    details = getattr(u, "completion_tokens_details", None)
    reasoning = getattr(details, "reasoning_tokens", 0) if details is not None else 0
    return Usage(int(getattr(u, "prompt_tokens", 0) or 0), int(getattr(u, "completion_tokens", 0) or 0),
                 int(reasoning or 0))


class RouterClient:
    def __init__(self, backend: ChatBackend, ledger: Ledger, cache: ResponseCache, max_retries: int = 6):
        self.backend = backend
        self.ledger = ledger
        self.cache = cache
        self.max_retries = max_retries

    def chat(self, spec: ModelSpec, prompt: str, *, max_tokens: int, temperature: float | None = None,
             purpose: str = "mirror", meta: dict[str, Any] | None = None,
             thinking: dict[str, Any] | None = None) -> ChatResult:
        """`thinking` overrides the family's thinking-off settings (used by `telltale mirrors probe`)."""
        t = spec.thinking() if thinking is None else thinking
        if t.get("prompt_suffix"):
            prompt = prompt + t["prompt_suffix"]
        messages = [{"role": "user", "content": prompt}]
        params: dict[str, Any] = {"max_tokens": max_tokens}
        if temperature is not None:
            params["temperature"] = round(float(temperature), 3)
        if t.get("reasoning_effort"):
            params["reasoning_effort"] = t["reasoning_effort"]
        if t.get("extra_body"):
            params["extra_body"] = t["extra_body"]

        key = self.cache.key(spec.route, messages, params)
        hit = self.cache.get(key)
        if hit is not None:
            u = hit.get("usage", {})
            return ChatResult(hit["text"], Usage(u.get("prompt_tokens", 0), u.get("completion_tokens", 0),
                                                 u.get("reasoning_tokens", 0)),
                              hit.get("cost_usd", 0.0), True, hit.get("finish_reason"))

        # worst case for this call: prompt (~4 chars/token) + full max_tokens of output
        worst = (len(prompt) / 3 * spec.price_in + max_tokens * spec.price_out) / 1e6
        self.ledger.check(worst)

        delay = 2.0
        for attempt in range(self.max_retries + 1):
            try:
                resp = self.backend.create(model=spec.route, messages=messages, **params)
                break
            except BudgetExceeded:
                raise
            except Exception as exc:  # noqa: BLE001
                if attempt >= self.max_retries or not _retryable(exc):
                    raise
                time.sleep(delay + random.random())
                delay = min(delay * 2, 60)

        choice = resp.choices[0]
        raw = choice.message.content or ""
        text = strip_thinking(raw)
        usage = _usage_from(resp)
        if not usage.reasoning_tokens:
            # provider did not report reasoning tokens: estimate from returned reasoning text (~4 chars/token)
            hidden = _reasoning_text(choice.message) + raw[: max(0, len(raw) - len(text))]
            usage.reasoning_tokens = len(hidden) // 4
        cost = self.ledger.charge(model=spec.model, provider=spec.provider, usage=usage,
                                  price_in=spec.price_in, price_out=spec.price_out, purpose=purpose, meta=meta)
        self.cache.put(key, {"text": text, "usage": usage.__dict__, "cost_usd": cost,
                             "finish_reason": getattr(choice, "finish_reason", None),
                             "route": spec.route, "params": {k: v for k, v in params.items() if k != "extra_body"}})
        return ChatResult(text, usage, cost, False, getattr(choice, "finish_reason", None))
