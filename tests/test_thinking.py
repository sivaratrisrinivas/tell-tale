"""Reproduces the M2 pilot failure (hidden reasoning eating max_tokens) and checks the fixes."""
from types import SimpleNamespace

import pandas as pd
import yaml

from telltale.config import load_models
from telltale.mirrors.client import RouterClient, strip_thinking
from telltale.mirrors.generate import MirrorGenerator, estimate_cost, pilot_unit_costs
from telltale.mirrors.ledger import Ledger, ResponseCache
from telltale.mirrors.probe import choose, run_probe

ANSWER = "Fix typo in net/http docs\n\n" + " ".join(["The doc comment for Server.Shutdown says recieve."] * 12)


class ReasoningBackend:
    """A provider that reasons ~1200 hidden tokens unless sent the switch it honours."""

    def __init__(self, honours: dict | None, reasoning: int = 1200, rejects_unknown: bool = False):
        self.honours = honours
        self.reasoning = reasoning
        self.rejects_unknown = rejects_unknown
        self.calls = []

    def create(self, *, model, messages, max_tokens, **params):
        sent = {k: v for k, v in params.items() if k in ("reasoning_effort", "extra_body")}
        if messages[0]["content"].endswith("/no_think"):
            sent["prompt_suffix"] = "\n\n/no_think"
        self.calls.append((sent, max_tokens))
        if self.rejects_unknown and sent and sent != self.honours:
            err = RuntimeError("400 unsupported parameter")
            err.status_code = 400
            raise err
        r = 0 if (self.honours is not None and sent == self.honours) else self.reasoning
        visible = max(0, max_tokens - r)
        text = ANSWER if visible >= 150 else ""
        finish = "stop" if visible >= 150 else "length"
        usage = SimpleNamespace(prompt_tokens=60, completion_tokens=min(max_tokens, r + 150),
                                completion_tokens_details=SimpleNamespace(reasoning_tokens=min(r, max_tokens)))
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=text), finish_reason=finish)],
                               usage=usage)


def _client(tmp_path, backend):
    return RouterClient(backend, Ledger(tmp_path / "l.jsonl", 18.0), ResponseCache(tmp_path / "c"))


def test_strip_inline_think_tags():
    assert strip_thinking("<think>plan plan</think>\nTitle\n\nBody") == "Title\n\nBody"
    assert strip_thinking("<think>never closed") == ""


def test_probe_finds_the_switch_the_provider_honours(tmp_path):
    models = load_models(thinking=None)
    switch = {"extra_body": {"thinking": {"type": "disabled"}}}
    be = ReasoningBackend(honours=switch, rejects_unknown=True)
    out = tmp_path / "thinking.yaml"
    res = run_probe(_client(tmp_path, be), models, ["deepseek"], out=out)
    pick = res["families"]["deepseek"]
    assert pick["status"] == "off" and pick["params"] == switch
    # the written file is picked up by load_models and sent on real calls
    m2 = load_models(thinking=out)
    assert m2["deepseek"].extra_body == {"thinking": {"type": "disabled"}}


def test_probe_falls_back_to_bigger_budget_when_nothing_works(tmp_path):
    models = load_models(thinking=None)
    be = ReasoningBackend(honours=None, reasoning=300)   # reasons 300 tokens whatever we send
    res = run_probe(_client(tmp_path, be), models, ["qwen"], out=tmp_path / "t.yaml")
    pick = res["families"]["qwen"]
    assert pick["status"] == "still_reasoning" and pick["mirror_max_tokens"] >= 900 + 300


def test_choose_handles_all_failures():
    assert choose([{"ok": False, "reasoning": None}])["status"] == "no_variant_worked"


def test_truncated_mirror_retries_with_bigger_budget(tmp_path):
    """The pilot bug: 900 max_tokens, ~1200 reasoning -> empty answer -> same-budget retry -> paid twice."""
    models = load_models(thinking=None)
    budget = {"max_tokens": {"topic": 300, "mirror": 900, "mirror_ceiling": 4000}}
    be = ReasoningBackend(honours=None, reasoning=1200)
    g = MirrorGenerator(_client(tmp_path, be), models, budget)
    item = {"id": "golang/go#1", "repo": "golang/go", "kind": "issue", "title": "net/http: typo",
            "body": " ".join(["the doc for Server.Shutdown has a typo recieve, should be receive."] * 10),
            "n_words": 110, "used_template": False, "headings": "[]"}
    row = g.mirror(item, "deepseek")
    mirror_calls = [mt for _, mt in be.calls if mt != 300]
    assert mirror_calls == [900, 1800]          # second attempt got room to answer
    assert row.reject is None


def test_estimate_uses_measured_cost_per_accepted_mirror():
    models = load_models(thinking=None)
    pilot = pd.DataFrame({"family": ["qwen"] * 4 + ["gemma"] * 4,
                          "reject": [None, "length", "length", "length"] + [None] * 4,
                          "cost_usd": [0.004] * 4 + [0.0001] * 4, "reasoning_tokens": [1600] * 4 + [0] * 4})
    u = pilot_unit_costs(pilot)
    assert u["qwen"]["usd_per_accepted"] == 0.016 and u["qwen"]["accept_rate"] == 0.25
    est = estimate_cost({str(i): "qwen" for i in range(100)}, models,
                        {"estimate": {"topic_in": 0, "topic_out": 0, "mirror_in": 1, "mirror_out": 1}}, unit_costs=u)
    assert est["mirrors"]["qwen"]["usd"] == 1.6 and est["mirrors"]["qwen"]["source"] == "pilot"
