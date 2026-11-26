"""Telltale GitHub Action entrypoint.

Security rules (spec):
  * Reads ONLY the title and body from the event payload. Never checks out or runs PR code.
  * Only ever: reads the config file via the API, adds/removes ONE label, creates/updates ONE comment.
  * Never closes, locks, requests changes, or blocks a merge.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # the telltale package ships with the action

import requests  # noqa: E402
import yaml  # noqa: E402

from telltale.text import compose, word_count  # noqa: E402

MARKER = "<!-- telltale-comment -->"
API = os.environ.get("GITHUB_API_URL", "https://api.github.com")
DEFAULTS: dict[str, Any] = {
    "threshold": None,          # None = use the calibrated threshold shipped with the model
    "min_words": 30,
    "label": "telltale: possibly machine-written",
    "comment": True,
    "exempt_users": [],
}
SKIP_ASSOCIATIONS = {"OWNER", "MEMBER", "COLLABORATOR"}
MODEL_FILES = ["model.int8.onnx", "tokenizer.json", "telltale_config.json"]


class GitHub:
    """Tiny REST client restricted to the endpoints Telltale is allowed to call."""

    def __init__(self, token: str, repo: str, session: requests.Session | None = None):
        self.repo = repo
        self.s = session or requests.Session()
        self.s.headers.update({"Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json",
                               "X-GitHub-Api-Version": "2022-11-28", "User-Agent": "telltale-action"})

    def _url(self, path: str) -> str:
        return f"{API}/repos/{self.repo}{path}"

    def get_config(self, path: str) -> dict:
        r = self.s.get(self._url(f"/contents/{path}"), headers={"Accept": "application/vnd.github.raw+json"})
        if r.status_code == 404:
            return {}
        r.raise_for_status()
        return yaml.safe_load(r.text) or {}

    def ensure_label(self, name: str) -> None:
        r = self.s.post(self._url("/labels"), json={"name": name, "color": "d4c5f9",
                                                     "description": "Telltale heuristic: description may be fully machine-written"})
        if r.status_code not in (201, 422):  # 422 = already exists
            r.raise_for_status()

    def add_label(self, number: int, name: str) -> None:
        self.s.post(self._url(f"/issues/{number}/labels"), json={"labels": [name]}).raise_for_status()

    def remove_label(self, number: int, name: str) -> None:
        r = self.s.delete(self._url(f"/issues/{number}/labels/{requests.utils.quote(name, safe='')}"))
        if r.status_code not in (200, 404):
            r.raise_for_status()

    def find_comment(self, number: int) -> dict | None:
        page = 1
        while True:
            r = self.s.get(self._url(f"/issues/{number}/comments"), params={"per_page": 100, "page": page})
            r.raise_for_status()
            items = r.json()
            for c in items:
                if MARKER in (c.get("body") or ""):
                    return c
            if len(items) < 100:
                return None
            page += 1

    def upsert_comment(self, number: int, body: str) -> str:
        existing = self.find_comment(number)
        if existing:
            self.s.patch(self._url(f"/issues/comments/{existing['id']}"), json={"body": body}).raise_for_status()
            return "updated"
        self.s.post(self._url(f"/issues/{number}/comments"), json={"body": body}).raise_for_status()
        return "created"


def load_config(gh: GitHub | None, path: str) -> dict:
    cfg = dict(DEFAULTS)
    if gh is not None:
        try:
            cfg.update({k: v for k, v in gh.get_config(path).items() if k in DEFAULTS})
        except Exception as exc:  # noqa: BLE001 — a bad config must not break the run
            print(f"::warning::could not read {path}: {exc}; using defaults")
    cfg["exempt_users"] = [u.lower() for u in cfg.get("exempt_users") or []]
    return cfg


def should_skip(item: dict, cfg: dict, text: str) -> str | None:
    user = item.get("user") or {}
    login = (user.get("login") or "").lower()
    if user.get("type") == "Bot" or login.endswith("[bot]"):
        return "author is a bot"
    if (item.get("author_association") or "").upper() in SKIP_ASSOCIATIONS:
        return f"author is {item['author_association'].lower()}"
    if login in cfg["exempt_users"]:
        return "author is exempt"
    if word_count(text) < int(cfg["min_words"]):
        return f"under {cfg['min_words']} words"
    return None


def comment_body(score, model_repo: str, kind: str) -> str:
    card = f"https://huggingface.co/{model_repo}"
    fam = ""
    if score.closest_family:
        fam = f"| Closest model family | `{score.closest_family}` ({score.family_prob:.0%}) |\n"
    return (
        f"{MARKER}\n"
        f"<details>\n<summary>🔍 Telltale: this {kind} description may be machine-written "
        f"(score {score.p_machine:.2f})</summary>\n\n"
        f"| | |\n|---|---|\n"
        f"| Score (p_machine) | **{score.p_machine:.3f}** (threshold {score.threshold:.3f}) |\n"
        f"{fam}\n"
        "This is an automated **heuristic**, not a verdict. It only looks at the written description, not the code, "
        "and it does not judge quality: a machine-written PR can still be a good PR, and AI-polished human writing "
        "is not what it is meant to flag. **A maintainer should decide.** Telltale never closes, locks or blocks "
        "anything.\n\n"
        f"Model card and known limitations: {card}\n</details>"
    )


def cleared_body(score, model_repo: str) -> str:
    return (f"{MARKER}\n<details>\n<summary>Telltale: no longer flagged after edit (score {score.p_machine:.2f})"
            f"</summary>\n\nModel card: https://huggingface.co/{model_repo}\n</details>")


def fetch_model(model_repo: str, revision: str, cache_dir: Path, hf_token: str | None) -> Path:
    d = cache_dir / model_repo.replace("/", "__") / revision
    d.mkdir(parents=True, exist_ok=True)
    headers = {"Authorization": f"Bearer {hf_token}"} if hf_token else {}
    for f in MODEL_FILES:
        p = d / f
        if p.exists() and p.stat().st_size > 0:
            continue
        url = f"https://huggingface.co/{model_repo}/resolve/{revision}/onnx/{f}"
        with requests.get(url, headers=headers, stream=True, timeout=120) as r:
            r.raise_for_status()
            tmp = p.with_suffix(".part")
            with open(tmp, "wb") as fh:
                for chunk in r.iter_content(1 << 20):
                    fh.write(chunk)
            tmp.replace(p)
    return d


def set_output(**kv: Any) -> None:
    out = os.environ.get("GITHUB_OUTPUT")
    lines = [f"{k}={v}" for k, v in kv.items()]
    if out:
        with open(out, "a") as f:
            f.write("\n".join(lines) + "\n")
    print("outputs: " + ", ".join(lines))


def run(event: dict, event_name: str, gh: GitHub | None, scorer_factory, model_repo: str,
        config_path: str = ".github/telltale.yml") -> dict:
    action = event.get("action")
    if event_name not in ("pull_request_target", "pull_request", "issues") or action not in ("opened", "edited"):
        return {"status": "skipped", "reason": f"event {event_name}/{action} not handled"}
    item = event.get("pull_request") or event.get("issue") or {}
    kind = "pull request" if "pull_request" in event or item.get("pull_request") else "issue"
    number = int(item["number"])
    text = compose(item.get("title") or "", item.get("body") or "")
    cfg = load_config(gh, config_path)
    reason = should_skip(item, cfg, text)
    if reason:
        return {"status": "skipped", "reason": reason}

    t0 = time.perf_counter()
    scorer = scorer_factory()
    score = scorer.score(item.get("title") or "", item.get("body") or "", threshold=cfg["threshold"])
    result = {"status": "scored", "number": number, "p_machine": round(score.p_machine, 4),
              "flagged": score.flagged, "closest_family": score.closest_family,
              "score_seconds": round(time.perf_counter() - t0, 3), "actions": []}
    if gh is None:
        return result
    labels = {(lab.get("name") or "") for lab in item.get("labels") or []}
    if score.flagged:
        gh.ensure_label(cfg["label"])
        gh.add_label(number, cfg["label"])
        result["actions"].append("label")
        if cfg["comment"]:
            result["actions"].append("comment_" + gh.upsert_comment(number, comment_body(score, model_repo, kind)))
    elif action == "edited":
        if cfg["label"] in labels:
            gh.remove_label(number, cfg["label"])
            result["actions"].append("unlabel")
        if cfg["comment"] and gh.find_comment(number):
            gh.upsert_comment(number, cleared_body(score, model_repo))
            result["actions"].append("comment_updated")
    return result


def main() -> int:
    event_path = os.environ.get("GITHUB_EVENT_PATH")
    if not event_path:
        print("::error::GITHUB_EVENT_PATH not set")
        return 1
    event = json.loads(Path(event_path).read_text())
    model_repo = os.environ["TELLTALE_MODEL_REPO"]
    revision = os.environ.get("TELLTALE_MODEL_REVISION", "main")
    cache = Path(os.environ.get("TELLTALE_CACHE", Path.home() / ".cache" / "telltale"))
    gh = GitHub(os.environ["GITHUB_TOKEN"], os.environ["GITHUB_REPOSITORY"])

    def factory():
        from telltale.inference import OnnxScorer

        local = os.environ.get("TELLTALE_MODEL_DIR")
        d = Path(local) if local else fetch_model(model_repo, revision, cache, os.environ.get("HF_TOKEN") or None)
        return OnnxScorer(d)

    res = run(event, os.environ.get("GITHUB_EVENT_NAME", ""), gh, factory, model_repo,
              os.environ.get("TELLTALE_CONFIG", ".github/telltale.yml"))
    print(json.dumps(res))
    set_output(status=res["status"], p_machine=res.get("p_machine", ""), flagged=str(res.get("flagged", False)).lower())
    return 0


if __name__ == "__main__":
    sys.exit(main())
