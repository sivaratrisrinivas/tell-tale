"""Text utilities shared by every stage.

The single most important rule in the spec: ONE normalizer for both classes.
`normalize()` is applied to human items at extraction time, to mirrors after
generation, and to live PR text inside the Action and the Space. Never apply
a cleaning step to only one class.
"""
from __future__ import annotations

import hashlib
import re

_TITLE_PREFIX = re.compile(r"^\s*(?:\*\*|__)?\s*(?:title|subject|pr title|issue title)\s*:\s*(?:\*\*|__)?\s*", re.I)
_BODY_PREFIX = re.compile(r"^\s*(?:\*\*|__)?\s*(?:body|description)\s*:\s*(?:\*\*|__)?\s*$", re.I)
_HTML_COMMENT = re.compile(r"<!--.*?-->", re.S)
_TRAILING_WS = re.compile(r"[ \t]+$", re.M)
_MANY_BLANKS = re.compile(r"\n{3,}")
_HEADING = re.compile(r"^(#{2,4})\s+(.+?)\s*#*\s*$", re.M)
_BOLD_HEADING = re.compile(r"^\*\*([^*\n]{2,60})\*\*:?\s*$", re.M)
_CHECKBOX = re.compile(r"^\s*[-*]\s+\[[ xX]\]\s+", re.M)
_WORD = re.compile(r"\S+")


def normalize_line_endings(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n")


def normalize_body(body: str | None) -> str:
    """Normalize a markdown body. Same function for human and machine text.

    - unify line endings
    - drop HTML comments (template instructions humans leave in and LLMs
      never write; keeping them would teach the model a formatting artifact)
    - strip trailing whitespace per line, collapse 3+ blank lines, trim
    """
    if not body:
        return ""
    t = normalize_line_endings(body)
    t = _HTML_COMMENT.sub("", t)
    t = _TRAILING_WS.sub("", t)
    t = _MANY_BLANKS.sub("\n\n", t)
    return t.strip()


def normalize_title(title: str | None) -> str:
    if not title:
        return ""
    t = normalize_line_endings(title).strip()
    t = t.lstrip("#").strip()
    t = _TITLE_PREFIX.sub("", t)
    # drop wrapping markdown emphasis / quotes / backticks around the whole title
    for a, b in (("**", "**"), ("__", "__"), ('"', '"'), ("`", "`"), ("'", "'")):
        if len(t) > len(a) + len(b) and t.startswith(a) and t.endswith(b):
            t = t[len(a):-len(b)].strip()
    return " ".join(t.split())


def compose(title: str, body: str) -> str:
    """The model input: normalized "{title}\\n\\n{body}"."""
    return f"{normalize_title(title)}\n\n{normalize_body(body)}".strip()


def split_generated(output: str) -> tuple[str, str]:
    """Parse an LLM mirror: title on first non-empty line, body after.

    Handles outputs wrapped in a ``` fence and "Title:" / "Body:" labels.
    """
    t = normalize_line_endings(output or "").strip()
    if t.startswith("```") and t.endswith("```") and t.count("```") == 2:
        t = t.split("\n", 1)[1] if "\n" in t else ""
        t = t.rsplit("```", 1)[0].strip()
    lines = t.split("\n")
    while lines and not lines[0].strip():
        lines.pop(0)
    if not lines:
        return "", ""
    title = normalize_title(lines[0])
    rest = lines[1:]
    while rest and (not rest[0].strip() or _BODY_PREFIX.match(rest[0])):
        rest.pop(0)
    return title, normalize_body("\n".join(rest))


def word_count(text: str) -> int:
    return len(_WORD.findall(text or ""))


def length_bucket(n_words: int) -> str:
    if n_words < 50:
        return "<50"
    if n_words <= 200:
        return "50-200"
    return ">200"


def template_headings(body: str) -> list[str]:
    heads = [m.group(2).strip() for m in _HEADING.finditer(body or "")]
    heads += [m.group(1).strip() for m in _BOLD_HEADING.finditer(body or "")]
    seen, out = set(), []
    for h in heads:
        k = h.lower()
        if k not in seen:
            seen.add(k)
            out.append(h)
    return out


def uses_template(body: str) -> bool:
    """Heuristic: a filled template has >=2 section headings or checkboxes."""
    return len(template_headings(body)) >= 2 or len(_CHECKBOX.findall(body or "")) >= 2


def ngrams(text: str, n: int = 8) -> set[tuple[str, ...]]:
    toks = (text or "").lower().split()
    return {tuple(toks[i:i + n]) for i in range(max(0, len(toks) - n + 1))}


def ngram_overlap(candidate: str, source: str, n: int = 8) -> float:
    """Fraction of the candidate's n-grams that also appear in the source."""
    c = ngrams(candidate, n)
    if not c:
        return 0.0
    return len(c & ngrams(source, n)) / len(c)


def stable_hash(*parts: object) -> str:
    h = hashlib.sha256()
    for p in parts:
        h.update(repr(p).encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()
