"""Mirror filters from the spec: refusals, empties, length window, 8-gram overlap."""
from __future__ import annotations

import re

from ..text import ngram_overlap, word_count

_REFUSAL = re.compile(
    r"^\s*(?:i'?m sorry|i am sorry|sorry,|i can(?:no|')t|i cannot|i won'?t|as an ai|i'm unable|i am unable|"
    r"unfortunately,? i)",
    re.I,
)


def is_refusal(text: str) -> bool:
    return bool(_REFUSAL.match(text or ""))


def check_mirror(title: str, body: str, source_text: str, *, max_overlap: float = 0.10) -> str | None:
    """Return a rejection reason, or None if the mirror is accepted."""
    full = f"{title}\n\n{body}".strip()
    if not title or not body:
        return "empty"
    if is_refusal(title) or is_refusal(body):
        return "refusal"
    src_n, n = word_count(source_text), word_count(full)
    if src_n and (n < 0.5 * src_n or n > 2.0 * src_n):
        return "length"
    if ngram_overlap(full, source_text, 8) > max_overlap:
        return "overlap"
    return None
