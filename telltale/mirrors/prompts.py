"""Prompts, verbatim from the spec."""
from __future__ import annotations

TOPIC_PROMPT = """Here is a GitHub {kind} from the {repo} project.

<item>
{title}

{body}
</item>

In 1 to 3 sentences, state what this {kind} is about: the problem or change,
the component involved, and concrete details a writer would need (versions,
error names, platforms). Do not quote the text. Output only the description."""

PERSONA_A = """Write a GitHub {kind} for the {repo} repository about the following:
{topic}

Make it about {n_words} words, like a typical {kind} on this project.
{template_hint}
Output the title on the first line, then the body. Nothing else."""

PERSONA_B = """You are an autonomous coding agent working on {repo}. You have just
{action}: {topic}

Open a GitHub {kind} describing it in about {n_words} words.
{template_hint}
Output the title on the first line, then the body. Nothing else."""

POLISH_PROMPT = """Fix grammar and spelling only. Change nothing else.

{text}"""


def kind_word(kind: str) -> str:
    return "pull request" if kind in ("pr", "pull_request", "pull request") else "issue"


def template_hint(headings: list[str]) -> str:
    if not headings:
        return ""
    return "Use this repository's template with these sections: " + ", ".join(headings[:12])


def topic_prompt(*, kind: str, repo: str, title: str, body: str, max_chars: int = 6000) -> str:
    return TOPIC_PROMPT.format(kind=kind_word(kind), repo=repo, title=title, body=body[:max_chars])


def mirror_prompt(*, persona: str, kind: str, repo: str, topic: str, n_words: int, headings: list[str]) -> str:
    kw = kind_word(kind)
    hint = template_hint(headings)
    if persona == "A":
        p = PERSONA_A.format(kind=kw, repo=repo, topic=topic, n_words=n_words, template_hint=hint)
    else:
        action = "made this change" if kw == "pull request" else "found this problem"
        p = PERSONA_B.format(kind=kw, repo=repo, action=action, topic=topic, n_words=n_words, template_hint=hint)
    # an empty template hint leaves a blank line; collapse it so prompts stay tidy
    return p.replace("\n\nOutput", "\nOutput") if not hint else p
