"""Telltale demo Space (Gradio, free CPU). Paste a PR/issue, get a score + model-family chart."""
from __future__ import annotations

import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))          # deployed Space: telltale/ is copied next to app.py
sys.path.insert(0, str(HERE.parent))   # local dev: repo root

import gradio as gr  # noqa: E402
import pandas as pd  # noqa: E402

from telltale.inference import OnnxScorer  # noqa: E402

MODEL_REPO = os.environ.get("TELLTALE_MODEL_REPO", "Srini410/telltale")
DISCLAIMER = (
    "**Telltale is a heuristic, not a verdict.** It estimates whether a GitHub PR or issue *description* was fully "
    "machine-written. It does not read code, does not judge quality, and is not meant to flag human writing that an "
    "AI only polished. Scores can be wrong, especially on short or non-native-English text. Never use it to punish "
    f"contributors. [Model card](https://huggingface.co/{MODEL_REPO})"
)
EXAMPLE_BODY = """## Summary

This PR introduces a comprehensive enhancement to the error handling mechanism in the parser module.
The changes ensure robust and consistent behavior when encountering malformed input, improving the overall
reliability and maintainability of the codebase.

## Changes
- Added validation for trailing commas
- Improved error messages for better developer experience
- Updated unit tests to cover edge cases

## Testing
All existing tests pass, and new tests have been added to verify the changes."""


def _load() -> OnnxScorer:
    local = os.environ.get("TELLTALE_MODEL_DIR")
    if local:
        return OnnxScorer(local)
    from huggingface_hub import snapshot_download

    d = snapshot_download(MODEL_REPO, allow_patterns=["onnx/*"], token=os.environ.get("HF_TOKEN"))
    return OnnxScorer(Path(d) / "onnx")


SCORER = _load()


def predict(title: str, body: str):
    if not (title or body).strip():
        return "Paste a title and body first.", None
    s = SCORER.score(title or "", body or "")
    verdict = "possibly machine-written" if s.flagged else "no strong machine signal"
    md = (f"### p_machine = {s.p_machine:.3f} → **{verdict}**\n"
          f"Threshold {s.threshold:.3f} (calibrated for ~1% false positives on held-out human text).")
    if s.closest_family:
        md += f"\n\nClosest model family: **{s.closest_family}** ({s.family_prob:.0%})"
    else:
        md += "\n\n_Model-family attribution is only shown when the score is above the threshold._"
    df = pd.DataFrame({"family": list(s.family_probs), "probability": list(s.family_probs.values())})
    return md, df.sort_values("probability", ascending=False)


with gr.Blocks(title="Telltale") as demo:
    gr.Markdown("# Telltale\nA slop detector for open-source PR and issue descriptions.")
    gr.Markdown(DISCLAIMER)
    with gr.Row():
        with gr.Column():
            title = gr.Textbox(label="Title", value="Enhance error handling in parser module")
            body = gr.Textbox(label="Body (markdown)", lines=14, value=EXAMPLE_BODY)
            btn = gr.Button("Score", variant="primary")
        with gr.Column():
            out = gr.Markdown()
            chart = gr.BarPlot(x="family", y="probability", title="Closest model family (if machine-written)",
                               y_lim=[0, 1], height=320)
    btn.click(predict, [title, body], [out, chart])

if __name__ == "__main__":
    demo.launch()
