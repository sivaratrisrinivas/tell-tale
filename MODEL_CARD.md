---
license: apache-2.0
language: [en]
library_name: onnx
base_model: answerdotai/ModernBERT-base
pipeline_tag: text-classification
datasets: [open-index/open-github-issues]
tags: [ai-text-detection, github, pull-requests, onnx, int8]
---

# Telltale

Telltale estimates whether a GitHub pull request or issue **description** was fully machine-written.
It is a ModernBERT-base encoder with two heads, exported to ONNX int8 for CPU use in a GitHub Action.

- `p_machine`: calibrated probability that the title + body were fully machine-written.
- `closest_family`: which of 10 model families the text most resembles (shown only above the threshold).

## Intended use

Triage help for maintainers: a label and a collapsible comment that say "a maintainer should look at this".

**Not for punishing contributors.** A flag is a heuristic, not proof. Do not use Telltale to ban, shame,
auto-close or rank people. It does not read code, does not judge quality (a machine-written PR can be a good
PR), and is not meant to flag human writing that an AI only polished for grammar.

## How it was trained

- **Human data:** GitHub issues and PRs created before 2022-01-01 (pre-ChatGPT, human by construction) from
  golang/go, python/cpython, rust-lang/rust, swiftlang/swift and mdn/content, via
  open-index/open-github-issues (ODC-By 1.0). Bots removed, bodies under 30 words dropped, near-duplicates
  removed with MinHash, templates kept as filled in.
- **Mirrors:** for each human item, gpt-oss-20b wrote a short topic; a rotating model family then wrote a new
  PR/issue on that topic, either as "a person who asked a chatbot" or "an autonomous coding agent".
  The same text normalizer is applied to both classes.
- **Hard negatives:** the model scored a 100K pre-2022 pool; the highest-scoring human items (false positives
  by construction) were mirrored and added to training, retraining from the base model each round.
- **Objective:** CE(human vs machine) + 0.3 × CE(family, machine rows only). Temperature-scaled on validation;
  threshold chosen for ≤1% false positives on validation human text.

### Generator models (Head B families)

DeepSeek-V4.1-Flash (MIT), GLM-5.3-Flash (MIT), Gemma-4-26B-A4B-it (Apache-2.0), gpt-oss-120b (Apache-2.0),
MiniMax-M3.1-Flash (MiniMax Community), Qwen3.8-27B (Apache-2.0), Inkling-2-Mini (Apache-2.0),
Nemotron-3.5-Lightning-30B-A3B (OpenMDW-1.1), Hy4-flash (Tencent Hy 1.0), Kimi-K3 (Kimi K3 license).
Held out for testing only: Mistral-Large-4. Llama-3.3-70B was excluded because its license requires models
trained on its outputs to carry "Llama" in their name.

## Evaluation

<!-- Paste runs/reports/final.md here after M7. -->

| Set | Measures | v1 target | Result |
|---|---|---|---|
| T1 held-out human, same repos | false positive rate | ≤1% (stretch 0.1%) | _TBD_ |
| T2 held-out mirrors, seen families | true positive rate | ≥90% | _TBD_ |
| T3 unseen family (Mistral-Large-4) | true positive rate | ≥80% | _TBD_ |
| T4 unseen repo (vuejs/core) | FPR / TPR | ≤2% / ≥85% | _TBD_ |
| T5 AI-polished human text | flagged | ≤5% | _TBD_ |
| T6 real-world, hand-checked (2025+) | precision / recall | report honestly | _TBD_ |
| T7 fingerprint, seen families | top-1 accuracy | report | _TBD_ |

### What the false positives have in common

<!-- Paste runs/hard_neg/round*_analysis.md findings: terse text, templates, non-native English,
generated changelogs, list-heavy bodies, short bodies... -->

_TBD after the hard-negative rounds._

## Limitations

- Trained on six large projects; other communities' writing styles may score differently.
- Short texts (<50 words) are the hardest; the Action skips texts under 30 words by default.
- Human training text is from before 2022, so it never mentions 2025+ versions or APIs. Mirrors share topics
  with their sources to limit this era leakage, and T6 checks real 2025+ text.
- New model families appear monthly. The pipeline is cheap to rerun with new generators.
- Non-native English writers may be over-flagged. Read the false-positive analysis above.

## Usage

```python
from telltale.inference import OnnxScorer
s = OnnxScorer("onnx/")              # folder with model.int8.onnx, tokenizer.json, telltale_config.json
print(s.score("Fix parser crash", "body markdown...").to_dict())
```

## Attribution

Human text from GitHub's public API via [open-index/open-github-issues](https://hf.co/datasets/open-index/open-github-issues),
licensed ODC-By 1.0. Method after Pangram's published technical reports.
