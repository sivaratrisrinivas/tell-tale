# Telltale

A slop detector for open-source PRs. Telltale is a small open model plus a GitHub Action. It flags pull
requests and issues whose written description looks **fully machine-written**, labels them
`telltale: possibly machine-written`, and leaves one collapsible comment. It never closes, locks or blocks anything.

It is trained the Pangram way: real pre-2022 GitHub text (human by construction) is paired with LLM-written
"mirrors" on the same topic from 10 model families, then the model is retrained on its own false positives.

```
pre-2022 GitHub text ─► topic (gpt-oss-20b) ─► mirrors from 10 families ─► 2-head ModernBERT
        └─ 100K pool ─► score ─► top-3K false positives ─► mirror them ─► retrain (2-3 rounds)
                                         └─► eval T1-T7 ─► ONNX int8 ─► Action + Space
```

## Layout

| Path | What |
|---|---|
| `telltale/data/` | extraction, bot/length/date filters, MinHash dedupe, splits, T6 candidate finder, dataset card |
| `telltale/mirrors/` | HF router client, prompts, filters, cost ledger with hard cap, response cache |
| `telltale/train/` | two-head model, training loop, calibration, hard-negative mining |
| `telltale/eval/` | T1-T7 harness and markdown report |
| `telltale/export/` | ONNX export, int8 quantization, parity and latency checks |
| `telltale/inference.py` | scoring API shared by eval, Action and Space (ONNX path needs no torch) |
| `action/` | `action.yml`, `entrypoint.py`, example workflow and config |
| `space/` | Gradio app + deploy script |
| `configs/` | `prices.yaml`, `models.yaml` (roster + license review), `budget.yaml`, `data.yaml`, `train.yaml` |
| `notebooks/kaggle_train.ipynb` | free-GPU training, eval, mining, export, push |
| `docs/DECISIONS.md` | every place the spec left something open, and what was chosen |

## Setup

Step-by-step guides: **[PILOT_GUIDE.md](PILOT_GUIDE.md)** (M0-M2) and **[TRAINING_GUIDE.md](TRAINING_GUIDE.md)** (dataset push + Kaggle training).

```bash
git clone https://github.com/sivaratrisrinivas/tell-tale.git && cd tell-tale
python3 -m venv .venv && source .venv/bin/activate
pip install -e .              # data + mirror generation only (M0-M3), no torch
pip install -e ".[dev,space]" # + training, ONNX, tests, Space
export HF_TOKEN=hf_...        # fine-grained token with "Make calls to Inference Providers" (+ write for pushes)
pytest -q                     # offline: unit tests + an end-to-end CPU smoke test with a fake LLM
```

## Run it, milestone by milestone

Paid steps go through the ledger (`runs/ledger.jsonl`). Every call is cached on disk, so reruns are free, and
everything halts at `cap_usd` (default **$18**, in `configs/budget.yaml`). `telltale ledger` shows spend by model.

| # | Command | Cost |
|---|---|---|
| M0 | `telltale test-call` (one cheap call that logs its cost) | ~$0.00001 |
| M1 | `telltale data extract` (prints counts by repo, kind and length; writes `runs/data/counts.md`) | free |
| M2 | `telltale mirrors probe` (finds each model's thinking-off switch, ~$0.10), then `telltale mirrors pilot` (200 mirrors across all 11 families, writes `runs/mirrors/pilot_review.md` with 20 pairs to review, and prints the v0 cost from measured per-mirror costs) | ~$0.20 |
| M3 | `telltale mirrors v0`, then `telltale data push <you>/telltale-data` (private) | ~$7.50 |
| M4 | `notebooks/kaggle_train.ipynb` (ROUND=0) or `telltale data build && telltale train` | free (Kaggle) |
| M5 | `telltale eval --model runs/model/v0/best` | free |
| M6 | `telltale mine --model … --round 1`, `telltale mirrors hardneg --round 1`, `telltale data build --with-hard-negatives`, retrain (repeat up to 3 rounds) | ~$2 per round |
| M7 | `telltale mirrors polish` (T5), `telltale data realworld`, hand-check `runs/data/t6_candidates.csv`, final `telltale eval` | ~$0.50 |
| M8 | `telltale export --model … --out runs/onnx` (prints latency + torch/ONNX parity) | free |
| M9 | add `action/example-workflow.yml` to a sandbox repo | free |
| M10 | `telltale push-model <you>/telltale`, `space/build_space.sh <you>/telltale` | free |

Stop at M2 and read the pilot pairs and the real cost (including reasoning tokens) before running M3.

## Using the Action

```yaml
# .github/workflows/telltale.yml
on:
  pull_request_target: {types: [opened, edited]}
  issues: {types: [opened, edited]}
permissions: {contents: read, pull-requests: write, issues: write}
jobs:
  telltale:
    runs-on: ubuntu-latest
    steps:
      - uses: sivaratrisrinivas/tell-tale/action@v0.1.0
        with: {model-repo: Srini410/telltale}
```

It reads only the title and body from the event payload, never checks out PR code, skips bots and
owners/members/collaborators, and skips texts under 30 words. Optional `.github/telltale.yml`: see
`action/telltale.example.yml` (threshold, min_words, label, comment, exempt_users). The model (~150 MB int8)
is cached with `actions/cache`. Measured on a 4-vCPU box with a ModernBERT-base-sized model: load 0.4 s,
scoring a 1,024-token description 1.5 s.

## Status

Everything in the spec is implemented and passes an offline end-to-end test (extract → mirrors → train →
eval T1-T7 → hard-negative round → ONNX int8 → Action with a mocked GitHub API). The paid and GPU steps
(M2 onward) have not been run yet: they need an `HF_TOKEN` and a Kaggle GPU. There are no real accuracy
numbers until then. `MODEL_CARD.md` has placeholders for them.

## Data and licenses

Human text: [open-index/open-github-issues](https://hf.co/datasets/open-index/open-github-issues), ODC-By 1.0
(attribution required). Generator licenses were reviewed for training-on-outputs; see `configs/models.yaml`.
Code: Apache-2.0.
