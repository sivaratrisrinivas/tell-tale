# Decisions where the spec left room

Each item: what the spec said, what was built, how to change it.

## Open questions for the owner (spec section), with the default chosen

1. **Credits on personal account or org?** Default: personal account. To bill an org, set `bill_to` in
   `configs/budget.yaml` (sends `X-HF-Bill-To`). HF Jobs is not used; training runs on Kaggle.
2. **English only or multilingual?** English only: `answerdotai/ModernBERT-base`. Switch with
   `base_model: jhu-clsp/mmBERT-small` in `configs/train.yaml`.
3. **Public or private on the Hub?** Private by default (`telltale data push` / `push-model` take `--public`).
   The Action and Space can read a private model with an `HF_TOKEN`, but a public model is simpler for adopters.
4. **Which repo for the Action pilot?** Not chosen. Any sandbox repo works; `action/example-workflow.yml` is ready.
5. **PRs only, or issues too?** Both, as the spec's trigger list says (`pull_request_target` and `issues`).

## Data

- **Which tables.** The `issues` table already contains PRs (`is_pull_request`) with title and body, so the human
  set reads only `issues`. `pull_requests` adds merge/diff stats, which Telltale ignores (description only).
- **Bots.** The `issues` table has no user type, so bots are matched by name: `[bot]` suffix, dependabot, renovate,
  github-actions, `robot-*`, names ending in `bot`, and known project bots (bors, miss-islington, bedevere,
  gopherbot, rustbot, ...). List in `configs/data.yaml`.
- **HTML comments are stripped by the normalizer** (both classes). Humans leave template instructions like
  `<!-- describe your change -->` in place and LLMs never write them, so keeping them would teach the
  formatting artifact the spec warns about. Everything else in the template is kept as filled in.
- **Template detection**: a body "used a template" if it has 2+ markdown section headings (or bold-line headings)
  or 2+ checkboxes. Those headings become `{template_hint}`.
- **Split sizes.** Core 10,000 → 8,000/1,000/1,000 stratified by repo+kind; vuejs/core: 500 items (T4);
  AI-polished sources: a separate 500 held-out items (not taken from the core test set, so T1 stays clean);
  pool: 100,000. Mirrors inherit their source item's split so a topic never crosses splits.
- **Mistral (T3) mirrors**: 800 extra mirrors written for core-test items, so T2 keeps all 1,000 seen-family
  test mirrors. Total mirrors ≈ 10,000 core + 500 vuejs + 800 Mistral. `telltale mirrors estimate` prints the cost.
- **T6 candidates.** In open-github-issues the `comments` table is populated only for mdn/content and vuejs/core
  among the six repos, so the keyword search over comments mostly finds candidates there; AI-related labels from
  `timeline_events` are also used. Only comments by OWNER/MEMBER/COLLABORATOR count. The arXiv study's data
  (2611.15159) is not wired in yet; check whether it has been released.

## Generators and licenses

Reviewed each Hub `LICENSE` on 26 Nov 2026 (spec rule: drop a model whose terms don't allow training on outputs):

- **Llama-3.3-70B: disabled.** Its license allows training on outputs but requires any such model to be named
  "Llama…". Set `enabled: true` in `configs/models.yaml` if you accept that.
- **MiniMax-M3.1-Flash: kept, flagged.** Grant is non-commercial and silent on outputs; fine for a free detector.
- Everything else is MIT, Apache-2.0, OpenMDW (explicitly no restriction on outputs), or MIT-like with MaaS
  clauses that don't apply.
- So **Head B is 10-way**, not 11. Its size comes from config, so re-enabling Llama makes it 11 automatically.
- Kimi keeps its 300-mirror quota. Llama's ~890 v0 slots are spread across the other families.
- Hard-negative rounds rotate across all families except quota-limited Kimi.

## Generation details

- Topic call: gpt-oss-20b, `reasoning_effort: low`, temperature 0.2, `max_tokens` 300. It is cached, so a retry
  or a second (Mistral) mirror of the same item reuses the topic for free.
- **Thinking off (changed after the first M2 pilot, 26 Nov).** The first pilot showed 8 of 11 families (deepseek,
  glm, hy4, inkling, kimi, minimax, qwen, mistral) still spending ~900+ hidden reasoning tokens per call. The
  `chat_template_kwargs` guesses only worked for Nemotron. With `max_tokens: 900` the reasoning used the whole
  budget, the visible answer came back empty or cut off, the mirror was rejected (96/200 accepted), and the retry
  paid again. Measured v0 cost at that point: ~$24, over the cap.
  Fixes:
  1. `telltale mirrors probe` tries 8 known switches per family (`reasoning_effort: none`, two
     `chat_template_kwargs` spellings, `thinking: {type: disabled}`, `reasoning: {enabled: false}`,
     `reasoning_effort: low`, a `/no_think` prompt suffix) on one short prompt, keeps the first that brings
     reasoning under 50 tokens, and writes `configs/thinking.yaml`, which overrides `models.yaml`. About $0.10.
  2. If no switch works for a family, the probe gives it a bigger `mirror_max_tokens` (900 + 1.5-2× its
     reasoning, max 4,000) so it pays once instead of twice.
  3. A mirror cut off by `finish_reason: length` retries once with double the budget instead of the same budget.
  4. Inline `<think>…</think>` blocks are stripped from every response.
  5. The pilot and `mirrors estimate --from-pilot` now project v0 cost from the **measured cost per accepted
     mirror** for each family, times the real v0 mix (Kimi 300, Mistral 800, ~1,130 for the rest). The old
     extrapolation multiplied the pilot's per-call average by 10,000, which ignored rejects and the v0 mix.
  Expected v0 cost: about **$4.50** if thinking turns off everywhere; about **$20** if it stays on for all eight
  (Kimi ~$7.80, Qwen ~$4.20, Mistral ~$3.40 of that).
- **If Kimi or Qwen still reason after the probe, disable them** (`enabled: false` in `configs/models.yaml`).
  That takes v0 back to roughly $8.50–9. Their mirrors are spread over the other families, and Head B drops to 9
  or 8 classes. Head B's size comes from config, so nothing else changes. T7 then reports attribution over the
  remaining families only. Mistral can't be dropped without losing T3; if it still reasons, cut
  `sizes.mistral_mirrors` in `configs/data.yaml` from 800 to 400 (about $1.70 instead of $3.40).
- `{n_words}` = source body word count rounded to 10 (min 30), so mirrors are length-matched.
- A rejected mirror (refusal, empty, outside 0.5–2× source length, >10% shared 8-grams) gets one retry with a
  new temperature; if that also fails it's kept in the JSONL with its `reject` reason and excluded from training.
- Persona A/B and temperature (uniform 0.6–1.0) are derived from a hash of the item id, so runs are reproducible
  and reruns hit the cache.
- Budget guard: before each call the ledger checks spend plus that call's worst case (prompt + full `max_tokens`)
  against the cap.

## Model and training

- Pooling: masked mean over the last hidden state; dropout 0.1; two linear heads.
- Custom training loop (not HF Trainer) for the masked CE_B and best-AUROC checkpointing.
- On a T4, if batch 16 × 1,024 tokens runs out of memory, use `--per-device-batch-size 8 --grad-accum 4`
  (same effective batch of 32).
- Threshold: the smallest value that keeps validation human FPR ≤ `target_fpr` (1%) after temperature scaling.

## Action

- If an edit brings a flagged item under the threshold, the label is removed and the same comment is updated to
  say so. The spec only says "update that same comment"; removing our own label seemed the honest reading.
- Config is read from the base repo through the contents API, never from the PR head.
- The model is downloaded over plain HTTPS from `huggingface.co/<repo>/resolve/<rev>/onnx/…` and cached by
  model repo + revision.

## Layout

The spec's tree (`data/ mirrors/ train/ eval/ export/`) lives under a `telltale/` Python package so it's
importable and pip-installable. `action/`, `space/`, `configs/` sit at the top level as specified.
