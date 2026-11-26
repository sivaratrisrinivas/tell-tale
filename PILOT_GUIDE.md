# Running the Telltale pilot on your own machine

This covers M0 (one test call), M1 (building the human dataset) and M2 (the 200-mirror pilot).
Expected spend is **about $0.20** of your HF credits, and the ledger hard-stops at $18.
You don't need a GPU, and these steps don't install PyTorch.

**What you need:** Python 3.10 or newer, git, about 2 GB of free disk and RAM, and a Hugging Face token.

## 1. Get a Hugging Face token

Go to https://huggingface.co/settings/tokens and choose **Create new token**, then **Fine-grained**.
Tick **"Make calls to Inference Providers"**. For the later `telltale data push`, also tick write access
to your repos. Copy the token, which starts with `hf_`.

## 2. Clone and install

```bash
git clone https://github.com/sivaratrisrinivas/tell-tale.git
```
(No git? Download `telltale.zip` from the project files and unzip it instead; then use `cd telltale` below.)

**macOS / Linux**
```bash
cd tell-tale
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e .
export HF_TOKEN=hf_your_token_here
```

**Windows (PowerShell)**
```powershell
cd tell-tale
py -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -e .
$env:HF_TOKEN = "hf_your_token_here"
```

Check that it installed with `telltale --help`. You should see `test-call`, `ledger`, `data`, `mirrors`, `train`, and so on.

Run every command below **from inside the `tell-tale/` folder**. Outputs go to `runs/` there.

## 3. M0: one test call (costs a tiny fraction of a cent)

```bash
telltale test-call
```

You should see a line of JSON with `usage` and `cost_usd`, then something like `ledger: $0.000010 spent of $18.00 cap`.
The `text` field can come back empty, because gpt-oss spends a few tokens reasoning first. That's fine.
What matters is that the cost was logged in `runs/ledger.jsonl`.

If this fails:
- `HF_TOKEN is not set`: set the token in the same terminal (step 2).
- `401` or `403`: the token is missing the Inference Providers permission.
- `402`: you're out of credits, or the credits sit on an organization. In that case, put `bill_to: <org-name>` in `configs/budget.yaml`.
- `404` or "model not supported by provider": a model or provider name changed since 26 Nov. Look the model up at
  https://huggingface.co/inference/models and fix `provider` in `configs/prices.yaml`.

## 4. M1: build the human dataset (free, about 10–30 minutes)

```bash
telltale data extract
```

This downloads the `issues` table for the six repos from `open-index/open-github-issues`, roughly 400 MB.
It keeps items from before 2022, drops bots and bodies under 30 words, removes near-duplicates, and makes the splits.
At the end it prints count tables by split × repo, kind and length. The same tables are saved to `runs/data/counts.md`.
Sanity check: `train` should be about 8,000, `val` and `test` about 1,000 each, `heldout_repo` (vuejs/core) about 500,
`polish` 500 and `pool` up to 100,000.

## 5. M2: the pilot (about $0.20)

Get a free estimate of the full v0 run first:
```bash
telltale mirrors estimate
```
`total_usd` should come out at a few dollars. If it says something wildly different, stop and check `configs/prices.yaml`.

Then run the pilot:
```bash
telltale mirrors pilot
```
This makes 200 mirrors spread across all 11 families (the 10 training families plus Mistral), 8 requests at a time.
It takes a few minutes. Every call is cached in `runs/cache/`, so if you rerun it, the calls it already made cost nothing.

When it finishes, it prints a table per family showing `n`, `rejected`, `cost`, mean output tokens (`out_tok`) and
mean `reasoning_tok`. It also prints the total spend and an **extrapolated cost for 10,000 mirrors**.

## 6. What to review before spending more

1. **Read `runs/mirrors/pilot_review.md`.** It has 20 human/mirror pairs side by side. For each one, ask:
   - Is the mirror on the same topic as the human item?
   - Does it read like a plausible PR or issue for that repo?
   - Where the human used a template, does the mirror use the same sections?
   - Is there formatting junk, like "Title:", code fences around everything, or preambles such as "Sure, here's…"?
2. **Look at `reasoning_tok` per family.** If a family shows hundreds of reasoning tokens per mirror, the
   "thinking off" flag isn't working for it. Its cost goes up, but the ledger still bills it correctly.
3. **Look at `rejected` per family.** A family that keeps getting rejected for length or refusals needs a look.
   You can read the failed rows in `runs/mirrors/pilot.jsonl` (check the `reject` column).
4. **Check the extrapolated cost against the budget.** The spec budgets about $7.50 for v0 mirrors. Run
   `telltale ledger` to see real spend broken down by model.

If it all looks right, the next step is `telltale mirrors v0` (M3). It spends about $7.50 and is still capped at $18.
Share `pilot_review.md` and the printed table in the thread first if you'd like a second opinion.
