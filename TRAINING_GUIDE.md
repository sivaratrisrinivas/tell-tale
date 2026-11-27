# From mirrors to a trained model (M3 → M4/M5)

You've finished `telltale mirrors v0`. These steps build the dataset, put it on the Hub (private), and train
the first model on Kaggle's free GPU.

**Cost:** about **$0.10** more in HF credits (step 2). Everything else is free.
**Time:** about 10 minutes on your machine, then 2–4 hours of Kaggle running on its own.

Run every command in your WSL terminal, inside the repo folder, with the venv active:
```bash
cd ~/workspace/github.com/sivaratrisrinivas/tell-tale
source .venv/bin/activate
git pull
```

---

## Step 1. Give your Hugging Face token write access (one time)

Pushing a dataset or model to the Hub needs **write** permission. Your current token only calls models.

1. Open https://huggingface.co/settings/tokens and click **Create new token**.
2. Choose the **Write** tab (the simple one), name it `telltale-write`, and create it. Copy the `hf_…` value.
3. In WSL, use it from now on. It can still make inference calls:
   ```bash
   export HF_TOKEN=hf_paste_the_write_token_here
   ```
   Optional, so you don't have to retype it in every new terminal:
   `echo 'export HF_TOKEN=hf_paste_the_write_token_here' >> ~/.bashrc`

Keep the token handy. Kaggle needs it in step 5.

## Step 2. Make the AI-polished test set (about $0.10)

This runs 500 held-out human texts through "Fix grammar and spelling only." on three models (test set T5).
Telltale must *not* flag these. They're never used for training.
```bash
telltale mirrors polish
```
**Check:** the last line says `polished 500 -> runs/mirrors/polished.jsonl` and shows the ledger total, which
should be about $6.80.

## Step 3. Build the training table (free, under a minute)

```bash
telltale data build
```
**Check:** it prints a table of rows per split, where `0` is human and `1` is machine. Roughly:

| split | 0 (human) | 1 (machine) |
|---|---|---|
| train | ~8,000 | ~7,500–8,000 |
| val | ~1,000 | ~950–1,000 |
| test | ~1,000 | ~1,150 (seen families + Mistral) |
| heldout_repo | ~500 | ~480 |

Machine counts sit a bit under human counts because rejected mirrors are left out. If any row shows 0 machine
rows, stop and send the output.

## Step 4. Push the dataset to the Hub (private, free)

```bash
telltale data push Srini410/telltale-data
```
**Check:** it prints `pushed https://huggingface.co/datasets/Srini410/telltale-data (private)`. Open that link.
Under **Files** you should see `human.parquet`, `mirrors.parquet`, `polished.parquet` and a `raw/` folder.
Kaggle reads `raw/`.

If you get a `401` or `403`, the token in `HF_TOKEN` isn't the write token from step 1.

## Step 5. Set up Kaggle (one time)

1. Sign in at https://www.kaggle.com, then go to **Settings → Phone verification**. Kaggle requires this before
   it gives you a GPU or internet access.
2. Create a GitHub token so Kaggle can download the private repo. Go to https://github.com/settings/personal-access-tokens/new,
   choose **Only select repositories → tell-tale**, set **Repository permissions → Contents → Read-only**,
   generate it, and copy it. (You can skip this once the repo is public.)
3. Get the notebook file over to Windows. In WSL run:
   ```bash
   explorer.exe notebooks
   ```
   A Windows Explorer window opens with `kaggle_train.ipynb` in it.
4. On Kaggle: **Create → New Notebook**, then **File → Import Notebook**, and upload `kaggle_train.ipynb`.
5. In the right-hand panel:
   - **Session options → Accelerator → GPU T4 x2**
   - **Session options → Internet → On**
   - **Add-ons → Secrets → Add secret**: name `HF_TOKEN`, value = the write token from step 1. Make sure it's ticked for this notebook.
   - **Add-ons → Secrets → Add secret**: name `GITHUB_TOKEN`, value = the GitHub token from item 2.

## Step 6. Train (free, 2–4 hours, runs on its own)

Click **Save Version** (top right), choose **Save & Run All (Commit)**, then **Save**.
That runs the whole notebook in the background, so you can close the tab. While it runs, check the
**Logs** on the version page. When it's done, read the cell outputs.

What each cell should show:

| Cell | Look for |
|---|---|
| setup | `Tesla T4` in the nvidia-smi line |
| pull data | a list of files under `runs/` (human.parquet, v0.jsonl, heldout.jsonl, polished.jsonl…) |
| build | the same table as step 3 |
| smoke test (~2 min) | `per-step batch …, grad accum …`, then `best val AUROC …` and a file list. If anything is broken, the run stops here |
| train | `eval step …: val AUROC 0.9…` lines every half epoch, ending in `best val AUROC … threshold=…` |
| eval | the T1–T7 table and `v1 acceptance (T1, T2, T5): …` |
| mine | `round 1: mined 3000` (for the next round; takes ~30 min) |
| export / push | latency JSON (`score_s_median` should be a few seconds or less), then `pushed to https://huggingface.co/Srini410/telltale (private)` |

The report is also saved to your dataset at `raw/reports/round0.md`, so you can read it on the Hub.

**Memory:** training uses fp16, gradient checkpointing and a per-step batch of 8 × 4 accumulation steps. Before
it starts, it tests a full-length batch and halves it automatically if it doesn't fit, so you shouldn't see
`CUDA out of memory`. If any command fails, the notebook stops at that cell with `Command failed …`. Scroll up
from there for the real error, and send it over.

**What "good" looks like for v0:** val AUROC above 0.98, T1 (false positives on real human text) at or under 1%,
and T2 (catching held-out mirrors) at or above 90%. T3 (Mistral, never seen in training) and T5 (polished human text)
are the interesting ones. Send the report back whatever the numbers are.

## What's next (not yet)

Hard-negative rounds (M6): the notebook already mined `round1` ids. Mirroring them costs about $1–2 per round on
your machine, then you retrain with `ROUND = 1`. That gets its own guide once v0 results are in.
