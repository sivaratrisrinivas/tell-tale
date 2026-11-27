# Round 1: fix the false positives (T1, T5) and the ONNX gap

Round 0 missed two targets: **T1** flagged 1.20% of real human text (target ≤ 1%) and **T5** flagged 6.00% of
AI-polished human text (target ≤ 5%). Everything else passed with room to spare (T2 99.5%, T3 99.6%).

This round does three things:

1. **Hard negatives.** The notebook already found the 3,000 real human texts the model scored as most
   machine-like (`round1_ids.csv`: mostly short Rust PRs). You generate mirrors for them, and they join training.
2. **Polished human text as human.** 1,500 *training* texts get the same "fix grammar and spelling only" pass as
   T5 and are labeled human. The 500 T5 test texts are still never trained on.
3. **A stricter threshold.** It is now picked for 0.75% FPR on validation instead of 1%, so ordinary
   val/test noise doesn't push T1 over 1%. This costs a little T2. The report now prints a table of
   T1 / T2 / T3 / T5 at 1%, 0.75% and 0.5%, so you can see that cost directly.

It also fixes the ONNX export (details at the end).

**Cost:** about **$2.10** in HF credits (steps 3 and 4). Your ledger should end near $9 of the $18 cap.
**Time:** about 30–60 minutes on your machine, then about 4–5 hours on Kaggle.

Run everything in WSL, inside the repo, with the venv active and your **write** token set:
```bash
cd ~/workspace/github.com/sivaratrisrinivas/tell-tale
source .venv/bin/activate
git pull
export HF_TOKEN=hf_your_write_token     # skip if it's already in ~/.bashrc
```

## Step 1. Update the install (free)

```bash
pip install -e .
```

## Step 2. Download the mined ids from your dataset (free)

```bash
telltale data pull Srini410/telltale-data
```
**Check:** it prints `hard_neg: round1_analysis.md, round1_ids.csv` and a `reports:` line.

## Step 3. Mirror the hard negatives (about $1.80)

```bash
telltale mirrors hardneg --round 1
```
This writes topics and mirrors for the 3,000 mined texts. Like v0, every call is cached, so if it stops you can
run the same command again and it picks up where it left off.
**Check:** the last line shows the ledger total, about **$8.60**.

## Step 4. Polish 1,500 training texts (about $0.30)

```bash
telltale mirrors polish --train 1500
```
**Check:** `polished 1500 -> runs/mirrors/polished_train.jsonl` and a ledger total of about **$8.90**.

## Step 5. Check the table, then push (free)

```bash
telltale data build --with-hard-negatives
```
**Check:** besides the split table, it prints a line like
`added to train: {'hard_negative': 3000, 'hard_negative_mirror': ~2900, 'polished_human': ~1500}`.
If any of those three is missing, stop and send the output.

```bash
telltale data push Srini410/telltale-data
```
**Check:** `pushed https://huggingface.co/datasets/Srini410/telltale-data (private)`.

## Step 6. Re-import the notebook and run it (free, about 4–5 hours)

The notebook changed, so Kaggle needs the new file. Re-running the old one won't pick up the changes.

1. In WSL: `explorer.exe notebooks`
2. On Kaggle, open your Telltale notebook, then **File → Import Notebook** and upload `kaggle_train.ipynb` again.
   It already says `ROUND = 1`.
3. Check the right-hand panel still has **GPU T4 x2**, **Internet On**, and both secrets (`HF_TOKEN`, `GITHUB_TOKEN`).
4. **Save Version → Save & Run All (Commit)**.

What's new in the output:

| Cell | Look for |
|---|---|
| build | the `added to train:` line from step 5. If you see `hardneg_round1.jsonl is missing`, step 5's push didn't land |
| train | same as round 0; roughly 25% more rows, so a bit longer |
| eval | the T1–T7 table, then **"Trade-off: thresholds picked on validation at other target FPRs"** |
| mine | `round 2: mined 3000` (for a possible round 2) |
| export | `int8_vs_fp32` numbers, then `"shipped": "int8"` or `"fp32"`, then a torch-vs-ONNX check on 200 texts |
| eval (shipped ONNX) | a second report, `round1_onnx.md`, scored with the exact file the Action will use (~40 min on CPU) |

Send both reports back: `raw/reports/round1.md` and `raw/reports/round1_onnx.md` in your dataset on the Hub.

**What to expect:** T1 at or under 1% with T2 still above 98%, and T5 lower than 6%. If T5 is still over 5%, the
trade-off table shows whether the 0.5% row would fix it, and what that costs in T2.

## About the ONNX gap (why the export changed)

In round 0, the export check printed a torch-vs-ONNX difference of up to 0.60 in p_machine, but only on two texts.

- **The export itself is correct.** An fp32 ONNX file matches the torch model exactly (zero difference, texts from
  3 to 1,500 words).
- **The gap came from int8 quantization**, and it is real, not just calibration scaling. Trained ModernBERT
  activations have large outliers, and plain dynamic int8 moves scores a lot. On top of that, the old check
  compared fp16 torch on the GPU, on just two texts. Flags agreed on those two, but the Action was using a
  threshold fitted on the torch model, not the file it ships.

What the export does now:
- quantizes only the encoder's matrix multiplies, per channel, and keeps the two heads in full precision
- measures int8 against fp32 on about 1,500 validation texts, and **ships fp32 automatically** if int8 costs more
  than 0.001 AUROC or changes more than 1% of flags (fp32 is about 600 MB instead of about 150 MB, and slower,
  but still well inside the Action's 30-second budget; the report prints the latency)
- **refits the temperature and threshold on the shipped file's own scores**, and stores them in its config
- evaluates the shipped file, so the numbers in `round1_onnx.md` are what users get
