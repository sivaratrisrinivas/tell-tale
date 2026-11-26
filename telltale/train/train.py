"""Training loop (M4 / M6). Runs on Kaggle's free T4/P100, or CPU for smoke tests."""
from __future__ import annotations

import json
import math
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from .calibrate import fit_temperature, p_machine, threshold_for_fpr
from .model import TwoHeadModel, build, save, two_head_loss


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class TextDS(Dataset):
    def __init__(self, df: pd.DataFrame):
        self.text = df["text"].tolist()
        self.label = df["label"].astype(int).tolist()
        self.family = df["family"].astype(int).tolist()

    def __len__(self) -> int:
        return len(self.text)

    def __getitem__(self, i: int) -> tuple[str, int, int]:
        return self.text[i], self.label[i], self.family[i]


def make_collate(tokenizer, max_length: int):
    def collate(batch):
        texts, labels, fams = zip(*batch)
        enc = tokenizer(list(texts), truncation=True, max_length=max_length, padding=True, return_tensors="pt")
        fam = torch.tensor(fams)  # -1 for human / held-out rows: masked out of CE_B
        return enc["input_ids"], enc["attention_mask"], torch.tensor(labels), fam
    return collate


@torch.no_grad()
def predict(model: TwoHeadModel, tokenizer, texts: list[str], max_length: int, batch_size: int = 32,
            device: str = "cpu") -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    la, lb = [], []
    use_amp = device.startswith("cuda")
    # sort by length for speed, then unsort
    order = np.argsort([len(t) for t in texts])
    for i in range(0, len(texts), batch_size):
        chunk = [texts[j] for j in order[i:i + batch_size]]
        enc = tokenizer(chunk, truncation=True, max_length=max_length, padding=True, return_tensors="pt").to(device)
        with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
            a, b = model(enc["input_ids"], enc["attention_mask"])
        la.append(a.float().cpu().numpy())
        lb.append(b.float().cpu().numpy())
    la_arr, lb_arr = np.concatenate(la), np.concatenate(lb)
    inv = np.empty_like(order)
    inv[order] = np.arange(len(order))
    return la_arr[inv], lb_arr[inv]


def val_auroc(model, tokenizer, val: pd.DataFrame, max_length: int, device: str) -> tuple[float, np.ndarray, np.ndarray]:
    la, lb = predict(model, tokenizer, val["text"].tolist(), max_length, device=device)
    y = val["label"].to_numpy()
    auc = roc_auc_score(y, p_machine(la)) if len(set(y)) > 1 else float("nan")
    return auc, la, lb


def train(table: pd.DataFrame, families: list[str], cfg: dict[str, Any], out_dir: str | Path,
          base_model: str | None = None, device: str | None = None, max_steps: int | None = None,
          log_every: int = 50, enc_kwargs: dict | None = None) -> dict[str, Any]:
    """Train from the base model (never from a previous checkpoint: spec M6 step 5)."""
    set_seed(cfg["seed"])
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    base = base_model or cfg["base_model"]
    max_len = cfg["max_length"]
    tr = table[table["split"] == "train"].reset_index(drop=True)
    va = table[table["split"] == "val"].reset_index(drop=True)
    print(f"train rows {len(tr)} (machine {tr['label'].mean():.1%}), val rows {len(va)}; device {device}")

    tok = AutoTokenizer.from_pretrained(base)
    model = build(base, len(families), **(enc_kwargs or {})).to(device)
    bs, accum = cfg["per_device_batch_size"], cfg["grad_accum"]
    dl = DataLoader(TextDS(tr), batch_size=bs, shuffle=True, collate_fn=make_collate(tok, max_len),
                    generator=torch.Generator().manual_seed(cfg["seed"]))
    steps_per_epoch = math.ceil(len(dl) / accum)
    total = steps_per_epoch * cfg["epochs"]
    if max_steps:
        total = min(total, max_steps)
    no_decay = ["bias", "norm", "LayerNorm"]
    groups = [
        {"params": [p for n, p in model.named_parameters() if not any(k in n for k in no_decay)],
         "weight_decay": cfg["weight_decay"]},
        {"params": [p for n, p in model.named_parameters() if any(k in n for k in no_decay)], "weight_decay": 0.0},
    ]
    opt = torch.optim.AdamW(groups, lr=cfg["lr"])
    sched = get_linear_schedule_with_warmup(opt, int(cfg["warmup_ratio"] * total), total)
    use_amp = bool(cfg.get("fp16")) and device.startswith("cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    eval_every = max(1, int(steps_per_epoch * cfg.get("eval_every_epochs", 0.5)))

    best = {"auroc": -1.0, "step": 0}
    history = []
    step, t0 = 0, time.time()
    out_dir = Path(out_dir)
    best_dir = out_dir / "best"
    model.train()
    done = False
    for epoch in range(cfg["epochs"]):
        for i, (ids, mask, y, fam) in enumerate(dl):
            ids, mask, y, fam = ids.to(device), mask.to(device), y.to(device), fam.to(device)
            with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
                la, lb = model(ids, mask)
            loss = two_head_loss(la, lb, y, fam, cfg["family_loss_weight"]) / accum
            scaler.scale(loss).backward()
            if (i + 1) % accum and (i + 1) != len(dl):
                continue
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            opt.zero_grad(set_to_none=True)
            sched.step()
            step += 1
            if step % log_every == 0:
                print(f"epoch {epoch} step {step}/{total} loss {loss.item() * accum:.4f} {time.time() - t0:.0f}s")
            if step % eval_every == 0 or step == total:
                auc, _, _ = val_auroc(model, tok, va, max_len, device)
                history.append({"step": step, "epoch": epoch, "val_auroc": auc})
                print(f"  eval step {step}: val AUROC {auc:.5f}")
                if auc > best["auroc"]:
                    best = {"auroc": auc, "step": step}
                    save(model, tok, best_dir, {"families": families, "max_length": max_len,
                                                "temperature": 1.0, "threshold": 0.5, "base_model": base})
                model.train()
            if step >= total:
                done = True
                break
        if done:
            break

    # calibrate the best checkpoint on validation
    from .model import load
    model, tok, meta = load(best_dir, device)
    _, la, lb = val_auroc(model, tok, va, max_len, device)
    y = va["label"].to_numpy()
    T = fit_temperature(la, y)
    p = p_machine(la, T)
    thr = threshold_for_fpr(p[y == 0], cfg["target_fpr"])
    meta.update({"temperature": T, "threshold": thr, "val_auroc": best["auroc"], "best_step": best["step"],
                 "target_fpr": cfg["target_fpr"], "train_rows": len(tr), "history": history})
    save(model, tok, best_dir, meta)
    (out_dir / "train_log.json").write_text(json.dumps(meta, indent=2, default=float))
    print(f"best val AUROC {best['auroc']:.5f} at step {best['step']}; T={T:.3f}; threshold={thr:.4f}")
    return meta
