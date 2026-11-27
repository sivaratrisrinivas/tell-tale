"""M8: ONNX export, int8 quantization with a drift check, threshold refit on the shipped model, CPU latency.

Why the drift check: an fp32 ONNX graph matches torch exactly (tested at 5-900 words), but dynamic int8
quantization of a *trained* ModernBERT can move scores a lot, because its activations have large outliers.
So export measures the int8 model against fp32 on the validation set, falls back to fp32 when int8 changes
rankings, and always refits temperature + threshold on the scores of the model it actually ships.
"""
from __future__ import annotations

import json
import shutil
import time
from pathlib import Path

import numpy as np
import torch

from ..inference import META_FILE, ONNX_FILE, OnnxScorer

FP32_FILE = "model.fp32.onnx"


class _Wrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids, attention_mask):
        return self.model(input_ids, attention_mask)


def export_fp32(model_dir: str | Path, out_dir: str | Path, opset: int = 17) -> Path:
    from ..train.model import load

    model, tok, meta = load(model_dir, "cpu")
    model.eval()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    sample = tok(["hello world " * 300], return_tensors="pt", truncation=True, max_length=meta["max_length"])
    with torch.no_grad():
        torch.onnx.export(
            _Wrapper(model), (sample["input_ids"], sample["attention_mask"]), str(out / FP32_FILE),
            input_names=["input_ids", "attention_mask"], output_names=["logits_machine", "logits_family"],
            dynamic_axes={"input_ids": {0: "batch", 1: "seq"}, "attention_mask": {0: "batch", 1: "seq"},
                          "logits_machine": {0: "batch"}, "logits_family": {0: "batch"}},
            opset_version=opset, dynamo=False,
        )
    tok.save_pretrained(out)
    meta = dict(meta)
    meta["pad_token"] = tok.pad_token
    (out / META_FILE).write_text(json.dumps(meta, indent=2, default=float))
    return out / FP32_FILE


def quantize_int8(fp32: Path, dst: Path) -> Path:
    """Per-channel int8 weights for the encoder's MatMuls only. The two heads (Gemm) and the embeddings stay
    fp32: they are tiny, and quantizing them is where most of the score drift comes from."""
    from onnxruntime.quantization import QuantType, quantize_dynamic

    quantize_dynamic(str(fp32), str(dst), weight_type=QuantType.QInt8, per_channel=True,
                     op_types_to_quantize=["MatMul"])
    return dst


def _auroc(y: np.ndarray, p: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score

    return float(roc_auc_score(y, p)) if len(set(y)) > 1 else float("nan")


def drift_report(ref_logits: np.ndarray, logits: np.ndarray, y: np.ndarray, threshold_ref: float,
                 temperature: float) -> dict:
    """How much a quantized/exported model moves validation scores relative to a reference."""
    from ..train.calibrate import p_machine

    pr, po = p_machine(ref_logits, temperature), p_machine(logits, temperature)
    d = np.abs(pr - po)
    return {"n": int(len(y)), "max_abs_diff": float(d.max()), "mean_abs_diff": float(d.mean()),
            "p99_abs_diff": float(np.quantile(d, 0.99)),
            "flag_agreement_at_ref_threshold": float(np.mean((pr >= threshold_ref) == (po >= threshold_ref))),
            "auroc_ref": _auroc(y, pr), "auroc": _auroc(y, po)}


def refit(meta: dict, logits: np.ndarray, y: np.ndarray, target_fpr: float) -> dict:
    """Temperature + threshold fitted on the shipped model's own validation logits."""
    from ..train.calibrate import fit_temperature, p_machine, threshold_for_fpr

    T = fit_temperature(logits, y)
    p = p_machine(logits, T)
    m = dict(meta)
    m.update({"temperature": T, "threshold": threshold_for_fpr(p[y == 0], target_fpr), "target_fpr": target_fpr,
              "torch_temperature": meta.get("temperature"), "torch_threshold": meta.get("threshold")})
    return m


def export(model_dir: str | Path, out_dir: str | Path, val_texts: list[str] | None = None,
           val_labels: np.ndarray | None = None, opset: int = 17, max_auroc_drop: float = 0.001,
           min_flag_agreement: float = 0.99, mode: str = "auto") -> dict:
    """Export, pick int8 or fp32, refit the threshold on the shipped file. mode: auto | int8 | fp32.

    Without validation texts it ships int8 with the training threshold (the old behaviour).
    """
    out = Path(out_dir)
    fp32 = export_fp32(model_dir, out, opset)
    meta = json.loads((out / META_FILE).read_text())
    int8 = quantize_int8(fp32, out / "model.q8.onnx") if mode in ("auto", "int8") else None
    report: dict = {"mode": mode}
    chosen = int8 if int8 is not None else fp32
    if val_texts is not None and len(val_texts):
        y = np.asarray(val_labels).astype(int)
        t0 = time.perf_counter()
        ref = OnnxScorer(out, onnx_file=FP32_FILE).logits(val_texts)[0]
        report["fp32_seconds"] = round(time.perf_counter() - t0, 1)
        T, thr = float(meta.get("temperature", 1.0)), float(meta.get("threshold", 0.5))
        if int8 is not None:
            q = OnnxScorer(out, onnx_file=int8.name).logits(val_texts)[0]
            report["int8_vs_fp32"] = drift_report(ref, q, y, thr, T)
            r = report["int8_vs_fp32"]
            ok = (r["auroc_ref"] - r["auroc"] <= max_auroc_drop
                  and r["flag_agreement_at_ref_threshold"] >= min_flag_agreement)
            if mode == "auto" and not ok:
                print("int8 changes validation scores too much; shipping fp32 instead")
                chosen = fp32
            shipped = q if chosen == int8 else ref
        else:
            shipped = ref
        meta = refit(meta, shipped, y, float(meta.get("target_fpr", 0.01)))
        report.update({"shipped": "int8" if chosen == int8 else "fp32",
                       "temperature": meta["temperature"], "threshold": meta["threshold"],
                       "torch_threshold": meta["torch_threshold"]})
    else:
        report["shipped"] = "int8" if chosen == int8 else "fp32"
    meta["onnx_precision"] = report["shipped"]
    meta["export_report"] = report
    (out / META_FILE).write_text(json.dumps(meta, indent=2, default=float))
    # The scorers always read ONNX_FILE; keep only the shipped weights so the Action downloads one file.
    shutil.move(str(chosen), out / ONNX_FILE)
    for f in (fp32, int8):
        if f is not None and f.exists():
            f.unlink()
    return report


def parity(model_dir: str | Path, onnx_dir: str | Path, texts: list[str], device: str = "cpu") -> dict:
    """Torch (fp32, CPU by default) vs the exported file, raw logits -> p_machine with the torch calibration.

    Compare on CPU: on a GPU, TorchScorer runs fp16 autocast, which is a third source of difference.
    """
    from ..inference import TorchScorer

    ts = TorchScorer(model_dir, device=device)
    os_ = OnnxScorer(onnx_dir)
    la_t, la_o = ts.logits(texts)[0], os_.logits(texts)[0]
    T = float(ts.meta.get("temperature", 1.0))
    r = drift_report(la_t, la_o, np.zeros(len(texts), dtype=int), float(ts.meta.get("threshold", 0.5)), T)
    r.pop("auroc_ref"), r.pop("auroc")
    r["max_logit_diff"] = float(np.abs(la_t - la_o).max())
    return r


def latency(onnx_dir: str | Path, n_words: int = 400, runs: int = 5) -> dict:
    """Cold load + per-text latency on CPU. Spec budget: whole Action run < 30 s."""
    t0 = time.perf_counter()
    s = OnnxScorer(onnx_dir)
    load_s = time.perf_counter() - t0
    text = ("This change fixes a crash in the parser when the input has a trailing comma. " * 40)
    text = " ".join(text.split()[:n_words])
    times = []
    for _ in range(runs):
        t1 = time.perf_counter()
        s.score("Fix parser crash", text)
        times.append(time.perf_counter() - t1)
    size_mb = (Path(onnx_dir) / ONNX_FILE).stat().st_size / 1e6
    return {"load_s": round(load_s, 3), "score_s_median": round(float(np.median(times)), 3),
            "model_mb": round(size_mb, 1), "n_words": n_words,
            "precision": s.meta.get("onnx_precision", "int8")}
