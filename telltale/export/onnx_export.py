"""M8: ONNX export + dynamic int8 quantization, and a CPU latency test."""
from __future__ import annotations

import json
import shutil
import time
from pathlib import Path

import numpy as np
import torch

from ..inference import META_FILE, ONNX_FILE, OnnxScorer


class _Wrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids, attention_mask):
        return self.model(input_ids, attention_mask)


def export(model_dir: str | Path, out_dir: str | Path, opset: int = 17, quantize: bool = True) -> Path:
    from ..train.model import load

    model, tok, meta = load(model_dir, "cpu")
    model.eval()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    fp32 = out / "model.fp32.onnx"
    sample = tok(["hello world " * 300], return_tensors="pt", truncation=True, max_length=meta["max_length"])
    with torch.no_grad():
        torch.onnx.export(
            _Wrapper(model), (sample["input_ids"], sample["attention_mask"]), str(fp32),
            input_names=["input_ids", "attention_mask"], output_names=["logits_machine", "logits_family"],
            dynamic_axes={"input_ids": {0: "batch", 1: "seq"}, "attention_mask": {0: "batch", 1: "seq"},
                          "logits_machine": {0: "batch"}, "logits_family": {0: "batch"}},
            opset_version=opset, dynamo=False,
        )
    tok.save_pretrained(out)
    meta = dict(meta)
    meta["pad_token"] = tok.pad_token
    (out / META_FILE).write_text(json.dumps(meta, indent=2, default=float))
    if quantize:
        from onnxruntime.quantization import QuantType, quantize_dynamic

        quantize_dynamic(str(fp32), str(out / ONNX_FILE), weight_type=QuantType.QInt8)
        fp32.unlink()
    else:
        shutil.move(fp32, out / ONNX_FILE)
    return out


def parity(model_dir: str | Path, onnx_dir: str | Path, texts: list[str]) -> dict:
    """Compare torch vs ONNX p_machine on a few texts (int8 drift check)."""
    from ..inference import TorchScorer

    t = TorchScorer(model_dir).score_texts(texts)
    o = OnnxScorer(onnx_dir).score_texts(texts)
    diff = np.abs(np.array([a.p_machine for a in t]) - np.array([b.p_machine for b in o]))
    agree = np.mean([a.flagged == b.flagged for a, b in zip(t, o)])
    return {"max_abs_diff": float(diff.max()), "mean_abs_diff": float(diff.mean()), "flag_agreement": float(agree)}


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
            "model_mb": round(size_mb, 1), "n_words": n_words}
