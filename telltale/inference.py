"""Scoring API used by eval, the Action and the Space.

Two backends with the same output:
  TorchScorer: a training checkpoint directory
  OnnxScorer:  an exported ONNX (int8) directory — CPU only, no torch needed
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .text import compose, normalize_body, normalize_title

ONNX_FILE = "model.int8.onnx"
META_FILE = "telltale_config.json"


@dataclass
class Score:
    p_machine: float
    flagged: bool
    threshold: float
    closest_family: str | None
    family_prob: float | None
    family_probs: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return self.__dict__.copy()


def _softmax(z: np.ndarray, t: float = 1.0) -> np.ndarray:
    z = z / t
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


class _Base:
    meta: dict

    def logits(self, texts: list[str]) -> tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError

    def score_texts(self, texts: list[str], threshold: float | None = None) -> list[Score]:
        thr = float(self.meta.get("threshold", 0.5) if threshold is None else threshold)
        fams = self.meta["families"]
        la, lb = self.logits(texts)
        pa = _softmax(la, float(self.meta.get("temperature", 1.0)))[:, 1]
        pb = _softmax(lb)
        out = []
        for p, fb in zip(pa, pb):
            probs = {f: float(x) for f, x in zip(fams, fb)}
            flagged = bool(p >= thr)
            top = int(fb.argmax()) if len(fams) else None
            out.append(Score(float(p), flagged, thr,
                             fams[top] if flagged and top is not None else None,
                             float(fb[top]) if flagged and top is not None else None, probs))
        return out

    def score(self, title: str, body: str, threshold: float | None = None) -> Score:
        return self.score_texts([compose(title, body)], threshold)[0]


class TorchScorer(_Base):
    def __init__(self, model_dir: str | Path, device: str = "cpu"):
        from .train.model import load

        self.model, self.tok, self.meta = load(model_dir, device)
        self.device = device

    def logits(self, texts):
        from .train.train import predict

        return predict(self.model, self.tok, texts, self.meta["max_length"], device=self.device)


class OnnxScorer(_Base):
    def __init__(self, model_dir: str | Path, onnx_file: str = ONNX_FILE, threads: int | None = None):
        import onnxruntime as ort
        from tokenizers import Tokenizer

        d = Path(model_dir)
        self.meta = json.loads((d / META_FILE).read_text())
        so = ort.SessionOptions()
        if threads:
            so.intra_op_num_threads = threads
        self.sess = ort.InferenceSession(str(d / onnx_file), so, providers=["CPUExecutionProvider"])
        self.tok = Tokenizer.from_file(str(d / "tokenizer.json"))
        self.max_len = int(self.meta["max_length"])
        self.tok.enable_truncation(self.max_len)
        pad_id = self.tok.token_to_id(self.meta.get("pad_token", "[PAD]"))
        self.tok.enable_padding(pad_id=pad_id if pad_id is not None else 0)
        self.input_names = {i.name for i in self.sess.get_inputs()}

    def logits(self, texts):
        las, lbs = [], []
        for t in texts:  # batch of 1 keeps padding free; the Action scores one text
            enc = self.tok.encode(t)
            feed = {"input_ids": np.array([enc.ids], dtype=np.int64),
                    "attention_mask": np.array([enc.attention_mask], dtype=np.int64)}
            a, b = self.sess.run(None, {k: v for k, v in feed.items() if k in self.input_names})
            las.append(a[0])
            lbs.append(b[0])
        return np.stack(las), np.stack(lbs)


def load_scorer(model_dir: str | Path) -> _Base:
    d = Path(model_dir)
    return OnnxScorer(d) if (d / ONNX_FILE).exists() else TorchScorer(d)


__all__ = ["Score", "TorchScorer", "OnnxScorer", "load_scorer", "normalize_title", "normalize_body"]
