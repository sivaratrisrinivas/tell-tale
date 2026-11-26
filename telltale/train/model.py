"""Shared encoder + two heads.

Head A: 2-way human vs machine.
Head B: N-way model family (N = number of training families). Its loss only
counts on machine rows (family label -100 elsewhere).
Total loss = CE_A + w * CE_B, w = 0.3 by default.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from torch import nn
from transformers import AutoConfig, AutoModel, AutoTokenizer

TELLTALE_CONFIG = "telltale_config.json"
WEIGHTS = "telltale_heads_and_encoder.pt"


class TwoHeadModel(nn.Module):
    def __init__(self, encoder: nn.Module, n_families: int, dropout: float = 0.1):
        super().__init__()
        self.encoder = encoder
        h = encoder.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.head_a = nn.Linear(h, 2)
        self.head_b = nn.Linear(h, n_families)

    def pool(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        hs = out.last_hidden_state
        m = attention_mask.unsqueeze(-1).to(hs.dtype)
        return (hs * m).sum(1) / m.sum(1).clamp(min=1.0)   # masked mean pooling

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.dropout(self.pool(input_ids, attention_mask))
        return self.head_a(z), self.head_b(z)


def two_head_loss(logits_a, logits_b, labels, families, family_weight: float = 0.3) -> torch.Tensor:
    ce_a = nn.functional.cross_entropy(logits_a.float(), labels)
    mask = families >= 0
    if mask.any():
        ce_b = nn.functional.cross_entropy(logits_b.float()[mask], families[mask])
    else:
        ce_b = logits_b.sum() * 0.0
    return ce_a + family_weight * ce_b


def build(base_model: str, n_families: int, **enc_kwargs: Any) -> TwoHeadModel:
    enc = AutoModel.from_pretrained(base_model, **enc_kwargs)
    return TwoHeadModel(enc, n_families)


def save(model: TwoHeadModel, tokenizer, out_dir: str | Path, meta: dict[str, Any]) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    model.encoder.config.save_pretrained(out)
    tokenizer.save_pretrained(out)
    torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()}, out / WEIGHTS)
    (out / TELLTALE_CONFIG).write_text(json.dumps(meta, indent=2))
    return out


def load(model_dir: str | Path, device: str = "cpu") -> tuple[TwoHeadModel, Any, dict[str, Any]]:
    d = Path(model_dir)
    meta = json.loads((d / TELLTALE_CONFIG).read_text())
    cfg = AutoConfig.from_pretrained(d)
    enc = AutoModel.from_config(cfg)
    model = TwoHeadModel(enc, len(meta["families"]))
    model.load_state_dict(torch.load(d / WEIGHTS, map_location=device))
    model.to(device).eval()
    tok = AutoTokenizer.from_pretrained(d)
    return model, tok, meta
