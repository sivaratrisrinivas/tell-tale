"""Temperature scaling on validation logits, then a threshold for a target FPR."""
from __future__ import annotations

import numpy as np
import torch


def fit_temperature(logits: np.ndarray, labels: np.ndarray, max_iter: int = 200) -> float:
    """logits: (N, 2) Head A logits. Returns T > 0 minimizing NLL."""
    lg = torch.tensor(logits, dtype=torch.float64)
    y = torch.tensor(labels, dtype=torch.long)
    log_t = torch.zeros(1, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=max_iter)

    def closure():
        opt.zero_grad()
        loss = torch.nn.functional.cross_entropy(lg / log_t.exp(), y)
        loss.backward()
        return loss

    opt.step(closure)
    return float(log_t.exp().item())


def p_machine(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    z = logits / temperature
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e[:, 1] / e.sum(axis=1)


def threshold_for_fpr(p_human: np.ndarray, target_fpr: float) -> float:
    """Smallest threshold t with FPR = mean(p_human >= t) <= target."""
    if len(p_human) == 0:
        return 0.5
    s = np.sort(p_human)[::-1]
    k = int(np.floor(target_fpr * len(s)))   # number of humans we may flag
    if k >= len(s):
        return float(s[-1])
    # flag strictly above the (k+1)-th highest score
    return float(np.nextafter(s[k], 1.0))
