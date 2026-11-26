"""Load the YAML configs in configs/."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = ROOT / "configs"


def load_yaml(name: str | Path) -> dict[str, Any]:
    p = Path(name)
    if not p.is_absolute() and not p.exists():
        p = CONFIG_DIR / p
    with open(p) as f:
        return yaml.safe_load(f) or {}


@dataclass
class ModelSpec:
    key: str            # short family name, e.g. "deepseek"
    model: str          # HF repo id
    provider: str
    role: str           # topic | mirror | heldout | polish
    price_in: float     # $ per 1M input tokens
    price_out: float    # $ per 1M output tokens
    enabled: bool = True
    license: str = ""
    license_ok: bool = False
    license_note: str = ""
    quota: int | None = None
    extra_body: dict[str, Any] = field(default_factory=dict)
    reasoning_effort: str | None = None

    @property
    def route(self) -> str:
        """The router model string; ':provider' pins provider and price."""
        return f"{self.model}:{self.provider}" if self.provider else self.model


def load_models(path: str | Path = "models.yaml", prices: str | Path = "prices.yaml") -> dict[str, ModelSpec]:
    cfg = load_yaml(path)
    price_tbl = load_yaml(prices).get("prices", {})
    out: dict[str, ModelSpec] = {}
    for key, m in cfg["models"].items():
        p = price_tbl.get(m["model"], {})
        out[key] = ModelSpec(
            key=key,
            model=m["model"],
            provider=m.get("provider", p.get("provider", "")),
            role=m["role"],
            price_in=float(p.get("in", m.get("price_in", 0.0))),
            price_out=float(p.get("out", m.get("price_out", 0.0))),
            enabled=m.get("enabled", True),
            license=m.get("license", ""),
            license_ok=m.get("license_ok", False),
            license_note=m.get("license_note", ""),
            quota=m.get("quota"),
            extra_body=m.get("extra_body", {}) or {},
            reasoning_effort=m.get("reasoning_effort"),
        )
    return out


def training_families(models: dict[str, ModelSpec]) -> list[str]:
    """Families that feed Head B: enabled mirror models with an OK license.

    Order is stable (config order) because it defines Head B's label ids.
    """
    return [k for k, m in models.items() if m.role == "mirror" and m.enabled and m.license_ok]


def heldout_families(models: dict[str, ModelSpec]) -> list[str]:
    return [k for k, m in models.items() if m.role == "heldout" and m.enabled and m.license_ok]
