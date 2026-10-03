"""Configuration loader.

All tunable numbers live in ``config.yaml`` at the repository root. This module exposes them
through a small attribute-access wrapper and resolves project paths so every other module can
stay free of hard-coded locations.
"""
from __future__ import annotations

import os
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(os.environ.get("ORCL_LAB_ROOT", Path(__file__).resolve().parents[2]))
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config.yaml"


class Config:
    """Read-only nested view over a dict supporting ``cfg.valuation.explicit_years`` access."""

    def __init__(self, data: dict[str, Any], root: Path = PROJECT_ROOT):
        object.__setattr__(self, "_data", data)
        object.__setattr__(self, "root", Path(root))

    # -- attribute / item access -------------------------------------------------------------
    def __getattr__(self, name: str) -> Any:
        try:
            value = self._data[name]
        except KeyError as exc:
            raise AttributeError(f"config has no key '{name}'") from exc
        return Config(value, self.root) if isinstance(value, dict) else value

    def __getitem__(self, key: str) -> Any:
        value = self._data[key]
        return Config(value, self.root) if isinstance(value, dict) else value

    def __contains__(self, key: str) -> bool:
        return key in self._data

    def get(self, key: str, default: Any = None) -> Any:
        return self[key] if key in self._data else default

    def to_dict(self) -> dict[str, Any]:
        return deepcopy(self._data)

    def with_overrides(self, **overrides: Any) -> "Config":
        """Return a copy with top-level sections shallow-merged (handy in tests / the dashboard)."""
        data = deepcopy(self._data)
        for key, value in overrides.items():
            if isinstance(value, dict) and isinstance(data.get(key), dict):
                data[key] = {**data[key], **value}
            else:
                data[key] = value
        return Config(data, self.root)

    # -- project paths -----------------------------------------------------------------------
    def path(self, *parts: str, create: bool = False) -> Path:
        p = self.root.joinpath(*parts)
        if create:
            p.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def raw_dir(self) -> Path:
        return self.path("data", "raw", create=True)

    @property
    def processed_dir(self) -> Path:
        return self.path("data", "processed", create=True)

    @property
    def manual_dir(self) -> Path:
        return self.path("data", "manual", create=True)

    @property
    def models_dir(self) -> Path:
        return self.path("models", create=True)

    @property
    def charts_dir(self) -> Path:
        return self.path("visualisations", create=True)

    @property
    def reports_dir(self) -> Path:
        return self.path("reports", create=True)


def load_config(path: str | os.PathLike | None = None) -> Config:
    cfg_path = Path(path) if path else DEFAULT_CONFIG_PATH
    with open(cfg_path, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    cfg = Config(data, cfg_path.parent)
    validate_config(cfg)
    return cfg


def validate_config(cfg: Config) -> None:
    """Fail fast on configuration mistakes that would silently corrupt results."""
    weights = cfg.score.weights.to_dict()
    total = sum(weights.values())
    if abs(total - 1.0) > 1e-9:
        raise ValueError(f"score.weights must sum to 1.0 (got {total:.6f})")
    if any(w < 0 for w in weights.values()):
        raise ValueError("score.weights must be non-negative")
    probs = cfg.valuation.scenario_probabilities.to_dict()
    if abs(sum(probs.values()) - 1.0) > 1e-9:
        raise ValueError("valuation.scenario_probabilities must sum to 1.0")
    mw = cfg.valuation.method_weights.to_dict()
    if abs(sum(mw.values()) - 1.0) > 1e-9:
        raise ValueError("valuation.method_weights must sum to 1.0")
    if cfg.risk.monte_carlo.n_sims < 10_000:
        raise ValueError("risk.monte_carlo.n_sims must be >= 10,000 (project requirement)")
    for name, g in cfg.valuation.terminal_growth.to_dict().items():
        if not (0.0 <= g < 0.05):
            raise ValueError(f"terminal growth '{name}'={g} outside the sane range [0, 5%)")
    lo, hi = cfg.event_study.estimation_window
    if not (lo < hi < 0):
        raise ValueError("event_study.estimation_window must be a negative [start, end] pair")
