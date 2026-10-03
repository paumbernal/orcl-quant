"""Typed access to the processed datasets so analysis code never touches file paths directly."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from ..config import Config, load_config
from .manifest import Manifest


class MissingDataError(RuntimeError):
    pass


class Store:
    def __init__(self, cfg: Config | None = None):
        self.cfg = cfg or load_config()
        self.dir = self.cfg.processed_dir

    # ---- helpers ----
    def _csv(self, name: str, **kw) -> pd.DataFrame:
        p = self.dir / name
        if not p.exists():
            raise MissingDataError(f"{name} not found - run `orcl-lab build` first (looked in {self.dir})")
        return pd.read_csv(p, **kw)

    def has_data(self) -> bool:
        return (self.dir / "prices_adj_close.csv").exists() and (self.dir / "fundamentals_quarterly.csv").exists()

    # ---- market data ----
    def prices(self) -> pd.DataFrame:
        return self._csv("prices_adj_close.csv", index_col=0, parse_dates=True)

    def closes(self) -> pd.DataFrame:
        return self._csv("prices_close.csv", index_col=0, parse_dates=True)

    def volumes(self) -> pd.DataFrame:
        return self._csv("prices_volume.csv", index_col=0, parse_dates=True)

    def ohlcv(self) -> pd.DataFrame:
        return self._csv(f"ohlcv_{self.cfg.project.ticker}.csv", index_col=0, parse_dates=True)

    def risk_free(self) -> pd.DataFrame:
        return self._csv("risk_free.csv", index_col=0, parse_dates=True)

    # ---- fundamentals ----
    def fundamentals(self) -> pd.DataFrame:
        return self._csv("fundamentals_quarterly.csv", index_col=0, parse_dates=["period_end", "avail_date", "filed", "release_date"])

    def annual(self) -> pd.DataFrame:
        return self._csv("fundamentals_annual.csv", index_col=0, parse_dates=True)

    def events(self) -> pd.DataFrame:
        return self._csv("earnings_events.csv", index_col=0, parse_dates=["period_end", "release_date", "reaction_date"])

    def kpis(self) -> pd.DataFrame:
        return self._csv("press_release_kpis.csv", index_col=0, parse_dates=True)

    # ---- forecasts / meta ----
    def consensus(self) -> dict:
        p = self.dir / "consensus_snapshot.json"
        if not p.exists():
            raise MissingDataError("consensus_snapshot.json not found - run `orcl-lab build`")
        return json.loads(p.read_text(encoding="utf-8"))

    def manifest(self) -> Manifest:
        return Manifest(self.cfg.path("data", "manifest.json"))

    def processed(self, name: str, **kw) -> pd.DataFrame:
        return self._csv(name, **kw)
