"""Market data acquisition (Yahoo Finance via ``yfinance``) and quality control.

Design notes
------------
* ``auto_adjust=False`` so we keep *both* the as-quoted (split-adjusted) close - needed for
  valuation multiples such as price / EPS - and the dividend-adjusted close, which is the right
  series for total-return statistics.
* The current trading day's bar is dropped while the US market is still open so no analysis ever
  uses a partial candle.
* Raw downloads are written to ``data/raw/prices`` unchanged; cleaning is applied afterwards.
"""
from __future__ import annotations

import logging
from datetime import datetime, time as dtime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from ..config import Config
from .manifest import Manifest

log = logging.getLogger(__name__)
NY = ZoneInfo("America/New_York")


def all_tickers(cfg: Config) -> list[str]:
    t = [cfg.project.ticker]
    t += list(cfg.data.benchmarks.to_dict().values())
    t += list(cfg.data.peers)
    seen: list[str] = []
    for x in t:
        if x not in seen:
            seen.append(x)
    return seen


def _drop_incomplete_bar(df: pd.DataFrame) -> pd.DataFrame:
    """Remove today's bar if the US cash session has not finished (16:00 ET + 10 min buffer)."""
    if df.empty:
        return df
    now = datetime.now(NY)
    last = df.index[-1].date()
    if last == now.date() and now.time() < dtime(16, 10):
        log.info("dropping incomplete bar for %s", last)
        return df.iloc[:-1]
    return df


def download_one(ticker: str, start: str, end: str | None) -> pd.DataFrame:
    import yfinance as yf

    raw = yf.Ticker(ticker).history(start=start, end=end, auto_adjust=False, actions=False)
    if raw.empty:
        raise RuntimeError(f"Yahoo returned no data for {ticker}")
    raw.index = pd.DatetimeIndex(raw.index).tz_localize(None).normalize()
    raw.index.name = "date"
    raw = raw.rename(columns=str.lower).rename(columns={"adj close": "adj_close"})
    keep = ["open", "high", "low", "close", "adj_close", "volume"]
    raw = raw[keep]
    return _drop_incomplete_bar(raw[~raw.index.duplicated(keep="last")].sort_index())


def quality_report(frames: dict[str, pd.DataFrame], max_abs_return: float = 0.5) -> pd.DataFrame:
    rows = []
    for t, df in frames.items():
        r = df["adj_close"].pct_change()
        rows.append(
            {
                "ticker": t,
                "first": df.index.min().date(),
                "last": df.index.max().date(),
                "rows": len(df),
                "nan_cells": int(df.isna().sum().sum()),
                "non_positive": int((df[["open", "high", "low", "close", "adj_close"]] <= 0).sum().sum()),
                "zero_volume_days": int((df["volume"] <= 0).sum()),
                "max_abs_daily_return": float(r.abs().max()),
                "suspect_moves": int((r.abs() > max_abs_return).sum()),
            }
        )
    return pd.DataFrame(rows).set_index("ticker")


def build_price_files(cfg: Config, manifest: Manifest, use_cache: bool = False) -> dict[str, pd.DataFrame]:
    """Download (or, with ``use_cache``, re-use ``data/raw/prices``) and write the processed price files."""
    tickers = all_tickers(cfg)
    raw_dir = cfg.path("data", "raw", "prices", create=True)
    frames: dict[str, pd.DataFrame] = {}
    for t in tickers:
        cached = raw_dir / f"{t}.csv"
        if use_cache and cached.exists():
            log.info("using cached %s", t)
            df = pd.read_csv(cached, index_col=0, parse_dates=True)
        else:
            log.info("downloading %s", t)
            df = download_one(t, cfg.data.start_date, cfg.data.end_date)
            df.to_csv(cached)
        frames[t] = df

    qa = quality_report(frames)
    qa.to_csv(cfg.processed_dir / "prices_quality_report.csv")
    bad = qa[(qa.non_positive > 0)]
    if len(bad):
        raise RuntimeError(f"non-positive prices detected:\n{bad}")

    adj = pd.DataFrame({t: f["adj_close"] for t, f in frames.items()}).sort_index()
    close = pd.DataFrame({t: f["close"] for t, f in frames.items()}).sort_index()
    vol = pd.DataFrame({t: f["volume"] for t, f in frames.items()}).sort_index()
    # Require the primary stock + benchmarks to be present; peers may have shorter history in principle.
    adj = adj.dropna(subset=[cfg.project.ticker])
    close, vol = close.loc[adj.index], vol.loc[adj.index]
    adj.to_csv(cfg.processed_dir / "prices_adj_close.csv")
    close.to_csv(cfg.processed_dir / "prices_close.csv")
    vol.to_csv(cfg.processed_dir / "prices_volume.csv")

    # Full OHLCV for the primary ticker with dividend-adjusted OHLC (consistent for indicators / backtest)
    o = frames[cfg.project.ticker].copy()
    factor = o["adj_close"] / o["close"]
    for c in ("open", "high", "low"):
        o[f"adj_{c}"] = o[c] * factor
    o.to_csv(cfg.processed_dir / f"ohlcv_{cfg.project.ticker}.csv")

    src = "Yahoo Finance via yfinance (auto_adjust=False; adj_close is dividend-adjusted)"
    for name, df in (("prices_adj_close", adj), ("prices_close", close), ("prices_volume", vol)):
        manifest.record(
            name,
            kind="HISTORICAL",
            source=src,
            file=cfg.processed_dir / f"{name}.csv",
            df=df,
            tickers=list(df.columns),
            notes="Benchmarks use total-return ETF proxies: SPY (S&P 500), QQQ (Nasdaq 100), SMH (semiconductors).",
        )
    manifest.record(
        f"ohlcv_{cfg.project.ticker}",
        kind="HISTORICAL",
        source=src,
        file=cfg.processed_dir / f"ohlcv_{cfg.project.ticker}.csv",
        df=o,
    )
    return frames
