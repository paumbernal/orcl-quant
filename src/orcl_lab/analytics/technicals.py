"""Technical indicators - kept deliberately small: each one answers a defined question.

    moving averages (20/50/100/200)  trend regime and its stability
    RSI (Wilder, 14)                 short-horizon overbought / oversold stretch
    MACD (12, 26, 9)                 trend momentum and its turning points
    ATR (Wilder, 14)                 daily price range -> position sizing / stop distance
    volume, relative volume          whether a price move was confirmed by participation

All indicators are computed on dividend-adjusted OHLC so they are consistent with the return series.
Every value at date t uses data up to and including t only.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def moving_averages(close: pd.Series, windows=(20, 50, 100, 200)) -> pd.DataFrame:
    return pd.DataFrame({f"ma_{w}": close.rolling(w, min_periods=w).mean() for w in windows})


def rsi(close: pd.Series, window: int = 14) -> pd.Series:
    """Wilder's RSI. avg gain / avg loss are Wilder-smoothed (EMA with alpha = 1/window)."""
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    avg_gain = gain.ewm(alpha=1.0 / window, adjust=False, min_periods=window).mean()
    avg_loss = loss.ewm(alpha=1.0 / window, adjust=False, min_periods=window).mean()
    rs = avg_gain / avg_loss
    out = 100.0 - 100.0 / (1.0 + rs)
    out = out.where(avg_loss != 0, 100.0)          # no losses at all -> RSI = 100
    out = out.where(~((avg_gain == 0) & (avg_loss == 0)), 50.0)
    return out.rename("rsi")


def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    ema_f = close.ewm(span=fast, adjust=False, min_periods=slow).mean()
    ema_s = close.ewm(span=slow, adjust=False, min_periods=slow).mean()
    line = ema_f - ema_s
    sig = line.ewm(span=signal, adjust=False, min_periods=signal).mean()
    return pd.DataFrame({"macd": line, "macd_signal": sig, "macd_hist": line - sig})


def true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    pc = close.shift(1)
    return pd.concat([high - low, (high - pc).abs(), (low - pc).abs()], axis=1).max(axis=1)


def atr(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
    tr = true_range(high, low, close)
    return tr.ewm(alpha=1.0 / window, adjust=False, min_periods=window).mean().rename("atr")


def relative_volume(volume: pd.Series, window: int = 20) -> pd.Series:
    """Today's volume divided by the average of the *previous* ``window`` sessions (excludes today)."""
    base = volume.rolling(window, min_periods=window).mean().shift(1)
    return (volume / base).rename("rel_volume")


def technical_frame(ohlcv: pd.DataFrame, cfg) -> pd.DataFrame:
    """All indicators for the primary ticker, built on dividend-adjusted OHLC."""
    a = cfg.analytics
    close, high, low = ohlcv["adj_close"], ohlcv["adj_high"], ohlcv["adj_low"]
    out = pd.DataFrame({"close": close, "volume": ohlcv["volume"]})
    out = out.join(moving_averages(close, list(a.ma_windows)))
    out["rsi"] = rsi(close, a.rsi_window)
    out = out.join(macd(close, a.macd.fast, a.macd.slow, a.macd.signal))
    out["atr"] = atr(high, low, close, a.atr_window)
    out["atr_pct"] = out["atr"] / close
    out["rel_volume"] = relative_volume(ohlcv["volume"], a.rel_volume_window)
    out["pct_from_52w_high"] = close / close.rolling(252, min_periods=60).max() - 1.0
    out["pct_from_52w_low"] = close / close.rolling(252, min_periods=60).min() - 1.0
    out["above_200dma"] = (close > out["ma_200"]).astype(float).where(out["ma_200"].notna())
    out["ma50_over_ma200"] = (out["ma_50"] > out["ma_200"]).astype(float).where(out["ma_200"].notna())
    return out


def trend_state(tech: pd.DataFrame) -> dict[str, object]:
    """Plain-language read of the latest bar - used by the dashboard and the report."""
    t = tech.dropna(subset=["ma_200"]).iloc[-1]
    state = {
        "date": tech.index[-1],
        "close": float(t["close"]),
        "above_200dma": bool(t["close"] > t["ma_200"]),
        "above_50dma": bool(t["close"] > t["ma_50"]),
        "ma50_over_ma200": bool(t["ma_50"] > t["ma_200"]),
        "rsi": float(t["rsi"]),
        "macd_hist": float(t["macd_hist"]),
        "atr_pct": float(t["atr_pct"]),
        "rel_volume": float(t["rel_volume"]),
        "pct_from_52w_high": float(t["pct_from_52w_high"]),
    }
    if state["rsi"] >= 70:
        state["rsi_read"] = "overbought (70 or above)"
    elif state["rsi"] <= 30:
        state["rsi_read"] = "oversold (30 or below)"
    else:
        state["rsi_read"] = "in the neutral 30-70 band"
    return state
