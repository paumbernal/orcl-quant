"""Run every analysis once and bundle the results.

``run_all_analyses`` is the single place where the pipeline's outputs are turned into research results. The dashboard, the
static charts and the written report all read the same ``AnalysisBundle`` (pickled to ``models/analysis_bundle.pkl``), so the
three can never disagree. Key tables are also written as CSV to ``models/`` for inspection on GitHub.
"""
from __future__ import annotations

import logging
import pickle
from dataclasses import dataclass, field
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from .. import __version__
from ..analytics import peers as P
from ..analytics import returns as R
from ..analytics import rpo as RP
from ..analytics import technicals as T
from ..config import Config, load_config
from ..data.store import Store
from ..models import backtest as BT
from ..models import factor as FM
from ..models import features as F
from ..models import score as SC
from ..models import signal as SG
from ..models.eventstudy import run_event_study
from ..risk import montecarlo as MC
from ..risk import var as V
from ..valuation.scenarios import ValuationRun, run_valuation, save_valuation
from .conclusion import build_conclusion

log = logging.getLogger(__name__)
BUNDLE_NAME = "analysis_bundle.pkl"


@dataclass
class AnalysisBundle:
    meta: dict
    sources: dict
    # market
    px: pd.DataFrame
    rets: pd.DataFrame
    perf: pd.DataFrame
    period_returns: pd.DataFrame
    calendar_returns: pd.DataFrame
    corr: pd.DataFrame
    roll_corr: pd.DataFrame
    roll_beta: pd.DataFrame
    roll_vol: pd.DataFrame
    drawdowns: pd.DataFrame
    dd_table: pd.DataFrame
    divergence: P.DivergenceResult
    tech: pd.DataFrame
    trend: dict
    capm: R.CAPMResult
    capm_nasdaq: R.CAPMResult
    # fundamentals
    q: pd.DataFrame
    a: pd.DataFrame
    rpo: RP.RPOAnalysis
    cash_conv: pd.DataFrame
    # valuation
    val: ValuationRun
    # statistical models
    panel: pd.DataFrame
    factor: FM.FactorModelResults
    score: pd.DataFrame
    buckets: SC.BucketResult
    weight_rob: pd.DataFrame
    # risk
    risk: dict
    # events, signal, backtest
    event: object
    signal: SG.SignalModel
    backtest: BT.BacktestResult
    conclusion: dict
    consensus: dict = field(default_factory=dict)


def _risk(st: Store, cfg: Config, rets: pd.DataFrame, capm: R.CAPMResult) -> dict:
    r = rets[cfg.project.ticker]
    rf = st.risk_free()
    mc = cfg.risk.monte_carlo
    rf_now = float(rf["rf_3m"].iloc[-1])
    drift = MC.drift_annual(mc.drift, r, rf_now, capm.beta, mc.equity_risk_premium)
    res = MC.run_all(r, drift, n_sims=mc.n_sims, horizon=mc.horizon_days, seed=mc.seed, block=mc.block_size)
    out = {
        "var_table": V.var_table(r, tuple(cfg.risk.var_levels), tuple(cfg.risk.var_horizons_days)),
        "tail": V.tail_diagnostics(r),
        "var_bt": {},
        "roll_var95": V.rolling_var_es(r, 252, 0.95),
        "roll_var99": V.rolling_var_es(r, 252, 0.99),
        "mc": res,
        "mc_summary": MC.summary_table(res, mc.start_value),
        "mc_tails": MC.tail_comparison(r, res),
        "mc_settings": {"drift_mode": mc.drift, "drift_annual": drift, "rf": rf_now, "beta": capm.beta, "erp": mc.equity_risk_premium,
                        "n_sims": mc.n_sims, "horizon_days": mc.horizon_days, "seed": mc.seed},
        "dd_stats": R.drawdown_stats(r),
        "downside_dev": R.downside_deviation(r),
        "returns": r,
    }
    for method in ("historical", "normal"):
        for lv in cfg.risk.var_levels:
            bt, frame = V.var_backtest(r, lv, 252, method)
            out["var_bt"][(method, lv)] = (bt, frame)
    out["mc_alt_drift"] = {m: MC.simulate(r, "garch_t", mc.n_sims, mc.horizon_days, MC.drift_annual(m, r, rf_now, capm.beta, mc.equity_risk_premium), mc.seed).summary(mc.start_value)
                           for m in ("capm", "historical", "zero")}
    return out


def run_all_analyses(cfg: Config | None = None, save: bool = True) -> AnalysisBundle:
    cfg = cfg or load_config()
    st = Store(cfg)
    tkr = cfg.project.ticker
    spy, qqq = cfg.data.benchmarks.sp500, cfg.data.benchmarks.nasdaq100
    px, q, a = st.prices(), st.fundamentals(), st.annual()
    rf = st.risk_free()
    rets = R.simple_returns(px)
    rfd = R.daily_rf(rf["rf_3m"])
    log.info("market analytics")
    perf = P.peer_table(px, rf["rf_3m"], spy)
    capm = R.capm(rets[tkr], rets[spy], rfd)
    capm_n = R.capm(rets[tkr], rets[qqq], rfd)
    roll_beta = pd.DataFrame({f"vs {spy} (S&P 500)": R.rolling_beta(rets[tkr], rets[spy], cfg.analytics.rolling_beta_window),
                              f"vs {qqq} (Nasdaq 100)": R.rolling_beta(rets[tkr], rets[qqq], cfg.analytics.rolling_beta_window)})
    roll_vol = pd.DataFrame({f"{n}-day": R.rolling_vol(rets[tkr], n) for n in cfg.analytics.vol_windows})
    dd = pd.DataFrame({c: R.drawdown_series(rets[c]) for c in rets.columns})
    div = P.divergence(px, tkr, list(cfg.data.peers), cfg.analytics.divergence.window, cfg.analytics.divergence.z_threshold, cfg.analytics.divergence.min_history)
    tech = T.technical_frame(st.ohlcv(), cfg)
    log.info("fundamentals / RPO")
    snap = st.consensus()
    rpo = RP.analyse(q, snap)
    log.info("valuation")
    val = run_valuation(cfg, st)
    save_valuation(val, cfg)
    log.info("features, factor models, score")
    panel = F.build_feature_panel(st, cfg)
    factor = FM.run_factor_models(st, cfg, panel)
    score = SC.build_score(panel, cfg)
    buckets = SC.bucket_analysis(score["score_live"].dropna(), panel["close"], tuple(cfg.score.forward_horizons_days), cfg.score.bearish_threshold, cfg.score.bullish_threshold)
    wrob = SC.weight_robustness(panel, cfg)
    log.info("risk engine + Monte Carlo")
    risk = _risk(st, cfg, rets, capm)
    log.info("event study, signal, backtest")
    event = run_event_study(px, st.events(), q, cfg)
    sig = SG.build_signal(panel, cfg)
    bt = BT.run_backtest(sig.frame, st.ohlcv(), px, rf["rf_3m"], cfg)
    score_latest = float(score["score_live"].dropna().iloc[-1])
    state_latest = str(sig.frame["state"].dropna().iloc[-1])
    concl = build_conclusion(cfg, val, score_latest, state_latest, q)
    man = st.manifest()
    sources = {
        "prices": man.source_line("prices_adj_close"), "rf": man.source_line("risk_free"),
        "fundamentals": man.source_line("fundamentals_quarterly_xbrl", "press_release_kpis"),
        "events": man.source_line("earnings_events"), "consensus": man.source_line("consensus_snapshot"),
        "peers": man.source_line("peer_fundamentals"),
    }
    meta = {
        "version": __version__, "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "price_date": str(px.index[-1].date()), "price_start": str(px.index[0].date()), "latest_quarter": q["label"].iloc[-1],
        "latest_quarter_end": str(q.index[-1].date()), "ticker": tkr, "price": float(st.closes()[tkr].iloc[-1]),
        "consensus_date": snap.get("retrieved_at_utc", "")[:10], "next_earnings": snap.get("next_earnings_date"),
    }
    bundle = AnalysisBundle(meta, sources, px, rets, perf, P.period_returns(px), P.calendar_year_returns(px), P.correlation_matrix(rets),
                            P.rolling_correlation(rets, tkr, cfg.analytics.rolling_corr_window), roll_beta, roll_vol, dd, P.drawdown_table(px), div,
                            tech, T.trend_state(tech), capm, capm_n, q, a, rpo, RP.cash_conversion(q), val, panel, factor, score, buckets, wrob,
                            risk, event, sig, bt, concl, snap)
    if save:
        d = cfg.models_dir
        with open(d / BUNDLE_NAME, "wb") as fh:
            pickle.dump(bundle, fh, protocol=5)
        _write_tables(bundle, cfg)
    return bundle


def _write_tables(b: AnalysisBundle, cfg: Config) -> None:
    d = cfg.models_dir
    b.perf.to_csv(d / "performance_summary.csv")
    b.corr.to_csv(d / "correlation_matrix.csv")
    b.rpo.scorecard.to_csv(d / "rpo_scorecard.csv")
    b.rpo.lead_lag.to_csv(d / "rpo_lead_lag.csv")
    b.rpo.growth_corr_spearman.to_csv(d / "growth_correlations_spearman.csv")
    b.factor.window_specs.to_csv(d / "factor_window_specifications.csv")
    b.factor.walk_forward.importance.to_csv(d / "factor_feature_importance.csv")
    b.factor.corr_target.to_csv(d / "factor_correlations_forward_return.csv")
    b.score.to_csv(d / "investment_score_history.csv")
    b.buckets.table.to_csv(d / "score_forward_return_buckets.csv")
    b.risk["var_table"].to_csv(d / "var_table.csv", index=False)
    b.risk["mc_summary"].to_csv(d / "monte_carlo_summary.csv")
    b.event.table.to_csv(d / "event_study_events.csv")
    b.event.summary.to_csv(d / "event_study_summary.csv")
    b.event.cross_section.to_csv(d / "event_study_cross_section.csv", index=False)
    b.backtest.metrics.to_csv(d / "backtest_metrics_full.csv")
    for k, t in b.backtest.by_period.items():
        t.to_csv(d / ("backtest_metrics_" + k.split(" ")[0].lower() + ".csv"))
    b.signal.frame.to_csv(d / "signal_history.csv")
    b.conclusion["votes"].to_csv(d / "conclusion_votes.csv", index=False)


def load_bundle(cfg: Config | None = None, rebuild_if_missing: bool = True) -> AnalysisBundle:
    cfg = cfg or load_config()
    p = cfg.models_dir / BUNDLE_NAME
    if p.exists():
        with open(p, "rb") as fh:
            return pickle.load(fh)
    if not rebuild_if_missing:
        raise FileNotFoundError(p)
    return run_all_analyses(cfg)
