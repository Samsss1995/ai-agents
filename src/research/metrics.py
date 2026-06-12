"""
Metric extraction - converts a backtesting.py stats object into the full
institutional metric set. Every metric is computed after costs (costs are applied
by the runner). In-sample vs out-of-sample labeling happens at the runner level.
"""

import math
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

ANNUALIZATION_MINUTES = 365 * 24 * 60


def _safe(value: Any) -> Optional[float]:
    """NaN/inf -> None so JSON storage and gate logic stay honest."""
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(f) or math.isinf(f):
        return None
    return f


def _equity_returns(equity_curve: pd.DataFrame, freq: str) -> pd.Series:
    eq = equity_curve["Equity"].resample(freq).last().dropna()
    return eq.pct_change().dropna()


def extract_metrics(stats: Any, benchmark_close: Optional[pd.Series] = None) -> Dict[str, Any]:
    """
    stats: the Series returned by backtesting.Backtest.run().
    benchmark_close: Close prices of the traded instrument over the same window,
                     used for buy-and-hold comparison and correlation.
    """
    trades: pd.DataFrame = stats["_trades"]
    equity: pd.DataFrame = stats["_equity_curve"]

    m: Dict[str, Any] = {
        "return_pct": _safe(stats["Return [%]"]),
        "buy_hold_return_pct": _safe(stats["Buy & Hold Return [%]"]),
        "annualized_return_pct": _safe(stats.get("Return (Ann.) [%]")),
        "max_drawdown_pct": _safe(stats["Max. Drawdown [%]"]),
        "avg_drawdown_pct": _safe(stats.get("Avg. Drawdown [%]")),
        "max_drawdown_duration": str(stats.get("Max. Drawdown Duration")),
        "sharpe": _safe(stats["Sharpe Ratio"]),
        "sortino": _safe(stats["Sortino Ratio"]),
        "calmar": _safe(stats["Calmar Ratio"]),
        "profit_factor": _safe(stats.get("Profit Factor")),
        "expectancy_pct": _safe(stats.get("Expectancy [%]")),
        "win_rate_pct": _safe(stats["Win Rate [%]"]),
        "sqn": _safe(stats.get("SQN")),
        "n_trades": int(stats["# Trades"]),
        "exposure_time_pct": _safe(stats["Exposure Time [%]"]),
        "best_trade_pct": _safe(stats["Best Trade [%]"]),
        "worst_trade_pct": _safe(stats["Worst Trade [%]"]),
        "avg_trade_pct": _safe(stats["Avg. Trade [%]"]),
        "start": str(stats["Start"]),
        "end": str(stats["End"]),
        "duration": str(stats["Duration"]),
        "final_equity": _safe(stats["Equity Final [$]"]),
    }

    # trade-level detail
    if len(trades) > 0:
        pnl = trades["PnL"]
        wins, losses = pnl[pnl > 0], pnl[pnl <= 0]
        m["avg_win_usd"] = _safe(wins.mean()) if len(wins) else None
        m["avg_loss_usd"] = _safe(losses.mean()) if len(losses) else None
        m["gross_profit_usd"] = _safe(wins.sum())
        m["gross_loss_usd"] = _safe(losses.sum())
        if m.get("profit_factor") is None and len(losses) and losses.sum() != 0:
            m["profit_factor"] = _safe(wins.sum() / abs(losses.sum()))
        m["long_trades"] = int((trades["Size"] > 0).sum())
        m["short_trades"] = int((trades["Size"] < 0).sum())
        m["long_pnl_usd"] = _safe(pnl[trades["Size"] > 0].sum())
        m["short_pnl_usd"] = _safe(pnl[trades["Size"] < 0].sum())
        if "Duration" in trades.columns:
            m["avg_holding_time"] = str(trades["Duration"].mean())
        # turnover: traded notional relative to final equity
        notional = (trades["Size"].abs() * trades["EntryPrice"]).sum()
        if m["final_equity"]:
            m["turnover_x"] = _safe(notional / m["final_equity"])
        # single-trade dependence: return contribution of the best trade
        m["best_trade_pnl_share"] = _safe(pnl.max() / pnl.sum()) if pnl.sum() > 0 else None
        m["return_without_best_trade_pct"] = _safe(
            (pnl.sum() - pnl.max()) / (m["final_equity"] - pnl.sum()) * 100
        ) if m["final_equity"] and (m["final_equity"] - pnl.sum()) > 0 else None
    else:
        m["avg_win_usd"] = m["avg_loss_usd"] = None
        m["long_trades"] = m["short_trades"] = 0

    # periodic worst cases from the equity curve
    if isinstance(equity.index, pd.DatetimeIndex) and len(equity) > 2:
        for label, freq in (("day", "1D"), ("week", "1W"), ("month", "1ME")):
            try:
                rets = _equity_returns(equity, freq)
                m[f"worst_{label}_pct"] = _safe(rets.min() * 100) if len(rets) else None
            except Exception:
                m[f"worst_{label}_pct"] = None
        # drawdown duration in bars
        eq = equity["Equity"]
        running_max = eq.cummax()
        in_dd = eq < running_max
        m["longest_drawdown_bars"] = int(
            (in_dd.groupby((~in_dd).cumsum()).cumsum()).max()
        ) if in_dd.any() else 0

    # benchmark correlation
    if benchmark_close is not None and isinstance(equity.index, pd.DatetimeIndex):
        try:
            eq_d = equity["Equity"].resample("1D").last().pct_change().dropna()
            bm_d = benchmark_close.resample("1D").last().pct_change().dropna()
            joined = pd.concat([eq_d, bm_d], axis=1).dropna()
            if len(joined) > 10:
                m["benchmark_correlation"] = _safe(joined.iloc[:, 0].corr(joined.iloc[:, 1]))
        except Exception:
            m["benchmark_correlation"] = None

    return m


def portfolio_metrics(stats_list: list, window_start=None) -> Dict[str, Any]:
    """
    Equal-weight portfolio of several single-instrument backtests of the SAME
    strategy: per-instrument equity curves normalized to 1.0 and averaged;
    trades pooled. window_start restricts curves and trades to a common
    evaluation window (cross-class portfolios MUST set it - component backtests
    may include warmup history before the window).

    Validity requirement: component curves must genuinely overlap in time.
    Non-overlapping windows produce forward-filled flat segments that fake
    diversification.
    """
    curves = []
    for s in stats_list:
        eq = s["_equity_curve"]["Equity"]
        if window_start is not None:
            eq = eq[eq.index >= window_start]
            if len(eq) == 0:
                continue
        curves.append(eq / eq.iloc[0])
    if not curves:
        return {"portfolio": True, "error": "no curves inside window"}
    joined = pd.concat(curves, axis=1).ffill().dropna()
    port = joined.mean(axis=1)
    if len(port) < 10:
        return {"portfolio": True, "error": "insufficient overlapping history"}

    rets = port.pct_change().dropna()
    years = max((port.index[-1] - port.index[0]).total_seconds() / (365.25 * 24 * 3600), 1e-9)
    periods_per_year = len(rets) / years
    total_return = port.iloc[-1] / port.iloc[0]
    ann_return = total_return ** (1 / years) - 1
    ann_vol = float(rets.std() * np.sqrt(periods_per_year))
    downside = rets[rets < 0]
    downside_vol = float(downside.std() * np.sqrt(periods_per_year)) if len(downside) > 1 else None
    peak = port.cummax()
    dd = (port - peak) / peak
    max_dd = float(dd.min())

    trades = pd.concat([s["_trades"] for s in stats_list], ignore_index=True)
    if window_start is not None and len(trades) and "EntryTime" in trades.columns:
        trades = trades[pd.to_datetime(trades["EntryTime"]) >= window_start]
    m: Dict[str, Any] = {
        "portfolio": True,
        "n_instruments": len(stats_list),
        "return_pct": _safe((total_return - 1) * 100),
        "annualized_return_pct": _safe(ann_return * 100),
        "max_drawdown_pct": _safe(max_dd * 100),
        "sharpe": _safe(rets.mean() * periods_per_year / ann_vol) if ann_vol > 0 else None,
        "sortino": _safe(rets.mean() * periods_per_year / downside_vol) if downside_vol else None,
        "calmar": _safe(ann_return / abs(max_dd)) if max_dd < 0 else None,
        "n_trades": int(len(trades)),
        "exposure_time_pct": _safe(np.mean([s["Exposure Time [%]"] for s in stats_list])),
        "buy_hold_return_pct": _safe(np.mean([s["Buy & Hold Return [%]"] for s in stats_list])),
        "start": str(port.index[0]), "end": str(port.index[-1]),
    }
    if len(trades):
        pnl = trades["PnL"]
        wins, losses = pnl[pnl > 0], pnl[pnl <= 0]
        m["win_rate_pct"] = _safe(len(wins) / len(trades) * 100)
        m["profit_factor"] = (_safe(wins.sum() / abs(losses.sum()))
                              if len(losses) and losses.sum() != 0 else None)
        m["expectancy_pct"] = _safe(trades["ReturnPct"].mean() * 100)
        m["best_trade_pct"] = _safe(trades["ReturnPct"].max() * 100)
        m["worst_trade_pct"] = _safe(trades["ReturnPct"].min() * 100)
        total_start_equity = sum(float(s["_equity_curve"]["Equity"].iloc[0]) for s in stats_list)
        total_final = sum(float(s["Equity Final [$]"]) for s in stats_list)
        base = total_final - pnl.sum()
        m["return_without_best_trade_pct"] = (
            _safe((pnl.sum() - pnl.max()) / base * 100) if base > 0 and pnl.sum() > 0 else
            _safe((pnl.sum() - pnl.max()) / total_start_equity * 100))
    return m


def trade_returns_pct(stats: Any) -> np.ndarray:
    """Per-trade return percentages, used by Monte Carlo reshuffling."""
    trades: pd.DataFrame = stats["_trades"]
    if len(trades) == 0:
        return np.array([])
    return trades["ReturnPct"].to_numpy() * 100.0
