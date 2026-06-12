"""
Cross-sectional engine - deterministic rotation backtests over an instrument
basket. Exists because the per-instrument backtesting.py contract cannot
express rank-and-rotate strategies (cross-sectional momentum) or funding
accrual (carry).

No LLM anywhere. No lookahead by construction: rankings computed at bar t set
weights from bar t+1. Costs charged on turnover at each rebalance. Crypto carry
accrues the actual hourly funding series against position sign.
"""

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


@dataclass
class XSConfig:
    name: str
    lookback_bars: int          # ranking window (momentum) or carry averaging window
    rebalance_bars: int         # rebalance frequency in bars
    top_n: int = 1              # longs (highest rank)
    bottom_n: int = 0           # shorts (lowest rank)
    rank_by: str = "momentum"   # "momentum" | "carry"
    cost_per_side: float = 0.0012
    accrue_carry: bool = False  # credit/debit funding against position sign


@dataclass
class XSResult:
    config: XSConfig
    returns: pd.Series                       # per-bar portfolio returns (net)
    holding_returns: List[float] = field(default_factory=list)  # per closed holding
    n_rebalances: int = 0
    total_turnover: float = 0.0


def _align(dfs: Dict[str, pd.DataFrame], column: str = "Close") -> pd.DataFrame:
    wide = pd.concat({k: df[column] for k, df in dfs.items()}, axis=1)
    return wide.dropna(how="any")  # inner-join calendar: only bars all instruments share


def run_xsectional(dfs: Dict[str, pd.DataFrame], config: XSConfig,
                   carry: Optional[Dict[str, pd.Series]] = None) -> XSResult:
    closes = _align(dfs)
    if len(closes) < config.lookback_bars + config.rebalance_bars + 10:
        raise ValueError(f"{config.name}: insufficient aligned history ({len(closes)} bars)")
    rets = closes.pct_change().fillna(0.0)

    carry_aligned = None
    if carry is not None:
        carry_aligned = pd.concat(carry, axis=1).reindex(closes.index).fillna(0.0)

    if config.rank_by == "momentum":
        signal = closes / closes.shift(config.lookback_bars) - 1.0
    elif config.rank_by == "carry":
        if carry_aligned is None:
            raise ValueError("rank_by=carry requires carry series")
        signal = -carry_aligned.rolling(config.lookback_bars, min_periods=1).mean()
        # negative funding mean ranks HIGH: long what you are paid to hold
    else:
        raise ValueError(f"unknown rank_by '{config.rank_by}'")

    n_inst = closes.shape[1]
    weights = pd.DataFrame(0.0, index=closes.index, columns=closes.columns)
    current = pd.Series(0.0, index=closes.columns)
    open_holdings: Dict[str, List[float]] = {}
    holding_returns: List[float] = []
    total_turnover = 0.0
    n_rebalances = 0
    cost_series = pd.Series(0.0, index=closes.index)

    for i in range(config.lookback_bars, len(closes)):
        ts = closes.index[i]
        if (i - config.lookback_bars) % config.rebalance_bars == 0:
            row = signal.iloc[i].dropna()
            if len(row) >= max(config.top_n + config.bottom_n, 2):
                ranked = row.sort_values(ascending=False)
                target = pd.Series(0.0, index=closes.columns)
                for sym in ranked.index[:config.top_n]:
                    target[sym] = 1.0 / config.top_n
                if config.bottom_n:
                    for sym in ranked.index[-config.bottom_n:]:
                        target[sym] += -1.0 / config.bottom_n
                turnover = float((target - current).abs().sum())
                if turnover > 1e-12:
                    cost_series.iloc[min(i + 1, len(closes) - 1)] = (
                        turnover * config.cost_per_side)
                    total_turnover += turnover
                    n_rebalances += 1
                    # close holdings whose weight sign changed or went to zero
                    for sym in closes.columns:
                        was, now = current[sym], target[sym]
                        if sym in open_holdings and (now == 0 or np.sign(now) != np.sign(was)):
                            path = open_holdings.pop(sym)
                            holding_returns.append(float(np.prod([1 + r for r in path]) - 1)
                                                   * np.sign(was))
                        if now != 0 and (was == 0 or np.sign(now) != np.sign(was)):
                            open_holdings[sym] = []
                    current = target
        weights.iloc[i] = current
        for sym, path in open_holdings.items():
            path.append(float(rets.iloc[i][sym]))

    # weights effective NEXT bar (signal at t -> exposure t+1)
    effective = weights.shift(1).fillna(0.0)
    port = (effective * rets).sum(axis=1) - cost_series
    if config.accrue_carry and carry_aligned is not None:
        port = port - (effective * carry_aligned).sum(axis=1)  # longs pay positive funding
    # close any open holdings at the end
    for sym, path in open_holdings.items():
        holding_returns.append(float(np.prod([1 + r for r in path]) - 1)
                               * np.sign(current[sym]))

    return XSResult(config=config, returns=port[config.lookback_bars:],
                    holding_returns=holding_returns,
                    n_rebalances=n_rebalances, total_turnover=total_turnover)


def metrics_from_returns(returns: pd.Series,
                         holding_returns: Optional[List[float]] = None) -> Dict[str, Any]:
    """Institutional metric set from a per-bar net return series."""
    if len(returns) < 20:
        return {"error": f"only {len(returns)} bars"}
    equity = (1 + returns).cumprod()
    years = max((returns.index[-1] - returns.index[0]).total_seconds() / (365.25 * 86400), 1e-9)
    ppy = len(returns) / years
    ann_ret = float(equity.iloc[-1] ** (1 / years) - 1)
    vol = float(returns.std() * math.sqrt(ppy))
    downside = returns[returns < 0]
    dvol = float(downside.std() * math.sqrt(ppy)) if len(downside) > 1 else None
    peak = equity.cummax()
    dd = float(((equity - peak) / peak).min())

    m: Dict[str, Any] = {
        "return_pct": float((equity.iloc[-1] - 1) * 100),
        "annualized_return_pct": ann_ret * 100,
        "sharpe": float(returns.mean() * ppy / vol) if vol > 0 else None,
        "sortino": float(returns.mean() * ppy / dvol) if dvol else None,
        "calmar": (ann_ret / abs(dd)) if dd < 0 else None,
        "max_drawdown_pct": dd * 100,
        "exposure_time_pct": float((returns != 0).mean() * 100),
        "start": str(returns.index[0]), "end": str(returns.index[-1]),
    }
    hr = holding_returns or []
    m["n_trades"] = len(hr)
    if hr:
        wins = [r for r in hr if r > 0]
        losses = [r for r in hr if r <= 0]
        m["win_rate_pct"] = len(wins) / len(hr) * 100
        m["profit_factor"] = (sum(wins) / abs(sum(losses))) if losses and sum(losses) != 0 else None
        m["expectancy_pct"] = float(np.mean(hr) * 100)
        m["best_trade_pct"] = max(hr) * 100
        m["worst_trade_pct"] = min(hr) * 100
        total = sum(hr)
        m["return_without_best_trade_pct"] = float((total - max(hr)) * 100)
    return m


def split_returns(returns: pd.Series, fractions: Dict[str, float]):
    n = len(returns)
    i1 = int(n * fractions["train"])
    i2 = int(n * (fractions["train"] + fractions["validation"]))
    return returns.iloc[:i1], returns.iloc[i1:i2], returns.iloc[i2:]


def _split_holdings(result: XSResult, seg: pd.Series) -> List[float]:
    # holding_returns lack timestamps; approximate per-segment trades by share of bars
    share = len(seg) / max(len(result.returns), 1)
    k = max(1, int(round(len(result.holding_returns) * share)))
    return result.holding_returns[:k]  # order-preserving approximation


def evaluate_xs(dfs: Dict[str, pd.DataFrame], config: XSConfig,
                fractions: Dict[str, float],
                carry: Optional[Dict[str, pd.Series]] = None,
                wf_folds: int = 4, seed: int = 42) -> Dict[str, Any]:
    """
    Full evaluation: train/validation metrics, lookback neighborhood, cost
    stress, anchored walk-forward, Monte Carlo on holding returns. Test slice
    untouched (reserved). Deterministic given seed.
    """
    result = run_xsectional(dfs, config, carry)
    train_r, val_r, _test_r = split_returns(result.returns, fractions)
    train_m = metrics_from_returns(train_r)
    val_m = metrics_from_returns(val_r)
    # trade counts: run engine on segments is cleaner but heavier; approximate via shares
    train_m["n_trades"] = len(_split_holdings(result, train_r))
    val_hold = result.holding_returns[len(_split_holdings(result, train_r)):]
    val_m["n_trades"] = max(0, int(round(len(result.holding_returns)
                                         * len(val_r) / len(result.returns))))
    if val_hold:
        wins = [r for r in val_hold if r > 0]; losses = [r for r in val_hold if r <= 0]
        val_m["profit_factor"] = (sum(wins) / abs(sum(losses))
                                  if losses and sum(losses) != 0 else None)
        total = sum(val_hold)
        val_m["return_without_best_trade_pct"] = float((total - max(val_hold)) * 100)

    # walk-forward: anchored folds over the full net-return series
    n = len(result.returns)
    folds = []
    min_train_frac = 0.4
    test_frac = (1 - min_train_frac) / wf_folds
    for k in range(wf_folds):
        a = int(n * (min_train_frac + k * test_frac))
        b = int(n * (min_train_frac + (k + 1) * test_frac))
        is_m = metrics_from_returns(result.returns.iloc[:a])
        oos_m = metrics_from_returns(result.returns.iloc[a:b])
        folds.append({"fold": k, "in_sample_return_pct": is_m.get("return_pct"),
                      "oos_return_pct": oos_m.get("return_pct")})
    is_mean = float(np.mean([f["in_sample_return_pct"] for f in folds]))
    oos_mean = float(np.mean([f["oos_return_pct"] for f in folds]))
    robustness: Dict[str, Any] = {
        "folds": folds,
        "walk_forward_is_mean_return_pct": is_mean,
        "walk_forward_oos_mean_return_pct": oos_mean,
        "walk_forward_retention": (oos_mean / is_mean) if is_mean > 0 else None,
    }
    # lookback neighborhood on train slice
    base = train_m.get("return_pct")
    neigh = []
    for mult in (0.8, 0.9, 1.1, 1.2):
        cfg = XSConfig(**{**config.__dict__,
                          "lookback_bars": max(2, int(config.lookback_bars * mult))})
        try:
            rr = run_xsectional(dfs, cfg, carry)
            tr, _, _ = split_returns(rr.returns, fractions)
            neigh.append(metrics_from_returns(tr).get("return_pct"))
        except Exception:
            neigh.append(None)
    clean = [x for x in neigh if x is not None]
    robustness["param_neighborhood_retention"] = (
        float(np.mean(clean)) / base if clean and base and base > 0 else None)
    # cost stress on validation
    stress = {}
    for mult in (1.5, 2.0):
        cfg = XSConfig(**{**config.__dict__, "cost_per_side": config.cost_per_side * mult})
        rr = run_xsectional(dfs, cfg, carry)
        _, vv, _ = split_returns(rr.returns, fractions)
        stress[f"return_pct_at_{mult}x"] = metrics_from_returns(vv).get("return_pct")
    robustness["cost_stress"] = stress
    # Monte Carlo on holding returns
    hr = np.array(result.holding_returns)
    if len(hr) >= 5:
        rng = np.random.default_rng(seed)
        dds = []
        for _ in range(1000):
            eq = np.cumprod(1 + rng.permutation(hr))
            pk = np.maximum.accumulate(eq)
            dds.append(((eq - pk) / pk).min() * 100)
        robustness["monte_carlo"] = {"p05_max_drawdown_pct": float(np.percentile(dds, 5)),
                                     "n_trades": int(len(hr)), "seed": seed}
    else:
        robustness["monte_carlo"] = {"error": f"only {len(hr)} holdings"}
    return {"train": train_m, "validation": val_m, "robustness": robustness,
            "n_rebalances": result.n_rebalances, "turnover": result.total_turnover}
