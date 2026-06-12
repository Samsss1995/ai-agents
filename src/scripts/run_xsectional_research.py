"""
Pre-registered cross-sectional momentum + crypto carry research run (2026-06-12).

Configs fixed before execution; no iteration. Per asset class: 90-day momentum,
weekly rebalance, long-short top1/bottom1 AND long-only top1 (both declared).
Cross-class: 15 instruments daily, top3/bottom3 and long-only top3. Crypto
carry: rank 3 coins by 3-day mean funding, daily rebalance, long lowest /
short highest, REAL funding accrual. FX/commodity/equity carry: untestable
without rate/term-structure data - excluded, stated.

Judged by the standard hard gates. Results recorded in the experiment store.
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from src.research.data_catalog import DataCatalog
from src.research.experiment_store import ExperimentStore
from src.research.factory_config import load_factory_config
from src.research.strategy_spec import StrategySpec
from src.research.validation_gates import evaluate_gates, hard_gates_passed
from src.research.xsectional import XSConfig, evaluate_xs

CLASSES = {
    "crypto":  (["BTC-USD-4h", "ETH-USD-4h", "SOL-USD-4h"], 540, 42, "crypto_perp"),
    "stocks":  (["AAPL-1d", "MSFT-1d", "JPM-1d"], 90, 5, "equity"),
    "indices": (["SPX-1d", "NDX-1d", "DJI-1d"], 90, 5, "index"),
    "commod":  (["WTI-1d", "GLD-1d", "SLV-1d"], 90, 5, "future"),
    "forex":   (["EURUSD-1d", "USDJPY-1d", "GBPUSD-1d"], 90, 5, "fx"),
}


def make_spec(name, family, aclass, instruments, timeframe, hypothesis) -> StrategySpec:
    return StrategySpec(
        name=name, family=family, asset_class=aclass, instruments=instruments,
        timeframe=timeframe, hypothesis=hypothesis,
        regime_assumptions="cross-sectional ranking is regime-agnostic by design",
        entry_logic=f"deterministic rotation per registered XSConfig ({name})",
        exit_logic="rebalance to new ranking; positions exit when they leave the rank band",
        stop_logic="none - rotation and diversification are the risk control",
        position_sizing="equal weight per ranked slot",
        risk_rules="weights bounded by construction; turnover costed per side",
        invalidation_rules="standard hard gates, one shot; no iteration permitted",
        expected_trade_frequency="swing", low_frequency=False,
        source="src/scripts/run_xsectional_research.py (pre-registered 2026-06-12)",
    )


def main() -> None:
    config = load_factory_config()
    store = ExperimentStore()
    catalog = DataCatalog(config)
    fractions = config["splits"]
    cost = config["costs"]["commission"] + config["costs"]["slippage_bps"] / 10_000

    runs = []
    # per-class momentum (LS and LO declared a priori)
    for cname, (datasets, lb, rb, aclass) in CLASSES.items():
        dfs = {d: catalog.require(d) for d in datasets}
        for variant, top, bot in (("LS", 1, 1), ("LO", 1, 0)):
            runs.append((f"XSMom{variant}_{cname}", "portfolio_allocation", aclass,
                         datasets, dfs, None,
                         XSConfig(name=f"XSMom{variant}_{cname}", lookback_bars=lb,
                                  rebalance_bars=rb, top_n=top, bottom_n=bot,
                                  rank_by="momentum", cost_per_side=cost),
                         "Winners keep winning over 1-6 month horizons because flows chase "
                         "performance and information diffuses slowly; ranking within a "
                         "basket nets out market beta."))
    # cross-class momentum: everything resampled to 1d
    xdfs = {}
    for cname, (datasets, _, _, _) in CLASSES.items():
        for d in datasets:
            df = catalog.require(d)
            if cname == "crypto":
                df = df.resample("1D").agg({"Open": "first", "High": "max",
                                            "Low": "min", "Close": "last",
                                            "Volume": "sum"}).dropna()
            xdfs[d] = df
    for variant, top, bot in (("LS", 3, 3), ("LO", 3, 0)):
        runs.append((f"XSMomXClass{variant}", "portfolio_allocation", "crypto_perp",
                     list(xdfs), xdfs, None,
                     XSConfig(name=f"XSMomXClass{variant}", lookback_bars=90,
                              rebalance_bars=5, top_n=top, bottom_n=bot,
                              rank_by="momentum", cost_per_side=cost),
                     "Cross-asset-class momentum: relative 90-day strength across 15 "
                     "instruments spanning crypto, stocks, indices, commodities and FX; "
                     "the classic CTA cross-sectional result."))
    # crypto funding carry with real accrual
    sig = {d: catalog.require(d) for d in ("BTC-SIG-1h", "ETH-SIG-1h", "SOL-SIG-1h")}
    carry = {d: df["FundingRate"] for d, df in sig.items()}
    runs.append(("FundingCarryLS_crypto", "funding_driven", "crypto_perp",
                 list(sig), sig, carry,
                 XSConfig(name="FundingCarryLS_crypto", lookback_bars=72,
                          rebalance_bars=24, top_n=1, bottom_n=1, rank_by="carry",
                          cost_per_side=cost, accrue_carry=True),
                 "Funding carry: be long the coin the market pays you to hold and short "
                 "the one longs overpay for; PnL includes the actual hourly funding "
                 "accrual, not just price."))

    print(f"{'config':24}{'val_ret%':>9}{'shrp':>6}{'srtno':>6}{'PF':>6}{'trades':>7}"
          f"{'WFret':>7}{'1.5x':>7}  verdict")
    for name, family, aclass, datasets, dfs, carry_s, cfg, hypo in runs:
        spec = make_spec(name, family, aclass, [d.split("-")[0] for d in datasets],
                         "1d" if "1h" not in datasets[0] else "1h", hypo)
        try:
            sid = store.add_spec(spec)
        except ValueError:
            sid = store.conn.execute("SELECT spec_id FROM specs WHERE fingerprint=?",
                                     (spec.fingerprint(),)).fetchone()["spec_id"]
            spec.spec_id = sid
        try:
            out = evaluate_xs(dfs, cfg, fractions, carry=carry_s,
                              seed=config["monte_carlo"]["seed"])
        except Exception as e:
            store.record_experiment(sid, "validation", False,
                                    error=f"{type(e).__name__}: {e}")
            print(f"{name:24}  FAILED: {e}")
            continue
        store.record_experiment(sid, "train", True, dataset_id="XS",
                                metrics=out["train"])
        store.record_experiment(sid, "validation", True, dataset_id="PORTFOLIO",
                                metrics=out["validation"])
        store.record_experiment(sid, "robustness", True, dataset_id="XS",
                                metrics=out["robustness"],
                                seed=config["monte_carlo"]["seed"])
        gates = evaluate_gates(spec, out["validation"], out["train"],
                               out["robustness"], static_review_clean=True)
        store.record_gate_results(sid, None, [g.to_dict() for g in gates])
        passed = hard_gates_passed(gates)
        failed = [g.name for g in gates if g.hard and not g.passed]
        if not passed:
            store.record_rejection(sid, "gates", f"failed: {failed}")
        v, r = out["validation"], out["robustness"]
        def f(x, p=2): return "-" if x is None else f"{x:.{p}f}"
        print(f"{name:24}{f(v.get('return_pct'),1):>9}{f(v.get('sharpe')):>6}"
              f"{f(v.get('sortino')):>6}{f(v.get('profit_factor')):>6}"
              f"{str(v.get('n_trades')):>7}{f(r.get('walk_forward_retention')):>7}"
              f"{f(r.get('cost_stress',{}).get('return_pct_at_1.5x'),1):>7}  "
              f"{'PASS ALL' if passed else f'{len(failed)} failed: ' + ','.join(failed)[:60]}")


if __name__ == "__main__":
    main()
