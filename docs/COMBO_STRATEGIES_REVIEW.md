# Review: Combination / Regime-Gated Strategies (grid, SMA/EMA/stochRSI/Bollinger)

Date: 2026-06-11. Ideas: `src/data/research_factory/ideas_combos.txt` (8 confluence/regime
ideas). First program evaluated at PORTFOLIO level: multi-instrument specs are judged on the
equal-weight basket of their per-instrument validation curves (pooled trades).

## Coverage
8 ideas x 5 asset classes = 40 spec-pipelines (39 evaluated; stocks grid died at codegen).
Run time ~58 min. All results in `research.db`; leaderboard regenerated.

## Outcome: 0 of 40 pass gates. Best cells (portfolio validation, after costs)

| Cell | Ret % | PF | Sharpe | Trades | Hard gates failed |
|---|---|---|---|---|---|
| C4 TrendPullbackStochastic / stocks | +7.0 | 1.70 | 1.06 | 153 | Sortino, Calmar, WF retention |
| C7 PyramidTrendConvexity / stocks | +38.2 | 2.06 | 1.04 | 250 | Sortino, Calmar, WF retention |
| C6 SqueezeTrendGate / stocks | +3.2 | 2.68 | 0.91 | 68 | Sharpe, Sortino, WF retention |
| C5 VolatilityTargetedEMACross / crypto | +40.7 | 1.58 | 0.96 | 325 (t+v) | maxDD, Sharpe, Sortino, WF retention |

Patterns: trend/pullback confluence on the STOCK basket is systematically the strongest
cell across all research to date (three Sharpe ~1 portfolios). Grid trading died in every
class (best -0.5%; trades a lot, pays fees, regime gate insufficient). Forex rejected all
eight ideas - fifth consecutive program where FX kills everything. Triple-confluence fades
underperform single-gate reversion (over-filtering removes the trades that paid).

## The C5 cross-class arc (methodology case study)

1. Crypto validation: +40.7%, PF 1.58, passed 10/14 gates - strongest single result of the
   research program.
2. Pre-registered criterion (set before other classes ran): if >=2 other classes positive
   with PF > 1.2, build the cross-class portfolio. Met: stocks (+24.7, PF 2.95), indices
   (+12.4, PF 2.23).
3. First cross-class construction was INVALID and caught: class validation windows do not
   overlap (crypto 2024-25 vs daily 2015-21); forward-fill faked diversification.
   Recorded as rejection `xclass_invalid`; portfolio_metrics now takes window_start and
   documents the overlap requirement.
4. Legitimate test: common holdout window (intersection of all test slices,
   2025-05-10 -> 2026-06-11), 15 instruments, criteria registered first
   (Sharpe>=1, Sortino>=1.5, PF>=1.2, DD<=25%, >=100 trades, positive).
5. **Result: Sharpe 0.26, Sortino 0.37, PF 1.10, +1.8%, negative without best trade -
   FAILED.** All five C5 specs retired.

This is the FOURTH confirmed validation-window artifact (after HA-commodities,
role-reversal-stocks, funding-divergence): strategies that look strong on one historical
window and evaporate on the adjacent one. Any future candidate whose strength is confined
to a single contiguous window inherits a strong prior of death.

## Standing conclusions after 101 specs / ~1,000 experiments

1. Published/retail price patterns - single-indicator AND confluence variants - do not
   survive institutional gates after 12bps costs on any of the five asset classes.
2. Portfolio-level judgment (added this program) materially helps Sharpe and drawdown but
   does not rescue strategies whose edge is window-local.
3. The stock-basket trend/pullback family is real-but-weak: consistently positive OOS,
   PF 1.7-2.7, Sharpe ~1, but walk-forward retention <0.5 everywhere - the returns
   concentrate in trending sub-periods. A regime-allocation layer is the only honest way
   to pursue it, and it must be pre-registered like everything else.
4. Differentiated data remains the credible path: L2/OI/funding collection live since
   2026-06-11 (signal studies ~early July; gate-grade backtests ~Sept; walk-forward
   grade ~Dec).
