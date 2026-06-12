# Review: "9 Profitable Trading Strategies" (R. Sadowski / HumbleTraders PDF)

Date: 2026-06-11. Source: `/Users/sam/Downloads/9 Profitable Trading Strategies.pdf` (42 pages).
Method: each strategy distilled to its mechanical core (`src/data/research_factory/ideas_pdf9.txt`),
LLM-coded to the factory contract, statically reviewed for lookahead, then backtested with
standardized costs (0.1% commission + 2bps slippage per side) on chronological
train/validation splits with walk-forward, Monte Carlo, cost-stress and parameter-neighborhood
batteries. All runs recorded in `src/data/research_factory/research.db`.

## Coverage (the "variants tested" matrix)

| Dimension | Coverage |
|---|---|
| Asset classes | crypto (BTC/ETH/SOL, 4h, 2021-2026), stocks (AAPL/MSFT/JPM, 1d, 2000-2026), indices (SPX/NDX/DJI, 1d, 2000-2026), commodities (WTI/GLD/SLV, 1d, 2000/05-2026), forex (EURUSD/USDJPY/GBPUSD, 1d, 2000/03-2026) |
| Spec-pipelines | 45 (9 strategies x 5 classes) |
| Parameter variants | every numeric parameter perturbed +/-10% and +/-20% per spec (parameter-neighborhood battery) |
| Backtests executed | ~600 of the store's 681 recorded experiments belong to this program |
| Promotion gate passes | **0 of 45** |

## Verdict grid (validation slice, after costs; "B&H" = buy-and-hold same slice)

Verdicts: REJECT (traded and failed broadly), NEAR-MISS (failed <=5 hard gates),
NO-TRADE (codegen never converged to a strategy that fires - implementation result,
not a hypothesis result).

| # | Strategy | Crypto 4h | Stocks 1d | Indices 1d | Commodities 1d | Forex 1d |
|---|---|---|---|---|---|---|
| 1 | Stochastic exhaustion fade | REJECT (-3.4%) | REJECT (-9.3%) | REJECT (0.0%) | **REJECT (-88.9%)** | REJECT (-3.4%) |
| 2 | Triple MA 20/60/100 cross | REJECT (+1.8%, PF 1.12) | REJECT (+15.7%, PF 4.34, **7 trades**) | REJECT (+9.3%, 9 trades) | NEAR-MISS (+2.3%, PF 1.25, 14 trades) | REJECT (-8.5%) |
| 3 | Heikin-Ashi exhaustion reversal | REJECT (+5.9%, PF 1.38) | REJECT (+4.3%) | REJECT (+4.3%) | REJECT (+24.0%, PF 3.66, 59 trades, 6 gates failed) | REJECT (-27.9%) |
| 4 | Swing compression continuation | NO-TRADE | NO-TRADE | NEAR-MISS (+1.0%, 12 trades) | REJECT (3 trades) | REJECT (-4.9%) |
| 5 | Candlestick reversals (engulf/hammer/shadow) | NEAR-MISS (-3.5%, 75 trades) | NO-TRADE | REJECT (-5.2%) | REJECT (+12.5%, PF 1.32) | REJECT (-0.5%) |
| 6 | Role reversal pullback | REJECT (3 trades) | **NEAR-MISS (+5.6%, PF 1.35, 30 trades, only Sharpe/Sortino/Calmar failed)** | REJECT (1 trade) | REJECT (-1.6%) | REJECT (2 trades) |
| 7 | Bollinger band squeeze | REJECT (+8.4%, PF 1.34, low trades) | REJECT (2 trades) | REJECT (-1.1%) | REJECT (-1.6%) | REJECT (-26.1%) |
| 8 | Narrow range NR4/NR7 breakout | REJECT (-0.7%) | REJECT (-10.8%) | REJECT (-5.9%) | REJECT (-55.4%) | REJECT (-0.8%) |
| 9 | RSI(2) momentum (as written: buy >90) | REJECT (-2.4%) | NEAR-MISS (-12.9%, 107 trades) | REJECT (-12.5%) | REJECT (+21.7%, PF 2.20, 67 trades) | REJECT (-18.5%) |

## Strategy-by-strategy assessment

1. **Momentum reversal (stochastic exhaustion fade)** - the PDF version depends on COT
   fundamentals that cannot be backtested; the mechanical core loses in all five classes and
   is catastrophic on commodities (-88.9%). Fading momentum extremes without a structural
   edge is paying the trend. **Dead in all variants.**
2. **Triple MA crossover** - the only strategy with positive OOS in 4 of 5 classes; trend
   following "works" directionally, but 20/60/100 on daily bars produces 7-29 trades per
   validation slice, so nothing is statistically establishable, and risk-adjusted returns
   (Sharpe 0.1-0.9) are far below gates. Stocks PF 4.34 on 7 trades is noise, not evidence.
   If pursued at all: faster parameters or a large multi-instrument portfolio to get the
   trade count up - that becomes a different (standard CTA) strategy.
3. **Heikin-Ashi exhaustion reversal** - best single cell of the program: commodities-1d
   +24.0%, PF 3.66, 59 trades; still failed 6 hard gates (Sharpe 0.56, walk-forward decay,
   parameter fragility). Loses badly on forex (-27.9%), the asset class the PDF targets.
   The commodities cell is the one result worth one pre-registered re-test
   (single-cell success across a 5x9 grid is exactly where multiple-comparisons luck lives).
4. **Swing compression continuation** - codegen never converged to a firing implementation
   on crypto and stocks (zero trades through 6 fix iterations); weak where it traded.
   The PDF describes a discretionary pattern ("lots and lots of overlap") that resists
   mechanical specification. Verdict: not falsified, but not implementable as specified.
5. **Candlestick reversal patterns** - closest on crypto (failed only Sharpe + walk-forward
   + parameter gates), but OOS negative there; positive only on commodities with broad gate
   failures. No cell justifies further spend.
6. **Role reversal pullback** - the best overall near-miss: on stocks it passed 13 of 16
   gates including trade count, positive OOS, profit factor, drawdown, walk-forward, cost
   stress and parameter stability - failing only the three risk-adjusted-return bars
   (Sharpe 0.31 vs 1.0 required). Honest read: a real but weak long-bias pullback effect on
   trending mega-cap stocks; underperforms buy-and-hold (5.6% vs 393.8% over the slice).
   Everywhere else it barely fires. Not promotable; not embarrassing.
7. **Bollinger squeeze** - crypto cell is positive (+8.4%, PF 1.34) but on too few trades
   with broad gate failures; forex (-26.1%) refutes the PDF's home turf. Reject.
8. **Narrow range breakout** - rejected in all five classes; commodities shows PF 1.77 with
   -55% return, i.e. occasional large wins swamped by sizing-asymmetric losses. The cleanest
   full-grid refutation in the program. Reject permanently.
9. **RSI(2) momentum** - tested as written (buy RSI2>90), which inverts the Connors
   mean-reversion convention; it loses nearly everywhere. The commodities cell (+21.7%,
   PF 2.20) is again a single-cell outlier. The natural variant - the mean-reversion
   inversion (buy <10 in an uptrend) - was NOT what the PDF prescribes and was not tested;
   it is the only follow-up with an a-priori literature basis.

## Caveats (read before quoting any number)

- Costs are standardized (12bps/side round-trip-ish), conservative for FX/indices and about
  right for crypto taker + stock commissions. A strategy that cannot survive these costs is
  not worth venue-specific cost modeling.
- Forex datasets have no real volume (Yahoo); volume-dependent filters on FX cells are inert.
- Gates evaluate the FIRST instrument of each class basket (BTC, AAPL, SPX, WTI, EURUSD);
  the other two instruments' runs are recorded in the store but do not drive pass/fail.
- Strategy 1 was reduced to its mechanical core (COT/fundamental overlay untestable).
- 2026 validation slices for daily data cover recent regimes; long-only underperformance vs
  B&H on stocks/indices partially reflects the secular bull in the test window.
- LLM coding variance: each cell is ONE implementation of the idea (plus up to 5 debug
  fixes and the parameter battery). NO-TRADE cells are implementation failures, not
  hypothesis falsifications.

## Bottom line

None of the nine PDF strategies, in any of the 45 class-variants tested, comes close to the
promotion gates after realistic costs. The PDF's own home market (forex) is where most of
them fail hardest. The two cells that warrant any further attention - Heikin-Ashi on
commodities and role-reversal on stocks - are single-cell positives in a 45-cell grid and
must be treated as multiple-comparisons suspects: one pre-registered re-test each, at most,
with the falsification criterion written before running (as done for FundingDivergenceV2,
which died exactly that way).

Program artifacts: leaderboard + rejected/promoted CSVs in
`src/data/research_factory/reports/`, every run queryable by spec_id in `research.db`.

## Addendum (2026-06-11): pre-registered re-tests - both dead

Protocol: cross-instrument replication of the exact original modules, zero parameter
changes, criteria written before inspection.

**HA-commodities (spec_789c9b2538)** - criterion: GLD AND SLV validation return > 0 AND
PF >= 1.2 AND >= 30 trades. Result: GLD +7.65%/PF 1.56/51 trades (pass);
SLV +4.24%/PF 1.107/50 trades (**fail on PF**). Additionally, train returns were negative
on all three commodities (-18.8% to -26.6%): the strategy is profitable only inside the
validation window. Verdict: regime artifact; the WTI +24%/PF 3.66 cell was the luckiest
draw of a regime-dependent pattern. RETIRED.

**Role-reversal-stocks (spec_5d6e36850c)** - criterion: MSFT AND JPM validation return > 0
AND PF >= 1.2. Result: MSFT +5.59%/PF 1.35 (pass); JPM **-5.30%/PF 0.69** (fail). Train
negative on 2 of 3 stocks. Verdict: a mega-cap-tech-in-the-test-window effect, not a role
reversal edge. RETIRED.

Final tally for the PDF program: 9 strategies, 45 class-variants, 2 pre-registered
re-tests, **0 survivors**.
