#!/bin/sh
# Generic driver: run an ideas file across all 5 asset classes sequentially.
# Usage: sh src/scripts/run_ideas_cycles.sh <ideas-file>
# Two passes per class (pass 2 mops up transient failures; processed ideas skip).
set -u
cd /Users/sam/ai-agents
PY=.venv/bin/python
IDEAS="${1:?usage: run_ideas_cycles.sh <ideas-file>}"
LOG_FILTER='Backtest.run|bar/s|invalid value encountered|c /= stddev|_function_base_impl|UserWarning|bt = Backtest|warnings.warn|np.where'

run_cycle() {
  name="$1"; tf="$2"; shift 2
  echo "CYCLE $name START $(date -u +%H:%M:%S)"
  for pass in 1 2; do
    $PY src/agents/quant_research_agent.py --mode research-cycle \
      --ideas-file "$IDEAS" --timeframe "$tf" --instruments "$@" 2>&1 \
      | grep -vE "$LOG_FILTER"
  done
  echo "CYCLE $name DONE $(date -u +%H:%M:%S)"
}

run_cycle crypto      4h BTC-USD ETH-USD SOL-USD
run_cycle stocks      1d AAPL MSFT JPM
run_cycle indices     1d SPX NDX DJI
run_cycle commodities 1d WTI GLD SLV
run_cycle forex       1d EURUSD USDJPY GBPUSD

echo "ALL_CYCLES_DONE $(date -u +%H:%M:%S)"
