#!/bin/sh
set -u
cd /Users/sam/ai-agents
PY=.venv/bin/python
F='Backtest.run|bar/s|invalid value|c /= stddev|_function_base|UserWarning|bt = Backtest|warnings.warn'
echo "EVID-1 ratio reversion START $(date -u +%H:%M:%S)"
$PY src/agents/quant_research_agent.py --mode research-cycle --ideas-file src/data/research_factory/ideas_rv.txt --timeframe 1h --datasets ETHBTC-1h SOLBTC-1h 2>&1 | grep -vE "$F"
echo "EVID-2 funding settlement START $(date -u +%H:%M:%S)"
$PY src/agents/quant_research_agent.py --mode research-cycle --ideas-file src/data/research_factory/ideas_fsettle.txt --timeframe 1h --datasets BTC-SIG-1h ETH-SIG-1h SOL-SIG-1h 2>&1 | grep -vE "$F"
echo "EVID-4 lead-lag START $(date -u +%H:%M:%S)"
$PY src/agents/quant_research_agent.py --mode research-cycle --ideas-file src/data/research_factory/ideas_leadlag.txt --timeframe 1h --datasets ETH-XL-1h SOL-XL-1h 2>&1 | grep -vE "$F"
echo "EVID-3 volume shock wide START $(date -u +%H:%M:%S)"
$PY src/scripts/run_wide_universe.py --name VolumeShockWide --module src/data/research_factory/strategies_v2/volume_shock_drift.py --suffix -1d 2>&1 | grep -vE "$F"
echo "EVID-5 idio gap wide START $(date -u +%H:%M:%S)"
$PY src/scripts/run_wide_universe.py --name IdioGapWide --module src/data/research_factory/strategies_v2/idio_gap_reversion.py --suffix -XG-1d 2>&1 | grep -vE "$F"
echo "EVIDENCE_BATCH_DONE $(date -u +%H:%M:%S)"
