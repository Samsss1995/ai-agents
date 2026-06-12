"""
Quant Research Factory - research -> backtest -> validate -> promote pipeline.

Modules:
    factory_config    - YAML config loading (configs/research_factory.yaml etc.)
    strategy_spec     - StrategySpec structure, validation, fingerprinting
    data_catalog      - OHLCV inventory, quality validation, refusal logic
    experiment_store  - SQLite store: ideas, specs, code versions, all runs, gates, promotions
    codegen           - LLM idea->spec and spec->code generation (ModelFactory)
    backtest_runner   - costed backtesting.py execution on train/validation/test slices
    metrics           - institutional metric set extraction
    validation_gates  - mechanical promotion gates (configs/validation_gates.yaml)
    robustness        - walk-forward, parameter neighborhood, Monte Carlo, cost stress
    leaderboard       - composite scoring + CSV outputs
    promotion         - status state machine, manual-approval enforcement
    report_writer     - markdown reports
"""
