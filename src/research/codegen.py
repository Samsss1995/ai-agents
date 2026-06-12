"""
Codegen - the only place the LLM touches the research pipeline.

Roles: idea -> StrategySpec extraction, spec -> strategy module generation,
bounded debug fixes. The LLM never sets thresholds, never sees gate logic, and
its output is always re-validated mechanically (spec.validate(), static review,
contract check, backtest execution).
"""

import json
import re
from pathlib import Path
from typing import Dict, Optional, Tuple

from src.models.model_factory import ModelFactory
from src.research.strategy_spec import StrategySpec, VALID_FAMILIES, VALID_ASSET_CLASSES

SPEC_SYSTEM_PROMPT = """You are a senior quantitative researcher. Convert a raw trading idea
into a rigorous StrategySpec JSON object. Be skeptical: if the idea has no plausible market
mechanism, say so in the hypothesis honestly rather than inventing one.

Return ONLY a JSON object (no markdown fences, no commentary) with exactly these keys:
name (CamelCase, two concrete words derived from the actual logic - NEVER a placeholder),
family (one of: {families}),
asset_class (one of: {asset_classes}),
instruments (list, e.g. ["BTC-USD"]),
timeframe (e.g. "15m"),
hypothesis (the market mechanism: WHO is on the losing side and WHY, >= 2 sentences),
regime_assumptions (when it works, when it fails),
entry_logic, exit_logic, stop_logic, position_sizing, risk_rules,
invalidation_rules (what observation would kill the hypothesis),
expected_trade_frequency (one of: intraday, swing, position, low_frequency),
low_frequency (bool),
intentionally_single_asset (bool),
forbidden_assumptions (list of strings, e.g. ["no same-bar fills", "no future data"]),
parameters (dict of named numeric tunables with sensible default values, 2-6 entries),
costs_to_model (dict, usually empty; set {{"exclusive_orders": false}} ONLY for grid/pyramiding
strategies that need multiple concurrent entries),
data_requirements (dict with keys: timeframe, min_bars).
"""

CODE_SYSTEM_PROMPT = """You are a senior Python quant developer. Generate a SINGLE Python module
implementing the given StrategySpec as a backtesting.py strategy. Follow this contract EXACTLY:

1. Imports allowed: backtesting, talib, pandas_ta, pandas, numpy, math. Nothing else.
2. Define a dict PARAMS with the spec's numeric parameters and defaults.
3. Define STRATEGY_CLASS = <YourClass>, where <YourClass> subclasses backtesting.Strategy.
   Every PARAMS key MUST also be a class attribute with the same default so the harness can
   override it via bt.run(param=value).
4. The module must NOT load any data, NOT call Backtest(), NOT print, NOT plot, and NOT
   reference any file paths. The harness supplies data and costs.
5. Indicators: compute in init() with self.I() wrapping talib/pandas_ta/numpy functions.
   NEVER compute indicators in next().
6. Absolutely NO lookahead: never index future bars, never use .shift(-n), never use
   centered rolling windows, never normalize by full-dataset statistics, no randomness,
   no datetime.now(). Use self.data.X[-1] for current and [-2] for previous values.
7. Position sizing: ALWAYS a fraction of equity: self.buy(size=f) / self.sell(size=f) with
   strictly 0 < f < 1. NEVER pass unit counts, never any value >= 1, never equity/price math.
   Map the spec's sizing rule onto a clamped fraction, e.g.
   f = min(0.5, max(0.01, intended_risk_fraction)). Volatility scaling adjusts f, still < 1.
8. Stops: use the sl=/tp= arguments of self.buy()/self.sell() so fills are modeled by the engine.
9. Also define a pure function generate_signal(df) -> dict with keys
   {"action": "BUY"|"SELL"|"NOTHING", "confidence": 0-100, "reasoning": str} that evaluates
   the SAME logic on a pandas OHLCV DataFrame (columns Open/High/Low/Close/Volume); used later
   for paper trading. It must only look at the last completed bar and earlier.
10. Deterministic: same data in, same signals out. No network, no I/O.
11. One position at a time (the engine closes the prior position on a new order) UNLESS the
    spec's costs_to_model sets exclusive_orders=false - only then may the strategy hold
    multiple concurrent entries (grid/pyramid). Grid strategies MUST include a hard
    flatten-all kill condition (regime exit or max adverse excursion).
12. If the spec's data_requirements lists signal_columns (e.g. FundingRate, Premium),
    those exist as EXTRA data columns: self.data.FundingRate in the Strategy (wrap with
    self.I(lambda x: x, self.data.FundingRate) if you need array history), and
    df["FundingRate"] inside generate_signal(df). Treat NaN values as no-signal.
    Never invent columns not listed.

Return ONLY the Python code (no markdown fences)."""

DEBUG_SYSTEM_PROMPT = """You are a senior Python quant developer. The following backtesting.py
strategy module failed. Fix the error while preserving the strategy logic and the module
contract (PARAMS dict, STRATEGY_CLASS, generate_signal(df), no data loading, no prints,
no lookahead). Return ONLY the corrected full module code (no markdown fences)."""


class CodegenError(Exception):
    pass


def _get_model(config: Dict):
    factory = ModelFactory()
    llm = config["llm"]
    model = factory.get_model(llm["provider"], llm.get("model"))
    if model is None:
        raise CodegenError(
            f"ModelFactory could not provide '{llm['provider']}' - check API keys in .env"
        )
    return model


def _strip_fences(text: str) -> str:
    text = text.strip()
    m = re.match(r"^```(?:python|json)?\s*(.*?)\s*```$", text, re.DOTALL)
    return m.group(1) if m else text


def idea_to_spec(idea_text: str, config: Dict, default_timeframe: str = "15m",
                 default_instruments: Optional[list] = None) -> StrategySpec:
    model = _get_model(config)
    llm = config["llm"]
    system = SPEC_SYSTEM_PROMPT.format(
        families=", ".join(sorted(VALID_FAMILIES)),
        asset_classes=", ".join(sorted(VALID_ASSET_CLASSES)),
    )
    instruments = default_instruments or ["BTC-USD", "ETH-USD", "SOL-USD"]
    response = model.generate_response(
        system_prompt=system,
        user_content=f"Trading idea:\n{idea_text}\n\n"
                     f"This research cycle targets timeframe {default_timeframe} and "
                     f"instruments {instruments}. Use BOTH (multi-asset spec, "
                     f"intentionally_single_asset=false) unless the idea inherently "
                     f"requires a specific timeframe or a single instrument.",
        temperature=llm["temperature"],
        max_tokens=llm["max_tokens"],
    )
    raw = _strip_fences(response.content)
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        raise CodegenError(f"LLM did not return valid spec JSON: {e}\nFirst 500 chars:\n{raw[:500]}")
    data["source"] = idea_text[:2000]
    data.setdefault("costs_to_model", {})
    spec = StrategySpec.from_dict(data)
    errors = spec.validate()
    if errors:
        raise CodegenError(f"generated spec invalid: {errors}")
    return spec


def spec_to_code(spec: StrategySpec, output_dir: Path, config: Dict) -> Tuple[Path, str]:
    """Generate the strategy module. Returns (path, model_name_used)."""
    model = _get_model(config)
    llm = config["llm"]
    response = model.generate_response(
        system_prompt=CODE_SYSTEM_PROMPT,
        user_content=f"StrategySpec:\n{spec.to_json()}",
        temperature=llm["temperature"],
        max_tokens=llm["max_tokens"],
    )
    code = _strip_fences(response.content)
    if "STRATEGY_CLASS" not in code or "PARAMS" not in code:
        raise CodegenError("generated code violates contract (missing STRATEGY_CLASS/PARAMS)")
    output_dir.mkdir(parents=True, exist_ok=True)
    safe_name = "".join(c for c in spec.name if c.isalnum() or c == "_")
    path = output_dir / f"{spec.spec_id or 'spec'}_{safe_name}.py"
    path.write_text(code)
    return path, getattr(model, "model_name", llm["provider"])


def debug_fix(module_path: Path, error_message: str, config: Dict) -> Path:
    """One bounded LLM fix iteration. Caller enforces max_debug_iterations."""
    model = _get_model(config)
    llm = config["llm"]
    code = Path(module_path).read_text()
    response = model.generate_response(
        system_prompt=DEBUG_SYSTEM_PROMPT,
        user_content=f"ERROR:\n{error_message[:4000]}\n\nMODULE:\n{code}",
        temperature=llm["temperature"],
        max_tokens=llm["max_tokens"],
    )
    fixed = _strip_fences(response.content)
    if "STRATEGY_CLASS" not in fixed or "PARAMS" not in fixed:
        raise CodegenError("debug fix violated contract (missing STRATEGY_CLASS/PARAMS)")
    Path(module_path).write_text(fixed)
    return Path(module_path)
