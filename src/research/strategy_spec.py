"""
StrategySpec - the structured contract every strategy idea must be converted into
before any compute is spent on it.

A spec is rejected at validation time if required fields are missing or if its
semantic fingerprint duplicates an existing spec in the experiment store.
"""

import hashlib
import json
import re
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

VALID_ASSET_CLASSES = {"crypto_perp", "crypto_spot", "equity", "etf", "index", "future", "fx"}
VALID_FREQUENCIES = {"intraday", "swing", "position", "low_frequency"}
VALID_FAMILIES = {
    "trend_following", "momentum_breakout", "mean_reversion", "vwap_deviation",
    "session_behavior", "volatility_breakout", "funding_driven", "liquidation_driven",
    "oi_driven", "regime_switching", "portfolio_allocation", "other",
}

REQUIRED_TEXT_FIELDS = [
    "name", "hypothesis", "entry_logic", "exit_logic", "stop_logic",
    "position_sizing", "risk_rules", "invalidation_rules",
]


@dataclass
class StrategySpec:
    # identity
    name: str
    family: str                       # one of VALID_FAMILIES
    asset_class: str                  # one of VALID_ASSET_CLASSES
    instruments: List[str]            # e.g. ["BTC-USD"]
    timeframe: str                    # e.g. "15m"

    # the actual strategy contract
    hypothesis: str                   # WHY this should make money (market mechanism)
    regime_assumptions: str           # in which regimes it is expected to work / fail
    entry_logic: str
    exit_logic: str
    stop_logic: str
    position_sizing: str
    risk_rules: str
    invalidation_rules: str           # what observation kills the hypothesis

    # research constraints
    expected_trade_frequency: str = "intraday"   # one of VALID_FREQUENCIES
    low_frequency: bool = False                  # relaxes min-trade gate (recorded, never silent)
    intentionally_single_asset: bool = False
    costs_to_model: Dict[str, Any] = field(default_factory=dict)   # overrides, conservative-only
    data_requirements: Dict[str, Any] = field(default_factory=dict)  # {symbols, timeframe, min_bars}
    forbidden_assumptions: List[str] = field(default_factory=list)
    parameters: Dict[str, Any] = field(default_factory=dict)        # named tunables with defaults

    # provenance
    source: str = ""                  # original idea text / file / URL
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    spec_id: Optional[str] = None     # assigned by the experiment store

    def validate(self) -> List[str]:
        """Return a list of validation errors. Empty list means valid."""
        errors: List[str] = []
        for f_name in REQUIRED_TEXT_FIELDS:
            value = getattr(self, f_name, "")
            if not isinstance(value, str) or len(value.strip()) < 5:
                errors.append(f"field '{f_name}' missing or too short")
        if self.family not in VALID_FAMILIES:
            errors.append(f"family '{self.family}' not in {sorted(VALID_FAMILIES)}")
        if self.asset_class not in VALID_ASSET_CLASSES:
            errors.append(f"asset_class '{self.asset_class}' not in {sorted(VALID_ASSET_CLASSES)}")
        if self.expected_trade_frequency not in VALID_FREQUENCIES:
            errors.append(
                f"expected_trade_frequency '{self.expected_trade_frequency}' "
                f"not in {sorted(VALID_FREQUENCIES)}"
            )
        if not self.instruments:
            errors.append("instruments list is empty")
        if not self.timeframe:
            errors.append("timeframe missing")
        # placeholder-name guard: the legacy pipeline shipped strategies literally named
        # "Uniquetwo-wordname" because the LLM echoed the prompt template.
        lowered = self.name.lower().replace(" ", "")
        for marker in ("unique", "placeholder", "unknown", "yourstrategy", "two-word"):
            if marker in lowered:
                errors.append(f"name '{self.name}' looks like an LLM template placeholder")
                break
        if self.low_frequency and self.expected_trade_frequency == "intraday":
            errors.append("low_frequency=true contradicts expected_trade_frequency='intraday'")
        return errors

    def fingerprint(self) -> str:
        """
        Semantic fingerprint used for duplicate detection. Name is deliberately
        excluded - the legacy pipeline produced 50+ near-identical strategies under
        thesaurus names. Logic text is canonicalized (lowercase, alphanumeric token
        sort) so trivial rewording maps to the same fingerprint.
        """
        def canon(text: str) -> str:
            tokens = re.findall(r"[a-z0-9]+", text.lower())
            return " ".join(sorted(set(tokens)))

        basis = json.dumps({
            "family": self.family,
            "asset_class": self.asset_class,
            "timeframe": self.timeframe,
            "entry": canon(self.entry_logic),
            "exit": canon(self.exit_logic),
            "stop": canon(self.stop_logic),
        }, sort_keys=True)
        return hashlib.sha256(basis.encode()).hexdigest()[:16]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_json())

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "StrategySpec":
        known = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        unknown = set(data) - known
        if unknown:
            raise ValueError(f"StrategySpec got unknown fields: {sorted(unknown)}")
        return cls(**data)

    @classmethod
    def load(cls, path: Path) -> "StrategySpec":
        return cls.from_dict(json.loads(Path(path).read_text()))
