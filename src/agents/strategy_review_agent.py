"""
Strategy Review Agent - lookahead/bias review of generated strategy modules.

Two layers:
  1. STATIC checks (deterministic, hard): regex/AST scans for lookahead patterns,
     nondeterminism, I/O, and contract violations. A hard finding fails the
     static_review_clean validation gate.
  2. LLM review (advisory, soft): a second pair of eyes for logic-level issues.
     The LLM can never pass a strategy - only flag additional concerns.

Standalone:
  python src/agents/strategy_review_agent.py --file PATH [--llm] [--spec-id ID]
"""

import argparse
import ast
import re
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

try:
    from termcolor import cprint
except ImportError:  # minimal envs: plain print, no color
    def cprint(text, *_args, **_kwargs):
        print(text)


@dataclass
class Finding:
    severity: str          # "hard" | "soft"
    rule: str
    detail: str
    line: Optional[int] = None


# (pattern, rule, detail, severity)
STATIC_PATTERNS = [
    (r"\.shift\(\s*-", "negative_shift",
     "shift(-n) reads future bars", "hard"),
    (r"center\s*=\s*True", "centered_window",
     "centered rolling window uses future data", "hard"),
    (r"\[\s*i\s*\+\s*1\s*\]", "forward_index",
     "indexing i+1 may read a future bar", "soft"),
    (r"datetime\.now\(|time\.time\(", "wall_clock",
     "wall-clock time makes the strategy nondeterministic", "hard"),
    (r"\brandom\.|np\.random\.|default_rng", "randomness",
     "randomness in signal logic is forbidden", "hard"),
    (r"open\(|read_csv|to_csv|Path\(", "file_io",
     "strategy modules must not perform file I/O", "hard"),
    (r"requests\.|urllib|httpx|aiohttp", "network",
     "strategy modules must not perform network I/O", "hard"),
    (r"\.iloc\[\s*-1\s*\]\s*.*\.max\(\)|\.min\(\)\s*\)?\s*$", "global_stat",
     "possible normalization by full-dataset statistic", "soft"),
    (r"print\(", "print_call",
     "no prints in strategy modules (harness parses nothing from stdout)", "soft"),
    (r"Backtest\(", "self_backtest",
     "module must not run its own Backtest() - harness controls costs/data", "hard"),
    (r"/Users/|C:\\\\", "hardcoded_path",
     "hardcoded filesystem path (legacy failure mode: 3,699 dead-path files)", "hard"),
]


def _ast_checks(code: str) -> List[Finding]:
    findings: List[Finding] = []
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return [Finding("hard", "syntax_error", str(e), e.lineno)]
    has_strategy_class = False
    has_params = False
    has_generate_signal = False
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name) and t.id == "STRATEGY_CLASS":
                    has_strategy_class = True
                if isinstance(t, ast.Name) and t.id == "PARAMS":
                    has_params = True
        if isinstance(node, ast.FunctionDef) and node.name == "generate_signal":
            has_generate_signal = True
    if not has_strategy_class:
        findings.append(Finding("hard", "contract", "missing STRATEGY_CLASS assignment"))
    if not has_params:
        findings.append(Finding("hard", "contract", "missing PARAMS dict"))
    if not has_generate_signal:
        findings.append(Finding("soft", "contract",
                                "missing generate_signal(df) - paper wrapper will not work"))
    return findings


def static_review(code: str) -> List[Finding]:
    findings = _ast_checks(code)
    for i, line in enumerate(code.splitlines(), start=1):
        stripped = line.strip()
        if stripped.startswith("#"):
            continue
        for pattern, rule, detail, severity in STATIC_PATTERNS:
            if re.search(pattern, line):
                findings.append(Finding(severity, rule, detail, i))
    return findings


def is_clean(findings: List[Finding]) -> bool:
    return not any(f.severity == "hard" for f in findings)


LLM_REVIEW_PROMPT = """You are a skeptical senior quant code reviewer. Review this
backtesting.py strategy module for: lookahead bias, same-bar impossible fills,
indicator misuse, position-sizing errors, stop-loss logic that cannot fill realistically,
and divergence between STRATEGY_CLASS logic and generate_signal(df). List concrete issues
with line references. If you find none, say "NO ADDITIONAL ISSUES". You cannot approve a
strategy; you can only flag problems."""


def llm_review(code: str, config: Dict) -> str:
    from src.models.model_factory import ModelFactory
    factory = ModelFactory()
    model = factory.get_model(config["llm"]["provider"], config["llm"].get("model"))
    if model is None:
        raise RuntimeError("ModelFactory could not provide the review model - check .env keys")
    response = model.generate_response(
        system_prompt=LLM_REVIEW_PROMPT,
        user_content=code,
        temperature=config["llm"]["temperature"],
        max_tokens=config["llm"]["max_tokens"],
    )
    return response.content


def review_file(path: Path, use_llm: bool = False,
                spec_id: Optional[str] = None,
                store=None) -> Dict:
    code = Path(path).read_text()
    findings = static_review(code)
    clean = is_clean(findings)
    result = {
        "file": str(path),
        "static_clean": clean,
        "findings": [asdict(f) for f in findings],
        "llm_review": None,
    }
    if use_llm:
        from src.research.factory_config import load_factory_config
        result["llm_review"] = llm_review(code, load_factory_config())
    if store is not None and spec_id is not None:
        store.record_experiment(
            spec_id, "review", clean,
            metrics={"findings": result["findings"],
                     "llm_review": (result["llm_review"] or "")[:4000]},
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Static + LLM strategy code review")
    parser.add_argument("--file", required=True, help="strategy module to review")
    parser.add_argument("--llm", action="store_true", help="add advisory LLM review")
    parser.add_argument("--spec-id", default=None, help="record result against this spec")
    args = parser.parse_args()

    store = None
    if args.spec_id:
        from src.research.experiment_store import ExperimentStore
        store = ExperimentStore()

    result = review_file(Path(args.file), use_llm=args.llm,
                         spec_id=args.spec_id, store=store)
    color = "green" if result["static_clean"] else "red"
    cprint(f"static review: {'CLEAN' if result['static_clean'] else 'HARD FINDINGS'}", color)
    for f in result["findings"]:
        line = f" (line {f['line']})" if f["line"] else ""
        cprint(f"  [{f['severity']}] {f['rule']}{line}: {f['detail']}",
               "red" if f["severity"] == "hard" else "yellow")
    if result["llm_review"]:
        cprint("\nLLM advisory review:", "cyan")
        print(result["llm_review"])
    sys.exit(0 if result["static_clean"] else 1)


if __name__ == "__main__":
    main()
