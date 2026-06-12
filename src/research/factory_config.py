"""
Configuration loading for the research factory.

All paths in YAML are relative to the project root (the directory containing src/).
Fails loudly if a config file is missing or malformed - no silent defaults for
files that should exist.
"""

from pathlib import Path
from typing import Any, Dict

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CONFIGS_DIR = PROJECT_ROOT / "configs"

FACTORY_CONFIG_PATH = CONFIGS_DIR / "research_factory.yaml"
GATES_CONFIG_PATH = CONFIGS_DIR / "validation_gates.yaml"
BROKER_PROFILES_PATH = CONFIGS_DIR / "broker_profiles.yaml"


def _load_yaml(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Required config file missing: {path}")
    with open(path, "r") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Config file {path} did not parse to a mapping")
    return data


def load_factory_config() -> Dict[str, Any]:
    return _load_yaml(FACTORY_CONFIG_PATH)


def load_gates_config() -> Dict[str, Any]:
    return _load_yaml(GATES_CONFIG_PATH)


def load_broker_profiles() -> Dict[str, Any]:
    return _load_yaml(BROKER_PROFILES_PATH)


def resolve_path(relative: str) -> Path:
    """Resolve a config-relative path against the project root."""
    p = Path(relative)
    return p if p.is_absolute() else PROJECT_ROOT / p


def factory_root(config: Dict[str, Any]) -> Path:
    root = resolve_path(config["paths"]["root"])
    root.mkdir(parents=True, exist_ok=True)
    return root


def python_executable(config: Dict[str, Any]) -> str:
    import sys

    exe = (config.get("python_executable") or "").strip()
    return exe if exe else sys.executable
