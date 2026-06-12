"""
Experiment store - the research memory layer.

SQLite database recording EVERY idea, spec, code version, backtest run (success,
failure, and rejection), gate evaluation, and promotion. Nothing is deleted;
rejected work stays queryable so researcher bias is measurable.
"""

import hashlib
import json
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.research.factory_config import load_factory_config, factory_root
from src.research.strategy_spec import StrategySpec

SCHEMA = """
CREATE TABLE IF NOT EXISTS ideas (
    idea_id     TEXT PRIMARY KEY,
    source      TEXT NOT NULL,
    text        TEXT NOT NULL,
    created_at  TEXT NOT NULL,
    status      TEXT NOT NULL DEFAULT 'new'
);
CREATE TABLE IF NOT EXISTS specs (
    spec_id     TEXT PRIMARY KEY,
    idea_id     TEXT,
    name        TEXT NOT NULL,
    fingerprint TEXT NOT NULL UNIQUE,
    spec_json   TEXT NOT NULL,
    created_at  TEXT NOT NULL,
    status      TEXT NOT NULL DEFAULT 'research_only'
);
CREATE TABLE IF NOT EXISTS code_versions (
    code_id     TEXT PRIMARY KEY,
    spec_id     TEXT NOT NULL,
    version     INTEGER NOT NULL,
    file_path   TEXT NOT NULL,
    code_hash   TEXT NOT NULL,
    model_used  TEXT,
    created_at  TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS experiments (
    exp_id      TEXT PRIMARY KEY,
    spec_id     TEXT NOT NULL,
    code_id     TEXT,
    kind        TEXT NOT NULL,   -- train|validation|test|walk_forward|param_neighborhood|monte_carlo|cost_stress|debug
    dataset_id  TEXT,
    params_json TEXT,
    metrics_json TEXT,
    success     INTEGER NOT NULL,
    error       TEXT,
    config_hash TEXT,
    seed        INTEGER,
    created_at  TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS gate_results (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    spec_id     TEXT NOT NULL,
    code_id     TEXT,
    gate_name   TEXT NOT NULL,
    passed      INTEGER NOT NULL,
    value       TEXT,
    threshold   TEXT,
    hard        INTEGER NOT NULL,
    created_at  TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS promotions (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    spec_id     TEXT NOT NULL,
    from_status TEXT NOT NULL,
    to_status   TEXT NOT NULL,
    approved_by TEXT,
    notes       TEXT,
    created_at  TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS rejections (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    spec_id     TEXT,
    code_id     TEXT,
    stage       TEXT NOT NULL,
    reason      TEXT NOT NULL,
    created_at  TEXT NOT NULL
);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:10]}"


class ExperimentStore:
    def __init__(self, db_path: Optional[Path] = None):
        if db_path is None:
            db_path = factory_root(load_factory_config()) / "research.db"
        self.db_path = Path(db_path)
        self.conn = sqlite3.connect(str(self.db_path))
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(SCHEMA)
        self.conn.commit()

    # ---------- ideas ----------
    def add_idea(self, text: str, source: str) -> str:
        idea_id = _new_id("idea")
        self.conn.execute(
            "INSERT INTO ideas (idea_id, source, text, created_at) VALUES (?,?,?,?)",
            (idea_id, source, text, _now()),
        )
        self.conn.commit()
        return idea_id

    def set_idea_status(self, idea_id: str, status: str) -> None:
        self.conn.execute("UPDATE ideas SET status=? WHERE idea_id=?", (status, idea_id))
        self.conn.commit()

    # ---------- specs ----------
    def add_spec(self, spec: StrategySpec, idea_id: Optional[str] = None) -> str:
        """Insert a spec. Raises ValueError on duplicate fingerprint (dedup gate)."""
        fingerprint = spec.fingerprint()
        existing = self.conn.execute(
            "SELECT spec_id, name FROM specs WHERE fingerprint=?", (fingerprint,)
        ).fetchone()
        if existing:
            raise ValueError(
                f"duplicate spec: fingerprint {fingerprint} already exists as "
                f"{existing['spec_id']} ('{existing['name']}')"
            )
        spec_id = _new_id("spec")
        spec.spec_id = spec_id
        self.conn.execute(
            "INSERT INTO specs (spec_id, idea_id, name, fingerprint, spec_json, created_at) "
            "VALUES (?,?,?,?,?,?)",
            (spec_id, idea_id, spec.name, fingerprint, spec.to_json(), _now()),
        )
        self.conn.commit()
        return spec_id

    def get_spec(self, spec_id: str) -> StrategySpec:
        row = self.conn.execute("SELECT spec_json FROM specs WHERE spec_id=?", (spec_id,)).fetchone()
        if row is None:
            raise KeyError(f"spec '{spec_id}' not found")
        return StrategySpec.from_dict(json.loads(row["spec_json"]))

    def get_spec_status(self, spec_id: str) -> str:
        row = self.conn.execute("SELECT status FROM specs WHERE spec_id=?", (spec_id,)).fetchone()
        if row is None:
            raise KeyError(f"spec '{spec_id}' not found")
        return row["status"]

    def set_spec_status(self, spec_id: str, status: str) -> None:
        self.conn.execute("UPDATE specs SET status=? WHERE spec_id=?", (status, spec_id))
        self.conn.commit()

    def list_specs(self, status: Optional[str] = None) -> List[Dict[str, Any]]:
        q = "SELECT spec_id, name, fingerprint, status, created_at FROM specs"
        params: tuple = ()
        if status:
            q += " WHERE status=?"
            params = (status,)
        return [dict(r) for r in self.conn.execute(q + " ORDER BY created_at", params)]

    # ---------- code versions ----------
    def add_code_version(self, spec_id: str, file_path: Path, model_used: str) -> str:
        code = Path(file_path).read_text()
        code_hash = hashlib.sha256(code.encode()).hexdigest()[:16]
        row = self.conn.execute(
            "SELECT COALESCE(MAX(version), 0) AS v FROM code_versions WHERE spec_id=?", (spec_id,)
        ).fetchone()
        version = row["v"] + 1
        code_id = _new_id("code")
        self.conn.execute(
            "INSERT INTO code_versions (code_id, spec_id, version, file_path, code_hash, "
            "model_used, created_at) VALUES (?,?,?,?,?,?,?)",
            (code_id, spec_id, version, str(file_path), code_hash, model_used, _now()),
        )
        self.conn.commit()
        return code_id

    def latest_code(self, spec_id: str) -> Optional[Dict[str, Any]]:
        row = self.conn.execute(
            "SELECT * FROM code_versions WHERE spec_id=? ORDER BY version DESC LIMIT 1", (spec_id,)
        ).fetchone()
        return dict(row) if row else None

    # ---------- experiments ----------
    def record_experiment(self, spec_id: str, kind: str, success: bool,
                          code_id: Optional[str] = None, dataset_id: Optional[str] = None,
                          params: Optional[Dict] = None, metrics: Optional[Dict] = None,
                          error: Optional[str] = None, seed: Optional[int] = None,
                          config_hash: Optional[str] = None) -> str:
        exp_id = _new_id("exp")
        self.conn.execute(
            "INSERT INTO experiments (exp_id, spec_id, code_id, kind, dataset_id, params_json, "
            "metrics_json, success, error, config_hash, seed, created_at) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
            (exp_id, spec_id, code_id, kind, dataset_id,
             json.dumps(params or {}), json.dumps(metrics or {}, default=str),
             int(success), error, config_hash, seed, _now()),
        )
        self.conn.commit()
        return exp_id

    def experiments_for(self, spec_id: str, kind: Optional[str] = None) -> List[Dict[str, Any]]:
        q = "SELECT * FROM experiments WHERE spec_id=?"
        params: list = [spec_id]
        if kind:
            q += " AND kind=?"
            params.append(kind)
        rows = self.conn.execute(q + " ORDER BY created_at", params).fetchall()
        out = []
        for r in rows:
            d = dict(r)
            d["metrics"] = json.loads(d.pop("metrics_json") or "{}")
            d["params"] = json.loads(d.pop("params_json") or "{}")
            out.append(d)
        return out

    # ---------- gates ----------
    def record_gate_results(self, spec_id: str, code_id: Optional[str],
                            results: List[Dict[str, Any]]) -> None:
        batch_ts = _now()  # one timestamp for the whole batch so latest_gate_results
        for r in results:  # retrieves the complete evaluation, not just the last row
            self.conn.execute(
                "INSERT INTO gate_results (spec_id, code_id, gate_name, passed, value, "
                "threshold, hard, created_at) VALUES (?,?,?,?,?,?,?,?)",
                (spec_id, code_id, r["name"], int(r["passed"]), str(r.get("value")),
                 str(r.get("threshold")), int(r.get("hard", True)), batch_ts),
            )
        self.conn.commit()

    def latest_gate_results(self, spec_id: str) -> List[Dict[str, Any]]:
        rows = self.conn.execute(
            "SELECT * FROM gate_results WHERE spec_id=? AND created_at = "
            "(SELECT MAX(created_at) FROM gate_results WHERE spec_id=?)",
            (spec_id, spec_id),
        ).fetchall()
        return [dict(r) for r in rows]

    def gates_passed(self, spec_id: str) -> bool:
        results = self.latest_gate_results(spec_id)
        if not results:
            return False
        return all(r["passed"] for r in results if r["hard"])

    # ---------- promotions / rejections ----------
    def record_promotion(self, spec_id: str, from_status: str, to_status: str,
                         approved_by: Optional[str], notes: str = "") -> None:
        self.conn.execute(
            "INSERT INTO promotions (spec_id, from_status, to_status, approved_by, notes, "
            "created_at) VALUES (?,?,?,?,?,?)",
            (spec_id, from_status, to_status, approved_by, notes, _now()),
        )
        self.conn.commit()

    def record_rejection(self, spec_id: Optional[str], stage: str, reason: str,
                         code_id: Optional[str] = None) -> None:
        self.conn.execute(
            "INSERT INTO rejections (spec_id, code_id, stage, reason, created_at) "
            "VALUES (?,?,?,?,?)",
            (spec_id, code_id, stage, reason, _now()),
        )
        self.conn.commit()

    def list_rejections(self) -> List[Dict[str, Any]]:
        return [dict(r) for r in self.conn.execute(
            "SELECT * FROM rejections ORDER BY created_at DESC")]

    # ---------- summary ----------
    def summary(self) -> Dict[str, Any]:
        def count(table: str) -> int:
            return self.conn.execute(f"SELECT COUNT(*) AS c FROM {table}").fetchone()["c"]

        status_rows = self.conn.execute(
            "SELECT status, COUNT(*) AS c FROM specs GROUP BY status").fetchall()
        return {
            "ideas": count("ideas"),
            "specs": count("specs"),
            "code_versions": count("code_versions"),
            "experiments": count("experiments"),
            "experiments_failed": self.conn.execute(
                "SELECT COUNT(*) AS c FROM experiments WHERE success=0").fetchone()["c"],
            "rejections": count("rejections"),
            "promotions": count("promotions"),
            "specs_by_status": {r["status"]: r["c"] for r in status_rows},
        }

    def close(self) -> None:
        self.conn.close()
