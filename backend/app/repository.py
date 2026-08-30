from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import DB_PATH


SCHEMA = """
CREATE TABLE IF NOT EXISTS datasets (
  dataset_id TEXT PRIMARY KEY, file_name TEXT NOT NULL, uploaded_at TEXT NOT NULL,
  first_date TEXT NOT NULL, latest_date TEXT NOT NULL, row_count INTEGER NOT NULL,
  raw_path TEXT NOT NULL, clean_path TEXT NOT NULL, dq_path TEXT NOT NULL,
  status TEXT NOT NULL, metadata_json TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS dq_summary (
  dataset_id TEXT PRIMARY KEY, valid_count INTEGER, missing_count INTEGER,
  suspect_count INTEGER, invalid_count INTEGER, summary_json TEXT NOT NULL,
  FOREIGN KEY(dataset_id) REFERENCES datasets(dataset_id)
);
CREATE TABLE IF NOT EXISTS model_runs (
  run_id TEXT PRIMARY KEY, dataset_id TEXT NOT NULL, origin_date TEXT NOT NULL,
  model_version TEXT NOT NULL, status TEXT NOT NULL, created_at TEXT NOT NULL,
  audit_json TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS predictions (
  run_id TEXT NOT NULL, dataset_id TEXT NOT NULL, target TEXT NOT NULL,
  horizon INTEGER NOT NULL, target_date TEXT NOT NULL, value REAL,
  response_json TEXT NOT NULL, PRIMARY KEY(run_id,target,horizon)
);
CREATE TABLE IF NOT EXISTS reliability_errors (
  target TEXT NOT NULL, horizon INTEGER NOT NULL, origin_date TEXT NOT NULL,
  target_date TEXT NOT NULL, expert_id TEXT NOT NULL, error REAL NOT NULL,
  sq_error REAL NOT NULL, source_run_id TEXT NOT NULL, matured_at TEXT NOT NULL,
  PRIMARY KEY(target,horizon,origin_date,expert_id)
);
"""


class Repository:
    def __init__(self, path: Path = DB_PATH):
        self.path = path
        with self.connect() as db:
            db.executescript(SCHEMA)

    def connect(self) -> sqlite3.Connection:
        db = sqlite3.connect(self.path)
        db.row_factory = sqlite3.Row
        return db

    def save_dataset(self, record: dict[str, Any], dq_summary: dict[str, Any]) -> None:
        with self.connect() as db:
            db.execute(
                """INSERT INTO datasets VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    record["dataset_id"], record["file_name"], record["uploaded_at"],
                    record["first_date"], record["latest_date"], record["row_count"],
                    record["raw_path"], record["clean_path"], record["dq_path"],
                    record["status"], json.dumps(record.get("metadata", {}), ensure_ascii=False),
                ),
            )
            db.execute(
                """INSERT INTO dq_summary VALUES (?,?,?,?,?,?)""",
                (
                    record["dataset_id"], dq_summary["valid"], dq_summary["missing"],
                    dq_summary["suspect"], dq_summary["invalid"],
                    json.dumps(dq_summary, ensure_ascii=False),
                ),
            )

    def get_dataset(self, dataset_id: str) -> dict[str, Any] | None:
        with self.connect() as db:
            row = db.execute("SELECT * FROM datasets WHERE dataset_id=?", (dataset_id,)).fetchone()
        if not row:
            return None
        result = dict(row)
        result["metadata"] = json.loads(result.pop("metadata_json"))
        return result

    def get_dq_summary(self, dataset_id: str) -> dict[str, Any] | None:
        with self.connect() as db:
            row = db.execute("SELECT summary_json FROM dq_summary WHERE dataset_id=?", (dataset_id,)).fetchone()
        return json.loads(row[0]) if row else None

    def update_dataset_runtime(
        self, dataset_id: str, *, status: str, metadata: dict[str, Any]
    ) -> None:
        """Persist runtime metadata without changing the immutable source files."""
        record = self.get_dataset(dataset_id)
        if not record:
            raise KeyError("DATASET_NOT_FOUND")
        merged = dict(record.get("metadata", {}))
        merged.update(metadata)
        with self.connect() as db:
            db.execute(
                "UPDATE datasets SET status=?, metadata_json=? WHERE dataset_id=?",
                (status, json.dumps(merged, ensure_ascii=False), dataset_id),
            )

    def save_forecast(self, run_id: str, dataset_id: str, response: dict[str, Any], audit: dict[str, Any]) -> None:
        created_at = datetime.now(timezone.utc).isoformat()
        with self.connect() as db:
            db.execute(
                "INSERT INTO model_runs VALUES (?,?,?,?,?,?,?)",
                (run_id, dataset_id, response["origin_date"], response["model"]["version"], response["status"], created_at, json.dumps(audit, ensure_ascii=False)),
            )
            for item in response["methane"]["predictions"]:
                db.execute(
                    "INSERT INTO predictions VALUES (?,?,?,?,?,?,?)",
                    (run_id, dataset_id, "CH4_m3d_observed", item["horizon"], item["target_date"], item.get("value"), json.dumps(item, ensure_ascii=False)),
                )
            if response.get("purity_next_day"):
                item = response["purity_next_day"]
                db.execute(
                    "INSERT INTO predictions VALUES (?,?,?,?,?,?,?)",
                    (run_id, dataset_id, "purity_derived", 1, item["target_date"], item.get("value_pct"), json.dumps(item, ensure_ascii=False)),
                )

    def latest_forecast(
        self,
        dataset_id: str,
        *,
        model_version: str | None = None,
        origin_date: str | None = None,
    ) -> dict[str, Any] | None:
        clauses = ["dataset_id=?"]
        values: list[Any] = [dataset_id]
        if model_version is not None:
            clauses.append("model_version=?")
            values.append(model_version)
        if origin_date is not None:
            clauses.append("origin_date=?")
            values.append(origin_date)
        with self.connect() as db:
            row = db.execute(
                "SELECT audit_json FROM model_runs WHERE "
                + " AND ".join(clauses)
                + " ORDER BY created_at DESC LIMIT 1",
                tuple(values),
            ).fetchone()
        if not row:
            return None
        return json.loads(row[0]).get("api_response")

    def mature_forecast_errors(self, actual: Any) -> int:
        """Mature prior expert forecasts only when target-date truth now exists."""
        truth = actual.set_index("date")
        inserted = 0
        with self.connect() as db:
            runs = db.execute("SELECT run_id,audit_json FROM model_runs").fetchall()
            for run in runs:
                audit = json.loads(run["audit_json"])
                for forecast in audit.get("forecasts", []):
                    target = forecast.get("target")
                    target_date = forecast.get("target_date")
                    if target not in truth.columns or target_date not in truth.index.strftime("%Y-%m-%d"):
                        continue
                    row = truth.loc[truth.index.strftime("%Y-%m-%d") == target_date]
                    if row.empty:
                        continue
                    y_true = row.iloc[-1][target]
                    try:
                        y_true = float(y_true)
                    except (TypeError, ValueError):
                        continue
                    if not __import__("math").isfinite(y_true):
                        continue
                    for expert, prediction in forecast.get("expert_predictions", {}).items():
                        try:
                            prediction = float(prediction)
                        except (TypeError, ValueError):
                            continue
                        if not __import__("math").isfinite(prediction):
                            continue
                        error = prediction - y_true
                        cursor = db.execute(
                            """INSERT OR IGNORE INTO reliability_errors
                               VALUES (?,?,?,?,?,?,?,?,?)""",
                            (
                                target, int(forecast["horizon"]), forecast["origin_date"], target_date,
                                expert, error, error * error, run["run_id"],
                                datetime.now(timezone.utc).isoformat(),
                            ),
                        )
                        inserted += int(cursor.rowcount > 0)
        return inserted

    def reliability_errors(self, target: str, horizon: int, origin_date: str) -> list[dict[str, Any]]:
        with self.connect() as db:
            rows = db.execute(
                """SELECT expert_id,error FROM reliability_errors
                   WHERE target=? AND horizon=? AND target_date<=?""",
                (target, horizon, origin_date),
            ).fetchall()
        return [dict(row) for row in rows]
