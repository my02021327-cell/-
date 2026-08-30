"""Keep the test suite out of the production dataset store.

The API tests exercise the real upload path, which writes workbooks, parquet
files, prediction audits and sqlite rows. Without this fixture every run leaves
another dataset behind in backend/storage. Only artefacts created *during* the
session are removed; anything that existed beforehand is left untouched.
"""

from __future__ import annotations

import sqlite3

import pytest

from backend.app.config import DB_PATH, PREDICTION_DIR, PROCESSED_DIR, UPLOAD_DIR

TRACKED_DIRS = (UPLOAD_DIR, PROCESSED_DIR, PREDICTION_DIR)
TRACKED_TABLES = ("predictions", "model_runs", "reliability_errors", "dq_summary", "datasets")


def _files() -> set:
    return {path for directory in TRACKED_DIRS for path in directory.glob("*") if path.is_file()}


def _dataset_ids() -> set:
    if not DB_PATH.is_file():
        return set()
    with sqlite3.connect(DB_PATH) as db:
        return {row[0] for row in db.execute("SELECT dataset_id FROM datasets")}


@pytest.fixture(scope="session", autouse=True)
def isolate_storage():
    before_files, before_ids = _files(), _dataset_ids()
    yield
    for path in _files() - before_files:
        path.unlink(missing_ok=True)
    created = _dataset_ids() - before_ids
    if not created:
        return
    placeholders = ",".join("?" for _ in created)
    values = list(created)
    with sqlite3.connect(DB_PATH) as db:
        runs = [
            row[0] for row in db.execute(
                f"SELECT run_id FROM model_runs WHERE dataset_id IN ({placeholders})", values
            )
        ]
        if runs:
            # reliability_errors is keyed by run, not by dataset.
            db.executemany(
                "DELETE FROM reliability_errors WHERE source_run_id = ?", [(run,) for run in runs]
            )
        for table in TRACKED_TABLES:
            columns = {row[1] for row in db.execute(f"PRAGMA table_info({table})")}
            if "dataset_id" not in columns:
                continue
            db.execute(f"DELETE FROM {table} WHERE dataset_id IN ({placeholders})", values)
        db.commit()
