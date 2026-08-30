"""Immutable Phase 04 split, baseline, and metric contracts."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Iterable

import pandas as pd


HORIZONS = (1, 3, 7, 14, 30)
PAIR_KEYS = ("target_name", "horizon", "origin_date", "target_date")
TARGET_COLUMNS = {
    "CH4_m3d_observed": "CH4_m3d_observed",
    "biogas_AB_m3d": "biogas_AB_m3d",
}

DEVELOPMENT_START = pd.Timestamp("2018-01-01")
DEVELOPMENT_END = pd.Timestamp("2021-12-31")
VALIDATION_START = pd.Timestamp("2022-01-01")
VALIDATION_END = pd.Timestamp("2022-12-31")
FINAL_TEST_START = pd.Timestamp("2023-01-01")
FINAL_TEST_END = pd.Timestamp("2023-09-17")


def baseline_registry() -> pd.DataFrame:
    """Return the closed deterministic baseline-family registry."""

    common = {
        "uses_target_history": True,
        "causal": True,
        "allowed_targets": "CH4_m3d_observed;biogas_AB_m3d",
        "allowed_horizons": "1;3;7;14;30",
        "status": "ACTIVE",
    }
    rows = [
        {
            "baseline_id": "B00_EXPANDING_MEAN",
            "baseline_name": "Expanding observed-history mean",
            "baseline_family": "CLIMATOLOGY",
            "definition": "mean of observed y(s) for s<=t",
            "required_history": "all observed target values through origin t",
            "requires_y_at_origin": False,
            "minimum_observations": 30,
            "season_period": "",
            "notes": "fixed 30-observation warm-up; never a full-data mean",
        },
        {
            "baseline_id": "B01_EXPANDING_MEDIAN",
            "baseline_name": "Expanding observed-history median",
            "baseline_family": "CLIMATOLOGY",
            "definition": "median of observed y(s) for s<=t",
            "required_history": "all observed target values through origin t",
            "requires_y_at_origin": False,
            "minimum_observations": 30,
            "season_period": "",
            "notes": "fixed 30-observation warm-up; never a full-data median",
        },
        {
            "baseline_id": "B10_PERSISTENCE_STRICT",
            "baseline_name": "Strict same-origin persistence",
            "baseline_family": "TARGET_INERTIA",
            "definition": "yhat(t+h|t)=y(t)",
            "required_history": "observed y at exact origin t",
            "requires_y_at_origin": True,
            "minimum_observations": 1,
            "season_period": "",
            "notes": "immutable canonical Skill_vs_persistence reference",
            "status": "CANONICAL_REFERENCE",
        },
        {
            "baseline_id": "B11_LAST_OBSERVED",
            "baseline_name": "Last observed target",
            "baseline_family": "TARGET_INERTIA",
            "definition": "yhat(t+h|t)=y(s*), s*=max{s<=t:y(s) observed}",
            "required_history": "at least one observed target on or before t",
            "requires_y_at_origin": False,
            "minimum_observations": 1,
            "season_period": "",
            "notes": "kept distinct from strict persistence; age recorded",
        },
        {
            "baseline_id": "B20_MOVING_AVG_7D",
            "baseline_name": "Observed-value calendar moving average, 7 d",
            "baseline_family": "TARGET_INERTIA",
            "definition": "mean observed y in [t-6,t]",
            "required_history": "calendar window [t-6,t]",
            "requires_y_at_origin": False,
            "minimum_observations": 1,
            "season_period": "7 calendar days",
            "notes": "right-aligned; observed values only",
        },
        {
            "baseline_id": "B21_MOVING_AVG_14D",
            "baseline_name": "Observed-value calendar moving average, 14 d",
            "baseline_family": "TARGET_INERTIA",
            "definition": "mean observed y in [t-13,t]",
            "required_history": "calendar window [t-13,t]",
            "requires_y_at_origin": False,
            "minimum_observations": 1,
            "season_period": "14 calendar days",
            "notes": "right-aligned; observed values only",
        },
        {
            "baseline_id": "B22_MOVING_AVG_30D",
            "baseline_name": "Observed-value calendar moving average, 30 d",
            "baseline_family": "TARGET_INERTIA",
            "definition": "mean observed y in [t-29,t]",
            "required_history": "calendar window [t-29,t]",
            "requires_y_at_origin": False,
            "minimum_observations": 1,
            "season_period": "30 calendar days",
            "notes": "right-aligned; observed values only",
        },
        {
            "baseline_id": "B30_WEEKLY_SEASONAL_NAIVE",
            "baseline_name": "Causal weekly seasonal naive",
            "baseline_family": "SEASONAL_NAIVE",
            "definition": "reference=t+h-7*ceil(h/7)<=t",
            "required_history": "observed target at the causal weekly reference",
            "requires_y_at_origin": False,
            "minimum_observations": 1,
            "season_period": "7 calendar days",
            "notes": "reference is never t+h-7 when that would be future",
        },
        {
            "baseline_id": "B31_ANNUAL_SEASONAL_NAIVE",
            "baseline_name": "Prior-calendar-year seasonal naive",
            "baseline_family": "SEASONAL_NAIVE",
            "definition": "reference=(t+h)-DateOffset(years=1)",
            "required_history": "observed target on prior calendar-year reference",
            "requires_y_at_origin": False,
            "minimum_observations": 1,
            "season_period": "1 calendar year",
            "notes": "pandas calendar-year offset; leap day maps to Feb 28",
        },
    ]
    frame = pd.DataFrame([{**common, **row} for row in rows])
    return frame.loc[
        :,
        [
            "baseline_id",
            "baseline_name",
            "baseline_family",
            "definition",
            "required_history",
            "uses_target_history",
            "causal",
            "requires_y_at_origin",
            "minimum_observations",
            "season_period",
            "allowed_targets",
            "allowed_horizons",
            "status",
            "notes",
        ],
    ]


def holdout_policy_text() -> str:
    """Return the exact sealed-holdout policy in YAML syntax."""

    return """development:
  start: 2018-01-01
  end: 2021-12-31

validation:
  start: 2022-01-01
  end: 2022-12-31

final_test:
  start: 2023-01-01
  end: 2023-09-17
  status: SEALED

allow_test_metric_reporting_in_phase04: false
allow_test_model_selection: false
allow_test_hyperparameter_selection: false
split_membership_rule: origin_and_target_date_within_same_period
"""


def sha256_file(path: Path) -> str:
    """Hash a source artifact without interpreting sealed-holdout target values."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_version_record(source_paths: Iterable[Path]) -> dict[str, object]:
    """Build the reproducibility/version record for the immutable benchmark."""

    registry = baseline_registry()
    definitions = dict(zip(registry["baseline_id"], registry["definition"], strict=True))
    metric_definitions = {
        "RMSE": "sqrt(mean((y_pred-y_true)^2))",
        "MAE": "mean(abs(y_pred-y_true))",
        "sMAPE": "200*mean(abs(y_pred-y_true)/(abs(y_true)+abs(y_pred))); both-zero term=0",
        "R2_level": "1-SSE_level/SST_level; NaN for n<2 or zero SST",
        "R2_delta": "R2(y_true-y_origin, y_pred-y_origin); requires observed y_origin",
        "Bias": "mean(y_pred-y_true); positive means overprediction",
        "Skill_vs_persistence": "1-MSE_model/MSE_persistence on exact paired support",
    }
    version_payload = {
        "baseline_definitions": definitions,
        "metric_definitions": metric_definitions,
        "horizons": list(HORIZONS),
        "splits": {
            "development": ["2018-01-01", "2021-12-31"],
            "validation": ["2022-01-01", "2022-12-31"],
            "final_test": ["2023-01-01", "2023-09-17", "SEALED"],
        },
        "split_membership_rule": "origin_and_target_date_within_same_period",
    }
    canonical = json.dumps(version_payload, sort_keys=True, separators=(",", ":"))
    version_id = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return {
        "phase": "04_baseline_modeling",
        "version_id": version_id,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_file_hashes": {
            str(path): sha256_file(path) for path in source_paths
        },
        **version_payload,
        "git_commit": None,
        "holdout_metric_access": "NONE",
    }
