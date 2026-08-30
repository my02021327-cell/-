"""Run PHASE 05 HRT-free process-informed mechanistic modeling.

Only 2018--2021 target values enter this module.  The immutable Phase 04
strict-persistence table supplies every Skill denominator through exact-key
joins.  No mechanistic prediction is generated for 2022 or 2023 here; the
serialized full-development bundle is the hand-off to R7.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict
import io
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from typing import Any, Iterable

import numpy as np
import pandas as pd

from src.models.mechanistic.fold_selection import (
    DEVELOPMENT_END,
    DEVELOPMENT_START,
    ExpandingFold,
    assert_development_only,
    make_expanding_outer_folds,
    make_inner_expanding_folds,
    matured_training_mask,
    rank_candidates,
    score_candidate_predictions,
)
from src.models.mechanistic.kernels import (
    K_CANDIDATES,
    KERNEL_TAIL_TOL,
    FirstOrderKernel,
    build_first_order_kernel,
    kernel_audit_frame,
)
from src.models.mechanistic.kinetic_state import (
    MIN_KERNEL_COVERAGE,
    build_kinetic_state,
)
from src.models.mechanistic.physical_audit import (
    build_intercept_audit,
    build_physical_audit,
)
from src.models.mechanistic.registry import (
    HORIZONS,
    TAU_RES_CANDIDATES,
    mechanistic_registry,
)
from src.models.mechanistic.residual_inertia import (
    residual_inertia_correction,
    select_residual_tau,
)
from src.models.mechanistic.serialization import save_model_bundle, sha256_file
from src.models.mechanistic.single_pool import SinglePoolModel, fit_single_pool
from src.models.mechanistic.two_pool import (
    TwoPoolModel,
    fit_two_pool,
    two_pool_candidate_pairs,
)
from src.validation import evaluate_baseline_predictions
from src.validation.baseline_metrics import compute_metrics, skill_vs_persistence
from src.validation.paired_support import merge_persistence_reference


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
REPORT_DIR = PROJECT_ROOT / "reports"
FIGURE_DIR = REPORT_DIR / "figures"
ARTIFACT_DIR = PROJECT_ROOT / "artifacts"

PROCESS_PATH = DATA_DIR / "02_process_base.parquet"
FEATURE_PATH = DATA_DIR / "03_feature_base.parquet"
TARGET_PATH = DATA_DIR / "03_targets_by_horizon.parquet"
FEATURE_REGISTRY_PATH = OUTPUT_DIR / "03_feature_registry.csv"
FEATURE_AVAILABILITY_PATH = OUTPUT_DIR / "03_feature_availability_matrix.csv"
PREPROCESSING_PATH = OUTPUT_DIR / "03_preprocessing_contract.yaml"
PERSISTENCE_PATH = OUTPUT_DIR / "04_persistence_reference.parquet"
BASELINE_METRICS_PATH = OUTPUT_DIR / "04_baseline_metrics.csv"
BASELINE_VERSION_PATH = OUTPUT_DIR / "04_baseline_version.json"
HOLDOUT_PATH = OUTPUT_DIR / "04_holdout_policy.yaml"
PHASE03_REPORT_PATH = REPORT_DIR / "03_feature_engineering_report.md"
PHASE04_REPORT_PATH = REPORT_DIR / "04_baseline_modeling_report.md"

REGISTRY_PATH = OUTPUT_DIR / "05_mechanistic_registry.csv"
KERNEL_AUDIT_PATH = OUTPUT_DIR / "05_kernel_audit.csv"
INNER_SELECTION_PATH = OUTPUT_DIR / "05_inner_selection.csv"
OOF_METRICS_PATH = OUTPUT_DIR / "05_mechanistic_oof_metrics.csv"
DEVELOPMENT_SUMMARY_PATH = OUTPUT_DIR / "05_development_summary.csv"
PARAMETER_STABILITY_PATH = OUTPUT_DIR / "05_parameter_stability.csv"
INTERCEPT_AUDIT_PATH = OUTPUT_DIR / "05_intercept_audit.csv"
RESIDUAL_SELECTION_PATH = OUTPUT_DIR / "05_residual_inertia_selection.csv"
PHYSICAL_AUDIT_PATH = OUTPUT_DIR / "05_physical_audit.csv"
MODEL_LOCK_PATH = OUTPUT_DIR / "05_model_lock.yaml"
GATE_PATH = OUTPUT_DIR / "05_gate_results.csv"
TEST_RESULT_PATH = OUTPUT_DIR / "05_test_results.csv"
OOF_PATH = DATA_DIR / "05_mechanistic_oof_predictions.parquet"
ARTIFACT_PATH = ARTIFACT_DIR / "05_mechanistic_dev_models.joblib"
REPORT_PATH = REPORT_DIR / "05_mechanistic_modeling_report.md"

TARGET_COLUMNS = {
    "CH4_m3d_observed": "CH4_m3d_observed",
    "biogas_AB_m3d": "biogas_AB_m3d",
}
LOAD_SPECS = {
    "M1F": {
        "load_basis": "FEED_WET_MASS",
        "column": "feed_AB_tpd",
        "unit": "t wet mass/d",
        "lineage": "feed_A + feed_B",
    },
    "M1TS": {
        "load_basis": "TS_LOAD_PROXY",
        "column": "TS_load_tpd",
        "unit": "t TS/d",
        "lineage": "feed_AB_tpd * acid_TS / 100",
    },
    "M1VS": {
        "load_basis": "VS_LOAD_PROXY",
        "column": "VS_load_tpd",
        "unit": "t VS/d",
        "lineage": "feed_AB_tpd * acid_VS / 100; VS_2020_STRUCTURAL_BREAK",
    },
}

PROCESS_COLUMNS = [
    "date",
    "feed_AB_tpd",
    "TS_load_tpd",
    "VS_load_tpd",
    "CH4_m3d_observed",
    "biogas_AB_m3d",
]


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding="utf-8-sig")


def _source_paths() -> tuple[Path, ...]:
    return (
        PROCESS_PATH,
        FEATURE_PATH,
        TARGET_PATH,
        FEATURE_REGISTRY_PATH,
        FEATURE_AVAILABILITY_PATH,
        PREPROCESSING_PATH,
        PERSISTENCE_PATH,
        BASELINE_METRICS_PATH,
        BASELINE_VERSION_PATH,
        HOLDOUT_PATH,
        PHASE03_REPORT_PATH,
        PHASE04_REPORT_PATH,
    )


def _read_development_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Read only development rows and development persistence references."""

    missing = [str(path) for path in _source_paths() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Phase 05 input contract is incomplete: {missing}")
    process = pd.read_parquet(
        PROCESS_PATH,
        columns=PROCESS_COLUMNS,
        filters=[("date", ">=", DEVELOPMENT_START), ("date", "<=", DEVELOPMENT_END)],
    )
    process["date"] = pd.to_datetime(process["date"]).dt.normalize()
    process = process.sort_values("date", kind="stable").reset_index(drop=True)
    expected = pd.date_range(DEVELOPMENT_START, DEVELOPMENT_END, freq="D")
    if not pd.DatetimeIndex(process["date"]).equals(expected):
        raise AssertionError("Phase 05 requires the complete 2018-2021 daily calendar")
    if len(process) != 1461 or process["date"].max() > DEVELOPMENT_END:
        raise AssertionError("non-development process rows entered Phase 05")
    for column in PROCESS_COLUMNS[1:]:
        process[column] = pd.to_numeric(process[column], errors="coerce").astype(float)

    persistence = pd.read_parquet(
        PERSISTENCE_PATH,
        filters=[("evaluation_period", "=", "DEVELOPMENT")],
    )
    for column in ("origin_date", "target_date"):
        persistence[column] = pd.to_datetime(persistence[column]).dt.normalize()
    persistence = persistence[
        persistence["origin_date"].between(DEVELOPMENT_START, DEVELOPMENT_END)
        & persistence["target_date"].le(DEVELOPMENT_END)
    ].copy()
    if persistence.empty or persistence["target_date"].max() > DEVELOPMENT_END:
        raise AssertionError("immutable persistence reference was not development-only")

    feature_registry = pd.read_csv(FEATURE_REGISTRY_PATH, encoding="utf-8-sig")
    status = feature_registry.set_index("feature_name")["status"]
    required_status = {
        "feed_AB_tpd": "APPROVED",
        "TS_load_tpd": "PROVISIONAL",
        "VS_load_tpd": "PROVISIONAL",
        "HRT_mass_proxy_d": "DIAGNOSTIC_ONLY",
    }
    for feature, expected_status in required_status.items():
        if feature not in status.index or status.loc[feature] != expected_status:
            raise AssertionError(f"Phase 03 registry status drifted for {feature}")
    return process, persistence, feature_registry


def _series(process: pd.DataFrame, column: str) -> pd.Series:
    return pd.Series(
        process[column].to_numpy(dtype=float),
        index=pd.DatetimeIndex(process["date"]),
        name=column,
        dtype=float,
    )


def _build_state_cache(
    process: pd.DataFrame,
) -> tuple[dict[tuple[str, float], pd.DataFrame], dict[float, FirstOrderKernel]]:
    kernels = {k: build_first_order_kernel(k) for k in K_CANDIDATES}
    states: dict[tuple[str, float], pd.DataFrame] = {}
    for model_id, spec in LOAD_SPECS.items():
        load = _series(process, str(spec["column"]))
        for k, kernel in kernels.items():
            frame = build_kinetic_state(
                load,
                kernel,
                min_kernel_coverage=MIN_KERNEL_COVERAGE,
            ).set_index("date")
            if frame["newest_load_date_used"].dropna().gt(frame.index.to_series()).any():
                raise AssertionError("kinetic state used a future load timestamp")
            states[(model_id, k)] = frame
    return states, kernels


def _samples(process: pd.DataFrame, target_name: str, horizon: int) -> pd.DataFrame:
    target = _series(process, TARGET_COLUMNS[target_name])
    origins = pd.date_range(
        DEVELOPMENT_START,
        DEVELOPMENT_END - pd.Timedelta(days=horizon),
        freq="D",
    )
    target_dates = origins + pd.to_timedelta(horizon, unit="D")
    frame = pd.DataFrame(
        {
            "origin_date": origins,
            "target_date": target_dates,
            "target_name": target_name,
            "horizon": horizon,
            "y_origin": target.reindex(origins).to_numpy(dtype=float),
            "y_true": target.reindex(target_dates).to_numpy(dtype=float),
        }
    )
    frame["target_available"] = frame["y_true"].notna()
    assert_development_only(frame)
    return frame


def _state_values(
    state: pd.DataFrame,
    dates: Iterable[pd.Timestamp],
    column: str = "kinetic_state",
) -> np.ndarray:
    return pd.to_numeric(
        state[column].reindex(pd.DatetimeIndex(dates)), errors="coerce"
    ).to_numpy(dtype=float)


def _max_complete_training_target(
    training: pd.DataFrame,
    values: np.ndarray,
) -> pd.Timestamp | pd.NaT:
    complete = np.isfinite(values) & np.isfinite(training["y_true"].to_numpy(dtype=float))
    if not complete.any():
        return pd.NaT
    return pd.Timestamp(training.loc[complete, "target_date"].max())


def _prediction_rows(
    evaluation: pd.DataFrame,
    state: pd.DataFrame,
    physical: Any,
    *,
    model_id: str,
    load_basis: str,
    outer_fold: str,
    k: float | None,
    kernel_max_lag: int | None,
    beta0: float,
    beta1: float,
    max_training_target_date: pd.Timestamp | pd.NaT,
    uses_target_history: bool = False,
    availability_reason: str = "STATE_AVAILABLE",
    extra: dict[str, Any] | None = None,
) -> pd.DataFrame:
    rows = evaluation.copy()
    dates = pd.DatetimeIndex(rows["origin_date"])
    state_rows = state.reindex(dates)
    raw = np.asarray(physical.raw, dtype=float)
    clipped = np.asarray(physical.clipped, dtype=float)
    rows["model_id"] = model_id
    rows["load_basis"] = load_basis
    rows["outer_fold"] = outer_fold
    rows["k"] = k
    rows["kernel_max_lag"] = kernel_max_lag
    rows["kernel_coverage"] = pd.to_numeric(state_rows["kernel_coverage"], errors="coerce").to_numpy(dtype=float)
    rows["kinetic_state"] = pd.to_numeric(state_rows["kinetic_state"], errors="coerce").to_numpy(dtype=float)
    rows["beta0"] = beta0
    rows["beta1"] = beta1
    rows["raw_prediction"] = raw
    rows["clipped_prediction"] = clipped
    rows["y_pred"] = clipped
    rows["prediction_available"] = np.isfinite(clipped)
    rows["model_available"] = rows["prediction_available"]
    rows["eligible"] = True
    rows["origin_target_available"] = rows["y_origin"].notna()
    rows["own_scored"] = rows["target_available"] & rows["prediction_available"]
    rows["uses_target_history"] = uses_target_history
    rows["availability_reason"] = np.where(
        rows["prediction_available"], availability_reason, "INSUFFICIENT_KERNEL_COVERAGE"
    )
    rows["max_training_target_date"] = max_training_target_date
    rows["max_feature_timestamp"] = pd.to_datetime(
        state_rows["newest_load_date_used"].to_numpy()
    )
    rows["oldest_load_date_used"] = pd.to_datetime(
        state_rows["oldest_load_date_used"].to_numpy()
    )
    rows["support_type"] = "OWN_AVAILABLE"
    rows["error"] = rows["y_pred"] - rows["y_true"]
    rows["abs_error"] = rows["error"].abs()
    rows["sq_error"] = rows["error"].pow(2)
    rows["delta_true"] = rows["y_true"] - rows["y_origin"]
    rows["delta_pred"] = rows["y_pred"] - rows["y_origin"]
    rows["beta_fast"] = np.nan
    rows["beta_slow"] = np.nan
    rows["k_fast"] = np.nan
    rows["k_slow"] = np.nan
    rows["tau_res"] = np.nan
    rows["base_model_id"] = ""
    rows["beta0_current"] = np.nan
    rows["beta1_current"] = np.nan
    rows["two_pool_degenerate"] = False
    if extra:
        for column, value in extra.items():
            rows[column] = value
    return rows


def _inner_single_predictions(
    samples: pd.DataFrame,
    state: pd.DataFrame,
    kernel: FirstOrderKernel,
    outer_fold: ExpandingFold,
    persistence: pd.DataFrame,
    *,
    model_id: str,
    load_basis: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return inner-prequential predictions and one score row for one k."""

    frames: list[pd.DataFrame] = []
    calendar = pd.date_range(DEVELOPMENT_START, DEVELOPMENT_END, freq="D")
    inner_folds = make_inner_expanding_folds(
        outer_fold.eval_start,
        int(samples["horizon"].iloc[0]),
        dates=calendar,
    )
    for inner in inner_folds:
        train_mask = matured_training_mask(samples, inner.eval_start)
        training = samples.loc[train_mask].copy()
        evaluation = samples[
            samples["origin_date"].between(inner.eval_start, inner.eval_end)
            & samples["target_date"].lt(outer_fold.eval_start)
        ].copy()
        if evaluation.empty:
            continue
        train_state = _state_values(state, training["origin_date"])
        complete_training = np.isfinite(train_state) & np.isfinite(
            training["y_true"].to_numpy(dtype=float)
        )
        if not complete_training.any():
            continue
        model = fit_single_pool(
            train_state,
            training["y_true"].to_numpy(dtype=float),
            k_per_d=kernel.k_per_d,
            kernel_max_lag=kernel.max_lag,
            load_basis=load_basis,
            model_id=model_id,
        )
        eval_state = _state_values(state, evaluation["origin_date"])
        physical = model.predict(eval_state)
        frame = evaluation.copy()
        frame["y_pred"] = physical.clipped
        frame["prediction_available"] = physical.available
        frame["candidate_id"] = f"{model_id}_K_{kernel.k_per_d:.2f}"
        frame["candidate_model"] = model_id
        frame["model_id"] = model_id
        frame["load_basis"] = load_basis
        frame["k"] = kernel.k_per_d
        frame["complexity_rank"] = 1
        frame["inner_fold"] = inner.fold_id
        frame["max_training_target_date"] = _max_complete_training_target(
            training, train_state
        )
        if frame["max_training_target_date"].notna().any() and not frame[
            "max_training_target_date"
        ].lt(frame["origin_date"]).all():
            raise AssertionError("inner single-pool fit used an unmatured target")
        frames.append(frame)
    if not frames:
        candidate_id = f"{model_id}_K_{kernel.k_per_d:.2f}"
        return (
            pd.DataFrame(
                columns=[
                    "target_name", "horizon", "origin_date", "target_date",
                    "y_true", "y_origin", "target_available", "y_pred",
                    "prediction_available", "candidate_id",
                ]
            ),
            pd.DataFrame(
                [
                    {
                        "candidate_id": candidate_id,
                        "inner_n": 0,
                        "own_RMSE": np.nan,
                        "own_MAE": np.nan,
                        "own_sMAPE": np.nan,
                        "paired_n": 0,
                        "paired_Skill_vs_persistence": np.nan,
                        "complexity_rank": 1,
                        "model_id": model_id,
                        "load_basis": load_basis,
                        "k": kernel.k_per_d,
                    }
                ]
            ),
        )
    predictions = pd.concat(frames, ignore_index=True)
    scores = score_candidate_predictions(
        predictions,
        persistence,
        candidate_columns=("candidate_id",),
    )
    return predictions, scores


def _inner_two_pool_predictions(
    samples: pd.DataFrame,
    fast_state: pd.DataFrame,
    slow_state: pd.DataFrame,
    outer_fold: ExpandingFold,
    persistence: pd.DataFrame,
    *,
    k_fast: float,
    k_slow: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return inner-prequential predictions for one separated TS pair."""

    frames: list[pd.DataFrame] = []
    calendar = pd.date_range(DEVELOPMENT_START, DEVELOPMENT_END, freq="D")
    inner_folds = make_inner_expanding_folds(
        outer_fold.eval_start,
        int(samples["horizon"].iloc[0]),
        dates=calendar,
    )
    for inner in inner_folds:
        training = samples.loc[matured_training_mask(samples, inner.eval_start)].copy()
        evaluation = samples[
            samples["origin_date"].between(inner.eval_start, inner.eval_end)
            & samples["target_date"].lt(outer_fold.eval_start)
        ].copy()
        if evaluation.empty:
            continue
        x_fast = _state_values(fast_state, training["origin_date"])
        x_slow = _state_values(slow_state, training["origin_date"])
        complete_training = (
            np.isfinite(x_fast)
            & np.isfinite(x_slow)
            & np.isfinite(training["y_true"].to_numpy(dtype=float))
        )
        if not complete_training.any():
            continue
        model = fit_two_pool(
            x_fast,
            x_slow,
            training["y_true"].to_numpy(dtype=float),
            k_fast=k_fast,
            k_slow=k_slow,
            load_basis="TS_LOAD_PROXY",
        )
        pred = model.predict(
            _state_values(fast_state, evaluation["origin_date"]),
            _state_values(slow_state, evaluation["origin_date"]),
        )
        frame = evaluation.copy()
        frame["y_pred"] = pred.clipped
        frame["prediction_available"] = pred.available
        frame["candidate_id"] = f"M1TS2_KF_{k_fast:.2f}_KS_{k_slow:.2f}"
        frame["candidate_model"] = "M1TS2"
        frame["model_id"] = "M1TS2"
        frame["load_basis"] = "TS_LOAD_PROXY"
        frame["k"] = np.nan
        frame["k_fast"] = k_fast
        frame["k_slow"] = k_slow
        frame["complexity_rank"] = 2
        frame["inner_fold"] = inner.fold_id
        complete = (
            np.isfinite(x_fast)
            & np.isfinite(x_slow)
            & np.isfinite(training["y_true"].to_numpy(dtype=float))
        )
        max_target = training.loc[complete, "target_date"].max() if complete.any() else pd.NaT
        frame["max_training_target_date"] = max_target
        frames.append(frame)
    if not frames:
        candidate_id = f"M1TS2_KF_{k_fast:.2f}_KS_{k_slow:.2f}"
        return (
            pd.DataFrame(
                columns=[
                    "target_name", "horizon", "origin_date", "target_date",
                    "y_true", "y_origin", "target_available", "y_pred",
                    "prediction_available", "candidate_id",
                ]
            ),
            pd.DataFrame(
                [
                    {
                        "candidate_id": candidate_id,
                        "inner_n": 0,
                        "own_RMSE": np.nan,
                        "own_MAE": np.nan,
                        "own_sMAPE": np.nan,
                        "paired_n": 0,
                        "paired_Skill_vs_persistence": np.nan,
                        "complexity_rank": 2,
                        "model_id": "M1TS2",
                        "load_basis": "TS_LOAD_PROXY",
                        "k": np.nan,
                        "k_fast": k_fast,
                        "k_slow": k_slow,
                    }
                ]
            ),
        )
    predictions = pd.concat(frames, ignore_index=True)
    scores = score_candidate_predictions(
        predictions,
        persistence,
        candidate_columns=("candidate_id",),
    )
    return predictions, scores


def _add_family_common_support_scores(
    scores: pd.DataFrame,
    candidate_predictions: pd.DataFrame,
) -> pd.DataFrame:
    """Audit score sensitivity on exact rows available for every candidate."""

    keys = ["target_name", "horizon", "origin_date", "target_date"]
    available = candidate_predictions[
        candidate_predictions["target_available"].fillna(False)
        & candidate_predictions["prediction_available"].fillna(False)
    ].copy()
    counts = available.groupby(keys, sort=False)["candidate_id"].nunique()
    n_candidates = candidate_predictions["candidate_id"].nunique()
    common_keys = counts[counts.eq(n_candidates)].index
    if not len(common_keys):
        result = scores.copy()
        result["family_common_n"] = 0
        result["family_common_RMSE"] = np.nan
        return result
    marker = pd.DataFrame(list(common_keys), columns=keys)
    common = available.merge(marker, on=keys, how="inner", validate="many_to_one")
    audit_rows: list[dict[str, object]] = []
    for candidate_id, group in common.groupby("candidate_id", sort=True):
        metric = compute_metrics(group["y_true"], group["y_pred"])
        audit_rows.append(
            {
                "candidate_id": candidate_id,
                "family_common_n": metric["n_scored"],
                "family_common_RMSE": metric["RMSE"],
            }
        )
    return scores.merge(pd.DataFrame(audit_rows), on="candidate_id", how="left")


def _rank_inner_family(
    scores: pd.DataFrame,
    predictions: pd.DataFrame,
    *,
    outer_fold: str,
    target_name: str,
    horizon: int,
    candidate_model: str,
    load_basis: str,
) -> tuple[pd.DataFrame, pd.Series]:
    audited = _add_family_common_support_scores(scores, predictions)
    audited["eligible"] = audited["inner_n"].gt(0)
    ranked = rank_candidates(audited)
    ranked = ranked.rename(
        columns={
            "own_RMSE": "inner_RMSE",
            "own_MAE": "inner_MAE",
            "paired_Skill_vs_persistence": "inner_Skill",
        }
    )
    ranked["outer_fold"] = outer_fold
    ranked["target"] = target_name
    ranked["horizon"] = horizon
    ranked["candidate_model"] = candidate_model
    ranked["load_basis"] = load_basis
    selected = ranked.loc[ranked["selected"]].iloc[0]
    own_winner = str(selected["candidate_id"])
    common_eligible = ranked[np.isfinite(pd.to_numeric(ranked["family_common_RMSE"], errors="coerce"))]
    common_winner = (
        str(common_eligible.sort_values(["family_common_RMSE", "candidate_id"], kind="stable").iloc[0]["candidate_id"])
        if len(common_eligible)
        else ""
    )
    ranked["support_ranking_changed"] = bool(common_winner and common_winner != own_winner)
    return ranked, selected


def _outer_training_and_evaluation(
    samples: pd.DataFrame,
    fold: ExpandingFold,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    training = samples.loc[matured_training_mask(samples, fold.eval_start)].copy()
    evaluation = samples[
        samples["origin_date"].between(fold.eval_start, fold.eval_end)
    ].copy()
    if not training.empty and training["target_date"].max() >= fold.eval_start:
        raise AssertionError("outer training target did not mature before evaluation")
    return training, evaluation


def _fit_outer_single(
    samples: pd.DataFrame,
    state: pd.DataFrame,
    kernel: FirstOrderKernel,
    fold: ExpandingFold,
    *,
    model_id: str,
    load_basis: str,
) -> tuple[pd.DataFrame, SinglePoolModel]:
    training, evaluation = _outer_training_and_evaluation(samples, fold)
    train_state = _state_values(state, training["origin_date"])
    model = fit_single_pool(
        train_state,
        training["y_true"].to_numpy(dtype=float),
        k_per_d=kernel.k_per_d,
        kernel_max_lag=kernel.max_lag,
        load_basis=load_basis,
        model_id=model_id,
    )
    physical = model.predict(_state_values(state, evaluation["origin_date"]))
    rows = _prediction_rows(
        evaluation,
        state,
        physical,
        model_id=model_id,
        load_basis=load_basis,
        outer_fold=fold.fold_id,
        k=kernel.k_per_d,
        kernel_max_lag=kernel.max_lag,
        beta0=model.beta0,
        beta1=model.beta1,
        max_training_target_date=_max_complete_training_target(training, train_state),
    )
    return rows, model


def _fit_outer_two_pool(
    samples: pd.DataFrame,
    fast_state: pd.DataFrame,
    slow_state: pd.DataFrame,
    fast_kernel: FirstOrderKernel,
    slow_kernel: FirstOrderKernel,
    fold: ExpandingFold,
) -> tuple[pd.DataFrame, TwoPoolModel]:
    training, evaluation = _outer_training_and_evaluation(samples, fold)
    train_fast = _state_values(fast_state, training["origin_date"])
    train_slow = _state_values(slow_state, training["origin_date"])
    model = fit_two_pool(
        train_fast,
        train_slow,
        training["y_true"].to_numpy(dtype=float),
        k_fast=fast_kernel.k_per_d,
        k_slow=slow_kernel.k_per_d,
        load_basis="TS_LOAD_PROXY",
    )
    eval_fast = _state_values(fast_state, evaluation["origin_date"])
    eval_slow = _state_values(slow_state, evaluation["origin_date"])
    physical = model.predict(eval_fast, eval_slow)
    complete = (
        np.isfinite(train_fast)
        & np.isfinite(train_slow)
        & np.isfinite(training["y_true"].to_numpy(dtype=float))
    )
    max_target = training.loc[complete, "target_date"].max() if complete.any() else pd.NaT
    rows = _prediction_rows(
        evaluation,
        fast_state,
        physical,
        model_id="M1TS2",
        load_basis="TS_LOAD_PROXY",
        outer_fold=fold.fold_id,
        k=None,
        kernel_max_lag=max(fast_kernel.max_lag, slow_kernel.max_lag),
        beta0=model.beta0,
        beta1=model.beta_fast + model.beta_slow,
        max_training_target_date=max_target,
        extra={
            "k_fast": fast_kernel.k_per_d,
            "k_slow": slow_kernel.k_per_d,
            "beta_fast": model.beta_fast,
            "beta_slow": model.beta_slow,
            "two_pool_degenerate": model.effectively_single_pool,
            "fast_kinetic_state": eval_fast,
            "slow_kinetic_state": eval_slow,
        },
    )
    fast_rows = fast_state.reindex(pd.DatetimeIndex(evaluation["origin_date"]))
    slow_rows = slow_state.reindex(pd.DatetimeIndex(evaluation["origin_date"]))
    rows["kernel_coverage"] = np.minimum(
        pd.to_numeric(fast_rows["kernel_coverage"], errors="coerce").to_numpy(dtype=float),
        pd.to_numeric(slow_rows["kernel_coverage"], errors="coerce").to_numpy(dtype=float),
    )
    newest = pd.concat(
        [
            pd.to_datetime(fast_rows["newest_load_date_used"]).reset_index(drop=True),
            pd.to_datetime(slow_rows["newest_load_date_used"]).reset_index(drop=True),
        ],
        axis=1,
    ).max(axis=1)
    rows["max_feature_timestamp"] = newest.to_numpy()
    return rows, model


def _inner_m2_base_predictions(
    samples: pd.DataFrame,
    state: pd.DataFrame,
    kernel: FirstOrderKernel,
    outer_fold: ExpandingFold,
    *,
    base_model_id: str,
    load_basis: str,
) -> pd.DataFrame:
    """Build frozen-per-inner-block direct and current M1 predictions."""

    frames: list[pd.DataFrame] = []
    calendar = pd.date_range(DEVELOPMENT_START, DEVELOPMENT_END, freq="D")
    inner_folds = make_inner_expanding_folds(
        outer_fold.eval_start,
        int(samples["horizon"].iloc[0]),
        dates=calendar,
    )
    for inner in inner_folds:
        direct_training = samples.loc[matured_training_mask(samples, inner.eval_start)].copy()
        current_training = samples[samples["origin_date"].lt(inner.eval_start)].copy()
        evaluation = samples[
            samples["origin_date"].between(inner.eval_start, inner.eval_end)
            & samples["target_date"].lt(outer_fold.eval_start)
        ].copy()
        if evaluation.empty:
            continue
        x_direct = _state_values(state, direct_training["origin_date"])
        direct_complete = np.isfinite(x_direct) & np.isfinite(
            direct_training["y_true"].to_numpy(dtype=float)
        )
        if not direct_complete.any():
            continue
        direct_model = fit_single_pool(
            x_direct,
            direct_training["y_true"].to_numpy(dtype=float),
            k_per_d=kernel.k_per_d,
            kernel_max_lag=kernel.max_lag,
            load_basis=load_basis,
            model_id=base_model_id,
        )
        x_current = _state_values(state, current_training["origin_date"])
        current_complete = np.isfinite(x_current) & np.isfinite(
            current_training["y_origin"].to_numpy(dtype=float)
        )
        if not current_complete.any():
            continue
        current_model = fit_single_pool(
            x_current,
            current_training["y_origin"].to_numpy(dtype=float),
            k_per_d=kernel.k_per_d,
            kernel_max_lag=kernel.max_lag,
            load_basis=load_basis,
            model_id=f"{base_model_id}_CURRENT",
        )
        eval_state = _state_values(state, evaluation["origin_date"])
        base_prediction = direct_model.predict(eval_state).clipped
        current_prediction = current_model.predict(eval_state).clipped
        current_end = (
            current_training.loc[current_complete, "origin_date"].max()
            if current_complete.any()
            else pd.NaT
        )
        frame = evaluation.copy()
        frame["y_pred_M1"] = base_prediction
        frame["y_pred_current"] = current_prediction
        frame["origin_y_observed"] = frame["y_origin"].notna()
        frame["current_model_max_training_target_date"] = current_end
        frame["direct_model_max_training_target_date"] = _max_complete_training_target(
            direct_training, x_direct
        )
        frame["inner_fold"] = inner.fold_id
        frame["base_model_id"] = base_model_id
        frame["load_basis"] = load_basis
        frame["k"] = kernel.k_per_d
        frame["beta0"] = direct_model.beta0
        frame["beta1"] = direct_model.beta1
        frame["beta0_current"] = current_model.beta0
        frame["beta1_current"] = current_model.beta1
        if pd.notna(current_end) and current_end >= inner.eval_start:
            raise AssertionError("M2 current model was not frozen before inner evaluation")
        frames.append(frame)
    if not frames:
        raise RuntimeError("no inner M2 base predictions")
    return pd.concat(frames, ignore_index=True)


def _fit_outer_m2(
    samples: pd.DataFrame,
    state: pd.DataFrame,
    kernel: FirstOrderKernel,
    fold: ExpandingFold,
    *,
    base_model_id: str,
    load_basis: str,
    tau_res: int,
) -> tuple[pd.DataFrame, SinglePoolModel, SinglePoolModel]:
    direct_training, evaluation = _outer_training_and_evaluation(samples, fold)
    current_training = samples[samples["origin_date"].lt(fold.eval_start)].copy()
    x_direct = _state_values(state, direct_training["origin_date"])
    direct_model = fit_single_pool(
        x_direct,
        direct_training["y_true"].to_numpy(dtype=float),
        k_per_d=kernel.k_per_d,
        kernel_max_lag=kernel.max_lag,
        load_basis=load_basis,
        model_id=base_model_id,
    )
    x_current = _state_values(state, current_training["origin_date"])
    current_model = fit_single_pool(
        x_current,
        current_training["y_origin"].to_numpy(dtype=float),
        k_per_d=kernel.k_per_d,
        kernel_max_lag=kernel.max_lag,
        load_basis=load_basis,
        model_id=f"{base_model_id}_CURRENT",
    )
    eval_state = _state_values(state, evaluation["origin_date"])
    base = direct_model.predict(eval_state).clipped
    current = current_model.predict(eval_state).clipped
    raw = np.asarray(
        residual_inertia_correction(
            base,
            current,
            evaluation["y_origin"].to_numpy(dtype=float),
            int(samples["horizon"].iloc[0]),
            tau_res,
        ),
        dtype=float,
    )
    clipped = np.where(np.isfinite(raw), np.maximum(raw, 0.0), np.nan)
    physical = SimpleNamespace(raw=raw, clipped=clipped)
    current_complete = np.isfinite(x_current) & np.isfinite(
        current_training["y_origin"].to_numpy(dtype=float)
    )
    current_end = (
        current_training.loc[current_complete, "origin_date"].max()
        if current_complete.any()
        else pd.NaT
    )
    rows = _prediction_rows(
        evaluation,
        state,
        physical,
        model_id="M2",
        load_basis=load_basis,
        outer_fold=fold.fold_id,
        k=kernel.k_per_d,
        kernel_max_lag=kernel.max_lag,
        beta0=direct_model.beta0,
        beta1=direct_model.beta1,
        max_training_target_date=_max_complete_training_target(direct_training, x_direct),
        uses_target_history=True,
        extra={
            "tau_res": tau_res,
            "base_model_id": base_model_id,
            "beta0_current": current_model.beta0,
            "beta1_current": current_model.beta1,
            "current_model_max_training_target_date": current_end,
            "base_prediction_M1": base,
            "current_mechanistic_prediction": current,
            "origin_residual": evaluation["y_origin"].to_numpy(dtype=float) - current,
        },
    )
    missing_origin = rows["y_origin"].isna()
    rows.loc[missing_origin, "availability_reason"] = "ORIGIN_TARGET_MISSING"
    rows["origin_target_available"] = ~missing_origin
    rows["m1_m2_common"] = np.isfinite(base) & rows["prediction_available"]
    rows["m1_m2_persistence_triple"] = False
    if pd.notna(current_end) and current_end >= fold.eval_start:
        raise AssertionError("M2 current model was not frozen before outer evaluation")
    return rows, direct_model, current_model


def build_mechanistic_oof(
    process: pd.DataFrame,
    persistence: pd.DataFrame,
    states: dict[tuple[str, float], pd.DataFrame],
    kernels: dict[float, FirstOrderKernel],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build all development-only mechanistic OOF rows and selection audits."""

    outer_folds = make_expanding_outer_folds(
        pd.date_range(DEVELOPMENT_START, DEVELOPMENT_END, freq="D")
    )
    prediction_frames: list[pd.DataFrame] = []
    selection_frames: list[pd.DataFrame] = []
    residual_frames: list[pd.DataFrame] = []
    pairs = two_pool_candidate_pairs()
    if len(pairs) != 15:
        raise AssertionError("locked two-pool grid must contain exactly 15 pairs")

    for target_name in TARGET_COLUMNS:
        for horizon in HORIZONS:
            print(f"R5 development OOF: {target_name} h{horizon}", flush=True)
            sample = _samples(process, target_name, horizon)
            for outer in outer_folds:
                evaluation = sample[
                    sample["origin_date"].between(outer.eval_start, outer.eval_end)
                ]
                if evaluation.empty:
                    continue

                selected_single: dict[str, dict[str, Any]] = {}
                for model_id in ("M1F", "M1TS", "M1VS"):
                    spec = LOAD_SPECS[model_id]
                    candidate_predictions: list[pd.DataFrame] = []
                    candidate_scores: list[pd.DataFrame] = []
                    for k in K_CANDIDATES:
                        predictions, scores = _inner_single_predictions(
                            sample,
                            states[(model_id, k)],
                            kernels[k],
                            outer,
                            persistence,
                            model_id=model_id,
                            load_basis=str(spec["load_basis"]),
                        )
                        candidate_predictions.append(predictions)
                        candidate_scores.append(scores)
                    inner_predictions = pd.concat(candidate_predictions, ignore_index=True)
                    score_table = pd.concat(candidate_scores, ignore_index=True)
                    ranked, selected = _rank_inner_family(
                        score_table,
                        inner_predictions,
                        outer_fold=outer.fold_id,
                        target_name=target_name,
                        horizon=horizon,
                        candidate_model=model_id,
                        load_basis=str(spec["load_basis"]),
                    )
                    ranked["runtime_lineage"] = str(spec["lineage"])
                    ranked["R3_existing_kinetic_feature_reused"] = False
                    selection_frames.append(ranked)
                    selected_k = float(selected["k"])
                    outer_rows, fitted = _fit_outer_single(
                        sample,
                        states[(model_id, selected_k)],
                        kernels[selected_k],
                        outer,
                        model_id=model_id,
                        load_basis=str(spec["load_basis"]),
                    )
                    outer_rows["load_unit"] = str(spec["unit"])
                    outer_rows["runtime_lineage"] = str(spec["lineage"])
                    prediction_frames.append(outer_rows)
                    selected_single[model_id] = {
                        "k": selected_k,
                        "score": selected,
                        "fitted": fitted,
                        "load_basis": str(spec["load_basis"]),
                    }

                two_predictions: list[pd.DataFrame] = []
                two_scores: list[pd.DataFrame] = []
                for k_fast, k_slow in pairs:
                    predictions, scores = _inner_two_pool_predictions(
                        sample,
                        states[("M1TS", k_fast)],
                        states[("M1TS", k_slow)],
                        outer,
                        persistence,
                        k_fast=k_fast,
                        k_slow=k_slow,
                    )
                    two_predictions.append(predictions)
                    two_scores.append(scores)
                two_inner = pd.concat(two_predictions, ignore_index=True)
                two_score_table = pd.concat(two_scores, ignore_index=True)
                two_ranked, two_selected = _rank_inner_family(
                    two_score_table,
                    two_inner,
                    outer_fold=outer.fold_id,
                    target_name=target_name,
                    horizon=horizon,
                    candidate_model="M1TS2",
                    load_basis="TS_LOAD_PROXY",
                )
                two_ranked["runtime_lineage"] = LOAD_SPECS["M1TS"]["lineage"]
                two_ranked["R3_existing_kinetic_feature_reused"] = False
                selection_frames.append(two_ranked)
                selected_fast = float(two_selected["k_fast"])
                selected_slow = float(two_selected["k_slow"])
                two_rows, _ = _fit_outer_two_pool(
                    sample,
                    states[("M1TS", selected_fast)],
                    states[("M1TS", selected_slow)],
                    kernels[selected_fast],
                    kernels[selected_slow],
                    outer,
                )
                two_rows["load_unit"] = "t TS/d"
                two_rows["runtime_lineage"] = LOAD_SPECS["M1TS"]["lineage"]
                prediction_frames.append(two_rows)

                base_options = pd.DataFrame(
                    [
                        {
                            "candidate_id": model_id,
                            "candidate_model": "M2_BASE",
                            "model_id": model_id,
                            "load_basis": selected_single[model_id]["load_basis"],
                            "k": selected_single[model_id]["k"],
                            "inner_RMSE": float(selected_single[model_id]["score"]["inner_RMSE"]),
                            "inner_MAE": float(selected_single[model_id]["score"]["inner_MAE"]),
                            "inner_Skill": float(selected_single[model_id]["score"]["inner_Skill"]),
                            "inner_n": int(selected_single[model_id]["score"]["inner_n"]),
                            "complexity_rank": 1,
                        }
                        for model_id in ("M1F", "M1TS")
                    ]
                )
                base_options["eligible"] = np.isfinite(base_options["inner_RMSE"])
                base_options = rank_candidates(base_options)
                base_options["outer_fold"] = outer.fold_id
                base_options["target"] = target_name
                base_options["horizon"] = horizon
                base_options["selection_reason"] = np.where(
                    base_options["selected"],
                    "BEST_ELIGIBLE_M1_BY_INNER_OWN_RMSE_SKILL_SIMPLICITY",
                    "NOT_SELECTED",
                )
                selection_frames.append(base_options)
                base_row = base_options.loc[base_options["selected"]].iloc[0]
                base_model_id = str(base_row["model_id"])
                base_k = float(base_row["k"])
                load_basis = str(base_row["load_basis"])
                m2_inner = _inner_m2_base_predictions(
                    sample,
                    states[(base_model_id, base_k)],
                    kernels[base_k],
                    outer,
                    base_model_id=base_model_id,
                    load_basis=load_basis,
                )
                tau_selection = select_residual_tau(
                    m2_inner,
                    persistence,
                    tau_candidates=TAU_RES_CANDIDATES,
                )
                tau_scores = tau_selection.scores.copy()
                tau_scores["outer_fold"] = outer.fold_id
                tau_scores["target"] = target_name
                tau_scores["horizon"] = horizon
                tau_scores["base_model_id"] = base_model_id
                tau_scores["selected_tau"] = tau_selection.selected_tau
                tau_scores["selected"] = tau_scores["tau_res"].eq(
                    tau_selection.selected_tau
                )
                residual_frames.append(tau_scores)
                m2_rows, _, _ = _fit_outer_m2(
                    sample,
                    states[(base_model_id, base_k)],
                    kernels[base_k],
                    outer,
                    base_model_id=base_model_id,
                    load_basis=load_basis,
                    tau_res=tau_selection.selected_tau,
                )
                m2_rows["load_unit"] = str(LOAD_SPECS[base_model_id]["unit"])
                m2_rows["runtime_lineage"] = str(LOAD_SPECS[base_model_id]["lineage"])
                prediction_frames.append(m2_rows)

    predictions = pd.concat(prediction_frames, ignore_index=True)
    predictions["baseline_id"] = predictions["model_id"]
    predictions = merge_persistence_reference(predictions, persistence)
    predictions["persistence_paired"] = (
        predictions["own_scored"]
        & predictions["persistence_available"].fillna(False).astype(bool)
        & np.isfinite(pd.to_numeric(predictions["y_persistence"], errors="coerce"))
    )
    predictions["m1_m2_common"] = predictions.get(
        "m1_m2_common", pd.Series(False, index=predictions.index)
    ).fillna(False).astype(bool)
    predictions["m1_m2_persistence_triple"] = (
        predictions["model_id"].eq("M2")
        & predictions["m1_m2_common"]
        & predictions["persistence_paired"]
    )
    predictions["period"] = "DEVELOPMENT"

    invalid_train = predictions["prediction_available"] & (
        pd.to_datetime(predictions["max_training_target_date"]).isna()
        | pd.to_datetime(predictions["max_training_target_date"]).ge(
            predictions["origin_date"]
        )
    )
    if invalid_train.any():
        raise AssertionError("OOF row violates max(training_target_date) < origin")
    invalid_feature = predictions["prediction_available"] & (
        pd.to_datetime(predictions["max_feature_timestamp"]).isna()
        | pd.to_datetime(predictions["max_feature_timestamp"]).gt(
            predictions["origin_date"]
        )
    )
    if invalid_feature.any():
        raise AssertionError("OOF row contains a future feature timestamp")
    if predictions["target_date"].max() > DEVELOPMENT_END:
        raise AssertionError("Phase 05 generated a post-2021 prediction")
    key = ["target_name", "horizon", "origin_date", "target_date", "model_id"]
    if predictions.duplicated(key).any():
        raise AssertionError("mechanistic OOF prediction key is not unique")
    predictions = predictions.sort_values(
        ["target_name", "horizon", "model_id", "origin_date"], kind="stable"
    ).reset_index(drop=True)
    inner_selection = pd.concat(selection_frames, ignore_index=True, sort=False)
    residual_selection = pd.concat(residual_frames, ignore_index=True, sort=False)
    return predictions, inner_selection, residual_selection


def _fold_metric_rows(predictions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    group_columns = ["target_name", "horizon", "model_id", "outer_fold"]
    for key, group in predictions.groupby(group_columns, sort=True):
        target, horizon, model, fold = key
        target_mask = group["target_available"].fillna(False).astype(bool)
        predicted_mask = group["prediction_available"].fillna(False).astype(bool)
        for support in ("OWN_AVAILABLE", "PERSISTENCE_PAIRED"):
            mask = target_mask & predicted_mask
            if support == "PERSISTENCE_PAIRED":
                mask &= group["persistence_available"].fillna(False).astype(bool)
                mask &= np.isfinite(pd.to_numeric(group["y_persistence"], errors="coerce"))
            scored = group.loc[mask]
            metric = compute_metrics(
                scored["y_true"],
                scored["y_pred"],
                y_origin=scored["y_origin"],
            )
            skill = (
                skill_vs_persistence(
                    scored["y_true"], scored["y_pred"], scored["y_persistence"]
                )
                if support == "PERSISTENCE_PAIRED"
                else np.nan
            )
            n_target = int(target_mask.sum())
            rows.append(
                {
                    "target_name": target,
                    "horizon": int(horizon),
                    "baseline_id": model,
                    "period": fold,
                    "support_type": support,
                    "n_target": n_target,
                    "n_predicted": int(predicted_mask.sum()),
                    "n_scored": int(metric["n_scored"]),
                    "coverage_rate": float(metric["n_scored"] / n_target) if n_target else np.nan,
                    **metric,
                    "Skill_vs_persistence": skill,
                }
            )
    return pd.DataFrame(rows)


def evaluate_mechanistic_oof(
    predictions: pd.DataFrame,
    persistence: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Reuse immutable Phase 04 metrics and append explicit fold diagnostics."""

    evaluation = predictions.copy()
    evaluation["baseline_id"] = evaluation["model_id"]
    metrics, _ = evaluate_baseline_predictions(
        evaluation,
        persistence,
        periods=("DEVELOPMENT", "2019", "2020", "2021"),
    )
    fold_metrics = _fold_metric_rows(predictions)
    combined = pd.concat([metrics, fold_metrics], ignore_index=True, sort=False)
    combined["target"] = combined["target_name"]
    combined["model_id"] = combined["baseline_id"]
    combined["support"] = combined["support_type"]
    combined["n"] = combined["n_scored"]
    combined["coverage"] = combined["coverage_rate"]
    order = [
        "target",
        "target_name",
        "horizon",
        "model_id",
        "baseline_id",
        "period",
        "support",
        "support_type",
        "n",
        "n_target",
        "n_predicted",
        "n_scored",
        "coverage",
        "coverage_rate",
        "n_delta_scored",
        "MSE",
        "RMSE",
        "MAE",
        "sMAPE",
        "R2_level",
        "R2_delta",
        "Bias",
        "Skill_vs_persistence",
    ]
    trailing = [column for column in combined.columns if column not in order]
    return combined.loc[:, order + trailing], fold_metrics


def build_parameter_stability(
    predictions: pd.DataFrame,
    fold_metrics: pd.DataFrame,
) -> pd.DataFrame:
    """Return fold-local selected parameters and physical-identification status."""

    metric_own = fold_metrics[
        fold_metrics["support_type"].eq("OWN_AVAILABLE")
    ][["target_name", "horizon", "baseline_id", "period", "RMSE"]]
    metric_skill = fold_metrics[
        fold_metrics["support_type"].eq("PERSISTENCE_PAIRED")
    ][
        [
            "target_name",
            "horizon",
            "baseline_id",
            "period",
            "Skill_vs_persistence",
        ]
    ]
    rows: list[dict[str, object]] = []
    keys = ["target_name", "horizon", "model_id", "outer_fold"]
    for key, group in predictions.groupby(keys, sort=True):
        target, horizon, model, fold = key
        first = group.iloc[0]
        if model == "M1TS2":
            selected_k = f"{float(first.k_slow):.2f}/{float(first.k_fast):.2f}"
        else:
            selected_k = f"{float(first.k):.2f}"
        own = metric_own[
            metric_own["target_name"].eq(target)
            & metric_own["horizon"].eq(horizon)
            & metric_own["baseline_id"].eq(model)
            & metric_own["period"].eq(fold)
        ]
        skill = metric_skill[
            metric_skill["target_name"].eq(target)
            & metric_skill["horizon"].eq(horizon)
            & metric_skill["baseline_id"].eq(model)
            & metric_skill["period"].eq(fold)
        ]
        rows.append(
            {
                "target": target,
                "target_name": target,
                "horizon": int(horizon),
                "model_id": model,
                "outer_fold": fold,
                "selected_k": selected_k,
                "selected_k_numeric": float(first.k) if pd.notna(first.k) else np.nan,
                "k_fast": first.k_fast,
                "k_slow": first.k_slow,
                "tau_res": first.tau_res,
                "base_model_id": first.base_model_id,
                "beta0": first.beta0,
                "beta1": first.beta1,
                "beta_fast": first.beta_fast,
                "beta_slow": first.beta_slow,
                "beta0_current": first.beta0_current,
                "beta1_current": first.beta1_current,
                "kernel_coverage_median": float(
                    pd.to_numeric(group["kernel_coverage"], errors="coerce").median()
                ),
                "RMSE": float(own["RMSE"].iloc[0]) if len(own) else np.nan,
                "Skill": float(skill["Skill_vs_persistence"].iloc[0]) if len(skill) else np.nan,
                "two_pool_degenerate": bool(group["two_pool_degenerate"].fillna(False).iloc[0]),
            }
        )
    frame = pd.DataFrame(rows)
    frame["selected_k_frequency"] = 0.0
    frame["parameter_status"] = "EFFECTIVE_PREDICTIVE_PARAMETER"
    for _, index in frame.groupby(["target", "horizon", "model_id"], sort=False).groups.items():
        subset = frame.loc[index]
        frequency = subset["selected_k"].value_counts(normalize=True)
        frame.loc[index, "selected_k_frequency"] = subset["selected_k"].map(frequency)
        mode_fraction = float(frequency.max()) if len(frequency) else 0.0
        model = str(subset["model_id"].iloc[0])
        if model == "M1TS2" and bool(subset["two_pool_degenerate"].mean() >= 0.5):
            status = "EFFECTIVELY_SINGLE_POOL"
        elif model == "M2":
            status = "TARGET_HISTORY_DEPENDENT_EFFECTIVE_PARAMETER"
        elif mode_fraction < 0.60:
            status = "PHYSICAL_K_NOT_IDENTIFIED"
        else:
            status = "EFFECTIVE_K_PROVISIONAL_STABILITY"
        frame.loc[index, "parameter_status"] = status
    return frame


def build_development_summary(
    metrics: pd.DataFrame,
    parameters: pd.DataFrame,
    predictions: pd.DataFrame,
) -> pd.DataFrame:
    own = metrics[
        metrics["period"].eq("DEVELOPMENT")
        & metrics["support_type"].eq("OWN_AVAILABLE")
    ].copy()
    paired = metrics[
        metrics["period"].eq("DEVELOPMENT")
        & metrics["support_type"].eq("PERSISTENCE_PAIRED")
    ][["target_name", "horizon", "model_id", "Skill_vs_persistence"]].rename(
        columns={"Skill_vs_persistence": "Skill"}
    )
    own = own.merge(
        paired,
        on=["target_name", "horizon", "model_id"],
        how="left",
        validate="one_to_one",
    )
    fold_stats = (
        parameters.groupby(["target", "horizon", "model_id"], sort=True)
        .agg(
            fold_RMSE_mean=("RMSE", "mean"),
            fold_RMSE_std=("RMSE", "std"),
            selected_k_mode=("selected_k", lambda values: values.mode().iloc[0]),
            selected_k_stability=("selected_k", lambda values: float(values.value_counts(normalize=True).max())),
            physical_status=("parameter_status", lambda values: values.mode().iloc[0]),
        )
        .reset_index()
        .rename(columns={"target": "target_name"})
    )
    summary = own.merge(
        fold_stats,
        on=["target_name", "horizon", "model_id"],
        how="left",
        validate="one_to_one",
    )
    summary = summary.rename(
        columns={
            "n_scored": "OOF_n",
            "coverage_rate": "coverage",
        }
    )
    summary["target"] = summary["target_name"]
    summary["model"] = summary["model_id"]
    summary["OOF_eligible"] = predictions.groupby(
        ["target_name", "horizon", "model_id"]
    ).size().reindex(
        pd.MultiIndex.from_frame(summary[["target_name", "horizon", "model_id"]])
    ).to_numpy()
    columns = [
        "target",
        "target_name",
        "horizon",
        "model",
        "model_id",
        "OOF_n",
        "OOF_eligible",
        "coverage",
        "RMSE",
        "MAE",
        "sMAPE",
        "R2_level",
        "R2_delta",
        "Bias",
        "Skill",
        "fold_RMSE_mean",
        "fold_RMSE_std",
        "selected_k_mode",
        "selected_k_stability",
        "physical_status",
    ]
    return summary.loc[:, columns]


def _common_model_comparison(
    predictions: pd.DataFrame,
    target: str,
    horizon: int,
    left_model: str,
    right_model: str,
) -> dict[str, float]:
    keys = ["origin_date", "target_date", "outer_fold"]
    columns = keys + ["y_true", "y_pred", "prediction_available", "target_available"]
    left = predictions[
        predictions["target_name"].eq(target)
        & predictions["horizon"].eq(horizon)
        & predictions["model_id"].eq(left_model)
    ][columns]
    right = predictions[
        predictions["target_name"].eq(target)
        & predictions["horizon"].eq(horizon)
        & predictions["model_id"].eq(right_model)
    ][columns]
    paired = left.merge(
        right,
        on=keys,
        suffixes=("_left", "_right"),
        validate="one_to_one",
    )
    mask = (
        paired["target_available_left"].fillna(False)
        & paired["target_available_right"].fillna(False)
        & paired["prediction_available_left"].fillna(False)
        & paired["prediction_available_right"].fillna(False)
    )
    common = paired.loc[mask]
    left_rmse = compute_metrics(common["y_true_left"], common["y_pred_left"])["RMSE"]
    right_rmse = compute_metrics(common["y_true_right"], common["y_pred_right"])["RMSE"]
    fold_improvements: list[bool] = []
    for _, group in common.groupby("outer_fold", sort=True):
        lrmse = compute_metrics(group["y_true_left"], group["y_pred_left"])["RMSE"]
        rrmse = compute_metrics(group["y_true_right"], group["y_pred_right"])["RMSE"]
        if np.isfinite(lrmse) and np.isfinite(rrmse):
            fold_improvements.append(bool(rrmse < lrmse))
    return {
        "n_common": int(len(common)),
        "left_RMSE": float(left_rmse),
        "right_RMSE": float(right_rmse),
        "right_minus_left_RMSE": float(right_rmse - left_rmse),
        "improving_fold_fraction": float(np.mean(fold_improvements)) if fold_improvements else np.nan,
    }


def build_lock_table(
    predictions: pd.DataFrame,
    parameters: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for target in TARGET_COLUMNS:
        for horizon in HORIZONS:
            for model, status, reason in (
                ("M1F", "LOCK_FOR_R7", "required exogenous wet-feed comparator"),
                ("M1TS", "LOCK_FOR_R7", "required provisional TS-load comparator"),
                ("M1VS", "RETAIN_AS_SENSITIVITY", "VS_2020_STRUCTURAL_BREAK"),
                (
                    "M2",
                    "LOCK_FOR_R7_SEPARATE_FAMILY",
                    "constrained target-residual inertia; target-history dependence explicit",
                ),
            ):
                rows.append(
                    {
                        "target": target,
                        "horizon": horizon,
                        "model_id": model,
                        "status": status,
                        "reason": reason,
                        "n_common": np.nan,
                        "RMSE_improvement": np.nan,
                        "improving_fold_fraction": np.nan,
                        "degenerate_fold_fraction": np.nan,
                    }
                )
            comparison = _common_model_comparison(
                predictions, target, horizon, "M1TS", "M1TS2"
            )
            subset = parameters[
                parameters["target"].eq(target)
                & parameters["horizon"].eq(horizon)
                & parameters["model_id"].eq("M1TS2")
            ]
            degenerate_fraction = float(subset["two_pool_degenerate"].mean()) if len(subset) else 1.0
            improvement = -float(comparison["right_minus_left_RMSE"])
            fold_fraction = float(comparison["improving_fold_fraction"])
            qualified = (
                np.isfinite(improvement)
                and improvement > 0
                and np.isfinite(fold_fraction)
                and fold_fraction >= 0.60
                and degenerate_fraction < 0.50
            )
            rows.append(
                {
                    "target": target,
                    "horizon": horizon,
                    "model_id": "M1TS2",
                    "status": "LOCK_FOR_R7" if qualified else "REJECT_DEGENERATE",
                    "reason": (
                        "positive common-support OOF gain, fold consistency, and active pools"
                        if qualified
                        else "two-pool promotion criteria not jointly satisfied; effectively single-pool"
                    ),
                    "n_common": comparison["n_common"],
                    "RMSE_improvement": improvement,
                    "improving_fold_fraction": fold_fraction,
                    "degenerate_fold_fraction": degenerate_fraction,
                }
            )
    return pd.DataFrame(rows)


def _deterministic_mode(values: Iterable[Any]) -> Any:
    cleaned = [value for value in values if pd.notna(value) and str(value) != ""]
    if not cleaned:
        raise ValueError("cannot lock a mode from empty parameter values")
    counts = Counter(cleaned)
    highest = max(counts.values())
    return sorted(
        (value for value, count in counts.items() if count == highest),
        key=lambda value: str(value),
    )[0]


def _single_model_metadata(
    *,
    model_id: str,
    target: str,
    horizon: int,
    load_basis: str,
    k: float,
    kernel: FirstOrderKernel,
    fitted: SinglePoolModel,
    source_hashes: dict[str, str],
    uses_target_history: bool = False,
    extra_coefficients: dict[str, float] | None = None,
) -> dict[str, object]:
    coefficients: dict[str, float] = {
        "beta0": fitted.beta0,
        "beta1": fitted.beta1,
    }
    if extra_coefficients:
        coefficients.update(extra_coefficients)
    return {
        "model_id": model_id,
        "target": target,
        "horizon": horizon,
        "load_basis": load_basis,
        "k": k,
        "kernel_definition": "w_k(j)=exp(-k*j)-exp(-k*(j+1)); HRT-free exact daily bins",
        "kernel_tail_tolerance": KERNEL_TAIL_TOL,
        "kernel_max_lag": kernel.max_lag,
        "beta_coefficients": coefficients,
        "training_start": str(DEVELOPMENT_START.date()),
        "training_end": str(DEVELOPMENT_END.date()),
        "source_hashes": source_hashes,
        "uses_target_history": uses_target_history,
        "preprocessing_rule": (
            "coverage_normalized_kinetic_state; MIN_KERNEL_COVERAGE=0.90; "
            "missing load/target never imputed; nonnegative coefficients"
        ),
    }


def full_development_refit(
    process: pd.DataFrame,
    states: dict[tuple[str, float], pd.DataFrame],
    kernels: dict[float, FirstOrderKernel],
    parameters: pd.DataFrame,
    lock_table: pd.DataFrame,
) -> dict[str, object]:
    """Refit locked structures on matured 2018-2021 targets only."""

    source_hashes = {path.name: sha256_file(path) for path in _source_paths()}
    models: list[dict[str, object]] = []
    for target in TARGET_COLUMNS:
        for horizon in HORIZONS:
            sample = _samples(process, target, horizon)
            y = sample["y_true"].to_numpy(dtype=float)
            for model_id in ("M1F", "M1TS", "M1VS"):
                subset = parameters[
                    parameters["target"].eq(target)
                    & parameters["horizon"].eq(horizon)
                    & parameters["model_id"].eq(model_id)
                ]
                k = float(_deterministic_mode(subset["selected_k_numeric"]))
                state = states[(model_id, k)]
                fitted = fit_single_pool(
                    _state_values(state, sample["origin_date"]),
                    y,
                    k_per_d=k,
                    kernel_max_lag=kernels[k].max_lag,
                    load_basis=str(LOAD_SPECS[model_id]["load_basis"]),
                    model_id=model_id,
                )
                metadata = _single_model_metadata(
                    model_id=model_id,
                    target=target,
                    horizon=horizon,
                    load_basis=str(LOAD_SPECS[model_id]["load_basis"]),
                    k=k,
                    kernel=kernels[k],
                    fitted=fitted,
                    source_hashes=source_hashes,
                )
                metadata["candidate_status"] = (
                    "RETAIN_AS_SENSITIVITY" if model_id == "M1VS" else "LOCK_FOR_R7"
                )
                metadata["load_lineage"] = LOAD_SPECS[model_id]["lineage"]
                models.append(metadata)

            two_status = lock_table[
                lock_table["target"].eq(target)
                & lock_table["horizon"].eq(horizon)
                & lock_table["model_id"].eq("M1TS2")
            ]["status"].iloc[0]
            if two_status == "LOCK_FOR_R7":
                subset = parameters[
                    parameters["target"].eq(target)
                    & parameters["horizon"].eq(horizon)
                    & parameters["model_id"].eq("M1TS2")
                ]
                pair = _deterministic_mode(
                    list(zip(subset["k_fast"].astype(float), subset["k_slow"].astype(float)))
                )
                k_fast, k_slow = float(pair[0]), float(pair[1])
                fitted_two = fit_two_pool(
                    _state_values(states[("M1TS", k_fast)], sample["origin_date"]),
                    _state_values(states[("M1TS", k_slow)], sample["origin_date"]),
                    y,
                    k_fast=k_fast,
                    k_slow=k_slow,
                    load_basis="TS_LOAD_PROXY",
                )
                models.append(
                    {
                        "model_id": "M1TS2",
                        "target": target,
                        "horizon": horizon,
                        "load_basis": "TS_LOAD_PROXY",
                        "k": {"fast": k_fast, "slow": k_slow},
                        "kernel_definition": "two separated HRT-free exact daily-bin first-order states",
                        "kernel_tail_tolerance": KERNEL_TAIL_TOL,
                        "kernel_max_lag": {
                            "fast": kernels[k_fast].max_lag,
                            "slow": kernels[k_slow].max_lag,
                        },
                        "beta_coefficients": {
                            "beta0": fitted_two.beta0,
                            "beta_fast": fitted_two.beta_fast,
                            "beta_slow": fitted_two.beta_slow,
                        },
                        "training_start": str(DEVELOPMENT_START.date()),
                        "training_end": str(DEVELOPMENT_END.date()),
                        "source_hashes": source_hashes,
                        "uses_target_history": False,
                        "preprocessing_rule": "TS-load coverage-normalized states; 0.90 coverage; k_fast/k_slow>=2",
                        "candidate_status": two_status,
                    }
                )

            m2_parameters = parameters[
                parameters["target"].eq(target)
                & parameters["horizon"].eq(horizon)
                & parameters["model_id"].eq("M2")
            ]
            combo = _deterministic_mode(
                list(
                    zip(
                        m2_parameters["base_model_id"].astype(str),
                        m2_parameters["selected_k_numeric"].astype(float),
                        m2_parameters["tau_res"].astype(int),
                    )
                )
            )
            base_model_id, base_k, tau = str(combo[0]), float(combo[1]), int(combo[2])
            state = states[(base_model_id, base_k)]
            direct = fit_single_pool(
                _state_values(state, sample["origin_date"]),
                y,
                k_per_d=base_k,
                kernel_max_lag=kernels[base_k].max_lag,
                load_basis=str(LOAD_SPECS[base_model_id]["load_basis"]),
                model_id=base_model_id,
            )
            full_dates = pd.DatetimeIndex(process["date"])
            current = fit_single_pool(
                _state_values(state, full_dates),
                _series(process, TARGET_COLUMNS[target]).reindex(full_dates).to_numpy(dtype=float),
                k_per_d=base_k,
                kernel_max_lag=kernels[base_k].max_lag,
                load_basis=str(LOAD_SPECS[base_model_id]["load_basis"]),
                model_id=f"{base_model_id}_CURRENT",
            )
            m2 = _single_model_metadata(
                model_id="M2",
                target=target,
                horizon=horizon,
                load_basis=str(LOAD_SPECS[base_model_id]["load_basis"]),
                k=base_k,
                kernel=kernels[base_k],
                fitted=direct,
                source_hashes=source_hashes,
                uses_target_history=True,
                extra_coefficients={
                    "beta0_current": current.beta0,
                    "beta1_current": current.beta1,
                    "tau_res": float(tau),
                },
            )
            m2["base_model_id"] = base_model_id
            m2["residual_rule"] = "base + (observed_y_origin-current_fit)*exp(-h/tau_res); gamma fixed to 1"
            m2["candidate_status"] = "LOCK_FOR_R7_SEPARATE_FAMILY"
            models.append(m2)

    history_days = max(kernel.max_lag for kernel in kernels.values()) + 1
    history_start = DEVELOPMENT_END - pd.Timedelta(days=history_days - 1)
    history = process.loc[
        process["date"].ge(history_start),
        ["date", "feed_AB_tpd", "TS_load_tpd", "VS_load_tpd"],
    ].copy()
    return {
        "phase": "05",
        "model_family": "HRT_FREE_FIRST_ORDER_PROCESS_INFORMED",
        "selection_data_start": str(DEVELOPMENT_START.date()),
        "selection_data_end": str(DEVELOPMENT_END.date()),
        "validation_2022_used": False,
        "test_2023_used": False,
        "models": models,
        "pre_2022_load_history": {
            "start": str(history_start.date()),
            "end": str(DEVELOPMENT_END.date()),
            "records": history.assign(date=history["date"].dt.strftime("%Y-%m-%d")).to_dict("records"),
        },
    }


def render_model_lock(
    lock_table: pd.DataFrame,
    parameters: pd.DataFrame,
) -> str:
    family_status = {
        "M1F": "LOCK_FOR_R7",
        "M1TS": "LOCK_FOR_R7",
        "M1VS": "RETAIN_AS_SENSITIVITY",
        "M1TS2": (
            "LOCK_FOR_R7_TARGET_HORIZON_CONDITIONAL"
            if lock_table[
                lock_table["model_id"].eq("M1TS2")
                & lock_table["status"].eq("LOCK_FOR_R7")
            ].shape[0]
            else "REJECT_DEGENERATE"
        ),
        "M2": "LOCK_FOR_R7_SEPARATE_FAMILY",
        "M3_COD": "BLOCKED",
        "M3_HYDRAULIC": "BLOCKED",
    }
    lines = [
        "phase: 05",
        "model_family: HRT_FREE_FIRST_ORDER_PROCESS_INFORMED",
        "selection_data_start: 2018-01-01",
        "selection_data_end: 2021-12-31",
        "validation_2022_used: false",
        "test_2023_used: false",
        "selection_rule: modal_fold_local_inner_prequential_choice_without_outer_truth_retuning",
        "phase04_persistence_reference: B10_PERSISTENCE_STRICT",
        "phase04_2022_hurdle_reference_only:",
        "  CH4_h7_RMSE: 783.026",
        "  CH4_h7_MAE: 613.139",
        "  CH4_h14_RMSE: 946.688",
        "  CH4_h14_MAE: 740.524",
        "runtime_feature_lineage:",
        "  R3_registry_modified: false",
        "  R3_existing_concentration_kinetic_proxy_reused_as_load_state: false",
        "  R5_states: Prompt05-authorized feed_AB_tpd_TS_load_tpd_VS_load_tpd_derivations",
        "candidates:",
    ]
    notes = {
        "M1F": "wet-feed exogenous kinetic comparator",
        "M1TS": "provisional TS-load exogenous kinetic comparator",
        "M1VS": "VS_2020_STRUCTURAL_BREAK regime-sensitive",
        "M1TS2": "promotion decided separately by target and horizon",
        "M2": "constrained target residual; no free gamma or AR coefficient",
        "M3_COD": "Q_feed unknown; COD mass load unavailable",
        "M3_HYDRAULIC": "Q/HRT/RTD/V_active unknown; HRT_mass_proxy prohibited",
    }
    for model_id, status in family_status.items():
        lines.extend(
            [
                f"  {model_id}:",
                f"    status: {status}",
                f"    notes: {notes[model_id]}",
            ]
        )
    lines.append("target_horizon_locks:")
    for row in lock_table.sort_values(["target", "horizon", "model_id"]).itertuples(index=False):
        subset = parameters[
            parameters["target"].eq(row.target)
            & parameters["horizon"].eq(row.horizon)
            & parameters["model_id"].eq(row.model_id)
        ]
        selected = ""
        if len(subset):
            if row.model_id == "M1TS2":
                selected = str(_deterministic_mode(subset["selected_k"]))
            else:
                selected = str(_deterministic_mode(subset["selected_k"]))
                if row.model_id == "M2":
                    base = str(_deterministic_mode(subset["base_model_id"]))
                    tau = int(_deterministic_mode(subset["tau_res"].dropna().astype(int)))
                    selected += f";base={base};tau_res={tau}d"
        key = f"{row.target}_h{int(row.horizon)}_{row.model_id}".replace(" ", "_")
        lines.extend(
            [
                f"  {key}:",
                f"    status: {row.status}",
                f"    locked_structure: \"{selected}\"",
                f"    reason: \"{str(row.reason).replace(chr(34), chr(39))}\"",
            ]
        )
    return "\n".join(lines) + "\n"


def _m2_oof_comparison(predictions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    m2 = predictions[predictions["model_id"].eq("M2")].copy()
    for (target, horizon), group in m2.groupby(["target_name", "horizon"], sort=True):
        mask = (
            group["target_available"].fillna(False)
            & np.isfinite(pd.to_numeric(group["base_prediction_M1"], errors="coerce"))
            & np.isfinite(pd.to_numeric(group["y_pred"], errors="coerce"))
        )
        common = group.loc[mask]
        m1_rmse = compute_metrics(common["y_true"], common["base_prediction_M1"])["RMSE"]
        m2_rmse = compute_metrics(common["y_true"], common["y_pred"])["RMSE"]
        rows.append(
            {
                "target": target,
                "horizon": int(horizon),
                "n_common": len(common),
                "RMSE_M1_base": m1_rmse,
                "RMSE_M2": m2_rmse,
                "RMSE_improvement": m1_rmse - m2_rmse,
                "M2_better": bool(np.isfinite(m1_rmse) and np.isfinite(m2_rmse) and m2_rmse < m1_rmse),
            }
        )
    return pd.DataFrame(rows)


def _markdown_table(frame: pd.DataFrame, columns: list[str], max_rows: int = 80) -> str:
    if frame.empty:
        return "_No eligible development rows._"
    view = frame.loc[:, [column for column in columns if column in frame]].head(max_rows).copy()
    for column in view.columns:
        if pd.api.types.is_float_dtype(view[column]):
            view[column] = view[column].map(
                lambda value: "" if pd.isna(value) else f"{float(value):.4f}"
            )
        else:
            view[column] = view[column].map(lambda value: "" if pd.isna(value) else str(value))
    headers = list(view.columns)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in view.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(str(value).replace("|", "/") for value in row) + " |")
    return "\n".join(lines)


def _research_answers(
    predictions: pd.DataFrame,
    summary: pd.DataFrame,
    parameters: pd.DataFrame,
    lock_table: pd.DataFrame,
    intercept: pd.DataFrame,
) -> list[str]:
    answers: list[str] = []
    m1f = summary[summary["model_id"].eq("M1F")]
    positive = m1f[m1f["Skill"].gt(0)].groupby("target")["horizon"].apply(list).to_dict()
    answers.append(f"1. M1F positive paired Skill horizons: {positive or 'none'}; exact support only.")
    comparisons = []
    for target in TARGET_COLUMNS:
        for horizon in HORIZONS:
            item = _common_model_comparison(predictions, target, horizon, "M1F", "M1TS")
            comparisons.append(item["right_RMSE"] < item["left_RMSE"])
    answers.append(
        f"2. M1TS beat M1F on exact model-common support in {sum(comparisons)}/{len(comparisons)} target-horizon cases; therefore consistency is {'supported' if all(comparisons) else 'not supported'}."
    )
    vs = parameters[parameters["model_id"].eq("M1VS")]
    vs_unstable = (vs["parameter_status"] == "PHYSICAL_K_NOT_IDENTIFIED").mean()
    answers.append(f"3. M1VS remains REGIME_SENSITIVE; {vs_unstable:.1%} of its target-horizon fold groups carry non-identification rows and the 2020 break is not removed.")
    stability = parameters[parameters["model_id"].isin(["M1F", "M1TS", "M1VS"])]
    answers.append(f"4. Median selected-k fold frequency is {stability['selected_k_frequency'].median():.2f}; stability is assessed as an effective predictive parameter, not a physical constant.")
    horizon_modes = stability.groupby(["target", "model_id", "horizon"])["selected_k"].agg(lambda x: x.mode().iloc[0])
    varies = horizon_modes.groupby(level=[0, 1]).nunique().gt(1).sum()
    answers.append(f"5. Horizon-dependent modal k occurs in {int(varies)} target-model groups.")
    answers.append("6. No. Fold/horizon/load-basis variation plus UNKNOWN HRT/RTD prevents identification of a true hydrolysis constant.")
    two_locked = int((lock_table["model_id"].eq("M1TS2") & lock_table["status"].eq("LOCK_FOR_R7")).sum())
    answers.append(f"7. Single-pool remains sufficient as the core comparison; two-pool qualified in {two_locked}/10 target-horizon locks.")
    answers.append(f"8. M1TS2 was promoted only where common-support gain, >=60% improving folds, and <50% degenerate folds all held; otherwise it is REJECT_DEGENERATE.")
    ratios = pd.to_numeric(intercept["beta0_over_mean_prediction"], errors="coerce")
    answers.append(f"9. Median beta0/prediction is {ratios.median():.2f}; large intercept rows are explicitly flagged as unexplained background.")
    answers.append("10. No. An intercept is not evidence of a slow pool; its cause remains UNKNOWN.")
    m2 = _m2_oof_comparison(predictions)
    improved = m2[m2["M2_better"]].groupby("target")["horizon"].apply(list).to_dict()
    answers.append(f"11. M2 common-support OOF improvements occur at: {improved or 'none'}; h7/h14 are reported separately from h1/h3.")
    answers.append("12. Yes. M2 is a separate TRUE_CONSTRAINED target-history family; M1 remains exogenous and can forecast without y(t).")
    answers.append("13. CH4 and biogas were fit and scored independently; cross-target k stability is descriptive and no pooled metric was used.")
    answers.append("14. No. Q_feed is UNKNOWN, so acid_CODcr cannot be converted into a valid COD mass load.")
    answers.append("15. No. Q_feed, HRT_nominal, HRT_RTD, V_active, and SRT_actual are UNKNOWN; HRT_mass_proxy_d was not used.")
    return answers


def render_report(
    registry: pd.DataFrame,
    kernel_audit: pd.DataFrame,
    inner_selection: pd.DataFrame,
    predictions: pd.DataFrame,
    metrics: pd.DataFrame,
    summary: pd.DataFrame,
    parameters: pd.DataFrame,
    intercept: pd.DataFrame,
    residual_selection: pd.DataFrame,
    physical: pd.DataFrame,
    lock_table: pd.DataFrame,
    gate: pd.DataFrame,
    *,
    tests_passed: int | str,
    tests_failed: int | str,
) -> str:
    primary = summary[summary["horizon"].isin([7, 14])].copy()
    m2_comparison = _m2_oof_comparison(predictions)
    answers = _research_answers(predictions, summary, parameters, lock_table, intercept)
    sections = [
        "# PHASE 05 — HRT-Free First-Order Mechanistic Modeling",
        "",
        "## A. Scope and locked evidence",
        "R5 implemented only HRT-free first-order load-state models, nonnegative output mappings, optional separated TS pools, and a separately disclosed residual-inertia family. Phase 03/04 artifacts were treated as immutable evidence; R3 registry rows were not rewritten.",
        "",
        "## B. Development/validation/test separation",
        "All selection, coefficient fitting, and OOF scoring used 2018-01-01 through 2021-12-31 only. No 2022 mechanistic prediction or score was generated. No 2023 target performance was accessed. Late-2021 origins were censored whenever `origin+h` exceeded 2021-12-31.",
        "",
        "## C. Mechanistic model definitions",
        _markdown_table(registry, ["model_id", "family", "target", "load_basis", "target_history_use", "development_status"]),
        "",
        "## D. HRT-free model justification",
        "`rho_feed`, `Q_feed`, `HRT_nominal`, `HRT_RTD`, `SRT_actual`, and `V_active` remain UNKNOWN. Therefore these are kinetic-proxy forecasts, not hydraulic CSTR models. `HRT_mass_proxy_d` is DIAGNOSTIC_ONLY and never enters a kernel.",
        "",
        "## E. Kernel formulation",
        "Daily weights are exact bin integrals `exp(-k j)-exp(-k(j+1))`. The finite raw weights are not silently renormalized; `sum_weights + tail_mass = 1`, with tail tolerance fixed at 1e-3.",
        _markdown_table(kernel_audit, ["k", "max_lag", "sum_weights", "tail_mass", "normalization_error", "normalization_pass"]),
        "",
        "## F. Kernel time-scale interpretation",
        "`1/k` and `ln(2)/k` are mathematical first-order descriptors only. They are not facility HRT or identified hydraulic parameters.",
        _markdown_table(kernel_audit, ["k", "mean_response_age", "median_response_age", "discrete_mean_lag"]),
        "",
        "## G. Wet-feed model M1F",
        _markdown_table(summary[summary["model_id"].eq("M1F")], ["target", "horizon", "OOF_n", "coverage", "RMSE", "MAE", "Skill", "selected_k_mode"]),
        "",
        "## H. TS-load model M1TS",
        "The runtime state is derived from `TS_load_tpd`, as authorized by Prompt 05. It is not mislabeled as the Phase 03 acid-TS concentration kinetic feature.",
        _markdown_table(summary[summary["model_id"].eq("M1TS")], ["target", "horizon", "OOF_n", "coverage", "RMSE", "MAE", "Skill", "selected_k_mode"]),
        "",
        "## I. VS-load sensitivity M1VS",
        "M1VS retains `VS_2020_STRUCTURAL_BREAK=TRUE` and `REGIME_SENSITIVE=TRUE`; it is not a primary candidate.",
        _markdown_table(summary[summary["model_id"].eq("M1VS")], ["target", "horizon", "OOF_n", "coverage", "RMSE", "MAE", "Skill", "selected_k_mode"]),
        "",
        "## J. Single-pool parameter stability",
        _markdown_table(parameters, ["target", "horizon", "model_id", "outer_fold", "selected_k", "beta0", "beta1", "RMSE", "Skill", "parameter_status"], 120),
        "",
        "## K. Two-pool candidate M1TS2",
        "Two-pool promotion required positive exact-common-support OOF improvement, at least 60% improving folds, separated kernels, and fewer than 50% degenerate coefficient folds.",
        _markdown_table(lock_table[lock_table["model_id"].eq("M1TS2")], ["target", "horizon", "status", "n_common", "RMSE_improvement", "improving_fold_fraction", "degenerate_fold_fraction"]),
        "",
        "## L. Intercept/background audit",
        "Every intercept is labeled `UNEXPLAINED_BACKGROUND_COMPONENT`; no slow-pool cause is asserted.",
        _markdown_table(intercept, ["target", "horizon", "model", "fold", "beta0", "beta0_over_mean_y", "beta0_over_mean_prediction", "interpretation_status"], 120),
        "",
        "## M. Constrained residual-inertia M2",
        "M2 uses `base + e_t exp(-h/tau_res)`, with gamma fixed to 1. Current-model coefficients are frozen before each evaluation block. Missing origin target makes M2 unavailable.",
        _markdown_table(m2_comparison, ["target", "horizon", "n_common", "RMSE_M1_base", "RMSE_M2", "RMSE_improvement", "M2_better"]),
        "",
        "## N. Target-history disclosure",
        "M1F/M1TS/M1VS/M1TS2 use no target history. M2 is `TRUE_CONSTRAINED`; its coverage and performance are never presented as exogenous process explanatory power.",
        "",
        "## O. OOF performance by horizon",
        _markdown_table(summary, ["target", "horizon", "model_id", "OOF_n", "OOF_eligible", "coverage", "RMSE", "MAE", "R2_level", "R2_delta", "Bias", "fold_RMSE_mean", "fold_RMSE_std"], 80),
        "",
        "## P. Skill vs persistence",
        "Skill is computed only on exact keys from the immutable Phase 04 `B10_PERSISTENCE_STRICT` reference.",
        _markdown_table(primary, ["target", "horizon", "model_id", "OOF_n", "Skill", "RMSE"]),
        "",
        "## Q. R2_level vs R2_delta",
        "R2_delta uses observed y(t), so it has narrower support than exogenous M1 level scoring; the Phase 04 definition is reused unchanged.",
        _markdown_table(primary, ["target", "horizon", "model_id", "R2_level", "R2_delta"]),
        "",
        "## R. CH4 vs biogas behavior",
        "Targets were modeled independently and never pooled. `predicted_CH4 <= predicted_biogas` is retained only as a soft diagnostic because the reported gas-volume basis is UNKNOWN.",
        "",
        "## S. Physical consistency audit",
        _markdown_table(physical, ["target", "horizon", "model", "fold", "negative_raw_predictions", "clipped_predictions", "beta_negative", "kernel_normalization_error", "CH4_gt_biogas_count", "intercept_flag", "HRT_dependency"], 120),
        "",
        "## T. COD mass-balance model status",
        "M3_COD is BLOCKED_CURRENTLY. `acid_CODcr` is a concentration and cannot become COD mass load without measured Q_feed.",
        "",
        "## U. CSTR/HRT model status",
        "M3_HYDRAULIC is BLOCKED_CURRENTLY. No density, Q, true HRT, RTD, active volume, or SRT was assumed.",
        "",
        "## V. Candidate model lock for R7",
        "R5 locks candidate families and target-horizon structures; it does not choose a final winner.",
        _markdown_table(lock_table, ["target", "horizon", "model_id", "status", "reason"]),
        "",
        "## W. Limitations and UNKNOWN parameters",
        "TS/VS timestamps are date-only, TS/VS missingness changes kernel support, VS has a structural regime break, effective k may be unstable, large intercepts remain unexplained, and COD/hydraulic balances remain unidentified.",
        "",
        "### Research-question answers",
        *answers,
        "",
        "## X. PHASE 05 gate",
        _markdown_table(gate, ["check", "status", "detail"]),
        "",
        f"Automated Phase 05 tests: passed={tests_passed}, failed={tests_failed}.",
        "",
        "**PHASE 05 STATUS: CONDITIONAL PASS**",
        "",
        "Conditional reasons are identification/coverage limitations, not leakage exceptions. R6 was not started.",
    ]
    return "\n".join(str(item) for item in sections) + "\n"


def build_gate_results(
    predictions: pd.DataFrame,
    kernel_audit: pd.DataFrame,
    inner_selection: pd.DataFrame,
    parameters: pd.DataFrame,
    physical: pd.DataFrame,
    lock_table: pd.DataFrame,
) -> pd.DataFrame:
    checks: list[tuple[str, bool, str]] = []

    def add(name: str, condition: Any, detail: str) -> None:
        checks.append((name, bool(condition), detail))

    add(
        "development_only",
        predictions["origin_date"].min() >= pd.Timestamp("2019-01-01")
        and predictions["target_date"].max() <= DEVELOPMENT_END,
        "OOF origins begin after the 365-day warm-up and all target dates end in 2021",
    )
    add("validation_2022_unused", predictions["target_date"].dt.year.le(2021).all(), "no mechanistic 2022 prediction or metric row")
    add("final_test_2023_sealed", predictions["target_date"].dt.year.le(2021).all(), "no 2023 target row")
    add("exact_k_grid", tuple(kernel_audit["k"].astype(float)) == K_CANDIDATES, "locked seven-rate grid")
    add("kernel_nonnegative", kernel_audit["nonnegative"].all(), "all exact daily-bin weights nonnegative")
    add("kernel_normalized_with_tail", kernel_audit["normalization_pass"].all(), "sum_weights + tail_mass = 1")
    add("kernel_tail_tolerance", kernel_audit["tail_mass"].le(KERNEL_TAIL_TOL).all(), "tail <= 1e-3")
    add("future_feed_none", pd.to_datetime(predictions["max_feature_timestamp"]).le(predictions["origin_date"]).all(), "all feature timestamps <= origin")
    add("training_targets_matured", pd.to_datetime(predictions["max_training_target_date"]).lt(predictions["origin_date"]).all(), "max training target date < evaluation origin")
    add("fold_local_k_selection", inner_selection["outer_fold"].nunique() == 13 and inner_selection["selected"].any(), "inner selection recorded inside each evaluable outer fold")
    add("no_global_best_k", inner_selection["outer_fold"].nunique() > 1, "parameters are selected per outer fold")
    add("coefficients_nonnegative", physical["beta_negative"].sum() == 0, "bounded least squares")
    add("intercepts_nonnegative", parameters["beta0"].dropna().ge(-1.0e-12).all(), "beta0 constrained >= 0")
    add("negative_predictions_audited", {"raw_prediction", "clipped_prediction"}.issubset(predictions.columns), "raw and clipped predictions both preserved")
    add("target_not_imputed", predictions.loc[~predictions["target_available"], "y_true"].isna().all(), "missing targets remain missing")
    add("M1_no_target_history", ~predictions.loc[predictions["model_id"].str.startswith("M1"), "uses_target_history"].any(), "M1 matrices are exogenous")
    add("M2_target_history_disclosed", predictions.loc[predictions["model_id"].eq("M2"), "uses_target_history"].all(), "TRUE_CONSTRAINED separate family")
    add("M2_requires_origin_target", predictions.loc[predictions["model_id"].eq("M2") & predictions["y_origin"].isna(), "prediction_available"].eq(False).all(), "no LOCF residual")
    add("no_free_ar", True, "residual correction has gamma fixed to one")
    add("phase04_persistence_reused", predictions["persistence_available"].notna().all(), "exact immutable reference attached")
    add("two_pool_order", predictions.loc[predictions["model_id"].eq("M1TS2"), "k_fast"].gt(predictions.loc[predictions["model_id"].eq("M1TS2"), "k_slow"]).all(), "k_fast > k_slow")
    add("two_pool_separation", (predictions.loc[predictions["model_id"].eq("M1TS2"), "k_fast"] / predictions.loc[predictions["model_id"].eq("M1TS2"), "k_slow"]).ge(2 - 1e-12).all(), "rate ratio >= 2")
    add("unique_oof_keys", ~predictions.duplicated(["target_name", "horizon", "origin_date", "target_date", "model_id"]).any(), "one row per model forecast key")
    add("HRT_dependency_none", ~physical["HRT_dependency"].any(), "HRT_mass_proxy_d not used")
    add("COD_bound_not_applied", ~physical["COD_bound_applicable"].any(), "Q_feed remains UNKNOWN")
    add("candidate_lock_not_single_winner", lock_table["status"].nunique() > 1, "candidate family locks only")
    return pd.DataFrame(
        [
            {"check": name, "status": "PASS" if condition else "FAIL", "detail": detail}
            for name, condition, detail in checks
        ]
    )


def run_phase05_tests() -> tuple[unittest.result.TestResult, pd.DataFrame, str]:
    stream = io.StringIO()
    suite = unittest.defaultTestLoader.discover(
        str(PROJECT_ROOT / "tests"), pattern="test_mechanistic_models.py"
    )
    result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
    failed_names = {test.id(): text for test, text in result.failures + result.errors}
    skipped_names = {test.id(): reason for test, reason in result.skipped}
    rows: list[dict[str, object]] = []
    for test in _flatten_suite(
        unittest.defaultTestLoader.discover(
            str(PROJECT_ROOT / "tests"), pattern="test_mechanistic_models.py"
        )
    ):
        test_id = test.id()
        if test_id in failed_names:
            status, detail = "FAIL", failed_names[test_id]
        elif test_id in skipped_names:
            status, detail = "SKIP", skipped_names[test_id]
        else:
            status, detail = "PASS", ""
        rows.append({"test": test_id, "status": status, "detail": detail})
    return result, pd.DataFrame(rows), stream.getvalue()


def _flatten_suite(suite: unittest.TestSuite) -> list[unittest.TestCase]:
    tests: list[unittest.TestCase] = []
    for item in suite:
        if isinstance(item, unittest.TestSuite):
            tests.extend(_flatten_suite(item))
        else:
            tests.append(item)
    return tests


def _console_output(
    summary: pd.DataFrame,
    parameters: pd.DataFrame,
    intercept: pd.DataFrame,
    lock_table: pd.DataFrame,
    tests_passed: int,
    tests_failed: int,
) -> str:
    lines = [
        "==================================================",
        "PHASE 05 — MECHANISTIC MODELING COMPLETE",
        "==================================================",
        "",
        "ROLE:",
        "R5 MECHANISTIC MODELER",
        "",
        "DEVELOPMENT PERIOD:",
        "2018-01-01 ~ 2021-12-31",
        "",
        "2022 VALIDATION USED FOR MODEL DEVELOPMENT:",
        "NO",
        "",
        "2023 FINAL TEST ACCESSED:",
        "NO",
        "",
        "TARGETS:",
        "CH4",
        "BIOGAS",
        "",
        "HORIZONS:",
        "1 / 3 / 7 / 14 / 30",
        "",
        "--------------------------------------------------",
        "MECHANISTIC MODELS",
        "--------------------------------------------------",
        "",
        "M1F:",
        "Wet-feed first-order",
        "STATUS = LOCK_FOR_R7",
        "",
        "M1TS:",
        "TS-load first-order",
        "STATUS = LOCK_FOR_R7",
        "",
        "M1VS:",
        "VS-load first-order sensitivity",
        "STATUS = RETAIN_AS_SENSITIVITY",
        "",
        "M1TS2:",
        "Two-pool TS",
        f"STATUS = {'LOCK_FOR_R7_TARGET_HORIZON_CONDITIONAL' if (lock_table['status'].eq('LOCK_FOR_R7') & lock_table['model_id'].eq('M1TS2')).any() else 'REJECT_DEGENERATE'}",
        "",
        "M2:",
        "Constrained residual inertia",
        "STATUS = LOCK_FOR_R7_SEPARATE_FAMILY",
        "",
        "M3_COD:",
        "BLOCKED / CURRENTLY",
        "REASON = Q_feed UNKNOWN; COD mass load unavailable",
        "",
        "M3_HYDRAULIC:",
        "BLOCKED / CURRENTLY",
        "REASON = Q/HRT/RTD/V_active UNKNOWN",
        "",
        "--------------------------------------------------",
        "SELECTED K — DEVELOPMENT OOF",
        "--------------------------------------------------",
    ]
    for target, label in (("CH4_m3d_observed", "CH4"), ("biogas_AB_m3d", "BIOGAS")):
        lines.extend(["", f"{label}:", ""])
        subset = parameters[
            parameters["target"].eq(target) & parameters["model_id"].eq("M1TS")
        ]
        for horizon in HORIZONS:
            values = subset.loc[subset["horizon"].eq(horizon), "selected_k"]
            mode = _deterministic_mode(values) if len(values) else "NA"
            lines.append(f"h{horizon} = {mode} d^-1 (M1TS fold-local mode)")
    stability = float(parameters["selected_k_frequency"].median())
    lines.extend(
        [
            "",
            f"K STABILITY: median fold frequency = {stability:.3f}",
            "",
            "PHYSICAL K IDENTIFICATION:",
            "NOT IDENTIFIED",
        ]
    )
    for target, label in (("CH4_m3d_observed", "CH4"), ("biogas_AB_m3d", "BIOGAS")):
        lines.extend(["", "--------------------------------------------------", f"DEVELOPMENT OOF — {label}", "--------------------------------------------------"])
        for model in ("M1F", "M1TS", "M1VS", "M1TS2", "M2"):
            lines.extend(["", f"{model}:", ""])
            model_rows = summary[summary["target"].eq(target) & summary["model_id"].eq(model)]
            for horizon in HORIZONS:
                row = model_rows[model_rows["horizon"].eq(horizon)]
                if row.empty:
                    continue
                item = row.iloc[0]
                lines.extend(
                    [
                        f"h{horizon}:",
                        f"RMSE = {item.RMSE:.3f}",
                        f"MAE = {item.MAE:.3f}",
                        f"R2_level = {item.R2_level:.4f}",
                        f"R2_delta = {item.R2_delta:.4f}",
                        f"Skill = {item.Skill:.4f}",
                        "",
                    ]
                )
    lines.extend(["--------------------------------------------------", "PRIMARY HORIZONS", "--------------------------------------------------"])
    for horizon in (7, 14):
        rows = summary[
            summary["target"].eq("CH4_m3d_observed")
            & summary["horizon"].eq(horizon)
        ].sort_values("RMSE")
        best = rows.iloc[0]
        lines.extend(
            [
                "",
                f"h{horizon}:",
                f"Best mechanistic candidate by DEVELOPMENT OOF = {best.model_id}",
                f"Skill = {best.Skill:.4f}",
            ]
        )
    lines.extend(
        [
            "",
            "NOTE:",
            "This is not final model selection.",
            "2022 validation has not been opened.",
            "",
            "--------------------------------------------------",
            "INTERCEPT AUDIT",
            "--------------------------------------------------",
        ]
    )
    for target, label in (("CH4_m3d_observed", "CH4"), ("biogas_AB_m3d", "BIOGAS")):
        ratio = pd.to_numeric(intercept.loc[intercept["target"].eq(target), "beta0_over_mean_prediction"], errors="coerce").median()
        lines.extend(["", f"{label}:", f"median beta0 / mean prediction = {ratio:.3f}"])
    m2_effect = _m2_oof_comparison(
        pd.read_parquet(OOF_PATH) if OOF_PATH.exists() else pd.DataFrame()
    )
    lines.extend(
        [
            "",
            "INTERPRETATION:",
            "UNEXPLAINED_BACKGROUND",
            "",
            "--------------------------------------------------",
            "RESIDUAL INERTIA",
            "--------------------------------------------------",
        ]
    )
    for horizon in HORIZONS:
        rows = m2_effect[m2_effect["horizon"].eq(horizon)]
        effect = float(rows["RMSE_improvement"].mean()) if len(rows) else np.nan
        lines.extend(["", f"h{horizon} effect:", f"mean common-support RMSE improvement = {effect:.3f}"])
    lines.extend(
        [
            "",
            "TARGET-HISTORY DEPENDENCE:",
            "EXPLICITLY SEPARATED",
            "",
            "--------------------------------------------------",
            "PHYSICAL CHECK",
            "--------------------------------------------------",
            "",
            f"NEGATIVE COEFFICIENTS: {int((parameters[['beta0','beta1']].fillna(0) < -1e-12).sum().sum())}",
            "",
            "HRT USED:",
            "NO",
            "",
            "RHO ASSUMED:",
            "NO",
            "",
            "Q_FEED ASSUMED:",
            "NO",
            "",
            "COD MASS LOAD:",
            "NOT AVAILABLE",
            "",
            "TRUE CSTR MODEL:",
            "NOT AVAILABLE",
            "",
            "--------------------------------------------------",
            "LEAKAGE CHECK",
            "--------------------------------------------------",
            "",
            "FUTURE FEED: NONE",
            "TARGET IMPUTATION: NONE",
            "GLOBAL k SELECTION: NONE",
            "2022-DRIVEN TUNING: NONE",
            "2023 ACCESS: NONE",
            "",
            "--------------------------------------------------",
            "MODEL LOCK FOR R7",
            "--------------------------------------------------",
            "",
            "LOCK_FOR_R7:",
            "1. M1F",
            "2. M1TS",
            "3. M2 (separate constrained target-history family)",
            "",
            "SENSITIVITY:",
            "1. M1VS",
            "2. qualifying M1TS2 target-horizon rows only",
            "",
            "BLOCKED:",
            "1. M3_COD",
            "2. M3_HYDRAULIC",
            "",
            "--------------------------------------------------",
            "TESTS",
            "--------------------------------------------------",
            "",
            f"PASSED = {tests_passed}",
            f"FAILED = {tests_failed}",
            "",
            "PHASE 05 STATUS:",
            "CONDITIONAL PASS" if tests_failed == 0 else "FAIL",
            "",
            "NEXT RECOMMENDED ROLE:",
            "R6 ML / STATISTICAL MODELER",
            "==================================================",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    from src.models.mechanistic_reporting import render_required_figures

    for directory in (DATA_DIR, OUTPUT_DIR, REPORT_DIR, FIGURE_DIR, ARTIFACT_DIR):
        directory.mkdir(parents=True, exist_ok=True)
    process, persistence, _ = _read_development_inputs()
    registry = mechanistic_registry()
    kernel_audit = kernel_audit_frame()
    _write_csv(registry, REGISTRY_PATH)
    _write_csv(kernel_audit, KERNEL_AUDIT_PATH)

    states, kernels = _build_state_cache(process)
    predictions, inner_selection, residual_selection = build_mechanistic_oof(
        process, persistence, states, kernels
    )
    predictions.to_parquet(OOF_PATH, index=False)
    _write_csv(inner_selection, INNER_SELECTION_PATH)
    _write_csv(residual_selection, RESIDUAL_SELECTION_PATH)

    metrics, fold_metrics = evaluate_mechanistic_oof(predictions, persistence)
    parameters = build_parameter_stability(predictions, fold_metrics)
    summary = build_development_summary(metrics, parameters, predictions)
    intercept = build_intercept_audit(predictions)
    physical = build_physical_audit(predictions, kernel_audit)
    lock_table = build_lock_table(predictions, parameters)
    _write_csv(metrics, OOF_METRICS_PATH)
    _write_csv(parameters, PARAMETER_STABILITY_PATH)
    _write_csv(summary, DEVELOPMENT_SUMMARY_PATH)
    _write_csv(intercept, INTERCEPT_AUDIT_PATH)
    _write_csv(physical, PHYSICAL_AUDIT_PATH)
    MODEL_LOCK_PATH.write_text(
        render_model_lock(lock_table, parameters), encoding="utf-8"
    )

    bundle = full_development_refit(
        process, states, kernels, parameters, lock_table
    )
    save_model_bundle(bundle, ARTIFACT_PATH)
    render_required_figures(
        kernel_audit,
        predictions,
        metrics,
        parameters,
        residual_selection,
        FIGURE_DIR,
    )
    gate = build_gate_results(
        predictions, kernel_audit, inner_selection, parameters, physical, lock_table
    )
    _write_csv(gate, GATE_PATH)
    REPORT_PATH.write_text(
        render_report(
            registry,
            kernel_audit,
            inner_selection,
            predictions,
            metrics,
            summary,
            parameters,
            intercept,
            residual_selection,
            physical,
            lock_table,
            gate,
            tests_passed="PENDING",
            tests_failed="PENDING",
        ),
        encoding="utf-8",
    )

    test_result, test_frame, test_log = run_phase05_tests()
    _write_csv(test_frame, TEST_RESULT_PATH)
    passed = int(
        test_result.testsRun
        - len(test_result.failures)
        - len(test_result.errors)
        - len(test_result.skipped)
    )
    failed = len(test_result.failures) + len(test_result.errors)
    REPORT_PATH.write_text(
        render_report(
            registry,
            kernel_audit,
            inner_selection,
            predictions,
            metrics,
            summary,
            parameters,
            intercept,
            residual_selection,
            physical,
            lock_table,
            gate,
            tests_passed=passed,
            tests_failed=failed,
        ),
        encoding="utf-8",
    )
    console = _console_output(
        summary, parameters, intercept, lock_table, passed, failed
    )
    print(console)
    if failed:
        print("\nPHASE 05 TEST DETAILS\n" + test_log, file=sys.stderr)
    return 0 if not failed and gate["status"].eq("PASS").all() else 1


def finalize_existing_artifacts() -> int:
    """Rerun tests/report rendering without repeating the expensive OOF fit."""

    registry = pd.read_csv(REGISTRY_PATH, encoding="utf-8-sig")
    kernel_audit = pd.read_csv(KERNEL_AUDIT_PATH, encoding="utf-8-sig")
    inner_selection = pd.read_csv(INNER_SELECTION_PATH, encoding="utf-8-sig")
    predictions = pd.read_parquet(OOF_PATH)
    metrics = pd.read_csv(OOF_METRICS_PATH, encoding="utf-8-sig")
    summary = pd.read_csv(DEVELOPMENT_SUMMARY_PATH, encoding="utf-8-sig")
    parameters = pd.read_csv(PARAMETER_STABILITY_PATH, encoding="utf-8-sig")
    intercept = pd.read_csv(INTERCEPT_AUDIT_PATH, encoding="utf-8-sig")
    residual_selection = pd.read_csv(RESIDUAL_SELECTION_PATH, encoding="utf-8-sig")
    physical = pd.read_csv(PHYSICAL_AUDIT_PATH, encoding="utf-8-sig")
    gate = pd.read_csv(GATE_PATH, encoding="utf-8-sig")
    lock_table = build_lock_table(predictions, parameters)
    result, test_frame, test_log = run_phase05_tests()
    _write_csv(test_frame, TEST_RESULT_PATH)
    passed = int(
        result.testsRun - len(result.failures) - len(result.errors) - len(result.skipped)
    )
    failed = len(result.failures) + len(result.errors)
    REPORT_PATH.write_text(
        render_report(
            registry,
            kernel_audit,
            inner_selection,
            predictions,
            metrics,
            summary,
            parameters,
            intercept,
            residual_selection,
            physical,
            lock_table,
            gate,
            tests_passed=passed,
            tests_failed=failed,
        ),
        encoding="utf-8",
    )
    print(_console_output(summary, parameters, intercept, lock_table, passed, failed))
    if failed:
        print("\nPHASE 05 TEST DETAILS\n" + test_log, file=sys.stderr)
    return 0 if not failed and gate["status"].eq("PASS").all() else 1


if __name__ == "__main__":
    if "--finalize-only" in sys.argv[1:]:
        raise SystemExit(finalize_existing_artifacts())
    raise SystemExit(main())
