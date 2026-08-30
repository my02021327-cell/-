"""Run PHASE 11 final integration and the one-time 2023 sealed test.

This is the first execution in the repository that opens the 2023 sealed
holdout. Everything upstream of this module (features, hyperparameters, the
A30 formula, expert pool, availability rules, PI methodology) is read-only
here: no new hyperparameter search, no new feature selection, no new model
comparison is performed. The only new work is (a) a single fixed-spec refit
of the five locked A30 experts (B10, B11, M2, M1F, RIDGE) over
2018-01-01..2022-12-31, and (b) a real prequential, one-time walk-forward of
the *unmodified* ``src.adaptive.phase08.simulate_adaptive_strategies`` and
``src.ensemble.phase09.add_past_residual_intervals`` functions over
2023-01-01..2023-09-17.

No existing Phase 01-10 file is written by this module.
"""

from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
import re
import unittest
from typing import Any

import numpy as np
import pandas as pd

from src.adaptive.phase08 import (
    EXPERTS,
    PAIR_KEY,
    build_adaptive_history,
    simulate_adaptive_strategies,
)
from src.ensemble.phase09 import add_past_residual_intervals
from src.models.baselines.last_observed import predict_last_observed
from src.models.baselines.persistence import predict_strict_persistence
from src.models.mechanistic.kernels import build_first_order_kernel
from src.models.mechanistic.kinetic_state import MIN_KERNEL_COVERAGE, build_kinetic_state
from src.models.mechanistic.single_pool import fit_single_pool
from src.models.ml_statistical.contracts import build_feature_contract, features_for
from src.models.ml_statistical.estimators import fit_estimator
from src.models.ml_statistical.preprocessing import FoldPreprocessor
from src.models.ml_statistical.workflow import selected_parameter_mode
from src.validation.baseline_metrics import compute_metrics, skill_from_mse


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data/processed"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
REPORT_DIR = PROJECT_ROOT / "reports"

FINAL_REFIT_START = pd.Timestamp("2018-01-01")
FINAL_REFIT_END = pd.Timestamp("2022-12-31")
FINAL_TEST_START = pd.Timestamp("2023-01-01")
FINAL_TEST_END = pd.Timestamp("2023-09-17")
TARGETS = ("CH4_m3d_observed", "biogas_AB_m3d")
HORIZONS = (1, 3, 7, 14, 30)
PRIMARY_HORIZONS = (7, 14)
DQ_FLAGS = (
    "flag_TA_sensor_fault", "flag_TB_sensor_fault", "flag_ALKB_corrupted",
    "flag_feed_frozen", "flag_VS_regime_change", "flag_structural_missing",
    "flag_target_CH4_invalid", "flag_range_error",
)

A30 = "A30_INVERSE_ERROR_RELIABILITY_WEIGHT"


# ---------------------------------------------------------------------------
# Section 0: file hashing (pre/post integrity check)
# ---------------------------------------------------------------------------

def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


LOCK_INTEGRITY_FILES = [
    "src/adaptive/phase08.py",
    "outputs/08_model_lock.yaml",
    "src/ensemble/phase09.py",
    "outputs/09_model_lock.yaml",
    "outputs/09_r10_handoff.yaml",
    "src/scientific/phase10.py",
    "outputs/10_final_integration_lock.yaml",
    "outputs/10_final_test_readiness.yaml",
    "data/processed/08_adaptive_predictions.parquet",
]


def hash_lock_files() -> dict[str, str]:
    return {name: sha256_file(PROJECT_ROOT / name) for name in LOCK_INTEGRITY_FILES}


# ---------------------------------------------------------------------------
# Section 1: locked hyperparameter lookups (read-only; no new search)
# ---------------------------------------------------------------------------

def _parse_model_lock_structure(text: str, key: str) -> str:
    marker = f"  {key}:"
    start = text.find(marker)
    if start < 0:
        raise KeyError(f"{key} not found in 05_model_lock.yaml")
    block = text[start:start + 400]
    field = "locked_structure:"
    at = block.find(field)
    if at < 0:
        raise KeyError(f"locked_structure missing for {key}")
    line = block[at + len(field):].splitlines()[0].strip()
    return line.strip('"')


def m1f_locked_k(lock_text: str, target: str, horizon: int) -> float:
    return float(_parse_model_lock_structure(lock_text, f"{target}_h{horizon}_M1F"))


def m2_locked_params(lock_text: str, target: str, horizon: int) -> tuple[float, int]:
    structure = _parse_model_lock_structure(lock_text, f"{target}_h{horizon}_M2")
    match = re.match(r"^([0-9.]+);base=M1F;tau_res=(\d+)d$", structure)
    if not match:
        raise ValueError(f"unexpected M2 locked_structure format: {structure!r}")
    return float(match.group(1)), int(match.group(2))


def ridge_locked_alpha(hyperparameters: pd.DataFrame, target: str, horizon: int) -> float:
    _, parameters = selected_parameter_mode(
        hyperparameters,
        target=target,
        horizon=horizon,
        model_id="S00_RIDGE",
        feature_set="SET-F",
        availability_track="TRACK-S_STRICT",
    )
    return float(parameters["alpha"])


# ---------------------------------------------------------------------------
# Section 2: B10 / B11 (no fitting; deterministic target-history baselines)
# ---------------------------------------------------------------------------

def build_b10_b11_2023(process: pd.DataFrame) -> dict[str, pd.DataFrame]:
    origins = pd.date_range(FINAL_TEST_START, FINAL_TEST_END, freq="D")
    out: dict[str, pd.DataFrame] = {}
    for target_name, column in zip(TARGETS, ("CH4_m3d_observed", "biogas_AB_m3d")):
        series = pd.Series(
            pd.to_numeric(process[column], errors="coerce").to_numpy(dtype=float),
            index=pd.DatetimeIndex(process["date"]),
            name=target_name,
        )
        for horizon in HORIZONS:
            b10 = predict_strict_persistence(series, horizon, origins)
            b10["target_name"] = target_name
            out[f"B10__{target_name}__h{horizon}"] = b10
            b11 = predict_last_observed(series, horizon, origins)
            b11["target_name"] = target_name
            out[f"B11__{target_name}__h{horizon}"] = b11
    return out


# ---------------------------------------------------------------------------
# Section 3: M1F / M2 fixed-spec 2018-2022 refit, predict-only for 2023
# ---------------------------------------------------------------------------

def build_m1f_m2_2023(
    process: pd.DataFrame,
    lock_text: str,
) -> tuple[dict[str, pd.DataFrame], dict[str, dict[str, float]]]:
    feed = pd.Series(
        pd.to_numeric(process["feed_AB_tpd"], errors="coerce").to_numpy(dtype=float),
        index=pd.DatetimeIndex(process["date"]),
        name="feed_AB_tpd",
    )
    target_series = {
        target_name: pd.Series(
            pd.to_numeric(process[column], errors="coerce").to_numpy(dtype=float),
            index=pd.DatetimeIndex(process["date"]),
            name=target_name,
        )
        for target_name, column in zip(TARGETS, ("CH4_m3d_observed", "biogas_AB_m3d"))
    }
    all_dates = pd.DatetimeIndex(process["date"])
    test_origins = all_dates[(all_dates >= FINAL_TEST_START) & (all_dates <= FINAL_TEST_END)]

    out: dict[str, pd.DataFrame] = {}
    lock_summary: dict[str, dict[str, float]] = {}
    kernel_cache: dict[float, Any] = {}
    state_cache: dict[float, pd.Series] = {}

    for target_name in TARGETS:
        series = target_series[target_name]
        for horizon in HORIZONS:
            k = m1f_locked_k(lock_text, target_name, horizon)
            m2_k, tau_res = m2_locked_params(lock_text, target_name, horizon)
            if m2_k != k:
                raise AssertionError(
                    f"M2 base k ({m2_k}) disagrees with locked M1F k ({k}) for "
                    f"{target_name} h{horizon}"
                )
            if k not in kernel_cache:
                kernel_cache[k] = build_first_order_kernel(k)
                state_cache[k] = build_kinetic_state(
                    feed, kernel_cache[k], min_kernel_coverage=MIN_KERNEL_COVERAGE
                ).set_index("date")["kinetic_state"]
            kernel = kernel_cache[k]
            state = state_cache[k]

            origin_all = all_dates[all_dates <= FINAL_REFIT_END]
            target_date_all = origin_all + pd.Timedelta(days=horizon)
            y_true_all = series.reindex(target_date_all).to_numpy(dtype=float)
            y_origin_all = series.reindex(origin_all).to_numpy(dtype=float)
            state_all = state.reindex(origin_all).to_numpy(dtype=float)

            # M1F direct base model: state(t) -> y_true(t+h), matured rows only.
            base_mask = np.isfinite(state_all) & np.isfinite(y_true_all)
            model_base = fit_single_pool(
                state_all[base_mask], y_true_all[base_mask],
                k_per_d=k, kernel_max_lag=kernel.max_lag,
                load_basis="FEED_WET_MASS", model_id="M1F",
            )
            # M2 "current" model: state(t) -> y_origin(t), matured rows only.
            current_mask = np.isfinite(state_all) & np.isfinite(y_origin_all)
            model_current = fit_single_pool(
                state_all[current_mask], y_origin_all[current_mask],
                k_per_d=k, kernel_max_lag=kernel.max_lag,
                load_basis="FEED_WET_MASS", model_id="M1F_CURRENT",
            )
            lock_summary[f"{target_name}__h{horizon}"] = {
                "k": k, "tau_res": tau_res,
                "beta0": model_base.beta0, "beta1": model_base.beta1,
                "beta0_current": model_current.beta0, "beta1_current": model_current.beta1,
                "n_train_base": int(base_mask.sum()),
                "n_train_current": int(current_mask.sum()),
            }

            state_test = state.reindex(test_origins).to_numpy(dtype=float)
            physical = model_base.predict(state_test)
            m1f = pd.DataFrame({
                "origin_date": test_origins, "target_date": test_origins + pd.Timedelta(days=horizon),
                "target_name": target_name, "horizon": horizon,
                "y_origin": series.reindex(test_origins).to_numpy(dtype=float),
                "y_true": series.reindex(test_origins + pd.Timedelta(days=horizon)).to_numpy(dtype=float),
                "raw_prediction": physical.raw, "y_pred": physical.clipped,
                "prediction_available": physical.available,
            })
            out[f"M1F__{target_name}__h{horizon}"] = m1f

            y_origin_test = series.reindex(test_origins).to_numpy(dtype=float)
            base_raw = model_base.beta0 + model_base.beta1 * state_test
            current_raw = model_current.beta0 + model_current.beta1 * state_test
            raw_m2 = base_raw + (y_origin_test - current_raw) * np.exp(-horizon / tau_res)
            valid_m2 = np.isfinite(y_origin_test) & np.isfinite(state_test)
            raw_m2 = np.where(valid_m2, raw_m2, np.nan)
            clipped_m2 = np.where(np.isfinite(raw_m2), np.maximum(raw_m2, 0.0), np.nan)
            m2 = pd.DataFrame({
                "origin_date": test_origins, "target_date": test_origins + pd.Timedelta(days=horizon),
                "target_name": target_name, "horizon": horizon,
                "y_origin": y_origin_test,
                "y_true": series.reindex(test_origins + pd.Timedelta(days=horizon)).to_numpy(dtype=float),
                "raw_prediction": raw_m2, "y_pred": clipped_m2,
                "prediction_available": np.isfinite(clipped_m2),
            })
            out[f"M2__{target_name}__h{horizon}"] = m2
    return out, lock_summary


# ---------------------------------------------------------------------------
# Section 4: RIDGE (S00_RIDGE, SET-A, TRACK-S_STRICT) fixed-spec refit
# ---------------------------------------------------------------------------

def build_ridge_2023(
    features: pd.DataFrame,
    process: pd.DataFrame,
    registry: pd.DataFrame,
    hyperparameters: pd.DataFrame,
) -> tuple[dict[str, pd.DataFrame], dict[str, dict[str, Any]]]:
    contract = build_feature_contract(registry, list(features.columns))
    target_series = {
        target_name: pd.Series(
            pd.to_numeric(process[column], errors="coerce").to_numpy(dtype=float),
            index=pd.DatetimeIndex(process["date"]),
            name=target_name,
        )
        for target_name, column in zip(TARGETS, ("CH4_m3d_observed", "biogas_AB_m3d"))
    }
    base = features.rename(columns={"date": "origin_date"}).copy()
    base["origin_date"] = pd.to_datetime(base["origin_date"]).dt.normalize()

    out: dict[str, pd.DataFrame] = {}
    lock_summary: dict[str, dict[str, Any]] = {}
    for target_name in TARGETS:
        series = target_series[target_name]
        for horizon in HORIZONS:
            feature_names = features_for(
                contract, horizon=horizon, availability_track="TRACK-S_STRICT", feature_set="SET-A"
            )
            alpha = ridge_locked_alpha(hyperparameters, target_name, horizon)

            frame = base.copy()
            frame["target_date"] = frame["origin_date"] + pd.Timedelta(days=horizon)
            frame["y_true"] = series.reindex(frame["target_date"]).to_numpy(dtype=float)

            training = frame.loc[
                frame["target_date"].le(FINAL_REFIT_END) & np.isfinite(frame["y_true"])
            ].copy()
            preprocessor = FoldPreprocessor(model_id="S00_RIDGE").fit(
                training, feature_names, fit_end=training["origin_date"].max()
            )
            estimator = fit_estimator(
                "S00_RIDGE", {"alpha": alpha},
                preprocessor.transform(training),
                training["y_true"].to_numpy(dtype=float),
            )
            lock_summary[f"{target_name}__h{horizon}"] = {
                "alpha": alpha, "feature_set": "SET-A", "availability_track": "TRACK-S_STRICT",
                "n_train": len(training), "n_features": len(feature_names),
                "features": feature_names,
                "training_start": FINAL_REFIT_START.strftime("%Y-%m-%d"),
                "training_end": FINAL_REFIT_END.strftime("%Y-%m-%d"),
            }

            test = frame.loc[
                frame["origin_date"].between(FINAL_TEST_START, FINAL_TEST_END)
                & frame["target_date"].le(FINAL_TEST_END)
            ].copy()
            raw = np.asarray(estimator.predict(preprocessor.transform(test)), dtype=float).reshape(-1)
            clipped = np.where(np.isfinite(raw), np.maximum(raw, 0.0), np.nan)
            ridge = pd.DataFrame({
                "origin_date": test["origin_date"].to_numpy(),
                "target_date": test["target_date"].to_numpy(),
                "target_name": target_name, "horizon": horizon,
                "y_origin": series.reindex(test["origin_date"]).to_numpy(dtype=float),
                "y_true": test["y_true"].to_numpy(dtype=float),
                "raw_prediction": raw, "y_pred": clipped,
                "prediction_available": np.isfinite(clipped),
            })
            out[f"RIDGE__{target_name}__h{horizon}"] = ridge
    return out, lock_summary


# ---------------------------------------------------------------------------
# Section 5: assemble the extended adaptive history (2023 rows only; the
# 2019-2022 warm-up is read verbatim from the locked Phase 08 function).
# ---------------------------------------------------------------------------

def build_final_test_history(
    process: pd.DataFrame,
    b10_b11: dict[str, pd.DataFrame],
    m1f_m2: dict[str, pd.DataFrame],
    ridge: dict[str, pd.DataFrame],
    features: pd.DataFrame,
) -> pd.DataFrame:
    flag_columns = [c for c in features.columns if c.startswith("flag_")]
    context = features[["date", "VS_measurement_regime", *flag_columns]].copy()
    context["date"] = pd.to_datetime(context["date"]).dt.normalize()
    context["dq_context"] = np.where(
        context[flag_columns].fillna(0).astype(bool).any(axis=1), "DQ_FLAGGED", "DQ_CLEAN"
    )
    context = context[["date", "VS_measurement_regime", "dq_context"]].rename(columns={"date": "origin_date"})

    target_series = {
        target_name: pd.Series(
            pd.to_numeric(process[column], errors="coerce").to_numpy(dtype=float),
            index=pd.DatetimeIndex(process["date"]),
            name=target_name,
        )
        for target_name, column in zip(TARGETS, ("CH4_m3d_observed", "biogas_AB_m3d"))
    }

    rows: list[pd.DataFrame] = []
    for target_name in TARGETS:
        series = target_series[target_name]
        for horizon in HORIZONS:
            b10 = b10_b11[f"B10__{target_name}__h{horizon}"]
            b11 = b10_b11[f"B11__{target_name}__h{horizon}"]
            m1f = m1f_m2[f"M1F__{target_name}__h{horizon}"]
            m2 = m1f_m2[f"M2__{target_name}__h{horizon}"]
            rg = ridge[f"RIDGE__{target_name}__h{horizon}"]

            origins = pd.date_range(FINAL_TEST_START, FINAL_TEST_END, freq="D")
            frame = pd.DataFrame({
                "origin_date": origins,
                "target_date": origins + pd.Timedelta(days=horizon),
                "target_name": target_name,
                "horizon": horizon,
            })
            frame["y_origin"] = series.reindex(frame["origin_date"]).to_numpy(dtype=float)
            frame["y_true"] = series.reindex(frame["target_date"]).to_numpy(dtype=float)
            frame = frame.loc[
                frame["origin_date"].between(FINAL_TEST_START, FINAL_TEST_END)
                & frame["target_date"].le(FINAL_TEST_END)
            ].reset_index(drop=True)
            key = ["origin_date", "target_date"]
            frame = frame.merge(context, on="origin_date", how="left", validate="many_to_one")
            frame["current_target_observed"] = np.isfinite(pd.to_numeric(frame["y_origin"], errors="coerce"))
            frame["target_observed"] = np.isfinite(pd.to_numeric(frame["y_true"], errors="coerce"))
            frame["last_observation_age_d"] = b11.set_index(key)["last_observation_age_d"].reindex(
                pd.MultiIndex.from_frame(frame[key])
            ).to_numpy()
            frame["year"] = frame["origin_date"].dt.year
            frame["quarter"] = "Q" + frame["origin_date"].dt.quarter.astype(str)
            frame["source_locked"] = False

            per_expert = {
                "B10": b10.set_index(key)[["y_pred", "prediction_available"]],
                "B11": b11.set_index(key)[["y_pred", "prediction_available"]],
                "M2": m2.set_index(key)[["y_pred", "prediction_available"]],
                "M1F": m1f.set_index(key)[["y_pred", "prediction_available"]],
                "RIDGE": rg.set_index(key)[["y_pred", "prediction_available"]],
            }
            for expert in EXPERTS:
                block = frame.copy()
                joined = per_expert[expert].reindex(pd.MultiIndex.from_frame(block[key]))
                block["expert_id"] = expert
                block["y_pred"] = joined["y_pred"].to_numpy()
                block["prediction_available"] = joined["prediction_available"].fillna(False).to_numpy(dtype=bool)
                block["y_persistence"] = frame["y_origin"]
                rows.append(block)
    history = pd.concat(rows, ignore_index=True)

    history["prediction_available"] = (
        history["prediction_available"].fillna(False).astype(bool)
        & np.isfinite(pd.to_numeric(history["y_pred"], errors="coerce"))
    )
    requires_current = history["expert_id"].isin(["B10", "M2"])
    history.loc[requires_current & ~history["current_target_observed"], "prediction_available"] = False
    history.loc[~history["prediction_available"], "y_pred"] = np.nan
    history["sq_error"] = np.square(history["y_pred"] - history["y_true"])
    history["abs_error"] = np.abs(history["y_pred"] - history["y_true"])
    history["error"] = history["y_pred"] - history["y_true"]

    if history["target_date"].max() > FINAL_TEST_END:
        raise AssertionError("final test history exceeds the sealed 2023-09-17 boundary")
    if history[PAIR_KEY + ["expert_id"]].duplicated().any():
        raise AssertionError("duplicate final-test candidate forecast key")
    if set(history["expert_id"]) != set(EXPERTS):
        raise AssertionError("final-test candidate pool is incomplete")
    if history.loc[
        ~history["current_target_observed"] & history["expert_id"].isin(["B10", "M2"]),
        "prediction_available",
    ].any():
        raise AssertionError(
            "CRITICAL ERROR: B10 or M2 received a prediction with current CH4 missing"
        )
    return history.sort_values(PAIR_KEY + ["expert_id"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Section 6: metrics / diagnostics on the 2023 A30 test rows
# ---------------------------------------------------------------------------

def skill_vs_expert(y_true: pd.Series, y_pred: pd.Series, y_ref: pd.Series) -> float:
    mask = np.isfinite(y_true) & np.isfinite(y_pred) & np.isfinite(y_ref)
    if not mask.any():
        return float("nan")
    model_mse = float(np.mean(np.square(y_pred[mask] - y_true[mask])))
    ref_mse = float(np.mean(np.square(y_ref[mask] - y_true[mask])))
    return skill_from_mse(model_mse, ref_mse)


def _acf(values: pd.Series, lag: int) -> float:
    values = values.dropna().reset_index(drop=True)
    return float(values.autocorr(lag=lag)) if len(values) > lag + 2 else np.nan


def build_metrics_table(a30_2023: pd.DataFrame, wide_experts: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (target, horizon), g in a30_2023.groupby(["target_name", "horizon"], sort=True):
        scored = g.loc[g["y_true"].notna() & g["y_pred"].notna()]
        metric = compute_metrics(scored["y_true"], scored["y_pred"], y_origin=scored["y_origin"])
        merged = scored.merge(
            wide_experts.loc[
                (wide_experts.target_name == target) & (wide_experts.horizon == horizon)
            ],
            on=["target_name", "horizon", "origin_date", "target_date"],
            how="left",
        )
        rows.append({
            "target_name": target, "horizon": int(horizon),
            "n": metric["n_scored"], "RMSE": metric["RMSE"], "MAE": metric["MAE"],
            "bias": metric["Bias"], "R2_level": metric["R2_level"], "R2_delta": metric["R2_delta"],
            "skill_vs_B10": skill_vs_expert(merged["y_true"], merged["y_pred"], merged["B10"]),
            "skill_vs_M2": skill_vs_expert(merged["y_true"], merged["y_pred"], merged["M2"]),
        })
    return pd.DataFrame(rows)


def build_reliability_diagnostics(a30_2023: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (target, horizon, status), g in a30_2023.groupby(
        ["target_name", "horizon", "reliability_status"], sort=True
    ):
        e = (g["y_pred"] - g["y_true"]).dropna()
        rows.append({
            "target_name": target, "horizon": int(horizon), "reliability_status": status,
            "n": len(e), "RMSE": np.sqrt(np.mean(np.square(e))) if len(e) else np.nan,
            "MAE": e.abs().mean() if len(e) else np.nan,
        })
    return pd.DataFrame(rows)


def build_residual_diagnostics(a30_2023: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (target, horizon), g in a30_2023.groupby(["target_name", "horizon"], sort=True):
        e = (g["y_pred"] - g["y_true"]).dropna()
        rows.append({
            "target_name": target, "horizon": int(horizon), "n": len(e),
            "bias": e.mean() if len(e) else np.nan,
            "RMSE": np.sqrt(np.mean(np.square(e))) if len(e) else np.nan,
            "MAE": e.abs().mean() if len(e) else np.nan,
            "ACF_lag1": _acf(e, 1),
            "note": "overlapping multi-step forecast errors; ACF is descriptive, not an independence test",
        })
    return pd.DataFrame(rows)


def build_physical_consistency(a30_2023: pd.DataFrame) -> pd.DataFrame:
    keys = ["origin_date", "target_date", "horizon"]
    wide = a30_2023.pivot(index=keys, columns="target_name", values="y_pred").reset_index()
    wide = wide.rename(columns={"CH4_m3d_observed": "pred_CH4", "biogas_AB_m3d": "pred_biogas"}).dropna(
        subset=["pred_CH4", "pred_biogas"]
    )
    wide["negative_CH4"] = wide["pred_CH4"] < 0
    wide["negative_biogas"] = wide["pred_biogas"] < 0
    wide["CH4_gt_biogas"] = wide["pred_CH4"] > wide["pred_biogas"]
    bad = wide[["negative_CH4", "negative_biogas", "CH4_gt_biogas"]].any(axis=1)
    wide["physical_status"] = np.where(bad, "PROVISIONAL_VIOLATION", "PASS")

    raw = a30_2023["raw_prediction"] if "raw_prediction" in a30_2023 else a30_2023["y_pred"]
    nonneg_rows = []
    for (target, horizon), g in a30_2023.groupby(["target_name", "horizon"], sort=True):
        r = g["raw_prediction"] if "raw_prediction" in g else g["y_pred"]
        n = int(r.notna().sum())
        n_violation = int((r < 0).sum())
        nonneg_rows.append({
            "check": f"A30_NONNEG_{target}_h{horizon}", "n": n, "n_violation": n_violation,
            "status": "PASS" if n_violation == 0 else "FAIL",
        })
    cross_rows = []
    for horizon, g in wide.groupby("horizon"):
        n = len(g)
        n_violation = int(g["physical_status"].eq("PROVISIONAL_VIOLATION").sum())
        cross_rows.append({
            "check": f"CROSS_TARGET_h{horizon}", "n": n, "n_violation": n_violation,
            "status": "PASS" if n_violation == 0 else "PROVISIONAL",
        })
    return pd.DataFrame(nonneg_rows + cross_rows)


def build_uncertainty_table(a30_2023_pi: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (target, horizon), g in a30_2023_pi.groupby(["target_name", "horizon"], sort=True):
        ok = g["y_true"].notna() & g["PI80_lower"].notna() & g["PI95_lower"].notna()
        gg = g.loc[ok]
        if gg.empty:
            rows.append({
                "target_name": target, "horizon": int(horizon), "n": 0,
                "PI80_coverage": np.nan, "PI95_coverage": np.nan,
            })
            continue
        c80 = float(((gg["PI80_lower"] <= gg["y_true"]) & (gg["y_true"] <= gg["PI80_upper"])).mean())
        c95 = float(((gg["PI95_lower"] <= gg["y_true"]) & (gg["y_true"] <= gg["PI95_upper"])).mean())
        rows.append({
            "target_name": target, "horizon": int(horizon), "n": len(gg),
            "PI80_coverage": c80, "PI95_coverage": c95,
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Main orchestration
# ---------------------------------------------------------------------------

def main() -> int:
    for directory in (OUTPUT_DIR, REPORT_DIR):
        directory.mkdir(parents=True, exist_ok=True)

    hashes_before = hash_lock_files()

    process = pd.read_parquet(DATA_DIR / "02_process_base.parquet")
    process["date"] = pd.to_datetime(process["date"]).dt.normalize()
    features = pd.read_parquet(DATA_DIR / "03_feature_base.parquet")
    features["date"] = pd.to_datetime(features["date"]).dt.normalize()
    registry = pd.read_csv(OUTPUT_DIR / "03_feature_registry.csv", encoding="utf-8-sig")
    hyperparameters = pd.read_csv(OUTPUT_DIR / "06_hyperparameter_selection.csv", encoding="utf-8-sig")
    lock_text = (OUTPUT_DIR / "05_model_lock.yaml").read_text(encoding="utf-8")

    print("PHASE 11: fixed-spec 2018-2022 refit (B10/B11 need no fitting)...", flush=True)
    b10_b11 = build_b10_b11_2023(process)
    print("PHASE 11: M1F/M2 fixed-spec refit...", flush=True)
    m1f_m2, mech_lock = build_m1f_m2_2023(process, lock_text)
    print("PHASE 11: RIDGE fixed-spec refit...", flush=True)
    ridge, ridge_lock = build_ridge_2023(features, process, registry, hyperparameters)

    print("PHASE 11: assembling final-test history and running locked A30 replay...", flush=True)
    final_test_history = build_final_test_history(process, b10_b11, m1f_m2, ridge, features)
    history_2019_2022 = build_adaptive_history(PROJECT_ROOT)
    common_columns = [c for c in history_2019_2022.columns if c in final_test_history.columns]
    extended_history = pd.concat(
        [history_2019_2022[common_columns], final_test_history[common_columns]],
        ignore_index=True,
    ).sort_values(PAIR_KEY + ["expert_id"]).reset_index(drop=True)
    if extended_history[PAIR_KEY + ["expert_id"]].duplicated().any():
        raise AssertionError("extended history has duplicate keys after concatenation")

    predictions, trace, reliability = simulate_adaptive_strategies(extended_history)
    a30_all = predictions.loc[predictions["strategy"].eq(A30)].copy()

    # CRITICAL ERROR check (section 8): B10/M2 must never receive positive
    # weight when current CH4 is missing.
    missing_current = a30_all.loc[~a30_all["current_target_observed"]]
    if (missing_current["weight_B10"].abs() > 1e-12).any() or (
        missing_current["weight_M2"].abs() > 1e-12
    ).any():
        raise AssertionError(
            "CRITICAL ERROR: A30 assigned positive weight to B10/M2 while current CH4 was missing"
        )
    print("CRITICAL ERROR check: PASSED (B10/M2 weight is zero whenever current CH4 is missing).")

    print("PHASE 11: calibrating past-residual-only prediction intervals...", flush=True)
    a30_for_pi = a30_all.copy()
    a30_for_pi["ensemble_id"] = "A30"
    a30_for_pi["y_A30"] = a30_for_pi["y_pred"]
    a30_pi = add_past_residual_intervals(a30_for_pi)

    a30_2023 = a30_all.loc[
        a30_all["origin_date"].between(FINAL_TEST_START, FINAL_TEST_END)
        & a30_all["target_date"].between(FINAL_TEST_START, FINAL_TEST_END)
    ].copy()
    a30_2023_pi = a30_pi.loc[
        a30_pi["origin_date"].between(FINAL_TEST_START, FINAL_TEST_END)
        & a30_pi["target_date"].between(FINAL_TEST_START, FINAL_TEST_END)
    ].copy()
    if a30_2023.empty:
        raise RuntimeError("no 2023 A30 rows were produced")

    wide_rows = []
    for expert in EXPERTS:
        part = extended_history.loc[extended_history["expert_id"].eq(expert), PAIR_KEY + ["y_pred"]]
        part = part.rename(columns={"y_pred": expert})
        wide_rows.append(part.set_index(PAIR_KEY))
    wide_experts = pd.concat(wide_rows, axis=1).reset_index()

    print("PHASE 11: computing metrics and diagnostics...", flush=True)
    metrics_2023 = build_metrics_table(a30_2023, wide_experts)
    primary_metrics_2023 = metrics_2023.loc[metrics_2023["horizon"].isin(PRIMARY_HORIZONS)].copy()
    reliability_2023 = build_reliability_diagnostics(a30_2023)
    residual_2023 = build_residual_diagnostics(a30_2023)
    physical_2023 = build_physical_consistency(a30_2023)
    uncertainty_2023 = build_uncertainty_table(a30_2023_pi)

    # Outputs -----------------------------------------------------------
    print("PHASE 11: writing outputs...", flush=True)
    predictions_out = a30_2023.loc[:, [
        "target_name", "horizon", "origin_date", "target_date", "y_true", "y_pred",
        "y_origin", "y_persistence", "current_target_observed",
        "reliability_status", "reliability_score", "selected_model", "available_models",
        *[f"weight_{m}" for m in EXPERTS],
    ]].merge(
        a30_2023_pi[["target_name", "horizon", "origin_date", "target_date",
                      "PI80_lower", "PI80_upper", "PI95_lower", "PI95_upper", "PI_n_calibration"]],
        on=["target_name", "horizon", "origin_date", "target_date"], how="left",
    )
    predictions_out.to_parquet(OUTPUT_DIR / "11_final_predictions_2023.parquet", index=False)
    metrics_2023.to_csv(OUTPUT_DIR / "11_final_metrics_2023.csv", index=False, encoding="utf-8-sig")
    primary_metrics_2023.to_csv(
        OUTPUT_DIR / "11_final_primary_metrics_2023.csv", index=False, encoding="utf-8-sig"
    )
    reliability_2023.to_csv(
        OUTPUT_DIR / "11_final_reliability_calibration_2023.csv", index=False, encoding="utf-8-sig"
    )
    uncertainty_2023.to_csv(
        OUTPUT_DIR / "11_final_uncertainty_2023.csv", index=False, encoding="utf-8-sig"
    )
    physical_2023.to_csv(
        OUTPUT_DIR / "11_final_physical_consistency_2023.csv", index=False, encoding="utf-8-sig"
    )
    residual_2023.to_csv(
        OUTPUT_DIR / "11_final_residual_diagnostics_2023.csv", index=False, encoding="utf-8-sig"
    )

    refit_manifest = {
        "final_refit_period": {"start": "2018-01-01", "end": "2022-12-31"},
        "final_test_period": {"start": "2023-01-01", "end": "2023-09-17"},
        "mechanistic": mech_lock,
        "ridge": {key: {k: v for k, v in value.items() if k != "features"} for key, value in ridge_lock.items()},
        "ridge_features_by_target_horizon": {key: value["features"] for key, value in ridge_lock.items()},
    }
    (OUTPUT_DIR / "11_final_refit_manifest.yaml").write_text(
        _to_yaml(refit_manifest), encoding="utf-8"
    )

    hashes_after = hash_lock_files()
    unchanged = all(hashes_before[key] == hashes_after[key] for key in hashes_before)
    if not unchanged:
        raise AssertionError("a Phase 01-10 lock file changed during Phase 11 execution")

    artifact_hashes = {
        "11_final_predictions_2023.parquet": sha256_file(OUTPUT_DIR / "11_final_predictions_2023.parquet"),
        "11_final_refit_manifest.yaml": sha256_file(OUTPUT_DIR / "11_final_refit_manifest.yaml"),
    }
    final_lock = {
        "system_id": "A30_INVERSE_ERROR_RELIABILITY_WEIGHT",
        "expert_pool": list(EXPERTS),
        "features_locked": True,
        "hyperparameters_locked": True,
        "availability_rules_locked": True,
        "pi_method_locked": True,
        "final_refit_period": {"start": "2018-01-01", "end": "2022-12-31"},
        "final_test_period": {"start": "2023-01-01", "end": "2023-09-17"},
        "post_2023_retuning_allowed": False,
        "critical_error_check": "PASSED",
        "artifact_sha256": artifact_hashes,
        "phase01_10_lock_files_unchanged": unchanged,
    }
    (OUTPUT_DIR / "11_final_model_lock.yaml").write_text(_to_yaml(final_lock), encoding="utf-8")

    print("PHASE 11 COMPLETE.")
    print(f"Phase 01-10 lock files unchanged before/after: {unchanged}")
    return 0


def _scalar(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        return "null" if not np.isfinite(value) else f"{float(value):.10g}"
    text = str(value)
    if not text or any(token in text for token in (":", "#", "[", "]", "{", "}", "\n")):
        return json.dumps(text, ensure_ascii=False)
    return text


def _yaml_lines(value: Any, indent: int = 0) -> list[str]:
    prefix = " " * indent
    if isinstance(value, dict):
        lines: list[str] = []
        for key, item in value.items():
            if isinstance(item, (dict, list)):
                lines.append(f"{prefix}{key}:")
                lines.extend(_yaml_lines(item, indent + 2))
            else:
                lines.append(f"{prefix}{key}: {_scalar(item)}")
        return lines
    if isinstance(value, list):
        lines = []
        for item in value:
            if isinstance(item, (dict, list)):
                lines.append(f"{prefix}-")
                lines.extend(_yaml_lines(item, indent + 2))
            else:
                lines.append(f"{prefix}- {_scalar(item)}")
        return lines
    return [f"{prefix}{_scalar(value)}"]


def _to_yaml(value: dict[str, Any]) -> str:
    return "\n".join(_yaml_lines(value)) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
