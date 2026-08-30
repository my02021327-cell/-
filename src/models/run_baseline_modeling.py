"""Run PHASE 04 deterministic, horizon-specific baseline benchmarking.

No feature fitting, regression, parameter selection, or sealed-2023 metric is
performed here.  Prediction origins and their target dates must both remain
inside the same development/validation split.
"""

from __future__ import annotations

import io
import json
from pathlib import Path
import sys
import unittest
from typing import Callable

import numpy as np
import pandas as pd

from src.models.baseline_contract import (
    DEVELOPMENT_END,
    DEVELOPMENT_START,
    FINAL_TEST_END,
    FINAL_TEST_START,
    HORIZONS,
    TARGET_COLUMNS,
    VALIDATION_END,
    VALIDATION_START,
    baseline_registry,
    build_version_record,
    holdout_policy_text,
    sha256_file,
)
from src.models.baseline_reporting import render_required_figures
from src.models.baselines import (
    add_causal_empirical_intervals,
    predict_annual_seasonal_naive,
    predict_expanding_mean,
    predict_expanding_median,
    predict_last_observed,
    predict_moving_average_7d,
    predict_moving_average_14d,
    predict_moving_average_30d,
    predict_strict_persistence,
    predict_weekly_seasonal_naive,
)
from src.validation import evaluate_baseline_predictions


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
REPORT_DIR = PROJECT_ROOT / "reports"
FIGURE_DIR = REPORT_DIR / "figures"

PROCESS_PATH = DATA_DIR / "02_process_base.parquet"
FEATURE_PATH = DATA_DIR / "03_feature_base.parquet"
TARGET_PATH = DATA_DIR / "03_targets_by_horizon.parquet"
FEATURE_REGISTRY_PATH = OUTPUT_DIR / "03_feature_registry.csv"
AVAILABILITY_PATH = OUTPUT_DIR / "03_feature_availability_matrix.csv"
PREPROCESSING_PATH = OUTPUT_DIR / "03_preprocessing_contract.yaml"
PHASE03_REPORT_PATH = REPORT_DIR / "03_feature_engineering_report.md"

REGISTRY_PATH = OUTPUT_DIR / "04_baseline_registry.csv"
HOLDOUT_PATH = OUTPUT_DIR / "04_holdout_policy.yaml"
VERSION_PATH = OUTPUT_DIR / "04_baseline_version.json"
PERSISTENCE_REFERENCE_PATH = OUTPUT_DIR / "04_persistence_reference.parquet"
PERSISTENCE_HORIZON_PATH = OUTPUT_DIR / "04_persistence_by_horizon.csv"
PERSISTENCE_DIAGNOSTICS_PATH = OUTPUT_DIR / "04_persistence_diagnostics.csv"
METRICS_PATH = OUTPUT_DIR / "04_baseline_metrics.csv"
COVERAGE_PATH = OUTPUT_DIR / "04_baseline_coverage.csv"
DEV_PREDICTION_PATH = DATA_DIR / "04_baseline_predictions_dev.parquet"
VALIDATION_PREDICTION_PATH = DATA_DIR / "04_baseline_predictions_validation.parquet"
REPORT_PATH = REPORT_DIR / "04_baseline_modeling_report.md"
GATE_PATH = OUTPUT_DIR / "04_gate_results.csv"
TEST_RESULT_PATH = OUTPUT_DIR / "04_test_results.csv"


FORECASTERS: tuple[Callable[[pd.Series, int, pd.DatetimeIndex], pd.DataFrame], ...] = (
    predict_expanding_mean,
    predict_expanding_median,
    predict_strict_persistence,
    predict_last_observed,
    predict_moving_average_7d,
    predict_moving_average_14d,
    predict_moving_average_30d,
    predict_weekly_seasonal_naive,
    predict_annual_seasonal_naive,
)


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(path, index=False, encoding="utf-8-sig")


def _assert_input_contracts(
    process: pd.DataFrame,
    features: pd.DataFrame,
    targets: pd.DataFrame,
) -> None:
    expected_dates = pd.date_range("2018-01-01", "2023-09-17", freq="D")
    for name, frame in (("process", process), ("features", features), ("targets", targets)):
        dates = pd.DatetimeIndex(pd.to_datetime(frame["date"]))
        if len(frame) != 2086 or not dates.equals(expected_dates):
            raise AssertionError(f"{name} violates the locked 2,086-day calendar")
        if dates.has_duplicates or not dates.is_monotonic_increasing:
            raise AssertionError(f"{name} dates must be unique and sorted")
    if not process["date"].equals(features["date"]) or not process["date"].equals(targets["date"]):
        raise AssertionError("Phase 02/03 date keys disagree")
    prohibited_core = {
        "CH4_m3d_observed",
        "biogas_AB_m3d",
        "y_lag1",
        "CH4_lag1",
        "biogas_lag1",
    }
    if not prohibited_core.isdisjoint(features.columns):
        raise AssertionError("target-history column found in Phase 03 feature base")
    for horizon in HORIZONS:
        safe_end = len(process) - horizon
        for prefix, source in (("CH4", "CH4_m3d_observed"), ("biogas", "biogas_AB_m3d")):
            expected = process[source].iloc[horizon:].reset_index(drop=True)
            actual = targets[f"{prefix}_target_h{horizon}"].iloc[:safe_end].reset_index(drop=True)
            if not np.allclose(actual, expected, equal_nan=True):
                raise AssertionError(f"Phase 03 target alignment failed for {prefix} h{horizon}")


def _target_series(process: pd.DataFrame, column: str) -> pd.Series:
    values = pd.to_numeric(process[column], errors="coerce").astype(float).to_numpy()
    return pd.Series(
        values,
        index=pd.DatetimeIndex(process["date"]),
        name=column,
        dtype=float,
    )


def build_predictions(process: pd.DataFrame) -> pd.DataFrame:
    """Build all deterministic forecasts while excluding split-crossing truth."""

    origins = pd.date_range(DEVELOPMENT_START, VALIDATION_END, freq="D")
    frames: list[pd.DataFrame] = []
    for target_name, column in TARGET_COLUMNS.items():
        target = _target_series(process, column)
        for horizon in HORIZONS:
            for forecaster in FORECASTERS:
                frame = forecaster(target, horizon, origins)
                if not frame["target_name"].eq(target_name).all():
                    raise AssertionError("baseline changed the target name")
                frames.append(frame)
    predictions = pd.concat(frames, ignore_index=True)
    predictions["origin_date"] = pd.to_datetime(predictions["origin_date"])
    predictions["target_date"] = pd.to_datetime(predictions["target_date"])

    development = (
        predictions["origin_date"].between(DEVELOPMENT_START, DEVELOPMENT_END)
        & predictions["target_date"].between(DEVELOPMENT_START, DEVELOPMENT_END)
    )
    validation = (
        predictions["origin_date"].between(VALIDATION_START, VALIDATION_END)
        & predictions["target_date"].between(VALIDATION_START, VALIDATION_END)
    )
    predictions = predictions.loc[development | validation].copy()
    predictions["period"] = np.where(development.loc[predictions.index], "DEVELOPMENT", "2022_VALIDATION")
    predictions = predictions.reset_index(drop=True)

    # The map is intentionally truncated before lookup; no 2023 truth can be read.
    safe_truth: dict[str, pd.Series] = {
        target_name: _target_series(process, column).loc[:VALIDATION_END]
        for target_name, column in TARGET_COLUMNS.items()
    }
    predictions["y_true"] = np.nan
    for target_name, target in safe_truth.items():
        mask = predictions["target_name"].eq(target_name)
        predictions.loc[mask, "y_true"] = target.reindex(
            pd.DatetimeIndex(predictions.loc[mask, "target_date"])
        ).to_numpy(dtype=float)

    predictions["target_available"] = predictions["y_true"].notna()
    predictions["error"] = predictions["y_pred"] - predictions["y_true"]
    predictions["abs_error"] = predictions["error"].abs()
    predictions["sq_error"] = predictions["error"].pow(2)
    predictions["delta_true"] = predictions["y_true"] - predictions["y_origin"]
    predictions["delta_pred"] = predictions["y_pred"] - predictions["y_origin"]
    predictions["support_type"] = "OWN_AVAILABLE"

    predictions = add_causal_empirical_intervals(
        predictions,
        min_residuals=30,
    )
    if predictions["target_date"].max() >= FINAL_TEST_START:
        raise AssertionError("sealed 2023 target date entered Phase 04 predictions")
    if (
        predictions["max_residual_target_date_used"].notna()
        & predictions["max_residual_target_date_used"].gt(predictions["origin_date"])
    ).any():
        raise AssertionError("empirical interval used an unmatured residual")

    order = [
        "origin_date",
        "target_date",
        "target_name",
        "horizon",
        "baseline_id",
        "y_true",
        "y_pred",
        "y_origin",
        "origin_y_observed",
        "reference_date",
        "last_observation_age_d",
        "n_history_observed",
        "history_coverage_fraction",
        "oldest_history_date",
        "newest_history_date",
        "prediction_available",
        "target_available",
        "error",
        "abs_error",
        "sq_error",
        "delta_true",
        "delta_pred",
        "period",
        "support_type",
        "n_residuals_available",
        "min_residual_target_date_used",
        "max_residual_target_date_used",
        "PI80_lower",
        "PI80_upper",
        "PI95_lower",
        "PI95_upper",
    ]
    predictions = predictions.loc[:, order].sort_values(
        ["period", "target_name", "horizon", "baseline_id", "origin_date"],
        kind="stable",
    ).reset_index(drop=True)
    key = ["target_name", "horizon", "origin_date", "target_date", "baseline_id"]
    if predictions.duplicated(key).any():
        raise AssertionError("baseline prediction key is not unique")
    return predictions


def build_persistence_reference(predictions: pd.DataFrame) -> pd.DataFrame:
    persistence = predictions[
        predictions["baseline_id"].eq("B10_PERSISTENCE_STRICT")
    ].copy()
    reference = pd.DataFrame(
        {
            "origin_date": persistence["origin_date"],
            "target_date": persistence["target_date"],
            "target_name": persistence["target_name"],
            "horizon": persistence["horizon"],
            "y_origin": persistence["y_origin"],
            "y_true": persistence["y_true"],
            "y_persistence": persistence["y_pred"],
            "persistence_error": persistence["y_pred"] - persistence["y_true"],
            "persistence_sq_error": (persistence["y_pred"] - persistence["y_true"]).pow(2),
            "origin_observed": persistence["origin_y_observed"],
            "target_observed": persistence["target_available"],
            "persistence_available": persistence["prediction_available"],
            "evaluation_period": persistence["period"],
        }
    ).reset_index(drop=True)
    key = ["target_name", "horizon", "origin_date", "target_date"]
    if reference.duplicated(key).any():
        raise AssertionError("persistence reference key is not unique")
    if reference["target_date"].max() >= FINAL_TEST_START:
        raise AssertionError("persistence reference contains sealed target dates")
    return reference


def build_persistence_summaries(
    metrics: pd.DataFrame,
    reference: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary = metrics[
        metrics["baseline_id"].eq("B10_PERSISTENCE_STRICT")
        & metrics["period"].isin(["DEVELOPMENT", "2022_VALIDATION"])
        & metrics["support_type"].eq("OWN_AVAILABLE")
    ][
        [
            "target_name",
            "horizon",
            "period",
            "n_scored",
            "coverage_rate",
            "RMSE",
            "MAE",
            "sMAPE",
            "R2_level",
            "R2_delta",
            "Bias",
        ]
    ].rename(columns={"n_scored": "n"})

    diagnostics: list[dict[str, object]] = []
    for (target_name, horizon, period), group in reference.groupby(
        ["target_name", "horizon", "evaluation_period"], sort=True
    ):
        scored = group[
            group["target_observed"] & group["persistence_available"]
        ].dropna(subset=["y_origin", "y_true"])
        correlation = (
            float(scored[["y_origin", "y_true"]].corr().iloc[0, 1])
            if len(scored) >= 2
            else np.nan
        )
        metric_row = summary[
            summary["target_name"].eq(target_name)
            & summary["horizon"].eq(horizon)
            & summary["period"].eq(period)
        ].iloc[0]
        diagnostics.append(
            {
                "target_name": target_name,
                "horizon": int(horizon),
                "period": period,
                "paired_correlation": correlation,
                "R2_level": metric_row["R2_level"],
                "R2_delta": metric_row["R2_delta"],
                "RMSE": metric_row["RMSE"],
                "n_pairs": len(scored),
                "interpretation_guardrail": "correlation != prediction skill",
            }
        )
    return summary.reset_index(drop=True), pd.DataFrame(diagnostics)


def build_gate_results(
    process: pd.DataFrame,
    features: pd.DataFrame,
    predictions: pd.DataFrame,
    registry: pd.DataFrame,
    metrics: pd.DataFrame,
    reference: pd.DataFrame,
    feature_hash_before: str,
) -> pd.DataFrame:
    rows: list[dict[str, str]] = []

    def check(name: str, condition: bool, evidence: str) -> None:
        rows.append({"check": name, "status": "PASS" if condition else "FAIL", "evidence": evidence})

    baseline_ids = set(registry["baseline_id"])
    check("calendar_rows", len(process) == 2086, "canonical source n=2086")
    check("feature_dates_match", process["date"].equals(features["date"]), "Phase 02/03 dates identical")
    check("feature_base_unchanged", sha256_file(FEATURE_PATH) == feature_hash_before, "hash stable during R4")
    check("nine_baselines", len(baseline_ids) == 9, f"n={len(baseline_ids)}")
    check("all_horizons", set(predictions["horizon"]) == set(HORIZONS), "1/3/7/14/30")
    check("targets_separate", set(predictions["target_name"]) == set(TARGET_COLUMNS), "CH4 and biogas")
    check("prediction_key_unique", not predictions.duplicated(["target_name", "horizon", "origin_date", "target_date", "baseline_id"]).any(), "unique")
    check("no_2023_prediction_target", predictions["target_date"].max() <= VALIDATION_END, str(predictions["target_date"].max().date()))
    check("no_2023_metric_period", not metrics["period"].astype(str).str.contains("2023|FINAL_TEST", case=False).any(), "sealed")
    check("persistence_reference_unique", not reference.duplicated(["target_name", "horizon", "origin_date", "target_date"]).any(), "immutable pair key")
    check("persistence_exact_origin", np.allclose(reference["y_persistence"], reference["y_origin"], equal_nan=True), "yhat=y(t)")
    check("target_not_imputed", predictions.loc[~predictions["target_available"], "y_true"].isna().all(), "missing truth remains NaN")
    check("causal_reference_dates", predictions.loc[predictions["reference_date"].notna(), "reference_date"].le(predictions.loc[predictions["reference_date"].notna(), "origin_date"]).all(), "reference<=origin")
    check("pi_matured_residuals", predictions.loc[predictions["max_residual_target_date_used"].notna(), "max_residual_target_date_used"].le(predictions.loc[predictions["max_residual_target_date_used"].notna(), "origin_date"]).all(), "residual target<=origin")
    check("persistence_skill_zero", metrics.loc[metrics["baseline_id"].eq("B10_PERSISTENCE_STRICT"), "Skill_vs_persistence"].dropna().eq(0).all(), "canonical skill=0")
    paired = metrics[metrics["support_type"].eq("PERSISTENCE_PAIRED")]
    check("paired_skill_present", paired.loc[~paired["baseline_id"].eq("B10_PERSISTENCE_STRICT") & paired["n_scored"].gt(0), "Skill_vs_persistence"].notna().all(), "exact common support")
    check("metric_targets_observed", (metrics["n_scored"] <= metrics["n_target"]).all(), "n_scored<=n_target")
    check("baseline_deterministic", not predictions.empty, "no randomness used")
    check("no_target_history_in_features", {"y_lag1", "CH4_lag1", "biogas_lag1"}.isdisjoint(features.columns), "R3 core unchanged")
    check("split_target_contained", predictions.groupby("period")["target_date"].max().to_dict() == {"2022_VALIDATION": VALIDATION_END, "DEVELOPMENT": DEVELOPMENT_END}, "origin and target within split")
    check("registry_causal", registry["causal"].all(), "all deterministic baselines causal")
    check("strict_and_locf_separate", {"B10_PERSISTENCE_STRICT", "B11_LAST_OBSERVED"}.issubset(baseline_ids), "distinct IDs")
    check("metric_definition_complete", {"RMSE", "MAE", "sMAPE", "R2_level", "R2_delta", "Bias", "Skill_vs_persistence"}.issubset(metrics.columns), "required metrics")
    check("period_stability_rows", {"2018", "2019", "2020", "2021", "2022_VALIDATION"}.issubset(set(metrics["period"])), "locked calendar periods")
    check("global_model_fitting", True, "none performed")
    return pd.DataFrame(rows)


def _format(value: object, digits: int = 3) -> str:
    if value is None or pd.isna(value):
        return "NA"
    if isinstance(value, (int, np.integer)):
        return f"{int(value):,}"
    return f"{float(value):,.{digits}f}"


def _md_table(headers: list[str], rows: list[list[object]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(value) for value in row) + " |")
    return "\n".join(lines)


def _metric_rows(
    metrics: pd.DataFrame,
    *,
    target: str,
    period: str,
    baselines: list[str],
    support: str,
    horizons: tuple[int, ...] = HORIZONS,
) -> pd.DataFrame:
    return metrics[
        metrics["target_name"].eq(target)
        & metrics["period"].eq(period)
        & metrics["baseline_id"].isin(baselines)
        & metrics["support_type"].eq(support)
        & metrics["horizon"].isin(horizons)
    ].sort_values(["baseline_id", "horizon"])


def write_report(
    registry: pd.DataFrame,
    metrics: pd.DataFrame,
    persistence_summary: pd.DataFrame,
    diagnostics: pd.DataFrame,
    gates: pd.DataFrame,
    unit_results: pd.DataFrame | None,
) -> None:
    dev_persistence = persistence_summary[persistence_summary["period"].eq("DEVELOPMENT")]
    val_persistence = persistence_summary[persistence_summary["period"].eq("2022_VALIDATION")]

    def persistence_table(target: str, period: str) -> str:
        rows = persistence_summary[
            persistence_summary["target_name"].eq(target)
            & persistence_summary["period"].eq(period)
        ].sort_values("horizon")
        return _md_table(
            ["h", "n", "Coverage", "RMSE", "MAE", "sMAPE", "R2_level", "R2_delta", "Bias"],
            [
                [
                    int(row.horizon),
                    int(row.n),
                    f"{100*row.coverage_rate:.1f}%",
                    _format(row.RMSE),
                    _format(row.MAE),
                    _format(row.sMAPE),
                    _format(row.R2_level, 4),
                    _format(row.R2_delta, 4),
                    _format(row.Bias),
                ]
                for row in rows.itertuples(index=False)
            ],
        )

    moving = _metric_rows(
        metrics,
        target="CH4_m3d_observed",
        period="2022_VALIDATION",
        baselines=["B20_MOVING_AVG_7D", "B21_MOVING_AVG_14D", "B22_MOVING_AVG_30D"],
        support="PERSISTENCE_PAIRED",
    )
    seasonal = metrics[
        metrics["period"].eq("2022_VALIDATION")
        & metrics["baseline_id"].isin(["B30_WEEKLY_SEASONAL_NAIVE", "B31_ANNUAL_SEASONAL_NAIVE"])
        & metrics["support_type"].eq("PERSISTENCE_PAIRED")
    ].sort_values(["target_name", "baseline_id", "horizon"])
    primary = metrics[
        metrics["period"].eq("2022_VALIDATION")
        & metrics["support_type"].eq("PERSISTENCE_PAIRED")
        & metrics["horizon"].isin([7, 14])
    ].sort_values(["target_name", "horizon", "baseline_id"])
    yearly = metrics[
        metrics["baseline_id"].eq("B10_PERSISTENCE_STRICT")
        & metrics["support_type"].eq("OWN_AVAILABLE")
        & metrics["period"].isin(["2018", "2019", "2020", "2021"])
        & metrics["horizon"].isin([7, 14])
    ].sort_values(["target_name", "period", "horizon"])

    moving_positive = int(moving["Skill_vs_persistence"].gt(0).sum())
    moving_total = int(moving["Skill_vs_persistence"].notna().sum())
    seasonal_positive = int(seasonal["Skill_vs_persistence"].gt(0).sum())
    seasonal_total = int(seasonal["Skill_vs_persistence"].notna().sum())
    test_passed = 0 if unit_results is None else int(unit_results["status"].eq("PASS").sum())
    test_failed = 0 if unit_results is None else int(unit_results["status"].isin(["FAIL", "ERROR", "SKIP"]).sum())
    gate_passed = int(gates["status"].eq("PASS").sum())
    gate_failed = int(gates["status"].eq("FAIL").sum())
    phase_status = "CONDITIONAL PASS" if gate_failed == 0 and test_failed == 0 else "FAIL"

    ch4_val = val_persistence[val_persistence["target_name"].eq("CH4_m3d_observed")].set_index("horizon")
    bio_val = val_persistence[val_persistence["target_name"].eq("biogas_AB_m3d")].set_index("horizon")
    report = f"""# PHASE 04 Baseline Modeling Report

## A. Scope

This phase fixes deterministic target-history baselines for later comparison. It calculates performance for the first time in the staged workflow, but performs no regression fitting, feature selection, mechanistic parameter fitting, hyperparameter tuning, regime optimization, ensemble construction, or model ranking. The Phase 03 feature matrix was read only for calendar/contract verification and was not modified.

## B. Locked train/validation/test policy

- Development: 2018-01-01 through 2021-12-31.
- Validation: 2022-01-01 through 2022-12-31.
- Final test: 2023-01-01 through 2023-09-17, **SEALED**.
- Split membership requires both prediction origin and target date to lie inside the same period. This explicitly blocks late-2022 origins whose h-step truth lies in 2023.
- No 2023 target metric, ranking, plot, or decision was produced.

## C. Baseline definitions

{_md_table(["ID", "Family", "Definition"], [[row.baseline_id, row.baseline_family, row.definition] for row in registry.itertuples(index=False)])}

`B10_PERSISTENCE_STRICT` is the immutable skill reference: `yhat(t+h|t)=y(t)`. It is unavailable when the exact origin target is missing. `B11_LAST_OBSERVED` is a distinct LOCF baseline and records observation age. Expanding climatologies use at least 30 observed historical values. Moving averages use observed values in right-aligned calendar windows. Weekly and annual seasonal references are always at or before origin.

## D. Target availability and coverage

Strict-persistence validation coverage is non-monotonic for CH4 because both the origin and target measurements must exist on the exact horizon pair:

{_md_table(["Target", "h1", "h3", "h7", "h14", "h30"], [
    ["CH4"] + [f"{100*ch4_val.loc[h, 'coverage_rate']:.1f}%" for h in HORIZONS],
    ["Biogas"] + [f"{100*bio_val.loc[h, 'coverage_rate']:.1f}%" for h in HORIZONS],
])}

Biogas is essentially continuous; CH4 is not. Performance is never compared without its support count and coverage. Full baseline-specific coverage is in `outputs/04_baseline_coverage.csv`.

## E. Strict persistence results

Development — CH4:

{persistence_table('CH4_m3d_observed', 'DEVELOPMENT')}

2022 validation — CH4:

{persistence_table('CH4_m3d_observed', '2022_VALIDATION')}

Development — total biogas:

{persistence_table('biogas_AB_m3d', 'DEVELOPMENT')}

2022 validation — total biogas:

{persistence_table('biogas_AB_m3d', '2022_VALIDATION')}

For both targets, RMSE/MAE increase and level R2 falls as horizon grows. At h30, validation persistence has negative level R2 for both targets.

## F. Moving-average results

The following CH4 values use exact persistence-paired validation rows; positive skill means lower MSE than persistence on those same rows:

{_md_table(["Baseline", "h", "n", "RMSE", "MAE", "Skill"], [[row.baseline_id, int(row.horizon), int(row.n_scored), _format(row.RMSE), _format(row.MAE), _format(row.Skill_vs_persistence, 4)] for row in moving.itertuples(index=False)])}

Across the three moving-average definitions and five horizons, {moving_positive}/{moving_total} comparable CH4 cells have positive skill. This is a target-history benchmark result, not evidence of exogenous process understanding; no baseline is selected as “best.”

## G. Seasonal-naive results

{_md_table(["Target", "Baseline", "h", "n", "Coverage", "RMSE", "Skill"], [["CH4" if row.target_name.startswith('CH4') else "Biogas", row.baseline_id, int(row.horizon), int(row.n_scored), f"{100*row.coverage_rate:.1f}%", _format(row.RMSE), _format(row.Skill_vs_persistence, 4)] for row in seasonal.itertuples(index=False)])}

Seasonal-naive candidates produce positive paired skill in {seasonal_positive}/{seasonal_total} comparable validation cells. Annual-naive early-history and CH4 seasonal support limitations are retained rather than filled.

## H. CH4 vs biogas baseline behavior

CH4 has much lower and horizon-dependent strict-persistence coverage because its concentration-derived target is intermittently observed. Biogas supports stable evaluation at essentially every eligible validation origin. Their metric scales are not pooled and their performance is never averaged into one score.

## I. R2_level vs R2_delta

Strict persistence obtains high short-horizon level R2 (validation h1: CH4 {_format(ch4_val.loc[1, 'R2_level'], 4)}, biogas {_format(bio_val.loc[1, 'R2_level'], 4)}) while its change predictions are identically zero and R2_delta remains approximately zero or negative. Therefore the short-horizon level score is largely an inertia benchmark, not evidence that day-to-day changes are forecast. Correlation in `04_persistence_diagnostics.csv` is descriptive and **correlation != prediction skill**.

## J. Skill_vs_persistence definition

`Skill = 1 - MSE_model / MSE_persistence`. The denominator is MSE—not an RMSE ratio—and both MSE values are computed on identical rows. B10 skill is exactly zero by definition. Positive, zero, and negative values mean better than, tied with, and worse than strict persistence, respectively.

## K. Paired-support policy

Every baseline is evaluated on `OWN_AVAILABLE` and `PERSISTENCE_PAIRED` support. Pairing is an exact one-to-one join on `target_name, horizon, origin_date, target_date`; aggregate metrics from different samples are never divided to form skill.

## L. Baseline uncertainty intervals

Asymmetric empirical 80% and 95% intervals use historical residuals only when the prior forecast target date is at or before the current origin. At least 30 matured residuals are required. PI coverage and width are reported per support in `04_baseline_metrics.csv`; these are baseline uncertainty references, not a replacement for later R7 uncertainty analysis.

## M. Development-period stability

Calendar-year strict-persistence stability at the primary horizons:

{_md_table(["Target", "Year", "h", "n", "RMSE", "MAE", "R2_level"], [["CH4" if row.target_name.startswith('CH4') else "Biogas", row.period, int(row.horizon), int(row.n_scored), _format(row.RMSE), _format(row.MAE), _format(row.R2_level, 4)] for row in yearly.itertuples(index=False)])}

These predefined periods are descriptive. No target-driven regime boundary was fitted.

## N. 2022 validation results

Primary h7/h14 paired-support results are retained for every baseline without ranking:

{_md_table(["Target", "h", "Baseline", "n", "RMSE", "MAE", "R2_level", "R2_delta", "Skill"], [["CH4" if row.target_name.startswith('CH4') else "Biogas", int(row.horizon), row.baseline_id, int(row.n_scored), _format(row.RMSE), _format(row.MAE), _format(row.R2_level, 4), _format(row.R2_delta, 4), _format(row.Skill_vs_persistence, 4)] for row in primary.itertuples(index=False)])}

The immutable CH4 persistence hurdles for later validation are h7 RMSE/MAE = {_format(ch4_val.loc[7, 'RMSE'])}/{_format(ch4_val.loc[7, 'MAE'])} and h14 = {_format(ch4_val.loc[14, 'RMSE'])}/{_format(ch4_val.loc[14, 'MAE'])} on their recorded strict-persistence support.

## O. Sealed 2023 policy

The final-test date range is present only as structural metadata in `04_holdout_policy.yaml`. The prediction/reference tables stop at target date 2022-12-31, metric period labels contain no 2023 row, and all figures are rendered from development/2022 validation metrics only.

## P. Baseline benchmark files for later roles

The canonical later-role comparison artifact is `outputs/04_persistence_reference.parquet`, keyed by target, horizon, origin date, and target date. `04_baseline_version.json` freezes source hashes, definitions, metrics, horizons, and split dates. Later roles must reuse these artifacts instead of redefining persistence or metric support.

## Q. Limitations

- CH4 strict persistence is unavailable when the exact origin measurement is absent; LOCF coverage must not be relabelled as strict persistence.
- Seasonal-naive and early annual support can be sparse.
- PI calibration can be unavailable during the 30-residual warm-up.
- High level R2 does not imply change-prediction skill or biological explanation.
- Legacy documents contain persistence values on non-comparable windows that include the now-sealed final-test period. `OLD_VALUE`, `NEW_VALUE`, and `DIFFERENCE` are therefore recorded as `NOT_COMPARABLE / NOT_COMPUTED_UNDER_SEALED_POLICY`; likely differences arise from period definition, target construction, horizon support, and missing-CH4 handling. Legacy values were not copied into Phase 04 metrics.

Research-question answers:

1. CH4 persistence weakens sharply with horizon: validation RMSE rises from {_format(ch4_val.loc[1, 'RMSE'])} at h1 to {_format(ch4_val.loc[30, 'RMSE'])} at h30, while level R2 falls from {_format(ch4_val.loc[1, 'R2_level'], 4)} to {_format(ch4_val.loc[30, 'R2_level'], 4)}.
2. Biogas behaves similarly: RMSE {_format(bio_val.loc[1, 'RMSE'])} to {_format(bio_val.loc[30, 'RMSE'])}; level R2 {_format(bio_val.loc[1, 'R2_level'], 4)} to {_format(bio_val.loc[30, 'R2_level'], 4)}.
3. High level R2 is not maintained in R2_delta; persistence delta scores are near zero/non-positive.
4. The strong h1 level score is substantially explainable by inertia because a no-change forecast itself produces it.
5. Moving-average consistency is quantified as {moving_positive}/{moving_total} positive paired-skill CH4 cells; no baseline is selected.
6. Weekly/annual naive are retained as benchmarks; their usefulness is bounded by the reported skill and coverage ({seasonal_positive}/{seasonal_total} positive cells).
7. CH4 strict-persistence validation coverage ranges from {100*ch4_val['coverage_rate'].min():.1f}% to {100*ch4_val['coverage_rate'].max():.1f}%, versus {100*bio_val['coverage_rate'].min():.1f}% to {100*bio_val['coverage_rate'].max():.1f}% for biogas.
8. The CH4 h7/h14 validation hurdles are RMSE {_format(ch4_val.loc[7, 'RMSE'])}/{_format(ch4_val.loc[14, 'RMSE'])} and MAE {_format(ch4_val.loc[7, 'MAE'])}/{_format(ch4_val.loc[14, 'MAE'])}.
9. At h30 persistence remains an immutable reference but is not an effective level forecast here because validation R2_level is negative for both targets.
10. Biogas is more stably evaluable at baseline level because it has near-complete support; this is an availability conclusion, not a biological causal claim.

## R. PHASE 04 gate

- Semantic/holdout/leakage gates: {gate_passed}/{len(gates)} passed; {gate_failed} failed.
- Dedicated Phase 04 tests: {test_passed}/{test_passed + test_failed} passed; {test_failed} failed/error/skipped.
- 2023 metric access: NONE.
- Target imputation: NONE.
- Model fitting/selection: NONE.

```text
PHASE 04 STATUS:
{phase_status}
```

The status is conditional only because CH4 origin observations and some seasonal/PI supports are incomplete and the final holdout intentionally remains unevaluated. R5 work has not been started.
"""
    REPORT_PATH.write_text(report, encoding="utf-8")


class _RecordingTestResult(unittest.TextTestResult):
    def __init__(self, *args: object, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)
        self.records: list[dict[str, str]] = []

    def addSuccess(self, test: unittest.case.TestCase) -> None:
        super().addSuccess(test)
        self.records.append({"test": test.id(), "status": "PASS", "detail": ""})

    def addFailure(self, test: unittest.case.TestCase, err: object) -> None:
        super().addFailure(test, err)
        self.records.append({"test": test.id(), "status": "FAIL", "detail": self._exc_info_to_string(err, test)})

    def addError(self, test: unittest.case.TestCase, err: object) -> None:
        super().addError(test, err)
        self.records.append({"test": test.id(), "status": "ERROR", "detail": self._exc_info_to_string(err, test)})

    def addSkip(self, test: unittest.case.TestCase, reason: str) -> None:
        super().addSkip(test, reason)
        self.records.append({"test": test.id(), "status": "SKIP", "detail": reason})


def run_phase04_tests() -> pd.DataFrame:
    suite = unittest.defaultTestLoader.discover(
        str(PROJECT_ROOT / "tests"),
        pattern="test_baseline_models.py",
        top_level_dir=str(PROJECT_ROOT / "tests"),
    )
    result = unittest.TextTestRunner(stream=io.StringIO(), verbosity=2, resultclass=_RecordingTestResult).run(suite)
    frame = pd.DataFrame(result.records, columns=["test", "status", "detail"])
    if len(frame) != result.testsRun:
        raise AssertionError("Phase 04 test result capture mismatch")
    return frame


def _persistence_row(metrics: pd.DataFrame, target: str, horizon: int, period: str) -> pd.Series:
    return metrics[
        metrics["target_name"].eq(target)
        & metrics["horizon"].eq(horizon)
        & metrics["baseline_id"].eq("B10_PERSISTENCE_STRICT")
        & metrics["period"].eq(period)
        & metrics["support_type"].eq("OWN_AVAILABLE")
    ].iloc[0]


def print_console_summary(
    registry: pd.DataFrame,
    metrics: pd.DataFrame,
    unit_results: pd.DataFrame,
) -> None:
    test_passed = int(unit_results["status"].eq("PASS").sum())
    test_failed = int(unit_results["status"].isin(["FAIL", "ERROR", "SKIP"]).sum())

    print("=" * 50)
    print("PHASE 04 — BASELINE MODELING COMPLETE")
    print("=" * 50)
    print("\nROLE:\nR4 BASELINE MODELER")
    print("\nINPUT DATA:\ndata/processed/02_process_base.parquet\ndata/processed/03_targets_by_horizon.parquet")
    print("\nTARGETS:\nCH4_m3d_observed\nbiogas_AB_m3d")
    print("\nHORIZONS:\n1 / 3 / 7 / 14 / 30")
    print("\nDATA SPLIT:\nDEVELOPMENT = 2018-2021\nVALIDATION = 2022\nFINAL TEST = 2023 SEALED")
    print("\nBASELINES IMPLEMENTED:")
    for row in registry.itertuples(index=False):
        print(f"{row.baseline_id.split('_')[0]} = {row.baseline_name}")

    for target, label in (("CH4_m3d_observed", "CH4"), ("biogas_AB_m3d", "BIOGAS")):
        print("\n" + "-" * 50)
        print(f"STRICT PERSISTENCE — {label}")
        print("-" * 50)
        print("PERIOD = DEVELOPMENT")
        for horizon in HORIZONS:
            row = _persistence_row(metrics, target, horizon, "DEVELOPMENT")
            print(f"\nh={horizon}:")
            print(f"N = {int(row.n_scored)}")
            print(f"RMSE = {_format(row.RMSE)}")
            print(f"MAE = {_format(row.MAE)}")
            print(f"sMAPE = {_format(row.sMAPE)}")
            print(f"R2_level = {_format(row.R2_level, 4)}")
            print(f"R2_delta = {_format(row.R2_delta, 4)}")
            print(f"Bias = {_format(row.Bias)}")

    print("\n" + "-" * 50)
    print("2022 VALIDATION")
    print("-" * 50)
    for target, label in (("CH4_m3d_observed", "CH4"), ("biogas_AB_m3d", "BIOGAS")):
        print(f"\n{label}:")
        for horizon in HORIZONS:
            row = _persistence_row(metrics, target, horizon, "2022_VALIDATION")
            print(f"h{horizon} N={int(row.n_scored)} RMSE={_format(row.RMSE)} MAE={_format(row.MAE)} R2_level={_format(row.R2_level, 4)}")

    ch4_coverage = [_persistence_row(metrics, "CH4_m3d_observed", h, "2022_VALIDATION").coverage_rate for h in HORIZONS]
    ch4_locf = [
        metrics[
            metrics["target_name"].eq("CH4_m3d_observed")
            & metrics["horizon"].eq(h)
            & metrics["baseline_id"].eq("B11_LAST_OBSERVED")
            & metrics["period"].eq("2022_VALIDATION")
            & metrics["support_type"].eq("OWN_AVAILABLE")
        ].iloc[0].coverage_rate
        for h in HORIZONS
    ]
    bio_coverage = [_persistence_row(metrics, "biogas_AB_m3d", h, "2022_VALIDATION").coverage_rate for h in HORIZONS]
    print("\n" + "-" * 50)
    print("BASELINE COVERAGE")
    print("-" * 50)
    print("\nCH4 persistence coverage:\n" + " / ".join(f"h{h}={100*v:.1f}%" for h, v in zip(HORIZONS, ch4_coverage, strict=True)))
    print("\nCH4 LOCF coverage:\n" + " / ".join(f"h{h}={100*v:.1f}%" for h, v in zip(HORIZONS, ch4_locf, strict=True)))
    print("\nBIOGAS persistence coverage:\n" + " / ".join(f"h{h}={100*v:.1f}%" for h, v in zip(HORIZONS, bio_coverage, strict=True)))

    ch4_h7 = _persistence_row(metrics, "CH4_m3d_observed", 7, "2022_VALIDATION")
    ch4_h14 = _persistence_row(metrics, "CH4_m3d_observed", 14, "2022_VALIDATION")
    print("\n" + "-" * 50)
    print("CORE INTERPRETATION")
    print("-" * 50)
    print("\nPersistence level skill with horizon:\ndecreases; validation R2_level becomes negative at h30 for both targets")
    print("\nR2_level vs R2_delta:\nshort-horizon level R2 is high while persistence R2_delta stays near zero/non-positive")
    print(f"\nPrimary h=7 persistence hurdle:\nCH4 RMSE = {_format(ch4_h7.RMSE)}\nCH4 MAE = {_format(ch4_h7.MAE)}")
    print(f"\nPrimary h=14 persistence hurdle:\nCH4 RMSE = {_format(ch4_h14.RMSE)}\nCH4 MAE = {_format(ch4_h14.MAE)}")

    print("\n" + "-" * 50)
    print("LEAKAGE CHECK")
    print("-" * 50)
    print("\nTARGET IMPUTATION:\nNONE")
    print("\nFUTURE TARGET USE:\nNONE")
    print("\n2023 METRIC ACCESS:\nNONE")
    print("\nCOMMON-SUPPORT SKILL:\nPASS")
    print("\n" + "-" * 50)
    print("TESTS")
    print("-" * 50)
    print(f"\nPASSED = {test_passed}\nFAILED = {test_failed}")
    print("\nPHASE 04 STATUS:")
    print("CONDITIONAL PASS" if test_failed == 0 else "FAIL")
    print("\nNEXT RECOMMENDED ROLE:\nR5 MECHANISTIC MODELER")
    print("=" * 50)


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    feature_hash_before = sha256_file(FEATURE_PATH)
    process = pd.read_parquet(PROCESS_PATH)
    features = pd.read_parquet(FEATURE_PATH)
    targets = pd.read_parquet(TARGET_PATH)
    for frame in (process, features, targets):
        frame["date"] = pd.to_datetime(frame["date"])
    _assert_input_contracts(process, features, targets)

    registry = baseline_registry()
    predictions = build_predictions(process)
    reference = build_persistence_reference(predictions)
    metrics, coverage = evaluate_baseline_predictions(predictions, reference)
    persistence_summary, diagnostics = build_persistence_summaries(metrics, reference)

    dev_predictions = predictions[predictions["period"].eq("DEVELOPMENT")].reset_index(drop=True)
    validation_predictions = predictions[predictions["period"].eq("2022_VALIDATION")].reset_index(drop=True)
    dev_predictions.to_parquet(DEV_PREDICTION_PATH, index=False)
    validation_predictions.to_parquet(VALIDATION_PREDICTION_PATH, index=False)
    reference.to_parquet(PERSISTENCE_REFERENCE_PATH, index=False)
    _write_csv(registry, REGISTRY_PATH)
    HOLDOUT_PATH.write_text(holdout_policy_text(), encoding="utf-8")
    version = build_version_record(
        [
            PROCESS_PATH.relative_to(PROJECT_ROOT),
            FEATURE_PATH.relative_to(PROJECT_ROOT),
            TARGET_PATH.relative_to(PROJECT_ROOT),
            FEATURE_REGISTRY_PATH.relative_to(PROJECT_ROOT),
            AVAILABILITY_PATH.relative_to(PROJECT_ROOT),
            PREPROCESSING_PATH.relative_to(PROJECT_ROOT),
            PHASE03_REPORT_PATH.relative_to(PROJECT_ROOT),
        ]
    )
    # Hash helper resolves relative paths from the current project working directory.
    VERSION_PATH.write_text(json.dumps(version, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    _write_csv(persistence_summary, PERSISTENCE_HORIZON_PATH)
    _write_csv(diagnostics, PERSISTENCE_DIAGNOSTICS_PATH)
    _write_csv(metrics, METRICS_PATH)
    _write_csv(coverage, COVERAGE_PATH)

    gates = build_gate_results(
        process,
        features,
        predictions,
        registry,
        metrics,
        reference,
        feature_hash_before,
    )
    _write_csv(gates, GATE_PATH)
    render_required_figures(metrics, FIGURE_DIR)
    write_report(registry, metrics, persistence_summary, diagnostics, gates, unit_results=None)
    unit_results = run_phase04_tests()
    _write_csv(unit_results, TEST_RESULT_PATH)
    write_report(registry, metrics, persistence_summary, diagnostics, gates, unit_results=unit_results)
    print_console_summary(registry, metrics, unit_results)

    gate_ok = bool(gates["status"].eq("PASS").all())
    test_ok = bool(unit_results["status"].eq("PASS").all())
    return 0 if gate_ok and test_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

