"""Phase 06 metrics, incremental comparisons, stability, and candidate locks."""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd

from ...validation.baseline_metrics import compute_metrics, skill_vs_persistence
from .contracts import (
    FEATURE_SETS,
    MODEL_FAMILIES,
    TRACK_PROVISIONAL,
    TRACK_STRICT,
)


SUPPORT_TYPES = ("OWN_AVAILABLE", "PERSISTENCE_PAIRED")
PERIODS: dict[str, tuple[pd.Timestamp, pd.Timestamp]] = {
    "DEVELOPMENT": (pd.Timestamp("2018-01-01"), pd.Timestamp("2021-12-31")),
    "2018": (pd.Timestamp("2018-01-01"), pd.Timestamp("2018-12-31")),
    "2019": (pd.Timestamp("2019-01-01"), pd.Timestamp("2019-12-31")),
    "2020": (pd.Timestamp("2020-01-01"), pd.Timestamp("2020-12-31")),
    "2021": (pd.Timestamp("2021-01-01"), pd.Timestamp("2021-12-31")),
}
CONFIG_COLUMNS = [
    "target_name",
    "horizon",
    "model_id",
    "feature_set",
    "availability_track",
]


def _finite(series: pd.Series) -> pd.Series:
    return pd.Series(
        np.isfinite(pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)),
        index=series.index,
    )


def _period_mask(frame: pd.DataFrame, period: str) -> pd.Series:
    start, end = PERIODS[period]
    origin = pd.to_datetime(frame["origin_date"], errors="raise")
    target = pd.to_datetime(frame["target_date"], errors="raise")
    return origin.between(start, end, inclusive="both") & target.between(
        start, end, inclusive="both"
    )


def _evaluate_group(frame: pd.DataFrame, support_type: str) -> dict[str, Any]:
    target_available = _finite(frame["y_true"]) & frame[
        "target_available"
    ].fillna(False).astype(bool)
    prediction_available = _finite(frame["y_pred"]) & frame[
        "prediction_available"
    ].fillna(False).astype(bool)
    persistence_available = _finite(frame["y_persistence"]) & frame[
        "persistence_available"
    ].fillna(False).astype(bool)
    own = target_available & prediction_available
    scored = own if support_type == "OWN_AVAILABLE" else own & persistence_available
    selected = frame.loc[scored]
    metrics = compute_metrics(
        selected["y_true"],
        selected["y_pred"],
        y_origin=selected["y_origin"],
    )
    skill = (
        skill_vs_persistence(
            selected["y_true"], selected["y_pred"], selected["y_persistence"]
        )
        if support_type == "PERSISTENCE_PAIRED"
        else np.nan
    )
    n_origins = int(len(frame))
    n_target = int(target_available.sum())
    n_predicted = int(prediction_available.sum())
    n_persistence = int(persistence_available.sum())
    return {
        "n_origins": n_origins,
        "n_target": n_target,
        "n_predicted": n_predicted,
        "n_persistence": n_persistence,
        "n": int(metrics["n_scored"]),
        "coverage": float(metrics["n_scored"] / n_target) if n_target else np.nan,
        "prediction_coverage": float(n_predicted / n_origins) if n_origins else np.nan,
        "target_coverage": float(n_target / n_origins) if n_origins else np.nan,
        "persistence_coverage": float(n_persistence / n_origins) if n_origins else np.nan,
        "n_delta": int(metrics["n_delta_scored"]),
        "MSE": metrics["MSE"],
        "RMSE": metrics["RMSE"],
        "MAE": metrics["MAE"],
        "sMAPE": metrics["sMAPE"],
        "R2_level": metrics["R2_level"],
        "R2_delta": metrics["R2_delta"],
        "Bias": metrics["Bias"],
        "Skill_vs_persistence": skill,
        "negative_prediction_count": int(
            frame.get("negative_prediction_flag", False).fillna(False).astype(bool).sum()
            if isinstance(frame.get("negative_prediction_flag"), pd.Series)
            else 0
        ),
        "mean_n_features": float(
            pd.to_numeric(frame.get("n_features"), errors="coerce").mean()
        )
        if "n_features" in frame
        else np.nan,
    }


def evaluate_oof(predictions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return aggregate and outer-fold metrics on own and exact paired support."""

    if bool(
        pd.to_datetime(predictions["origin_date"]).dt.year.gt(2021).any()
        or pd.to_datetime(predictions["target_date"]).dt.year.gt(2021).any()
    ):
        raise ValueError("Phase 06 metrics cannot include post-2021 rows")
    aggregate_rows: list[dict[str, Any]] = []
    fold_rows: list[dict[str, Any]] = []
    for key, group in predictions.groupby(CONFIG_COLUMNS, sort=True, dropna=False):
        identifiers = dict(zip(CONFIG_COLUMNS, key, strict=True))
        for period in PERIODS:
            period_frame = group.loc[_period_mask(group, period)]
            for support_type in SUPPORT_TYPES:
                aggregate_rows.append(
                    {
                        **identifiers,
                        "target": identifiers["target_name"],
                        "period": period,
                        "support_type": support_type,
                        **_evaluate_group(period_frame, support_type),
                    }
                )
        for outer_fold, fold in group.groupby("outer_fold", sort=True):
            for support_type in SUPPORT_TYPES:
                fold_rows.append(
                    {
                        **identifiers,
                        "target": identifiers["target_name"],
                        "outer_fold": outer_fold,
                        "support_type": support_type,
                        **_evaluate_group(fold, support_type),
                    }
                )
    return pd.DataFrame(aggregate_rows), pd.DataFrame(fold_rows)


def _common_metrics(frame: pd.DataFrame, prediction_column: str) -> dict[str, Any]:
    valid = _finite(frame["y_true"]) & _finite(frame[prediction_column])
    scored = frame.loc[valid]
    metrics = compute_metrics(
        scored["y_true"], scored[prediction_column], y_origin=scored["y_origin"]
    )
    paired = scored.loc[
        scored["persistence_available"].fillna(False).astype(bool)
        & _finite(scored["y_persistence"])
    ]
    skill = skill_vs_persistence(
        paired["y_true"], paired[prediction_column], paired["y_persistence"]
    )
    return {**metrics, "Skill": skill, "paired_n": int(len(paired))}


def build_feature_set_comparison(
    predictions: pd.DataFrame,
    fold_metrics: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    steps = list(zip(FEATURE_SETS[:-1], FEATURE_SETS[1:], strict=True))
    grouping = ["target_name", "horizon", "model_id", "availability_track"]
    key = ["origin_date", "target_date", "outer_fold"]
    for identifiers, group in predictions.groupby(grouping, sort=True, dropna=False):
        base_id = dict(zip(grouping, identifiers, strict=True))
        for from_set, to_set in steps:
            before = group.loc[group["feature_set"].eq(from_set)].copy()
            after = group.loc[group["feature_set"].eq(to_set)].copy()
            merged = before.merge(
                after,
                on=key,
                how="inner",
                suffixes=("_before", "_after"),
                validate="one_to_one",
            )
            if merged.empty:
                continue
            common = merged.loc[
                _finite(merged["y_true_before"])
                & _finite(merged["y_pred_before"])
                & _finite(merged["y_pred_after"])
            ].copy()
            common = common.assign(
                y_true=common["y_true_before"],
                y_origin=common["y_origin_before"],
                y_persistence=common["y_persistence_before"],
                persistence_available=common["persistence_available_before"],
            )
            before_metrics = _common_metrics(common, "y_pred_before")
            after_metrics = _common_metrics(common, "y_pred_after")
            improvement_flags: list[bool] = []
            for _, fold in common.groupby("outer_fold", sort=True):
                left = _common_metrics(fold.assign(
                    y_true=fold["y_true_before"],
                    y_origin=fold["y_origin_before"],
                    y_persistence=fold["y_persistence_before"],
                    persistence_available=fold["persistence_available_before"],
                ), "y_pred_before")
                right = _common_metrics(fold.assign(
                    y_true=fold["y_true_before"],
                    y_origin=fold["y_origin_before"],
                    y_persistence=fold["y_persistence_before"],
                    persistence_available=fold["persistence_available_before"],
                ), "y_pred_after")
                if np.isfinite(left["RMSE"]) and np.isfinite(right["RMSE"]):
                    improvement_flags.append(right["RMSE"] < left["RMSE"])
            delta_rmse = after_metrics["RMSE"] - before_metrics["RMSE"]
            delta_mae = after_metrics["MAE"] - before_metrics["MAE"]
            delta_skill = after_metrics["Skill"] - before_metrics["Skill"]
            delta_r2 = after_metrics["R2_delta"] - before_metrics["R2_delta"]
            improving_fraction = (
                float(np.mean(improvement_flags)) if improvement_flags else np.nan
            )
            if not np.isfinite(delta_rmse) or len(common) < 20:
                status = "INSUFFICIENT_COMMON_SUPPORT"
            elif delta_rmse < 0 and improving_fraction >= 0.6:
                status = "GLOBAL_BENEFIT"
            elif delta_rmse < 0:
                status = "POTENTIAL_REGIME_SPECIFIC_VALUE"
            else:
                status = "NO_INCREMENTAL_VALUE"
            before_coverage = float(
                before["prediction_available"].fillna(False).astype(bool).mean()
            )
            after_coverage = float(
                after["prediction_available"].fillna(False).astype(bool).mean()
            )
            rows.append(
                {
                    **base_id,
                    "target": base_id["target_name"],
                    "model": base_id["model_id"],
                    "from_set": from_set,
                    "to_set": to_set,
                    "common_n": int(len(common)),
                    "RMSE_before": before_metrics["RMSE"],
                    "RMSE_after": after_metrics["RMSE"],
                    "delta_RMSE": delta_rmse,
                    "MAE_before": before_metrics["MAE"],
                    "MAE_after": after_metrics["MAE"],
                    "delta_MAE": delta_mae,
                    "Skill_before": before_metrics["Skill"],
                    "Skill_after": after_metrics["Skill"],
                    "delta_Skill": delta_skill,
                    "R2_delta_before": before_metrics["R2_delta"],
                    "R2_delta_after": after_metrics["R2_delta"],
                    "delta_R2_delta": delta_r2,
                    "improving_fold_fraction": improving_fraction,
                    "coverage_before": before_coverage,
                    "coverage_after": after_coverage,
                    "coverage_change": after_coverage - before_coverage,
                    "incremental_status": status,
                }
            )
    return pd.DataFrame(rows)


def _distribution(values: pd.Series) -> str:
    counts = values.astype(str).value_counts(dropna=False).sort_index()
    return json.dumps({str(key): int(value) for key, value in counts.items()}, sort_keys=True)


def build_hyperparameter_stability(hyperparameters: pd.DataFrame) -> pd.DataFrame:
    selected = hyperparameters.loc[
        hyperparameters["selected"].fillna(False).astype(bool)
    ].copy()
    rows: list[dict[str, Any]] = []
    grouping = ["target", "horizon", "model_id", "availability_track"]
    for key, group in selected.groupby(grouping, sort=True, dropna=False):
        identifiers = dict(zip(grouping, key, strict=True))
        parameter_dicts = [json.loads(value) for value in group["parameters"].astype(str)]
        parameter_names = sorted({name for params in parameter_dicts for name in params})
        record = {
            **identifiers,
            "model": identifiers["model_id"],
            "feature_set": "SET-F_TUNING_ANCHOR",
            "selected_candidate_distribution": _distribution(group["candidate_id"]),
            "n_outer_folds": int(group["outer_fold"].nunique()),
            "modal_candidate_frequency": float(
                group["candidate_id"].value_counts(normalize=True).max()
            ),
        }
        for name in parameter_names:
            record[f"selected_{name}_distribution"] = _distribution(
                pd.Series([params.get(name, "NA") for params in parameter_dicts])
            )
        rows.append(record)
    return pd.DataFrame(rows)


def build_feature_stability(
    selected_features: pd.DataFrame,
    importance_by_fold: pd.DataFrame,
) -> pd.DataFrame:
    selection_group = [
        "target",
        "horizon",
        "model",
        "feature_set",
        "availability_track",
        "feature",
    ]
    selection = (
        selected_features.groupby(selection_group, as_index=False)
        .agg(
            selection_frequency=("selected", "mean"),
            n_outer_folds=("outer_fold", "nunique"),
            selection_method=("selection_method", "first"),
        )
    )
    importance = (
        importance_by_fold.groupby(selection_group, as_index=False)
        .agg(
            importance_median=("importance", "median"),
            importance_q25=("importance", lambda x: float(np.nanquantile(x, 0.25))),
            importance_q75=("importance", lambda x: float(np.nanquantile(x, 0.75))),
            signed_importance_median=("signed_importance", "median"),
            sign_consistency=(
                "signed_importance",
                lambda x: float(
                    max((pd.to_numeric(x, errors="coerce") > 0).mean(),
                        (pd.to_numeric(x, errors="coerce") < 0).mean())
                ),
            ),
            importance_method=("importance_method", "first"),
        )
    )
    result = selection.merge(importance, on=selection_group, how="left")
    result["importance_IQR"] = result["importance_q75"] - result["importance_q25"]
    horizon_group = [
        "target",
        "model",
        "feature_set",
        "availability_track",
        "feature",
    ]
    horizon_consistency = (
        result.assign(_stable=result["selection_frequency"].ge(0.5))
        .groupby(horizon_group, as_index=False)
        .agg(horizon_consistency=("_stable", "mean"))
    )
    return result.merge(horizon_consistency, on=horizon_group, how="left")


def choose_locked_configurations(
    metrics: pd.DataFrame,
    fold_metrics: pd.DataFrame,
) -> pd.DataFrame:
    development = metrics.loc[
        metrics["period"].eq("DEVELOPMENT")
        & metrics["support_type"].eq("OWN_AVAILABLE")
        & np.isfinite(pd.to_numeric(metrics["RMSE"], errors="coerce"))
    ].copy()
    paired = metrics.loc[
        metrics["period"].eq("DEVELOPMENT")
        & metrics["support_type"].eq("PERSISTENCE_PAIRED"),
        CONFIG_COLUMNS + ["Skill_vs_persistence", "n"],
    ].rename(
        columns={
            "n": "paired_n",
            "Skill_vs_persistence": "paired_Skill_vs_persistence",
        }
    )
    development = development.drop(columns=["Skill_vs_persistence"])
    development = development.merge(
        paired, on=CONFIG_COLUMNS, how="left", validate="one_to_one"
    )
    development = development.rename(
        columns={"paired_Skill_vs_persistence": "Skill_vs_persistence"}
    )
    fold_own = fold_metrics.loc[
        fold_metrics["support_type"].eq("OWN_AVAILABLE")
    ]
    stability = (
        fold_own.groupby(CONFIG_COLUMNS, as_index=False)
        .agg(
            fold_RMSE_std=("RMSE", "std"),
            fold_RMSE_median=("RMSE", "median"),
            improving_fold_count=("Skill_vs_persistence", lambda _: 0),
        )
    )
    development = development.merge(stability, on=CONFIG_COLUMNS, how="left")
    set_rank = {name: index for index, name in enumerate(FEATURE_SETS)}
    choices: list[pd.Series] = []
    group_cols = ["target_name", "horizon", "model_id", "availability_track"]
    for _, group in development.groupby(group_cols, sort=True, dropna=False):
        minimum = float(group["RMSE"].min())
        shortlist = group.loc[group["RMSE"].le(minimum * 1.01)].copy()
        shortlist["_set_rank"] = shortlist["feature_set"].map(set_rank)
        shortlist = shortlist.sort_values(
            ["_set_rank", "mean_n_features", "RMSE"], kind="stable"
        )
        choices.append(shortlist.iloc[0].drop(labels=["_set_rank"]))
    lock = pd.DataFrame(choices).reset_index(drop=True)
    lock["development_status"] = "OOF_COMPLETE"
    lock["R7_status"] = "RETAIN_AS_SENSITIVITY"
    lock["lock_reason"] = "PROVISIONAL_DEPLOYMENT_AVAILABILITY"

    for (target, horizon), group in lock.loc[
        lock["availability_track"].eq(TRACK_STRICT)
    ].groupby(["target_name", "horizon"], sort=True):
        linear = group.loc[group["model_id"].isin(["S00_RIDGE", "S01_ELASTIC_NET"])]
        if not linear.empty:
            winner = linear.sort_values(["RMSE", "model_id"], kind="stable").iloc[0]
            selected_index = winner.name
            lock.loc[linear.index, "R7_status"] = "RETAIN_AS_LINEAR_REFERENCE"
            lock.loc[linear.index, "lock_reason"] = "STABLE_REGULARIZED_LINEAR_REFERENCE"
            lock.loc[selected_index, "R7_status"] = "LOCK_FOR_R7"
            lock.loc[selected_index, "lock_reason"] = "BEST_STABLE_LINEAR_DEVELOPMENT_OOF"
        forest = group.loc[group["model_id"].eq("S10_RANDOM_FOREST")]
        if not forest.empty:
            lock.loc[forest.index, "R7_status"] = "LOCK_FOR_R7"
            lock.loc[forest.index, "lock_reason"] = "TREE_BAGGING_CANDIDATE"
        boosting = group.loc[group["model_id"].isin(["S20_XGBOOST", "S21_LIGHTGBM"])]
        if not boosting.empty:
            winner = boosting.sort_values(["RMSE", "model_id"], kind="stable").iloc[0]
            lock.loc[boosting.index, "R7_status"] = "RETAIN_AS_SENSITIVITY"
            lock.loc[boosting.index, "lock_reason"] = "BOOSTING_COMPLEXITY_REFERENCE"
            lock.loc[winner.name, "R7_status"] = "LOCK_FOR_R7"
            lock.loc[winner.name, "lock_reason"] = "BEST_STABLE_BOOSTING_DEVELOPMENT_OOF"
    lock["uses_target_history"] = False
    lock["final_winner_selected"] = False
    return lock.sort_values(
        ["target_name", "horizon", "availability_track", "model_id"], kind="stable"
    ).reset_index(drop=True)


__all__ = [
    "CONFIG_COLUMNS",
    "PERIODS",
    "SUPPORT_TYPES",
    "build_feature_set_comparison",
    "build_feature_stability",
    "build_hyperparameter_stability",
    "choose_locked_configurations",
    "evaluate_oof",
]
