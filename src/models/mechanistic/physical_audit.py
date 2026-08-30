"""Physical and fitted-parameter audit tables for PHASE 05."""

from __future__ import annotations

import numpy as np
import pandas as pd


def build_intercept_audit(predictions: pd.DataFrame) -> pd.DataFrame:
    """Summarize the nonnegative intercept without assigning it a cause."""

    required = {
        "target_name",
        "horizon",
        "model_id",
        "outer_fold",
        "beta0",
        "y_true",
        "y_pred",
    }
    missing = required.difference(predictions.columns)
    if missing:
        raise KeyError(f"predictions missing intercept-audit columns: {sorted(missing)}")
    rows: list[dict[str, object]] = []
    keys = ["target_name", "horizon", "model_id", "outer_fold"]
    for key, group in predictions.groupby(keys, sort=True, dropna=False):
        target, horizon, model, fold = key
        beta0_values = pd.to_numeric(group["beta0"], errors="coerce").dropna()
        beta0 = float(beta0_values.iloc[0]) if len(beta0_values) else float("nan")
        mean_y = float(pd.to_numeric(group["y_true"], errors="coerce").mean())
        mean_prediction = float(pd.to_numeric(group["y_pred"], errors="coerce").mean())
        beta0_over_y = beta0 / mean_y if np.isfinite(mean_y) and mean_y != 0 else np.nan
        beta0_over_prediction = (
            beta0 / mean_prediction
            if np.isfinite(mean_prediction) and mean_prediction != 0
            else np.nan
        )
        rows.append(
            {
                "target": target,
                "horizon": int(horizon),
                "model": model,
                "fold": fold,
                "beta0": beta0,
                "mean_y": mean_y,
                "mean_prediction": mean_prediction,
                "beta0_over_mean_y": beta0_over_y,
                "beta0_over_mean_prediction": beta0_over_prediction,
                "interpretation_status": "UNEXPLAINED_BACKGROUND_COMPONENT",
            }
        )
    return pd.DataFrame(rows)


def _ch4_biogas_violation_counts(predictions: pd.DataFrame) -> pd.DataFrame:
    """Return soft CH4>biogas counts on exact common forecast keys."""

    common = ["origin_date", "target_date", "horizon", "model_id", "outer_fold"]
    usable = predictions[
        predictions["target_name"].isin(["CH4_m3d_observed", "biogas_AB_m3d"])
    ].copy()
    pivot = usable.pivot_table(
        index=common,
        columns="target_name",
        values="y_pred",
        aggfunc="first",
    ).reset_index()
    if not {"CH4_m3d_observed", "biogas_AB_m3d"}.issubset(pivot.columns):
        return pd.DataFrame(columns=["horizon", "model_id", "outer_fold", "CH4_gt_biogas_count"])
    finite = np.isfinite(pd.to_numeric(pivot["CH4_m3d_observed"], errors="coerce")) & np.isfinite(
        pd.to_numeric(pivot["biogas_AB_m3d"], errors="coerce")
    )
    pivot["violation"] = finite & pivot["CH4_m3d_observed"].gt(pivot["biogas_AB_m3d"])
    return (
        pivot.groupby(["horizon", "model_id", "outer_fold"], sort=True)["violation"]
        .sum()
        .rename("CH4_gt_biogas_count")
        .reset_index()
    )


def build_physical_audit(
    predictions: pd.DataFrame,
    kernel_audit: pd.DataFrame,
) -> pd.DataFrame:
    """Audit clipping, coefficient signs, kernels, and locked UNKNOWN inputs."""

    required = {
        "target_name",
        "horizon",
        "model_id",
        "outer_fold",
        "raw_prediction",
        "y_pred",
        "beta0",
        "beta1",
    }
    missing = required.difference(predictions.columns)
    if missing:
        raise KeyError(f"predictions missing physical-audit columns: {sorted(missing)}")
    kernel_negative = int((~kernel_audit["nonnegative"].astype(bool)).sum())
    normalization_error = float(pd.to_numeric(kernel_audit["normalization_error"], errors="coerce").max())
    violation = _ch4_biogas_violation_counts(predictions)
    rows: list[dict[str, object]] = []
    keys = ["target_name", "horizon", "model_id", "outer_fold"]
    for key, group in predictions.groupby(keys, sort=True, dropna=False):
        target, horizon, model, fold = key
        raw = pd.to_numeric(group["raw_prediction"], errors="coerce")
        clipped = pd.to_numeric(group["y_pred"], errors="coerce")
        coefficient_columns = [column for column in ("beta0", "beta1", "beta_fast", "beta_slow") if column in group]
        coefficient_values = pd.concat(
            [pd.to_numeric(group[column], errors="coerce") for column in coefficient_columns],
            ignore_index=True,
        )
        match = violation[
            violation["horizon"].eq(horizon)
            & violation["model_id"].eq(model)
            & violation["outer_fold"].eq(fold)
        ]
        intercept = float(pd.to_numeric(group["beta0"], errors="coerce").dropna().iloc[0]) if group["beta0"].notna().any() else np.nan
        mean_prediction = float(clipped.mean())
        ratio = intercept / mean_prediction if np.isfinite(intercept) and mean_prediction > 0 else np.nan
        rows.append(
            {
                "target": target,
                "horizon": int(horizon),
                "model": model,
                "fold": fold,
                "negative_raw_predictions": int(raw.lt(0).sum()),
                "clipped_predictions": int((raw.lt(0) & clipped.eq(0)).sum()),
                "beta_negative": int(coefficient_values.lt(-1.0e-12).sum()),
                "kernel_negative": kernel_negative,
                "kernel_normalization_error": normalization_error,
                "CH4_gt_biogas_count": int(match["CH4_gt_biogas_count"].iloc[0]) if len(match) else 0,
                "intercept_flag": "LARGE_UNEXPLAINED_BACKGROUND" if np.isfinite(ratio) and ratio >= 0.5 else "UNEXPLAINED_BACKGROUND",
                "VS_regime_dependency": bool(str(model).startswith("M1VS")),
                "COD_bound_applicable": False,
                "HRT_dependency": False,
            }
        )
    return pd.DataFrame(rows)

