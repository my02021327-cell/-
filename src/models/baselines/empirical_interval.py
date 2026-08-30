"""Leakage-safe empirical residual prediction intervals."""

from __future__ import annotations

from numbers import Integral, Real
from typing import Iterable, Sequence

import numpy as np
import pandas as pd


DEFAULT_LEVELS: tuple[float, ...] = (0.80, 0.95)
MIN_RESIDUALS = 30
DEFAULT_GROUP_COLUMNS: tuple[str, ...] = (
    "target_name",
    "horizon",
    "baseline_id",
)


def _validate_levels(levels: Sequence[float]) -> list[tuple[float, int]]:
    if not levels:
        raise ValueError("levels must contain at least one interval level")
    validated: list[tuple[float, int]] = []
    labels: set[int] = set()
    for level in levels:
        if isinstance(level, bool) or not isinstance(level, Real):
            raise TypeError("each interval level must be numeric")
        value = float(level)
        if not np.isfinite(value) or not 0.0 < value < 1.0:
            raise ValueError("each interval level must be finite and between 0 and 1")
        label = int(round(100.0 * value))
        if not np.isclose(value, label / 100.0) or label in labels:
            raise ValueError("interval levels must map to unique whole percentages")
        labels.add(label)
        validated.append((value, label))
    return validated


def empirical_residual_bounds(
    y_pred: float,
    residuals: Iterable[float],
    level: float,
    min_residuals: int = MIN_RESIDUALS,
) -> tuple[float, float]:
    """Return an asymmetric central empirical interval around one prediction.

    Quantiles use NumPy's deterministic linear interpolation.  ``NaN`` bounds
    are returned until at least ``min_residuals`` finite residuals exist.
    """

    validated_level = _validate_levels((level,))[0][0]
    if isinstance(min_residuals, bool) or not isinstance(min_residuals, Integral):
        raise TypeError("min_residuals must be a non-boolean integer")
    minimum = int(min_residuals)
    if minimum < 1:
        raise ValueError("min_residuals must be >= 1")
    prediction = float(y_pred)
    if not np.isfinite(prediction):
        return np.nan, np.nan
    values = np.asarray(list(residuals), dtype=float)
    if values.ndim != 1:
        raise ValueError("residuals must be one-dimensional")
    if np.any(np.isinf(values)):
        raise ValueError("residuals may contain finite numbers or NaN only")
    values = values[~np.isnan(values)]
    if len(values) < minimum:
        return np.nan, np.nan
    alpha = (1.0 - validated_level) / 2.0
    lower_q, upper_q = np.quantile(
        values,
        [alpha, 1.0 - alpha],
        method="linear",
    )
    return prediction + float(lower_q), prediction + float(upper_q)


def add_causal_empirical_intervals(
    predictions: pd.DataFrame,
    *,
    group_columns: Sequence[str] = DEFAULT_GROUP_COLUMNS,
    levels: Sequence[float] = DEFAULT_LEVELS,
    min_residuals: int = MIN_RESIDUALS,
    origin_column: str = "origin_date",
    target_date_column: str = "target_date",
    prediction_column: str = "y_pred",
    truth_column: str = "y_true",
) -> pd.DataFrame:
    """Add rowwise empirical PIs using only residuals matured by each origin.

    Residuals are grouped by target, horizon, and baseline by default.  For a
    forecast issued at origin ``t``, the eligible historical set is exactly
    rows in the same group with finite ``y_true-y_pred`` and
    ``residual_target_date <= t``.  The returned
    ``max_residual_target_date_used`` column makes that leakage rule auditable.
    Input row order is preserved.
    """

    if not isinstance(predictions, pd.DataFrame):
        raise TypeError("predictions must be a pandas DataFrame")
    groups = tuple(group_columns)
    if not groups or len(set(groups)) != len(groups):
        raise ValueError("group_columns must be non-empty and unique")
    required = {
        *groups,
        origin_column,
        target_date_column,
        prediction_column,
        truth_column,
    }
    missing = sorted(required - set(predictions.columns))
    if missing:
        raise ValueError(f"predictions missing required columns: {missing}")
    if isinstance(min_residuals, bool) or not isinstance(min_residuals, Integral):
        raise TypeError("min_residuals must be a non-boolean integer")
    minimum = int(min_residuals)
    if minimum < 1:
        raise ValueError("min_residuals must be >= 1")
    interval_levels = _validate_levels(levels)

    result = predictions.copy().reset_index(drop=True)
    try:
        result[origin_column] = pd.to_datetime(result[origin_column], errors="raise")
        result[target_date_column] = pd.to_datetime(
            result[target_date_column],
            errors="raise",
        )
    except (TypeError, ValueError) as exc:
        raise TypeError("origin and target dates must be valid datetimes") from exc
    for column in (origin_column, target_date_column):
        dates = pd.DatetimeIndex(result[column])
        if dates.tz is not None or dates.hasnans or not dates.equals(dates.normalize()):
            raise ValueError(f"{column} must contain timezone-naive midnight dates")
    if (result[target_date_column] <= result[origin_column]).any():
        raise ValueError("each residual target date must be later than its origin")
    prediction_key = [*groups, origin_column, target_date_column]
    duplicated = result.duplicated(prediction_key, keep=False)
    if duplicated.any():
        raise ValueError(
            "predictions must be unique by residual group, origin, and target date"
        )

    for column in (prediction_column, truth_column):
        try:
            result[column] = pd.to_numeric(result[column], errors="raise").astype(float)
        except (TypeError, ValueError) as exc:
            raise TypeError(f"{column} must be numeric or NaN") from exc
        values = result[column].to_numpy(dtype=float)
        if np.any(~np.isfinite(values) & ~np.isnan(values)):
            raise ValueError(f"{column} may contain finite numbers or NaN only")

    result["n_residuals_available"] = np.zeros(len(result), dtype=int)
    result["min_residual_target_date_used"] = pd.NaT
    result["max_residual_target_date_used"] = pd.NaT
    for _, label in interval_levels:
        result[f"PI{label}_lower"] = np.nan
        result[f"PI{label}_upper"] = np.nan

    result["_row_order"] = np.arange(len(result), dtype=int)
    grouped = result.groupby(list(groups), sort=False, dropna=False)
    for _, group in grouped:
        candidates = group.loc[
            group[prediction_column].notna() & group[truth_column].notna(),
            [target_date_column, prediction_column, truth_column],
        ].copy()
        candidates["_residual"] = (
            candidates[truth_column] - candidates[prediction_column]
        )
        candidates = candidates.sort_values(target_date_column, kind="mergesort")
        candidate_dates = candidates[target_date_column].to_numpy(
            dtype="datetime64[ns]"
        )
        candidate_residuals = candidates["_residual"].to_numpy(dtype=float)

        ordered_rows = group.sort_values(
            [origin_column, target_date_column, "_row_order"],
            kind="mergesort",
        )
        for row_index in ordered_rows.index:
            origin = result.at[row_index, origin_column]
            cutoff = int(
                np.searchsorted(
                    candidate_dates,
                    np.datetime64(origin),
                    side="right",
                )
            )
            matured = candidate_residuals[:cutoff]
            result.at[row_index, "n_residuals_available"] = cutoff
            if cutoff:
                result.at[row_index, "min_residual_target_date_used"] = pd.Timestamp(
                    candidate_dates[0]
                )
                result.at[row_index, "max_residual_target_date_used"] = pd.Timestamp(
                    candidate_dates[cutoff - 1]
                )
            if cutoff < minimum:
                continue
            prediction = result.at[row_index, prediction_column]
            if not np.isfinite(prediction):
                continue
            for level, label in interval_levels:
                lower, upper = empirical_residual_bounds(
                    prediction,
                    matured,
                    level,
                    minimum,
                )
                result.at[row_index, f"PI{label}_lower"] = lower
                result.at[row_index, f"PI{label}_upper"] = upper

    return result.drop(columns="_row_order")


causal_empirical_intervals = add_causal_empirical_intervals
