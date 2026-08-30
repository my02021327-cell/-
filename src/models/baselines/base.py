"""Shared contracts for deterministic, target-history baseline forecasts.

The helpers in this module deliberately require a complete daily calendar.
Missing target measurements must be represented by ``NaN`` values, not by
missing rows.  This makes horizon and calendar-window semantics unambiguous.
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral
from typing import Any, Iterable

import numpy as np
import pandas as pd


HORIZONS: tuple[int, ...] = (1, 3, 7, 14, 30)

COMMON_PREDICTION_COLUMNS: tuple[str, ...] = (
    "origin_date",
    "target_date",
    "target_name",
    "horizon",
    "baseline_id",
    "y_origin",
    "origin_y_observed",
    "reference_date",
    "last_observation_age_d",
    "n_history_observed",
    "history_coverage_fraction",
    "oldest_history_date",
    "newest_history_date",
    "y_pred",
    "prediction_available",
)


@dataclass(frozen=True)
class ForecastInputs:
    """Validated target series, forecast origins, and horizon."""

    target: pd.Series
    origins: pd.DatetimeIndex
    horizon: int


def validate_horizon(horizon: int) -> int:
    """Validate and return one of the locked Phase 04 horizons."""

    if isinstance(horizon, bool) or not isinstance(horizon, Integral):
        raise TypeError("horizon must be a non-boolean integer")
    value = int(horizon)
    if value not in HORIZONS:
        raise ValueError(f"horizon must be one of {HORIZONS}")
    return value


def validate_daily_target(target: pd.Series) -> pd.Series:
    """Return a float copy of a complete, sorted daily target series.

    The index must be a unique, timezone-naive ``DatetimeIndex`` at midnight
    with no calendar gaps.  Non-finite observed values are rejected; ``NaN``
    remains the sole representation of an unobserved target and is never
    filled by this validator.
    """

    if not isinstance(target, pd.Series):
        raise TypeError("target must be a pandas Series")
    if not isinstance(target.index, pd.DatetimeIndex):
        raise TypeError("target index must be a pandas DatetimeIndex")
    if target.empty:
        raise ValueError("target must contain at least one daily row")

    index = target.index
    if index.tz is not None:
        raise ValueError("target index must be timezone-naive")
    if index.hasnans:
        raise ValueError("target index must not contain NaT")
    if not index.equals(index.normalize()):
        raise ValueError("target index must contain midnight calendar dates")
    if index.has_duplicates or not index.is_monotonic_increasing:
        raise ValueError("target dates must be unique and strictly sorted")

    expected = pd.date_range(index[0], index[-1], freq="D")
    if not index.equals(expected):
        raise ValueError(
            "target must use a complete daily calendar; encode missing "
            "measurements as NaN values"
        )
    if pd.api.types.is_bool_dtype(target.dtype):
        raise TypeError("target values must be numeric, not boolean")
    try:
        numeric = pd.to_numeric(target, errors="raise").astype(float)
    except (TypeError, ValueError) as exc:
        raise TypeError("target values must be numeric or NaN") from exc
    observed = numeric.notna().to_numpy()
    values = numeric.to_numpy(dtype=float)
    if np.any(observed & ~np.isfinite(values)):
        raise ValueError("observed target values must be finite")
    numeric.index = index.copy()
    numeric.name = target.name
    return numeric


def validate_origins(
    origins: Iterable[Any] | pd.DatetimeIndex | None,
    target_index: pd.DatetimeIndex,
) -> pd.DatetimeIndex:
    """Validate a sorted, unique subset of target calendar dates."""

    if origins is None:
        return target_index.copy()
    try:
        parsed = pd.DatetimeIndex(pd.to_datetime(list(origins)))
    except (TypeError, ValueError) as exc:
        raise TypeError("origins must be parseable calendar dates") from exc
    if parsed.tz is not None:
        raise ValueError("origins must be timezone-naive")
    if parsed.hasnans:
        raise ValueError("origins must not contain NaT")
    if not parsed.equals(parsed.normalize()):
        raise ValueError("origins must contain midnight calendar dates")
    if parsed.has_duplicates or not parsed.is_monotonic_increasing:
        raise ValueError("origins must be unique and strictly sorted")
    if len(parsed) and not parsed.isin(target_index).all():
        raise ValueError("every origin must be present in the target index")
    return parsed


def prepare_inputs(
    target: pd.Series,
    horizon: int,
    origins: Iterable[Any] | pd.DatetimeIndex | None = None,
) -> ForecastInputs:
    """Validate the shared inputs for a baseline forecast."""

    clean_target = validate_daily_target(target)
    clean_origins = validate_origins(origins, clean_target.index)
    return ForecastInputs(
        target=clean_target,
        origins=clean_origins,
        horizon=validate_horizon(horizon),
    )


def initialize_forecast(inputs: ForecastInputs, baseline_id: str) -> pd.DataFrame:
    """Create a common prediction frame without inspecting future targets."""

    if not isinstance(baseline_id, str) or not baseline_id.strip():
        raise ValueError("baseline_id must be a non-empty string")
    origin_values = inputs.target.reindex(inputs.origins).to_numpy(dtype=float)
    n_rows = len(inputs.origins)
    target_name = str(inputs.target.name) if inputs.target.name is not None else "target"
    return pd.DataFrame(
        {
            "origin_date": inputs.origins,
            "target_date": inputs.origins
            + pd.to_timedelta(inputs.horizon, unit="D"),
            "target_name": target_name,
            "horizon": np.full(n_rows, inputs.horizon, dtype=int),
            "baseline_id": baseline_id,
            "y_origin": origin_values,
            "origin_y_observed": np.isfinite(origin_values),
            "reference_date": pd.Series(
                pd.NaT,
                index=range(n_rows),
                dtype="datetime64[ns]",
            ),
            "last_observation_age_d": np.full(n_rows, np.nan, dtype=float),
            "n_history_observed": np.zeros(n_rows, dtype=int),
            "history_coverage_fraction": np.full(n_rows, np.nan, dtype=float),
            "oldest_history_date": pd.Series(
                pd.NaT,
                index=range(n_rows),
                dtype="datetime64[ns]",
            ),
            "newest_history_date": pd.Series(
                pd.NaT,
                index=range(n_rows),
                dtype="datetime64[ns]",
            ),
            "y_pred": np.full(n_rows, np.nan, dtype=float),
            "prediction_available": np.zeros(n_rows, dtype=bool),
        }
    )


def set_predictions(frame: pd.DataFrame, predictions: Any) -> pd.DataFrame:
    """Set finite predictions and their availability mask on a frame copy."""

    values = np.asarray(predictions, dtype=float)
    if values.ndim != 1 or len(values) != len(frame):
        raise ValueError("predictions must be one-dimensional and row-aligned")
    if np.any(~np.isfinite(values) & ~np.isnan(values)):
        raise ValueError("predictions may contain finite numbers or NaN only")
    result = frame.copy()
    result["y_pred"] = values
    result["prediction_available"] = np.isfinite(values)
    return result.loc[:, COMMON_PREDICTION_COLUMNS]


def observed_history_bounds(target: pd.Series) -> tuple[pd.Series, pd.Series]:
    """Return causal first/most-recent observed dates at every target date."""

    dates = pd.Series(target.index, index=target.index, dtype="datetime64[ns]")
    observed_dates = dates.where(target.notna())
    first = pd.Series(pd.NaT, index=target.index, dtype="datetime64[ns]")
    if target.notna().any():
        first_observed = target.index[target.notna()][0]
        first.loc[first_observed:] = first_observed
    latest = observed_dates.ffill()
    return first, latest
