"""Causal weekly and annual seasonal-naive baselines."""

from __future__ import annotations

from typing import Any, Iterable

import numpy as np
import pandas as pd

from .base import (
    initialize_forecast,
    prepare_inputs,
    set_predictions,
    validate_horizon,
)


WEEKLY_BASELINE_ID = "B30_WEEKLY_SEASONAL_NAIVE"
ANNUAL_BASELINE_ID = "B31_ANNUAL_SEASONAL_NAIVE"


def weekly_reference_date(origin: Any, horizon: int) -> pd.Timestamp:
    """Return ``t+h-7*ceil(h/7)``, which is never later than origin ``t``."""

    h = validate_horizon(horizon)
    timestamp = pd.Timestamp(origin)
    if (
        pd.isna(timestamp)
        or timestamp.tz is not None
        or timestamp != timestamp.normalize()
    ):
        raise ValueError("origin must be a timezone-naive midnight date")
    weeks = (h + 6) // 7
    reference = timestamp + pd.Timedelta(days=h - 7 * weeks)
    if reference > timestamp:  # Defensive assertion of the causal contract.
        raise RuntimeError("weekly seasonal reference would use future target data")
    return reference


def annual_reference_date(target_date: Any) -> pd.Timestamp:
    """Return the same calendar date one year earlier.

    The explicit leap policy follows ``pandas.DateOffset``: February 29 maps
    to February 28 when the previous calendar year is not a leap year.
    """

    timestamp = pd.Timestamp(target_date)
    if (
        pd.isna(timestamp)
        or timestamp.tz is not None
        or timestamp != timestamp.normalize()
    ):
        raise ValueError("target_date must be a timezone-naive midnight date")
    return timestamp - pd.DateOffset(years=1)


def _seasonal_frame(
    target: pd.Series,
    horizon: int,
    origins: Iterable[Any] | pd.DatetimeIndex | None,
    baseline_id: str,
    references: pd.DatetimeIndex,
) -> pd.DataFrame:
    inputs = prepare_inputs(target, horizon, origins)
    if len(references) != len(inputs.origins):
        raise ValueError("references must align one-to-one with forecast origins")
    if np.any(references > inputs.origins):
        raise RuntimeError("seasonal reference dates must not exceed origins")

    frame = initialize_forecast(inputs, baseline_id)
    predictions = inputs.target.reindex(references).to_numpy(dtype=float)
    available = np.isfinite(predictions)
    age = (inputs.origins - references).days.astype(float)

    frame["reference_date"] = references
    frame["last_observation_age_d"] = age
    frame["n_history_observed"] = available.astype(int)
    frame["history_coverage_fraction"] = available.astype(float)
    frame.loc[available, "oldest_history_date"] = references[available]
    frame.loc[available, "newest_history_date"] = references[available]
    return set_predictions(frame, predictions)


def predict_weekly_seasonal_naive(
    target: pd.Series,
    horizon: int,
    origins: Iterable[Any] | pd.DatetimeIndex | None = None,
) -> pd.DataFrame:
    """Forecast from the locked causal weekly reference for each horizon."""

    inputs = prepare_inputs(target, horizon, origins)
    weeks = (inputs.horizon + 6) // 7
    references = inputs.origins + pd.to_timedelta(
        inputs.horizon - 7 * weeks,
        unit="D",
    )
    return _seasonal_frame(
        inputs.target,
        inputs.horizon,
        inputs.origins,
        WEEKLY_BASELINE_ID,
        references,
    )


def predict_annual_seasonal_naive(
    target: pd.Series,
    horizon: int,
    origins: Iterable[Any] | pd.DatetimeIndex | None = None,
) -> pd.DataFrame:
    """Forecast from the target date's same calendar date one year earlier."""

    inputs = prepare_inputs(target, horizon, origins)
    target_dates = inputs.origins + pd.to_timedelta(inputs.horizon, unit="D")
    references = pd.DatetimeIndex(
        [annual_reference_date(date) for date in target_dates]
    )
    return _seasonal_frame(
        inputs.target,
        inputs.horizon,
        inputs.origins,
        ANNUAL_BASELINE_ID,
        references,
    )


weekly_seasonal_naive = predict_weekly_seasonal_naive
annual_seasonal_naive = predict_annual_seasonal_naive
