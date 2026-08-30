"""Causal expanding mean and median climatology baselines."""

from __future__ import annotations

from typing import Any, Iterable, Literal

import numpy as np
import pandas as pd

from .base import (
    initialize_forecast,
    observed_history_bounds,
    prepare_inputs,
    set_predictions,
)


MIN_HISTORY_OBSERVATIONS = 30
BASELINE_IDS: dict[str, str] = {
    "mean": "B00_EXPANDING_MEAN",
    "median": "B01_EXPANDING_MEDIAN",
}


def predict_expanding_climatology(
    target: pd.Series,
    horizon: int,
    statistic: Literal["mean", "median"],
    origins: Iterable[Any] | pd.DatetimeIndex | None = None,
) -> pd.DataFrame:
    """Forecast from all observations at or before ``t``, after 30 values.

    The 30-observation threshold is a fixed Phase 04 warm-up rule rather than
    a fitted or selectable parameter.
    """

    if statistic not in BASELINE_IDS:
        raise ValueError("statistic must be 'mean' or 'median'")
    inputs = prepare_inputs(target, horizon, origins)
    frame = initialize_forecast(inputs, BASELINE_IDS[statistic])

    expanding = inputs.target.expanding(min_periods=MIN_HISTORY_OBSERVATIONS)
    estimates = expanding.mean() if statistic == "mean" else expanding.median()
    counts = inputs.target.notna().cumsum()
    first_observed, latest_observed = observed_history_bounds(inputs.target)

    predictions = estimates.reindex(inputs.origins).to_numpy(dtype=float)
    history_count = counts.reindex(inputs.origins).to_numpy(dtype=int)
    first = pd.to_datetime(first_observed.reindex(inputs.origins)).reset_index(drop=True)
    latest = pd.to_datetime(latest_observed.reindex(inputs.origins)).reset_index(drop=True)
    origin_series = frame["origin_date"].reset_index(drop=True)
    elapsed_days = (
        (inputs.origins - inputs.target.index[0]).days.astype(float) + 1.0
    )

    frame["reference_date"] = latest
    frame["last_observation_age_d"] = (origin_series - latest).dt.days.astype(float)
    frame["n_history_observed"] = history_count
    frame["history_coverage_fraction"] = history_count.astype(float) / elapsed_days
    frame["oldest_history_date"] = first
    frame["newest_history_date"] = latest
    return set_predictions(frame, predictions)


def predict_expanding_mean(
    target: pd.Series,
    horizon: int,
    origins: Iterable[Any] | pd.DatetimeIndex | None = None,
) -> pd.DataFrame:
    """Return canonical B00 expanding-mean forecasts."""

    return predict_expanding_climatology(target, horizon, "mean", origins)


def predict_expanding_median(
    target: pd.Series,
    horizon: int,
    origins: Iterable[Any] | pd.DatetimeIndex | None = None,
) -> pd.DataFrame:
    """Return canonical B01 expanding-median forecasts."""

    return predict_expanding_climatology(target, horizon, "median", origins)

