"""Causal last-observed-value baseline with observation-age metadata."""

from __future__ import annotations

from typing import Any, Iterable

import numpy as np
import pandas as pd

from .base import initialize_forecast, prepare_inputs, set_predictions


BASELINE_ID = "B11_LAST_OBSERVED"


def predict_last_observed(
    target: pd.Series,
    horizon: int,
    origins: Iterable[Any] | pd.DatetimeIndex | None = None,
) -> pd.DataFrame:
    """Forecast from the most recent observed target on or before each origin.

    No maximum-age cutoff is applied.  ``last_observation_age_d`` is the exact
    calendar-day distance from origin ``t`` to the selected observation
    ``s* = max{s <= t: y(s) observed}``.
    """

    inputs = prepare_inputs(target, horizon, origins)
    frame = initialize_forecast(inputs, BASELINE_ID)

    date_values = pd.Series(inputs.target.index, index=inputs.target.index)
    observed_dates = date_values.where(inputs.target.notna()).ffill()
    last_values = inputs.target.ffill()
    references = pd.to_datetime(observed_dates.reindex(inputs.origins)).reset_index(
        drop=True
    )
    predictions = last_values.reindex(inputs.origins).to_numpy(dtype=float)
    origin_series = frame["origin_date"].reset_index(drop=True)
    age = (origin_series - references).dt.days.astype(float)
    available = np.isfinite(predictions)

    frame["reference_date"] = references
    frame["last_observation_age_d"] = age
    frame["n_history_observed"] = available.astype(int)
    frame["history_coverage_fraction"] = available.astype(float)
    frame["oldest_history_date"] = references
    frame["newest_history_date"] = references
    return set_predictions(frame, predictions)


last_observed = predict_last_observed
