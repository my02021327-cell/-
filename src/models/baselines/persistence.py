"""Canonical strict-persistence baseline."""

from __future__ import annotations

from typing import Any, Iterable

import numpy as np
import pandas as pd

from .base import initialize_forecast, prepare_inputs, set_predictions


BASELINE_ID = "B10_PERSISTENCE_STRICT"


def predict_strict_persistence(
    target: pd.Series,
    horizon: int,
    origins: Iterable[Any] | pd.DatetimeIndex | None = None,
) -> pd.DataFrame:
    """Forecast ``y(t+h)`` as the observed value at the exact origin ``y(t)``.

    The same origin value is used for every supported horizon.  If ``y(t)``
    is missing, the prediction is missing; this function never falls back to
    a prior observation and therefore remains distinct from LOCF.
    """

    inputs = prepare_inputs(target, horizon, origins)
    frame = initialize_forecast(inputs, BASELINE_ID)
    observed = frame["origin_y_observed"].to_numpy(dtype=bool)
    origin_dates = pd.DatetimeIndex(frame["origin_date"])

    frame["reference_date"] = origin_dates
    frame.loc[observed, "last_observation_age_d"] = 0.0
    frame["n_history_observed"] = observed.astype(int)
    frame["history_coverage_fraction"] = observed.astype(float)
    frame.loc[observed, "oldest_history_date"] = origin_dates[observed]
    frame.loc[observed, "newest_history_date"] = origin_dates[observed]
    return set_predictions(frame, frame["y_origin"].to_numpy(dtype=float))


strict_persistence = predict_strict_persistence
