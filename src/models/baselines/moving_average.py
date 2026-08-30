"""Right-aligned calendar moving-average baselines."""

from __future__ import annotations

from numbers import Integral
from typing import Any, Iterable

import numpy as np
import pandas as pd

from .base import initialize_forecast, prepare_inputs, set_predictions


BASELINE_IDS: dict[int, str] = {
    7: "B20_MOVING_AVG_7D",
    14: "B21_MOVING_AVG_14D",
    30: "B22_MOVING_AVG_30D",
}


def predict_moving_average(
    target: pd.Series,
    horizon: int,
    window_days: int,
    origins: Iterable[Any] | pd.DatetimeIndex | None = None,
) -> pd.DataFrame:
    """Forecast with the observed mean over calendar window ``[t-w+1, t]``.

    Only observed values inside the right-aligned window contribute to the
    mean.  At least one observation is required.  The coverage denominator is
    the locked calendar-window width, including during dataset warm-up; the
    operation does not fill missing target values.
    """

    if (
        isinstance(window_days, bool)
        or not isinstance(window_days, Integral)
        or int(window_days) not in BASELINE_IDS
    ):
        raise ValueError(f"window_days must be one of {tuple(BASELINE_IDS)}")
    inputs = prepare_inputs(target, horizon, origins)
    frame = initialize_forecast(inputs, BASELINE_IDS[int(window_days)])

    predictions = np.full(len(frame), np.nan, dtype=float)
    counts = np.zeros(len(frame), dtype=int)
    oldest = pd.Series(pd.NaT, index=frame.index, dtype="datetime64[ns]")
    newest = pd.Series(pd.NaT, index=frame.index, dtype="datetime64[ns]")
    one_day = pd.Timedelta(days=1)
    width = int(window_days)

    for row, origin in enumerate(inputs.origins):
        start = origin - (width - 1) * one_day
        observed = inputs.target.loc[start:origin].dropna()
        counts[row] = len(observed)
        if len(observed):
            predictions[row] = float(observed.mean())
            oldest.iloc[row] = observed.index[0]
            newest.iloc[row] = observed.index[-1]

    frame["n_history_observed"] = counts
    frame["history_coverage_fraction"] = counts.astype(float) / width
    frame["oldest_history_date"] = oldest
    frame["newest_history_date"] = newest
    return set_predictions(frame, predictions)


def predict_moving_average_7d(
    target: pd.Series,
    horizon: int,
    origins: Iterable[Any] | pd.DatetimeIndex | None = None,
) -> pd.DataFrame:
    """Return canonical B20 seven-calendar-day moving-average forecasts."""

    return predict_moving_average(target, horizon, 7, origins)


def predict_moving_average_14d(
    target: pd.Series,
    horizon: int,
    origins: Iterable[Any] | pd.DatetimeIndex | None = None,
) -> pd.DataFrame:
    """Return canonical B21 fourteen-calendar-day moving-average forecasts."""

    return predict_moving_average(target, horizon, 14, origins)


def predict_moving_average_30d(
    target: pd.Series,
    horizon: int,
    origins: Iterable[Any] | pd.DatetimeIndex | None = None,
) -> pd.DataFrame:
    """Return canonical B22 thirty-calendar-day moving-average forecasts."""

    return predict_moving_average(target, horizon, 30, origins)
