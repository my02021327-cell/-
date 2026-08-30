"""Deterministic causal baselines for Phase 04 benchmarking."""

from .base import (
    COMMON_PREDICTION_COLUMNS,
    HORIZONS,
    ForecastInputs,
    validate_daily_target,
    validate_horizon,
    validate_origins,
)
from .empirical_interval import (
    DEFAULT_GROUP_COLUMNS,
    DEFAULT_LEVELS,
    MIN_RESIDUALS,
    add_causal_empirical_intervals,
    causal_empirical_intervals,
    empirical_residual_bounds,
)
from .expanding_climatology import (
    MIN_HISTORY_OBSERVATIONS,
    predict_expanding_climatology,
    predict_expanding_mean,
    predict_expanding_median,
)
from .last_observed import last_observed, predict_last_observed
from .moving_average import (
    predict_moving_average,
    predict_moving_average_7d,
    predict_moving_average_14d,
    predict_moving_average_30d,
)
from .persistence import predict_strict_persistence, strict_persistence
from .seasonal_naive import (
    annual_reference_date,
    annual_seasonal_naive,
    predict_annual_seasonal_naive,
    predict_weekly_seasonal_naive,
    weekly_reference_date,
    weekly_seasonal_naive,
)


__all__ = [
    "COMMON_PREDICTION_COLUMNS",
    "DEFAULT_GROUP_COLUMNS",
    "DEFAULT_LEVELS",
    "ForecastInputs",
    "HORIZONS",
    "MIN_HISTORY_OBSERVATIONS",
    "MIN_RESIDUALS",
    "add_causal_empirical_intervals",
    "annual_reference_date",
    "annual_seasonal_naive",
    "causal_empirical_intervals",
    "empirical_residual_bounds",
    "last_observed",
    "predict_annual_seasonal_naive",
    "predict_expanding_climatology",
    "predict_expanding_mean",
    "predict_expanding_median",
    "predict_last_observed",
    "predict_moving_average",
    "predict_moving_average_7d",
    "predict_moving_average_14d",
    "predict_moving_average_30d",
    "predict_strict_persistence",
    "predict_weekly_seasonal_naive",
    "strict_persistence",
    "validate_daily_target",
    "validate_horizon",
    "validate_origins",
    "weekly_reference_date",
    "weekly_seasonal_naive",
]
