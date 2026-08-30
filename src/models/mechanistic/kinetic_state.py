"""Causal gap-aware kinetic-state construction for daily load series."""

from __future__ import annotations

from numbers import Real

import numpy as np
import pandas as pd

from .kernels import FirstOrderKernel


MIN_KERNEL_COVERAGE = 0.90


def validate_daily_load(load: pd.Series) -> pd.Series:
    """Return a float copy of a complete daily nonnegative load series.

    Calendar gaps are not silently repaired.  Missing measurements must be
    represented as ``NaN`` values and remain unfilled during state creation.
    """

    if not isinstance(load, pd.Series):
        raise TypeError("load must be a pandas Series")
    if not isinstance(load.index, pd.DatetimeIndex):
        raise TypeError("load index must be a pandas DatetimeIndex")
    if load.empty:
        raise ValueError("load must contain at least one daily row")
    index = load.index
    if index.tz is not None or index.hasnans:
        raise ValueError("load dates must be timezone-naive and nonmissing")
    if not index.equals(index.normalize()):
        raise ValueError("load dates must be midnight calendar dates")
    if index.has_duplicates or not index.is_monotonic_increasing:
        raise ValueError("load dates must be unique and strictly sorted")
    expected = pd.date_range(index[0], index[-1], freq="D")
    if not index.equals(expected):
        raise ValueError("load must use a complete daily calendar without row gaps")
    if pd.api.types.is_bool_dtype(load.dtype):
        raise TypeError("load values must be numeric, not boolean")
    try:
        numeric = pd.to_numeric(load, errors="raise").astype(float)
    except (TypeError, ValueError) as exc:
        raise TypeError("load values must be numeric or NaN") from exc
    values = numeric.to_numpy(dtype=float)
    if np.any(np.isinf(values)):
        raise ValueError("load values may contain finite numbers or NaN only")
    if np.any(values[np.isfinite(values)] < 0.0):
        raise ValueError("observed load values must be nonnegative")
    numeric.index = index.copy()
    numeric.name = load.name
    return numeric


def _coverage_threshold(value: Real) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError("min_kernel_coverage must be a real scalar")
    threshold = float(value)
    if not np.isfinite(threshold) or not 0.0 < threshold <= 1.0:
        raise ValueError("min_kernel_coverage must be finite and in (0, 1]")
    return threshold


def build_kinetic_state(
    load: pd.Series,
    kernel: FirstOrderKernel,
    min_kernel_coverage: Real = MIN_KERNEL_COVERAGE,
) -> pd.DataFrame:
    """Build a causal coverage-normalized first-order load state.

    For date ``t``, only load values at ``t-j`` for nonnegative kernel lags
    are inspected.  Missing and pre-series load days contribute no observed
    weight.  The state is the observed weighted sum divided by observed
    kernel weight and is released only when observed/full-truncated kernel
    weight is at least ``min_kernel_coverage``.
    """

    if not isinstance(kernel, FirstOrderKernel):
        raise TypeError("kernel must be a FirstOrderKernel")
    if not kernel.nonnegative or kernel.sum_weights <= 0.0:
        raise ValueError("kernel must have nonnegative weights and positive mass")
    threshold = _coverage_threshold(min_kernel_coverage)
    clean = validate_daily_load(load)
    values = clean.to_numpy(dtype=float)
    n_rows = len(clean)
    width = len(kernel.weights)

    state = np.full(n_rows, np.nan, dtype=float)
    coverage = np.zeros(n_rows, dtype=float)
    observed_weight = np.zeros(n_rows, dtype=float)
    observed_count = np.zeros(n_rows, dtype=int)
    oldest = pd.Series(pd.NaT, index=range(n_rows), dtype="datetime64[ns]")
    newest = pd.Series(pd.NaT, index=range(n_rows), dtype="datetime64[ns]")

    for position in range(n_rows):
        available_lags = min(position + 1, width)
        lagged = values[position - available_lags + 1 : position + 1][::-1]
        weights = kernel.weights[:available_lags]
        observed = np.isfinite(lagged)
        weight = float(np.sum(weights[observed], dtype=float))
        observed_weight[position] = weight
        observed_count[position] = int(observed.sum())
        coverage[position] = weight / kernel.sum_weights
        if observed.any():
            observed_lags = np.flatnonzero(observed)
            newest.iloc[position] = clean.index[position - int(observed_lags.min())]
            oldest.iloc[position] = clean.index[position - int(observed_lags.max())]
        if coverage[position] + 1.0e-12 >= threshold and weight > 0.0:
            state[position] = float(np.dot(weights[observed], lagged[observed]) / weight)

    available = np.isfinite(state)
    return pd.DataFrame(
        {
            "date": clean.index,
            "coverage_normalized_kinetic_state": state,
            "kinetic_state": state,
            "kernel_coverage": coverage,
            "observed_kernel_weight": observed_weight,
            "truncated_kernel_weight": np.full(n_rows, kernel.sum_weights),
            "n_kernel_observed": observed_count,
            "n_kernel_expected": np.full(n_rows, width, dtype=int),
            "state_available": available,
            "oldest_load_date_used": oldest,
            "newest_load_date_used": newest,
            "k": np.full(n_rows, kernel.k_per_d),
            "kernel_max_lag": np.full(n_rows, kernel.max_lag, dtype=int),
            "minimum_kernel_coverage": np.full(n_rows, threshold),
        }
    )


coverage_normalized_kinetic_state = build_kinetic_state

