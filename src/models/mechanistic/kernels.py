"""HRT-free first-order hydrolysis/degradation kernel utilities.

The daily weights are exact integrals of ``k * exp(-k*t)`` over each lag
bin.  Truncation is controlled only by the locked residual-tail tolerance;
no target observations or forecast scores enter kernel construction.
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral, Real
from typing import Iterable

import numpy as np
import pandas as pd


K_CANDIDATES: tuple[float, ...] = (0.03, 0.05, 0.07, 0.10, 0.15, 0.20, 0.35)
KERNEL_TAIL_TOL = 1.0e-3
NORMALIZATION_ATOL = 1.0e-12


def _finite_positive(value: Real, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real scalar")
    numeric = float(value)
    if not np.isfinite(numeric) or numeric <= 0.0:
        raise ValueError(f"{name} must be finite and > 0")
    return numeric


def _tail_tolerance(value: Real) -> float:
    tolerance = _finite_positive(value, "tail_tolerance")
    if tolerance >= 1.0:
        raise ValueError("tail_tolerance must be < 1")
    return tolerance


def minimal_inclusive_max_lag(
    k_per_d: Real,
    tail_tolerance: Real = KERNEL_TAIL_TOL,
) -> int:
    """Return minimal ``K`` satisfying ``exp[-k*(K+1)] <= tolerance``."""

    k = _finite_positive(k_per_d, "k_per_d")
    tolerance = _tail_tolerance(tail_tolerance)
    max_lag = max(0, int(np.ceil(-np.log(tolerance) / k)) - 1)

    # Correct the analytic result at floating-point equality boundaries.
    while np.exp(-k * (max_lag + 1)) > tolerance:
        max_lag += 1
    while max_lag > 0 and np.exp(-k * max_lag) <= tolerance:
        max_lag -= 1
    return max_lag


def exact_daily_weights(k_per_d: Real, max_lag: int) -> np.ndarray:
    """Return ``exp(-k*j) - exp(-k*(j+1))`` for inclusive lags ``0..K``."""

    k = _finite_positive(k_per_d, "k_per_d")
    if isinstance(max_lag, bool) or not isinstance(max_lag, Integral):
        raise TypeError("max_lag must be a non-boolean integer")
    inclusive_lag = int(max_lag)
    if inclusive_lag < 0:
        raise ValueError("max_lag must be >= 0")
    lags = np.arange(inclusive_lag + 1, dtype=float)
    weights = np.exp(-k * lags) - np.exp(-k * (lags + 1.0))
    weights.setflags(write=False)
    return weights


@dataclass(frozen=True)
class FirstOrderKernel:
    """Immutable exact daily-bin first-order kernel and audit properties."""

    k_per_d: float
    max_lag: int
    lags_d: np.ndarray
    weights: np.ndarray
    tail_tolerance: float
    tail_mass: float
    sum_weights: float
    normalization_error: float
    nonnegative: bool
    normalization_pass: bool
    mean_response_age_d: float
    median_response_age_d: float
    discrete_mean_lag_d: float
    continuous_mean_time_d: float
    continuous_median_time_d: float
    minimal_truncation: bool

    def audit_record(self) -> dict[str, object]:
        """Return the locked kernel-audit row expected by Phase 05."""

        return {
            "k": self.k_per_d,
            "max_lag": self.max_lag,
            "sum_weights": self.sum_weights,
            "tail_mass": self.tail_mass,
            "mean_response_age": self.mean_response_age_d,
            "median_response_age": self.median_response_age_d,
            "nonnegative": self.nonnegative,
            "normalization_error": self.normalization_error,
            "normalization_pass": self.normalization_pass,
            "minimal_truncation": self.minimal_truncation,
            "tail_tolerance": self.tail_tolerance,
            "discrete_mean_lag": self.discrete_mean_lag_d,
            "continuous_mean_time": self.continuous_mean_time_d,
            "continuous_median_time": self.continuous_median_time_d,
        }


def build_first_order_kernel(
    k_per_d: Real,
    tail_tolerance: Real = KERNEL_TAIL_TOL,
    max_lag: int | None = None,
) -> FirstOrderKernel:
    """Build an exact HRT-free daily response kernel.

    With the default ``max_lag=None``, the returned inclusive maximum lag is
    the smallest integer whose unrepresented exponential tail is no greater
    than ``tail_tolerance``.  Supplying ``max_lag`` is supported for explicit
    mathematical audits, but the resulting object records whether that lag is
    the deterministic minimal truncation.
    """

    k = _finite_positive(k_per_d, "k_per_d")
    tolerance = _tail_tolerance(tail_tolerance)
    minimal_lag = minimal_inclusive_max_lag(k, tolerance)
    if max_lag is None:
        inclusive_lag = minimal_lag
    else:
        if isinstance(max_lag, bool) or not isinstance(max_lag, Integral):
            raise TypeError("max_lag must be a non-boolean integer")
        inclusive_lag = int(max_lag)
        if inclusive_lag < 0:
            raise ValueError("max_lag must be >= 0")

    weights = exact_daily_weights(k, inclusive_lag)
    lags = np.arange(inclusive_lag + 1, dtype=int)
    lags.setflags(write=False)
    tail = float(np.exp(-k * (inclusive_lag + 1)))
    weight_sum = float(np.sum(weights, dtype=float))
    normalization_error = abs(weight_sum + tail - 1.0)
    nonnegative = bool(np.all(weights >= 0.0))
    tail_pass = bool(tail <= tolerance)
    normalization_pass = bool(
        nonnegative
        and tail_pass
        and normalization_error <= NORMALIZATION_ATOL
    )
    discrete_mean = float(np.dot(lags.astype(float), weights) / weight_sum)
    cumulative = np.cumsum(weights, dtype=float) / weight_sum
    discrete_median = float(lags[np.searchsorted(cumulative, 0.5, side="left")])

    return FirstOrderKernel(
        k_per_d=k,
        max_lag=inclusive_lag,
        lags_d=lags,
        weights=weights,
        tail_tolerance=tolerance,
        tail_mass=tail,
        sum_weights=weight_sum,
        normalization_error=normalization_error,
        nonnegative=nonnegative,
        normalization_pass=normalization_pass,
        mean_response_age_d=discrete_mean,
        median_response_age_d=discrete_median,
        discrete_mean_lag_d=discrete_mean,
        continuous_mean_time_d=1.0 / k,
        continuous_median_time_d=float(np.log(2.0) / k),
        minimal_truncation=bool(inclusive_lag == minimal_lag),
    )


def kernel_audit_frame(
    k_values: Iterable[Real] = K_CANDIDATES,
    tail_tolerance: Real = KERNEL_TAIL_TOL,
) -> pd.DataFrame:
    """Return deterministic audit properties for an ordered rate grid."""

    rates = tuple(k_values)
    if not rates:
        raise ValueError("k_values must contain at least one rate")
    kernels = [build_first_order_kernel(k, tail_tolerance) for k in rates]
    return pd.DataFrame([kernel.audit_record() for kernel in kernels])


first_order_daily_weights = exact_daily_weights
