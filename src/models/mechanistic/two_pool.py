"""Optional separated-rate two-pool nonnegative output model."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Real
from typing import Any, Iterable

import numpy as np
import pandas as pd

from .kernels import K_CANDIDATES
from .nonnegative_fit import (
    NonnegativeLinearFit,
    PhysicalPredictions,
    fit_nonnegative_linear,
)


MIN_RATE_RATIO = 2.0


def _grid_rate(value: Real, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real scalar")
    rate = float(value)
    if not np.isfinite(rate) or rate not in K_CANDIDATES:
        raise ValueError(f"{name} must be one of {K_CANDIDATES}")
    return rate


def validate_two_pool_rates(
    k_fast: Real,
    k_slow: Real,
    min_ratio: Real = MIN_RATE_RATIO,
) -> tuple[float, float]:
    """Validate locked grid membership, ordering, and rate separation."""

    fast = _grid_rate(k_fast, "k_fast")
    slow = _grid_rate(k_slow, "k_slow")
    if isinstance(min_ratio, bool) or not isinstance(min_ratio, Real):
        raise TypeError("min_ratio must be a real scalar")
    ratio = float(min_ratio)
    if not np.isfinite(ratio) or ratio < MIN_RATE_RATIO:
        raise ValueError(f"min_ratio must be finite and >= {MIN_RATE_RATIO:g}")
    if fast <= slow:
        raise ValueError("k_fast must be strictly greater than k_slow")
    if fast / slow + 1.0e-12 < ratio:
        raise ValueError("k_fast / k_slow must satisfy the minimum ratio")
    return fast, slow


def two_pool_candidate_pairs(
    k_values: Iterable[Real] = K_CANDIDATES,
    min_ratio: Real = MIN_RATE_RATIO,
) -> tuple[tuple[float, float], ...]:
    """Return deterministic ``(k_fast, k_slow)`` grid pairs."""

    rates = tuple(float(value) for value in k_values)
    if rates != K_CANDIDATES:
        raise ValueError("two-pool candidates must use the exact locked k grid")
    pairs: list[tuple[float, float]] = []
    for slow in rates:
        for fast in rates:
            try:
                pairs.append(validate_two_pool_rates(fast, slow, min_ratio))
            except ValueError:
                continue
    return tuple(pairs)


def _assert_series_alignment(*values: Any) -> None:
    series = [value for value in values if isinstance(value, pd.Series)]
    if series and any(not series[0].index.equals(item.index) for item in series[1:]):
        raise ValueError("fast state, slow state, and target indexes must align")


def _two_column_matrix(fast_state: Any, slow_state: Any) -> pd.DataFrame | np.ndarray:
    if isinstance(fast_state, pd.Series) and isinstance(slow_state, pd.Series):
        if not fast_state.index.equals(slow_state.index):
            raise ValueError("fast and slow state indexes must align")
        return pd.DataFrame(
            {
                "fast_kinetic_state": fast_state,
                "slow_kinetic_state": slow_state,
            },
            index=fast_state.index,
        )
    fast = np.asarray(fast_state, dtype=float)
    slow = np.asarray(slow_state, dtype=float)
    if fast.ndim != 1 or slow.ndim != 1 or len(fast) != len(slow):
        raise ValueError("fast_state and slow_state must be aligned 1D arrays")
    return np.column_stack((fast, slow))


@dataclass(frozen=True)
class TwoPoolModel:
    """Fitted nonnegative fast/slow two-pool output mapping."""

    fitted: NonnegativeLinearFit
    k_fast: float
    k_slow: float
    load_basis: str | None = None
    model_id: str = "M1TS2"
    uses_target_history: bool = False

    @property
    def beta0(self) -> float:
        return self.fitted.intercept

    @property
    def beta_fast(self) -> float:
        return float(self.fitted.coefficients[0])

    @property
    def beta_slow(self) -> float:
        return float(self.fitted.coefficients[1])

    @property
    def n_observations(self) -> int:
        return self.fitted.n_observations

    @property
    def residual_sum_squares(self) -> float:
        return self.fitted.residual_sum_squares

    @property
    def effectively_single_pool(self) -> bool:
        scale = max(self.beta_fast, self.beta_slow, 1.0)
        tolerance = 1.0e-10 * scale
        return self.beta_fast <= tolerance or self.beta_slow <= tolerance

    def predict(self, fast_state: Any, slow_state: Any) -> PhysicalPredictions:
        """Return raw and explicitly clipped predictions."""

        design = _two_column_matrix(fast_state, slow_state)
        return self.fitted.predict(design)

    def predict_raw(self, fast_state: Any, slow_state: Any) -> np.ndarray:
        return self.predict(fast_state, slow_state).raw

    def predict_clipped(self, fast_state: Any, slow_state: Any) -> np.ndarray:
        return self.predict(fast_state, slow_state).clipped


def fit_two_pool(
    fast_state: Any,
    slow_state: Any,
    y: Any,
    *,
    k_fast: Real,
    k_slow: Real,
    fit_intercept: bool = True,
    load_basis: str | None = None,
    model_id: str = "M1TS2",
) -> TwoPoolModel:
    """Fit a separated-rate nonnegative two-pool model."""

    fast_rate, slow_rate = validate_two_pool_rates(k_fast, k_slow)
    _assert_series_alignment(fast_state, slow_state, y)
    if load_basis is not None and (
        not isinstance(load_basis, str) or not load_basis
    ):
        raise ValueError("load_basis must be a non-empty string or None")
    if not isinstance(model_id, str) or not model_id:
        raise ValueError("model_id must be a non-empty string")
    design = _two_column_matrix(fast_state, slow_state)
    fitted = fit_nonnegative_linear(design, y, fit_intercept=fit_intercept)
    if fitted.n_features_in != 2:
        raise RuntimeError("two-pool fitting requires exactly two kinetic states")
    return TwoPoolModel(
        fitted=fitted,
        k_fast=fast_rate,
        k_slow=slow_rate,
        load_basis=load_basis,
        model_id=model_id,
    )


def predict_two_pool(
    model: TwoPoolModel,
    fast_state: Any,
    slow_state: Any,
) -> PhysicalPredictions:
    """Functional prediction wrapper for a fitted two-pool model."""

    if not isinstance(model, TwoPoolModel):
        raise TypeError("model must be a TwoPoolModel")
    return model.predict(fast_state, slow_state)
