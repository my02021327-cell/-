"""Single-pool process-informed nonnegative output model."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral, Real
from typing import Any

import numpy as np
import pandas as pd

from .kernels import K_CANDIDATES
from .nonnegative_fit import (
    NonnegativeLinearFit,
    PhysicalPredictions,
    fit_nonnegative_linear,
)


def _aligned_series(left: Any, right: Any) -> None:
    if isinstance(left, pd.Series) and isinstance(right, pd.Series):
        if not left.index.equals(right.index):
            raise ValueError("state and target Series indexes must align exactly")


def _optional_grid_rate(value: Real | None) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError("k_per_d must be a real scalar or None")
    rate = float(value)
    if not np.isfinite(rate) or rate not in K_CANDIDATES:
        raise ValueError(f"k_per_d must be one of {K_CANDIDATES}")
    return rate


def _optional_max_lag(value: int | None) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError("kernel_max_lag must be a non-boolean integer or None")
    lag = int(value)
    if lag < 0:
        raise ValueError("kernel_max_lag must be >= 0")
    return lag


@dataclass(frozen=True)
class SinglePoolModel:
    """Fitted ``beta0 + beta1 * kinetic_state`` model."""

    fitted: NonnegativeLinearFit
    k_per_d: float | None = None
    kernel_max_lag: int | None = None
    load_basis: str | None = None
    model_id: str = "M1"
    uses_target_history: bool = False

    @property
    def beta0(self) -> float:
        return self.fitted.intercept

    @property
    def beta1(self) -> float:
        return float(self.fitted.coefficients[0])

    @property
    def intercept(self) -> float:
        return self.beta0

    @property
    def coefficient(self) -> float:
        return self.beta1

    @property
    def n_observations(self) -> int:
        return self.fitted.n_observations

    @property
    def residual_sum_squares(self) -> float:
        return self.fitted.residual_sum_squares

    def predict(self, kinetic_state: Any) -> PhysicalPredictions:
        """Return raw and nonnegative-clipped predictions."""

        return self.fitted.predict(kinetic_state)

    def predict_raw(self, kinetic_state: Any) -> np.ndarray:
        return self.predict(kinetic_state).raw

    def predict_clipped(self, kinetic_state: Any) -> np.ndarray:
        return self.predict(kinetic_state).clipped


def fit_single_pool(
    kinetic_state: Any,
    y: Any,
    *,
    fit_intercept: bool = True,
    k_per_d: Real | None = None,
    kernel_max_lag: int | None = None,
    load_basis: str | None = None,
    model_id: str = "M1",
) -> SinglePoolModel:
    """Fit a deterministic nonnegative single-pool output mapping.

    Pair-incomplete rows are excluded by the underlying solver without
    imputation.  ``k_per_d`` is metadata only and, when supplied, must come
    from the locked candidate grid.
    """

    _aligned_series(kinetic_state, y)
    if load_basis is not None and (
        not isinstance(load_basis, str) or not load_basis
    ):
        raise ValueError("load_basis must be a non-empty string or None")
    if not isinstance(model_id, str) or not model_id:
        raise ValueError("model_id must be a non-empty string")
    fitted = fit_nonnegative_linear(
        kinetic_state,
        y,
        fit_intercept=fit_intercept,
    )
    if fitted.n_features_in != 1:
        raise ValueError("single-pool fitting requires exactly one kinetic state")
    return SinglePoolModel(
        fitted=fitted,
        k_per_d=_optional_grid_rate(k_per_d),
        kernel_max_lag=_optional_max_lag(kernel_max_lag),
        load_basis=load_basis,
        model_id=model_id,
    )


def predict_single_pool(
    model: SinglePoolModel,
    kinetic_state: Any,
) -> PhysicalPredictions:
    """Functional prediction wrapper for a fitted single-pool model."""

    if not isinstance(model, SinglePoolModel):
        raise TypeError("model must be a SinglePoolModel")
    return model.predict(kinetic_state)
