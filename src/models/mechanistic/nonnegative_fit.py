"""Deterministic small-p nonnegative least-squares without SciPy.

The global optimum is obtained by enumerating every active parameter subset,
solving ordinary least squares on that face, and choosing the feasible face
with minimum residual sum of squares.  This is intentionally limited to the
small one- and two-pool designs used in Phase 05.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Any

import numpy as np
import pandas as pd


FEASIBILITY_TOL = 1.0e-10
TIE_RTOL = 1.0e-12
TIE_ATOL = 1.0e-12
MAX_ENUMERATED_PARAMETERS = 12


def _readonly(values: Any, dtype: Any) -> np.ndarray:
    array = np.array(values, dtype=dtype, copy=True)
    array.setflags(write=False)
    return array


def _feature_matrix(X: Any) -> tuple[np.ndarray, tuple[str, ...]]:
    if isinstance(X, pd.DataFrame):
        names = tuple(str(column) for column in X.columns)
        values = X.to_numpy(dtype=float)
    elif isinstance(X, pd.Series):
        names = (str(X.name) if X.name is not None else "x0",)
        values = X.to_numpy(dtype=float).reshape(-1, 1)
    else:
        values = np.asarray(X, dtype=float)
        if values.ndim == 1:
            values = values.reshape(-1, 1)
        if values.ndim != 2:
            raise ValueError("X must be one- or two-dimensional")
        names = tuple(f"x{index}" for index in range(values.shape[1]))
    if values.ndim != 2 or values.shape[1] < 1:
        raise ValueError("X must contain at least one feature")
    if np.any(np.isinf(values)):
        raise ValueError("X may contain finite nonnegative values or NaN only")
    finite = values[np.isfinite(values)]
    if np.any(finite < 0.0):
        raise ValueError("mechanistic predictor values must be nonnegative")
    return values, names


def _target_vector(y: Any) -> np.ndarray:
    values = np.asarray(y, dtype=float)
    if values.ndim != 1:
        raise ValueError("y must be one-dimensional")
    if np.any(np.isinf(values)):
        raise ValueError("y may contain finite nonnegative values or NaN only")
    finite = values[np.isfinite(values)]
    if np.any(finite < 0.0):
        raise ValueError("observed target values must be nonnegative")
    return values


@dataclass(frozen=True)
class PhysicalPredictions:
    """Raw and nonnegative-clipped predictions with an explicit audit mask."""

    raw: np.ndarray
    clipped: np.ndarray
    available: np.ndarray
    negative_raw: np.ndarray
    clipped_count: int

    @property
    def y_pred(self) -> np.ndarray:
        """Canonical physical prediction array."""

        return self.clipped


@dataclass(frozen=True)
class NonnegativeLinearFit:
    """Immutable result of deterministic nonnegative least squares."""

    intercept: float
    coefficients: np.ndarray
    fit_intercept: bool
    feature_names: tuple[str, ...]
    n_features_in: int
    n_observations: int
    n_rows_received: int
    valid_row_mask: np.ndarray
    active_parameter_indices: tuple[int, ...]
    rank: int
    residual_sum_squares: float

    @property
    def beta0(self) -> float:
        return self.intercept

    @property
    def parameter_vector(self) -> np.ndarray:
        values = np.concatenate(([self.intercept], self.coefficients))
        values.setflags(write=False)
        return values

    def predict(self, X: Any) -> PhysicalPredictions:
        """Return both raw and explicitly clipped predictions."""

        return predict_nonnegative_linear(self, X)


def _candidate_is_better(
    sse: float,
    active: tuple[int, ...],
    best_sse: float,
    best_active: tuple[int, ...] | None,
) -> bool:
    if best_active is None:
        return True
    if sse < best_sse and not np.isclose(
        sse,
        best_sse,
        rtol=TIE_RTOL,
        atol=TIE_ATOL,
    ):
        return True
    if np.isclose(sse, best_sse, rtol=TIE_RTOL, atol=TIE_ATOL):
        return (len(active), active) < (len(best_active), best_active)
    return False


def fit_nonnegative_linear(
    X: Any,
    y: Any,
    *,
    fit_intercept: bool = True,
) -> NonnegativeLinearFit:
    """Fit intercept and coefficients constrained to be nonnegative.

    Rows with missing ``X`` or ``y`` are excluded pairwise and recorded in
    ``valid_row_mask``; values are never imputed.  Ties are resolved first in
    favor of fewer active parameters and then lexicographically, making rank-
    deficient solutions deterministic.
    """

    if not isinstance(fit_intercept, (bool, np.bool_)):
        raise TypeError("fit_intercept must be boolean")
    features, names = _feature_matrix(X)
    target = _target_vector(y)
    if len(features) != len(target):
        raise ValueError("X and y must have identical row counts")
    complete = np.isfinite(target) & np.isfinite(features).all(axis=1)
    if not complete.any():
        raise ValueError("at least one complete training row is required")
    X_valid = features[complete]
    y_valid = target[complete]
    design = (
        np.column_stack((np.ones(len(X_valid), dtype=float), X_valid))
        if bool(fit_intercept)
        else X_valid
    )
    parameter_count = design.shape[1]
    if parameter_count > MAX_ENUMERATED_PARAMETERS:
        raise ValueError(
            f"active-subset solver supports at most {MAX_ENUMERATED_PARAMETERS} "
            "parameters"
        )

    best_sse = float("inf")
    best_active: tuple[int, ...] | None = None
    best_parameters = np.zeros(parameter_count, dtype=float)
    best_rank = 0

    for active_count in range(parameter_count + 1):
        for active in combinations(range(parameter_count), active_count):
            parameters = np.zeros(parameter_count, dtype=float)
            if active:
                subdesign = design[:, active]
                solution, _, rank, _ = np.linalg.lstsq(
                    subdesign,
                    y_valid,
                    rcond=None,
                )
                if np.any(solution < -FEASIBILITY_TOL):
                    continue
                parameters[list(active)] = np.maximum(solution, 0.0)
                selected_rank = int(rank)
            else:
                selected_rank = 0
            residual = design @ parameters - y_valid
            sse = float(np.dot(residual, residual))
            if _candidate_is_better(sse, active, best_sse, best_active):
                best_sse = sse
                best_active = tuple(active)
                best_parameters = parameters
                best_rank = selected_rank

    if best_active is None:  # The all-zero face is always feasible.
        raise RuntimeError("nonnegative active-subset enumeration found no solution")
    if bool(fit_intercept):
        intercept = float(best_parameters[0])
        coefficients = best_parameters[1:]
    else:
        intercept = 0.0
        coefficients = best_parameters

    return NonnegativeLinearFit(
        intercept=max(intercept, 0.0),
        coefficients=_readonly(np.maximum(coefficients, 0.0), float),
        fit_intercept=bool(fit_intercept),
        feature_names=names,
        n_features_in=features.shape[1],
        n_observations=int(complete.sum()),
        n_rows_received=len(features),
        valid_row_mask=_readonly(complete, bool),
        active_parameter_indices=best_active,
        rank=best_rank,
        residual_sum_squares=best_sse,
    )


def predict_nonnegative_linear(
    fitted: NonnegativeLinearFit,
    X: Any,
) -> PhysicalPredictions:
    """Predict with a fitted nonnegative model without filling missing rows."""

    if not isinstance(fitted, NonnegativeLinearFit):
        raise TypeError("fitted must be a NonnegativeLinearFit")
    features, _ = _feature_matrix(X)
    if features.shape[1] != fitted.n_features_in:
        raise ValueError(
            f"X has {features.shape[1]} features; expected {fitted.n_features_in}"
        )
    available = np.isfinite(features).all(axis=1)
    raw = np.full(len(features), np.nan, dtype=float)
    raw[available] = (
        fitted.intercept + features[available] @ fitted.coefficients
    )
    negative = np.isfinite(raw) & (raw < 0.0)
    clipped = raw.copy()
    clipped[negative] = 0.0
    return PhysicalPredictions(
        raw=_readonly(raw, float),
        clipped=_readonly(clipped, float),
        available=_readonly(available, bool),
        negative_raw=_readonly(negative, bool),
        clipped_count=int(negative.sum()),
    )


fit_nonnegative_least_squares = fit_nonnegative_linear
