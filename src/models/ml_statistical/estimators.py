"""Deterministic bounded Phase 06 estimator families and candidate grids."""

from __future__ import annotations

import json
import warnings
from dataclasses import dataclass
from typing import Any

import numpy as np

from .contracts import bootstrap_local_dependencies


bootstrap_local_dependencies()

from sklearn.ensemble import RandomForestRegressor  # noqa: E402
from sklearn.exceptions import ConvergenceWarning  # noqa: E402
from sklearn.linear_model import ElasticNet, Ridge  # noqa: E402


MODEL_SEED = 606


@dataclass(frozen=True)
class CandidateSpec:
    candidate_id: str
    parameters: dict[str, Any]
    complexity_rank: int

    def parameters_json(self) -> str:
        return json.dumps(self.parameters, sort_keys=True, separators=(",", ":"))


def candidate_specs(model_id: str) -> tuple[CandidateSpec, ...]:
    """Return a small predeclared grid; no data-dependent search expansion."""

    grids: dict[str, tuple[CandidateSpec, ...]] = {
        "S00_RIDGE": tuple(
            CandidateSpec(f"RIDGE_A{alpha:g}", {"alpha": alpha}, rank)
            for rank, alpha in enumerate((0.1, 1.0, 10.0), start=1)
        ),
        "S01_ELASTIC_NET": (
            CandidateSpec("EN_A01_L01", {"alpha": 0.1, "l1_ratio": 0.1}, 1),
            CandidateSpec("EN_A1_L05", {"alpha": 1.0, "l1_ratio": 0.5}, 2),
            CandidateSpec("EN_A10_L09", {"alpha": 10.0, "l1_ratio": 0.9}, 3),
        ),
        "S10_RANDOM_FOREST": (
            CandidateSpec(
                "RF_D4_L14_F07",
                {"n_estimators": 64, "max_depth": 4, "min_samples_leaf": 14, "max_features": 0.7},
                1,
            ),
            CandidateSpec(
                "RF_D8_L7_F07",
                {"n_estimators": 64, "max_depth": 8, "min_samples_leaf": 7, "max_features": 0.7},
                2,
            ),
        ),
        "S20_XGBOOST": (
            CandidateSpec(
                "XGB_D2_LR03_C10_R10",
                {"max_depth": 2, "learning_rate": 0.03, "n_estimators": 60, "subsample": 0.8, "colsample_bytree": 0.7, "min_child_weight": 10, "reg_lambda": 10.0},
                1,
            ),
            CandidateSpec(
                "XGB_D3_LR07_C3_R10",
                {"max_depth": 3, "learning_rate": 0.07, "n_estimators": 100, "subsample": 0.8, "colsample_bytree": 0.7, "min_child_weight": 3, "reg_lambda": 10.0},
                2,
            ),
        ),
        "S21_LIGHTGBM": (
            CandidateSpec(
                "LGB_L7_LR03_C30_F07",
                {"num_leaves": 7, "learning_rate": 0.03, "n_estimators": 60, "min_child_samples": 30, "feature_fraction": 0.7, "bagging_fraction": 0.8},
                1,
            ),
            CandidateSpec(
                "LGB_L15_LR07_C30_F07",
                {"num_leaves": 15, "learning_rate": 0.07, "n_estimators": 100, "min_child_samples": 30, "feature_fraction": 0.7, "bagging_fraction": 0.8},
                2,
            ),
        ),
    }
    if model_id not in grids:
        raise ValueError(f"unknown Phase 06 model family: {model_id}")
    return grids[model_id]


def make_estimator(model_id: str, parameters: dict[str, Any]):
    """Construct exactly the requested family with deterministic settings."""

    params = dict(parameters)
    if model_id == "S00_RIDGE":
        return Ridge(**params)
    if model_id == "S01_ELASTIC_NET":
        return ElasticNet(
            **params,
            max_iter=100000,
            tol=1.0e-4,
            selection="cyclic",
            random_state=MODEL_SEED,
        )
    if model_id == "S10_RANDOM_FOREST":
        return RandomForestRegressor(
            **params,
            random_state=MODEL_SEED,
            n_jobs=1,
            bootstrap=True,
        )
    if model_id == "S20_XGBOOST":
        from xgboost import XGBRegressor

        return XGBRegressor(
            **params,
            objective="reg:squarederror",
            tree_method="hist",
            random_state=MODEL_SEED,
            n_jobs=1,
            verbosity=0,
        )
    if model_id == "S21_LIGHTGBM":
        from lightgbm import LGBMRegressor

        return LGBMRegressor(
            **params,
            random_state=MODEL_SEED,
            n_jobs=1,
            verbosity=-1,
            deterministic=True,
            force_col_wise=True,
            bagging_freq=1,
        )
    raise ValueError(f"unknown Phase 06 model family: {model_id}")


def fit_estimator(
    model_id: str,
    parameters: dict[str, Any],
    x_train: np.ndarray,
    y_train: np.ndarray,
):
    """Fit one candidate after validating finite targets and usable rows."""

    y = np.asarray(y_train, dtype=float)
    if x_train.ndim != 2 or len(x_train) != len(y):
        raise ValueError("training matrix/target shape mismatch")
    if not np.isfinite(y).all():
        raise ValueError("target imputation is prohibited; training targets must be finite")
    if model_id not in {"S20_XGBOOST", "S21_LIGHTGBM"} and not np.isfinite(x_train).all():
        raise ValueError("non-native estimator received missing predictors")
    estimator = make_estimator(model_id, parameters)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        estimator.fit(x_train, y)
    if any(issubclass(item.category, ConvergenceWarning) for item in caught):
        raise RuntimeError("candidate did not converge inside the locked iteration budget")
    return estimator


def raw_importance(estimator: Any, model_id: str, feature_names: list[str]) -> pd.DataFrame:
    """Return coefficients or built-in tree diagnostics without selecting on them."""

    import pandas as pd

    if hasattr(estimator, "coef_"):
        values = np.asarray(estimator.coef_, dtype=float).reshape(-1)
        method = "STANDARDIZED_COEFFICIENT"
    elif hasattr(estimator, "feature_importances_"):
        values = np.asarray(estimator.feature_importances_, dtype=float).reshape(-1)
        method = "BUILTIN_TREE_DIAGNOSTIC_ONLY"
    else:
        values = np.zeros(len(feature_names), dtype=float)
        method = "UNAVAILABLE"
    if len(values) != len(feature_names):
        raise ValueError("importance length does not match transformed features")
    return pd.DataFrame(
        {
            "transformed_feature": feature_names,
            "importance": values,
            "importance_method": method,
            "importance_used_for_refit": False,
            "model_id": model_id,
        }
    )


__all__ = [
    "CandidateSpec",
    "MODEL_SEED",
    "candidate_specs",
    "fit_estimator",
    "make_estimator",
    "raw_importance",
]
