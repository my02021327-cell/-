"""Constrained M2 residual-inertia correction and fold-local tau selection."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from ...validation.baseline_metrics import compute_metrics, skill_vs_persistence
from ...validation.paired_support import (
    OWN_AVAILABLE,
    PERSISTENCE_PAIRED,
    merge_persistence_reference,
    support_mask,
    validate_pair_keys,
)
from .fold_selection import (
    DEVELOPMENT_END,
    DEVELOPMENT_START,
    assert_development_only,
    rank_candidates,
    score_candidate_predictions,
    select_candidate,
)


TAU_RES_CANDIDATES = (1, 2, 3, 5, 7, 10, 14, 21, 30)


@dataclass(frozen=True)
class ResidualTauSelection:
    """Fold-local tau result with complete audit tables."""

    selected_tau: int
    scores: pd.DataFrame
    candidate_predictions: pd.DataFrame


def _positive_scalar(value: Any, name: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be numeric") from exc
    if not np.isfinite(parsed) or parsed <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return parsed


def residual_decay_factor(horizon: Any, tau_res: Any) -> Any:
    """Return the fixed coefficient ``exp(-h / tau_res)`` (no free gamma)."""

    tau = _positive_scalar(tau_res, "tau_res")
    horizon_array = np.asarray(horizon, dtype=float)
    if np.any(~np.isfinite(horizon_array)) or np.any(horizon_array <= 0):
        raise ValueError("horizon must contain finite positive values")
    result = np.exp(-horizon_array / tau)
    return float(result) if result.ndim == 0 else result


def residual_inertia_correction(
    base_forecast: Any,
    current_prediction: Any,
    y_origin: Any,
    horizon: Any,
    tau_res: Any,
) -> Any:
    """Apply ``base + (y(t)-current_fit(t))*exp(-h/tau)``.

    Rows lacking the actually observed origin target, current mechanistic
    estimate, or base forecast return NaN. There is no LOCF and no fitted AR
    multiplier.
    """

    base, current, origin, horizons = np.broadcast_arrays(
        np.asarray(base_forecast, dtype=float),
        np.asarray(current_prediction, dtype=float),
        np.asarray(y_origin, dtype=float),
        np.asarray(horizon, dtype=float),
    )
    decay = np.asarray(residual_decay_factor(horizons, tau_res), dtype=float)
    valid = (
        np.isfinite(base)
        & np.isfinite(current)
        & np.isfinite(origin)
        & np.isfinite(decay)
    )
    corrected = np.full(base.shape, np.nan, dtype=float)
    corrected[valid] = base[valid] + (origin[valid] - current[valid]) * decay[valid]
    return float(corrected) if corrected.ndim == 0 else corrected


def apply_residual_inertia(
    predictions: pd.DataFrame,
    tau_res: int,
    *,
    base_prediction_column: str = "y_pred_M1",
    current_prediction_column: str = "y_pred_current",
    origin_target_column: str = "y_origin",
    output_column: str = "y_pred",
    training_target_end_column: str | None = "current_model_max_training_target_date",
) -> pd.DataFrame:
    """Materialize an audited non-negative M2 prediction DataFrame."""

    tau = int(_positive_scalar(tau_res, "tau_res"))
    if tau != tau_res:
        raise ValueError("tau_res must be an integer number of days")
    required = [
        base_prediction_column,
        current_prediction_column,
        origin_target_column,
        "horizon",
    ]
    missing = [column for column in required if column not in predictions]
    if missing:
        raise KeyError(f"residual-inertia frame missing columns: {missing}")
    result = predictions.copy()
    if {"target_name", "origin_date", "target_date"}.issubset(result.columns):
        assert_development_only(result)

    base = pd.to_numeric(result[base_prediction_column], errors="coerce")
    current = pd.to_numeric(result[current_prediction_column], errors="coerce")
    origin = pd.to_numeric(result[origin_target_column], errors="coerce")
    horizon = pd.to_numeric(result["horizon"], errors="coerce")
    raw = np.asarray(
        residual_inertia_correction(base, current, origin, horizon, tau),
        dtype=float,
    )

    origin_observed = pd.Series(np.isfinite(origin), index=result.index)
    for flag in ("origin_y_observed", "origin_observed"):
        if flag in result:
            origin_observed &= result[flag].fillna(False).astype(bool)
    raw[~origin_observed.to_numpy(dtype=bool)] = np.nan

    if training_target_end_column and training_target_end_column in result:
        training_end = pd.to_datetime(
            result[training_target_end_column], errors="coerce"
        )
        origin_dates = pd.to_datetime(result["origin_date"], errors="coerce")
        available = np.isfinite(raw)
        invalid_training = available & (
            training_end.isna() | training_end.ge(origin_dates)
        )
        if bool(np.any(invalid_training)):
            raise ValueError(
                "current mechanistic model used an unmatured target at origin"
            )

    decay = np.asarray(residual_decay_factor(horizon.to_numpy(), tau), dtype=float)
    residual = origin.to_numpy(dtype=float) - current.to_numpy(dtype=float)
    residual[~origin_observed.to_numpy(dtype=bool)] = np.nan
    correction = residual * decay
    clipped = np.where(np.isfinite(raw), np.maximum(raw, 0.0), np.nan)

    if output_column == base_prediction_column:
        result["y_pred_M1"] = base
    result["base_prediction_M1"] = base
    result["current_mechanistic_prediction"] = current
    result["origin_residual"] = residual
    result["residual_decay_factor"] = decay
    result["residual_correction"] = correction
    result["raw_prediction"] = raw
    result[output_column] = clipped
    result["prediction_available"] = np.isfinite(clipped)
    result["negative_raw_prediction"] = np.isfinite(raw) & (raw < 0)
    result["clipped_prediction"] = clipped
    result["prediction_was_clipped"] = result["negative_raw_prediction"]
    result["tau_res"] = tau
    result["uses_target_history"] = "TRUE_CONSTRAINED"
    return result


def _validate_tau_grid(tau_candidates: Sequence[int]) -> tuple[int, ...]:
    parsed = tuple(int(_positive_scalar(value, "tau candidate")) for value in tau_candidates)
    if not parsed:
        raise ValueError("tau_candidates must contain at least one value")
    if any(float(raw) != parsed_value for raw, parsed_value in zip(tau_candidates, parsed)):
        raise ValueError("tau candidates must be integer days")
    if len(set(parsed)) != len(parsed):
        raise ValueError("tau candidates must be unique")
    return parsed


def score_residual_tau_candidates(
    inner_predictions: pd.DataFrame,
    persistence_reference: pd.DataFrame,
    *,
    tau_candidates: Sequence[int] = TAU_RES_CANDIDATES,
    base_prediction_column: str = "y_pred_M1",
    current_prediction_column: str = "y_pred_current",
    origin_target_column: str = "y_origin",
    training_target_end_column: str | None = "current_model_max_training_target_date",
    smape_epsilon: float = 1.0e-12,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return tau score rows and all fold-local M2 candidate predictions."""

    assert_development_only(inner_predictions)
    tau_grid = _validate_tau_grid(tau_candidates)
    candidate_frames: list[pd.DataFrame] = []
    for tau in tau_grid:
        candidate = apply_residual_inertia(
            inner_predictions,
            tau,
            base_prediction_column=base_prediction_column,
            current_prediction_column=current_prediction_column,
            origin_target_column=origin_target_column,
            output_column="y_pred",
            training_target_end_column=training_target_end_column,
        )
        candidate["candidate_id"] = f"M2_TAU_{tau:03d}"
        candidate["complexity_rank"] = 2
        candidate_frames.append(candidate)
    candidates = pd.concat(candidate_frames, ignore_index=True)
    scores = score_candidate_predictions(
        candidates,
        persistence_reference,
        candidate_columns=("candidate_id",),
        smape_epsilon=smape_epsilon,
    )

    # Add the requested M1/M2 common-support audit on exactly the M2 rows.
    development_reference = validate_pair_keys(
        persistence_reference,
        require_unique=True,
        frame_name="phase04_persistence_reference",
    )
    development_reference = development_reference.loc[
        development_reference["origin_date"].between(
            DEVELOPMENT_START, DEVELOPMENT_END, inclusive="both"
        )
        & development_reference["target_date"].le(DEVELOPMENT_END)
    ].copy()
    paired = pd.concat(
        [
            merge_persistence_reference(group, development_reference)
            for _, group in candidates.groupby("candidate_id", sort=True)
        ],
        ignore_index=True,
    )
    audit_rows: list[dict[str, Any]] = []
    for candidate_id, group in paired.groupby("candidate_id", sort=True):
        own = group.loc[support_mask(group, OWN_AVAILABLE)].copy()
        m1_metrics = compute_metrics(
            own["y_true"],
            own["base_prediction_M1"],
            y_origin=own[origin_target_column],
            smape_epsilon=smape_epsilon,
        )
        m2_metrics = compute_metrics(
            own["y_true"],
            own["y_pred"],
            y_origin=own[origin_target_column],
            smape_epsilon=smape_epsilon,
        )
        common = group.loc[support_mask(group, PERSISTENCE_PAIRED)].copy()
        skill_m1 = skill_vs_persistence(
            common["y_true"],
            common["base_prediction_M1"],
            common["y_persistence"],
        )
        skill_m2 = skill_vs_persistence(
            common["y_true"],
            common["y_pred"],
            common["y_persistence"],
        )
        audit_rows.append(
            {
                "candidate_id": candidate_id,
                "n_common": int(len(own)),
                "RMSE_M1": m1_metrics["RMSE"],
                "RMSE_M2": m2_metrics["RMSE"],
                "persistence_triple_n": int(len(common)),
                "Skill_M1": skill_m1,
                "Skill_M2": skill_m2,
            }
        )
    scores = scores.merge(pd.DataFrame(audit_rows), on="candidate_id", how="left")
    return scores, candidates


def select_residual_tau(
    inner_predictions: pd.DataFrame,
    persistence_reference: pd.DataFrame,
    *,
    tau_candidates: Sequence[int] = TAU_RES_CANDIDATES,
    base_prediction_column: str = "y_pred_M1",
    current_prediction_column: str = "y_pred_current",
    origin_target_column: str = "y_origin",
    training_target_end_column: str | None = "current_model_max_training_target_date",
    smape_epsilon: float = 1.0e-12,
) -> ResidualTauSelection:
    """Select tau by own RMSE, then paired skill, then deterministic simplicity."""

    scores, candidates = score_residual_tau_candidates(
        inner_predictions,
        persistence_reference,
        tau_candidates=tau_candidates,
        base_prediction_column=base_prediction_column,
        current_prediction_column=current_prediction_column,
        origin_target_column=origin_target_column,
        training_target_end_column=training_target_end_column,
        smape_epsilon=smape_epsilon,
    )
    selected = select_candidate(scores)
    ranked = rank_candidates(scores)
    selected_tau = int(selected["tau_res"])
    ranked["tau"] = ranked["tau_res"].astype(int)
    ranked["selected_tau"] = selected_tau
    for fold_column in ("outer_fold", "fold_id", "fold"):
        if fold_column in inner_predictions:
            values = inner_predictions[fold_column].dropna().drop_duplicates()
            if len(values) > 1:
                raise ValueError(
                    f"fold-local tau selection received multiple {fold_column} values"
                )
            if len(values) == 1:
                ranked[fold_column] = values.iloc[0]
    return ResidualTauSelection(
        selected_tau=selected_tau,
        scores=ranked,
        candidate_predictions=candidates,
    )


constrained_residual_inertia = residual_inertia_correction


__all__ = [
    "ResidualTauSelection",
    "TAU_RES_CANDIDATES",
    "apply_residual_inertia",
    "constrained_residual_inertia",
    "residual_decay_factor",
    "residual_inertia_correction",
    "score_residual_tau_candidates",
    "select_residual_tau",
]
