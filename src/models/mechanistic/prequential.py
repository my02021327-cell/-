"""Leakage-safe development prequential selection for mechanistic models.

The runner is deliberately model-agnostic.  Callers own construction of the
fold-local kinetic states and fitting of single- or two-pool candidates; this
module owns only chronology, exact-key prediction alignment, inner selection,
and the outer OOF audit trail.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from ...validation.paired_support import (
    OWN_AVAILABLE,
    PAIR_KEY,
    PERSISTENCE_PAIRED,
    merge_persistence_reference,
    support_mask,
    validate_pair_keys,
)
from .fold_selection import (
    DEVELOPMENT_END,
    DEVELOPMENT_START,
    ExpandingFold,
    assert_development_only,
    make_expanding_outer_folds,
    make_inner_expanding_folds,
    rank_candidates,
    score_candidate_predictions,
    select_candidate,
    split_fold_samples,
    validate_development_calendar,
)


FitCandidate = Callable[[pd.DataFrame, str, int, Mapping[str, Any]], Any]
PredictCandidate = Callable[[Any, pd.DataFrame], Any]

_FUTURE_OUTCOME_COLUMNS = {
    "y_true",
    "target_available",
    "error",
    "abs_error",
    "sq_error",
    "delta_true",
    "delta_pred",
    "y_pred",
    "raw_prediction",
    "prediction_available",
    "y_persistence",
    "persistence_available",
}
_PROTECTED_EVALUATION_COLUMNS = {
    *PAIR_KEY,
    "y_true",
    "target_available",
    "y_origin",
    "origin_observed",
    "origin_y_observed",
}
_RESERVED_CANDIDATE_COLUMNS = {
    *PAIR_KEY,
    *_FUTURE_OUTCOME_COLUMNS,
    "y_origin",
    "origin_observed",
    "origin_y_observed",
    "outer_fold",
    "inner_fold",
}


@dataclass(frozen=True)
class PrequentialResult:
    """Complete development-only result of nested prequential selection."""

    oof_predictions: pd.DataFrame
    inner_selection: pd.DataFrame
    fold_audit: pd.DataFrame
    inner_predictions: pd.DataFrame

    @property
    def predictions(self) -> pd.DataFrame:
        """Alias for the selected-candidate outer OOF predictions."""

        return self.oof_predictions


def normalize_candidate_specs(
    candidate_specs: pd.DataFrame | Iterable[Mapping[str, Any]],
) -> pd.DataFrame:
    """Validate and deterministically order fold-local candidate metadata.

    Each candidate needs a stable ``candidate_id`` (``model_id`` is accepted
    as its source).  If ``complexity_rank`` is omitted, single-pool candidates
    default to 1 and candidates containing both ``k_fast`` and ``k_slow``
    default to 2.  Supplying the rank explicitly is preferable for mixed model
    families such as constrained M2.
    """

    if isinstance(candidate_specs, pd.DataFrame):
        result = candidate_specs.copy()
    else:
        result = pd.DataFrame([dict(spec) for spec in candidate_specs])
    if result.empty:
        raise ValueError("candidate_specs must contain at least one candidate")
    if "candidate_id" not in result:
        if "model_id" not in result:
            raise KeyError("candidate_specs require candidate_id or model_id")
        result["candidate_id"] = result["model_id"]
    blank = result["candidate_id"].isna() | result["candidate_id"].astype(
        str
    ).str.strip().eq("")
    if bool(blank.any()):
        raise ValueError("candidate_id values must be non-empty")
    result["candidate_id"] = result["candidate_id"].astype(str)
    if bool(result["candidate_id"].duplicated().any()):
        raise ValueError("candidate_id values must be unique")

    forbidden = sorted(_RESERVED_CANDIDATE_COLUMNS.intersection(result.columns))
    if forbidden:
        raise ValueError(
            "candidate_specs cannot override prediction/sample columns: "
            f"{forbidden}"
        )
    if "model_id" not in result:
        result["model_id"] = result["candidate_id"]

    if "complexity_rank" not in result:
        two_pool = pd.Series(False, index=result.index)
        if {"k_fast", "k_slow"}.issubset(result.columns):
            two_pool = result["k_fast"].notna() & result["k_slow"].notna()
        result["complexity_rank"] = np.where(two_pool, 2.0, 1.0)
    complexity = pd.to_numeric(result["complexity_rank"], errors="coerce")
    if bool((~np.isfinite(complexity) | complexity.lt(0)).any()):
        raise ValueError("complexity_rank must be finite and non-negative")
    result["complexity_rank"] = complexity.astype(float)
    return result.sort_values("candidate_id", kind="stable").reset_index(drop=True)


def _normalize_samples(
    samples: pd.DataFrame,
) -> tuple[pd.DataFrame, str, int]:
    if "y_true" not in samples:
        raise KeyError("mechanistic samples require y_true")
    normalized = validate_pair_keys(
        samples,
        require_unique=True,
        frame_name="mechanistic_samples",
    )
    assert_development_only(normalized)
    targets = normalized["target_name"].drop_duplicates()
    horizons = normalized["horizon"].drop_duplicates()
    if len(targets) != 1 or len(horizons) != 1:
        raise ValueError(
            "run_prequential_selection handles exactly one target and horizon"
        )
    return normalized, str(targets.iloc[0]), int(horizons.iloc[0])


def _prediction_input(evaluation: pd.DataFrame) -> pd.DataFrame:
    """Remove held-out future outcomes before invoking a prediction callback."""

    withheld = [
        column for column in _FUTURE_OUTCOME_COLUMNS if column in evaluation
    ]
    return evaluation.drop(columns=withheld).copy()


def _physical_prediction_frame(output: Any, expected_rows: int) -> pd.DataFrame:
    """Adapt kernel model ``PhysicalPredictions``-like results by position."""

    if not hasattr(output, "y_pred"):
        raise TypeError(
            "predict_candidate must return a DataFrame, Series, array, or an "
            "object exposing y_pred"
        )
    prediction = np.asarray(getattr(output, "y_pred"), dtype=float)
    if prediction.ndim != 1 or len(prediction) != expected_rows:
        raise ValueError("callback prediction length must equal evaluation rows")
    payload: dict[str, Any] = {"y_pred": prediction}
    attribute_map = {
        "raw": "raw_prediction",
        "available": "prediction_available",
        "negative_raw": "negative_raw_prediction",
    }
    for attribute, column in attribute_map.items():
        if hasattr(output, attribute):
            values = np.asarray(getattr(output, attribute))
            if values.ndim != 1 or len(values) != expected_rows:
                raise ValueError(f"callback {attribute} length is misaligned")
            payload[column] = values
    return pd.DataFrame(payload)


def _callback_frame(output: Any, expected_rows: int) -> pd.DataFrame:
    if isinstance(output, pd.DataFrame):
        return output.copy()
    if isinstance(output, pd.Series):
        values = output.to_numpy()
        if len(values) != expected_rows:
            raise ValueError("callback prediction length must equal evaluation rows")
        return pd.DataFrame({"y_pred": values})
    if hasattr(output, "y_pred"):
        return _physical_prediction_frame(output, expected_rows)
    values = np.asarray(output)
    if values.ndim != 1 or len(values) != expected_rows:
        raise ValueError("callback prediction must be an aligned 1D result")
    return pd.DataFrame({"y_pred": values})


def _attach_callback_output(
    evaluation: pd.DataFrame,
    callback_output: Any,
) -> pd.DataFrame:
    """Align callback predictions without positional joins when keys exist."""

    output = _callback_frame(callback_output, len(evaluation))
    if "y_pred" not in output:
        raise KeyError("predict_candidate output requires y_pred")
    has_any_key = any(column in output for column in PAIR_KEY)
    has_all_keys = all(column in output for column in PAIR_KEY)
    if has_any_key and not has_all_keys:
        raise KeyError("keyed callback output must contain the complete pair key")

    protected = _PROTECTED_EVALUATION_COLUMNS.intersection(output.columns)
    payload_columns = [
        column
        for column in output.columns
        if column not in PAIR_KEY and column not in protected
    ]
    if "y_pred" not in payload_columns:
        payload_columns.append("y_pred")
    if has_all_keys:
        keyed = validate_pair_keys(
            output,
            require_unique=True,
            frame_name="callback_predictions",
        )
        expected_keys = evaluation.loc[:, list(PAIR_KEY)]
        membership = keyed.loc[:, list(PAIR_KEY)].merge(
            expected_keys,
            on=list(PAIR_KEY),
            how="left",
            indicator=True,
            validate="one_to_one",
        )
        if bool(membership["_merge"].ne("both").any()):
            raise ValueError("callback returned prediction keys outside evaluation")
        base = evaluation.drop(
            columns=[column for column in payload_columns if column in evaluation]
        )
        result = base.merge(
            keyed.loc[:, [*PAIR_KEY, *payload_columns]],
            on=list(PAIR_KEY),
            how="left",
            sort=False,
            validate="one_to_one",
        )
    else:
        if len(output) != len(evaluation):
            raise ValueError("unkeyed callback output must cover every evaluation row")
        result = evaluation.copy()
        for column in payload_columns:
            result[column] = output[column].to_numpy()

    prediction = pd.to_numeric(result["y_pred"], errors="coerce")
    finite_prediction = pd.Series(
        np.isfinite(prediction.to_numpy(dtype=float)), index=result.index
    )
    if "prediction_available" in result:
        available = result["prediction_available"].fillna(False).astype(bool)
        available &= finite_prediction
    else:
        available = finite_prediction
    result["y_pred"] = prediction.where(finite_prediction, np.nan)
    result["prediction_available"] = available
    if "raw_prediction" not in result:
        result["raw_prediction"] = result["y_pred"]
    if "clipped_prediction" not in result:
        result["clipped_prediction"] = result["y_pred"]
    raw_numeric = pd.to_numeric(result["raw_prediction"], errors="coerce")
    negative_raw = np.isfinite(raw_numeric.to_numpy(dtype=float)) & raw_numeric.lt(0)
    if "negative_raw_prediction" not in result:
        result["negative_raw_prediction"] = negative_raw
    if "prediction_was_clipped" not in result:
        result["prediction_was_clipped"] = negative_raw

    truth = pd.to_numeric(result["y_true"], errors="coerce")
    finite_truth = pd.Series(
        np.isfinite(truth.to_numpy(dtype=float)), index=result.index
    )
    if "target_available" in result:
        target_available = result["target_available"].fillna(False).astype(bool)
        target_available &= finite_truth
    else:
        target_available = finite_truth
    result["target_available"] = target_available
    scored = available & target_available
    error = pd.Series(np.nan, index=result.index, dtype=float)
    error.loc[scored] = result.loc[scored, "y_pred"] - truth.loc[scored]
    result["error"] = error
    result["abs_error"] = error.abs()
    result["sq_error"] = error.pow(2)

    result["delta_true"] = np.nan
    result["delta_pred"] = np.nan
    if "y_origin" in result:
        origin = pd.to_numeric(result["y_origin"], errors="coerce")
        delta_support = scored & np.isfinite(origin.to_numpy(dtype=float))
        result.loc[delta_support, "delta_true"] = (
            truth.loc[delta_support] - origin.loc[delta_support]
        )
        result.loc[delta_support, "delta_pred"] = (
            result.loc[delta_support, "y_pred"] - origin.loc[delta_support]
        )
    return result


def _model_metadata(model: Any) -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    for attribute in (
        "beta0",
        "beta1",
        "beta_fast",
        "beta_slow",
        "k_per_d",
        "k_fast",
        "k_slow",
        "kernel_max_lag",
        "load_basis",
        "model_id",
        "uses_target_history",
        "n_observations",
    ):
        if hasattr(model, attribute):
            value = getattr(model, attribute)
            if not callable(value):
                metadata[attribute] = value
    return metadata


def _fit_predict(
    training: pd.DataFrame,
    evaluation: pd.DataFrame,
    *,
    target_name: str,
    horizon: int,
    candidate_spec: Mapping[str, Any],
    fit_candidate: FitCandidate,
    predict_candidate: PredictCandidate,
) -> pd.DataFrame:
    if training.empty:
        raise ValueError("candidate fitting requires at least one matured row")
    max_target = pd.to_datetime(training["target_date"]).max()
    first_eval = pd.to_datetime(evaluation["origin_date"]).min()
    if max_target >= first_eval:
        raise AssertionError("unmatured target reached a candidate fit callback")
    model = fit_candidate(
        training.copy(),
        target_name,
        horizon,
        dict(candidate_spec),
    )
    callback_output = predict_candidate(model, _prediction_input(evaluation))
    result = _attach_callback_output(evaluation, callback_output)
    for column, value in candidate_spec.items():
        result[column] = value
    for column, value in _model_metadata(model).items():
        if column not in result:
            result[column] = value
    result["candidate_id"] = str(candidate_spec["candidate_id"])
    if "model_id" not in result:
        result["model_id"] = result["candidate_id"]
    result["max_training_target_date"] = max_target
    result["n_training_rows_received"] = int(len(training))
    return result


def _observed_target_count(frame: pd.DataFrame) -> int:
    target = pd.to_numeric(frame["y_true"], errors="coerce")
    available = pd.Series(
        np.isfinite(target.to_numpy(dtype=float)), index=frame.index
    )
    if "target_available" in frame:
        available &= frame["target_available"].fillna(False).astype(bool)
    return int(available.sum())


def _fold_audit_record(
    fold: ExpandingFold,
    horizon: int,
    target_name: str,
    training: pd.DataFrame,
    evaluation: pd.DataFrame,
) -> dict[str, Any]:
    record = fold.as_record(horizon)
    record.update(
        {
            "target_name": target_name,
            "n_training_rows": int(len(training)),
            "n_training_targets_observed": _observed_target_count(training),
            "n_evaluation_rows": int(len(evaluation)),
            "n_evaluation_targets_observed": _observed_target_count(evaluation),
            "actual_max_training_origin_date": (
                pd.to_datetime(training["origin_date"]).max()
                if not training.empty
                else pd.NaT
            ),
            "actual_max_training_target_date": (
                pd.to_datetime(training["target_date"]).max()
                if not training.empty
                else pd.NaT
            ),
        }
    )
    return record


def run_prequential_selection(
    samples: pd.DataFrame,
    candidate_specs: pd.DataFrame | Iterable[Mapping[str, Any]],
    persistence_reference: pd.DataFrame,
    *,
    fit_candidate: FitCandidate,
    predict_candidate: PredictCandidate,
    outer_folds: Sequence[ExpandingFold] | None = None,
    development_dates: Sequence[Any] | pd.DatetimeIndex | None = None,
    smape_epsilon: float = 1.0e-12,
) -> PrequentialResult:
    """Run nested expanding-window selection and selected-candidate OOF.

    Callback contract
    -----------------
    ``fit_candidate(train, target_name, horizon, candidate_spec)`` receives
    only rows whose target date is strictly earlier than the next evaluation
    origin. ``predict_candidate(model, eval_origins)`` never receives
    ``y_true`` or target-derived error columns. It may return an aligned 1D
    prediction, a ``PhysicalPredictions``-like object, or a DataFrame. A keyed
    DataFrame is joined only on the immutable Phase 04 pair key.

    Selection is pooled over each outer fold's inner OOF rows: lowest
    own-support RMSE, then higher exact persistence-paired skill, then the
    supplied complexity rank. No row with a target after 2021 is accepted.
    """

    frame, target_name, horizon = _normalize_samples(samples)
    specs = normalize_candidate_specs(candidate_specs)
    if development_dates is None:
        development_dates = pd.date_range(
            DEVELOPMENT_START, DEVELOPMENT_END, freq="D"
        )
    calendar = validate_development_calendar(development_dates)
    folds = (
        tuple(outer_folds)
        if outer_folds is not None
        else make_expanding_outer_folds(calendar)
    )
    if not folds:
        raise ValueError("outer_folds must contain at least one fold")

    oof_parts: list[pd.DataFrame] = []
    inner_prediction_parts: list[pd.DataFrame] = []
    inner_selection_parts: list[pd.DataFrame] = []
    audit_rows: list[dict[str, Any]] = []
    spec_records = specs.to_dict("records")

    for outer_fold in folds:
        outer_training, outer_evaluation = split_fold_samples(
            frame, outer_fold, horizon=horizon
        )
        audit = _fold_audit_record(
            outer_fold,
            horizon,
            target_name,
            outer_training,
            outer_evaluation,
        )
        if outer_evaluation.empty:
            audit.update(
                {
                    "fold_status": "NO_EVALUABLE_ORIGINS",
                    "selected_candidate_id": None,
                    "n_inner_folds": 0,
                }
            )
            audit_rows.append(audit)
            continue

        inner_folds = make_inner_expanding_folds(
            outer_fold.eval_start,
            horizon,
            calendar,
        )
        if not inner_folds:
            raise ValueError(
                f"{outer_fold.fold_id} has no eligible inner validation fold"
            )

        outer_inner_parts: list[pd.DataFrame] = []
        for spec in spec_records:
            candidate_id = str(spec["candidate_id"])
            for inner_fold in inner_folds:
                inner_training, inner_evaluation = split_fold_samples(
                    frame, inner_fold, horizon=horizon
                )
                if inner_evaluation.empty:
                    continue
                try:
                    prediction = _fit_predict(
                        inner_training,
                        inner_evaluation,
                        target_name=target_name,
                        horizon=horizon,
                        candidate_spec=spec,
                        fit_candidate=fit_candidate,
                        predict_candidate=predict_candidate,
                    )
                except Exception as exc:
                    raise RuntimeError(
                        f"inner fit/predict failed for {outer_fold.fold_id}/"
                        f"{inner_fold.fold_id}/{candidate_id}"
                    ) from exc
                prediction["selection_stage"] = "INNER"
                prediction["outer_fold"] = outer_fold.fold_id
                prediction["inner_fold"] = inner_fold.fold_id
                prediction["inner_eval_start"] = inner_fold.eval_start
                prediction["inner_eval_end"] = inner_fold.eval_end
                outer_inner_parts.append(prediction)

        if not outer_inner_parts:
            raise ValueError(f"{outer_fold.fold_id} produced no inner predictions")
        outer_inner = pd.concat(outer_inner_parts, ignore_index=True)
        scores = score_candidate_predictions(
            outer_inner,
            persistence_reference,
            candidate_columns=("candidate_id",),
            smape_epsilon=smape_epsilon,
        )
        missing_metadata = [
            column
            for column in specs.columns
            if column != "candidate_id" and column not in scores.columns
        ]
        if missing_metadata:
            scores = scores.merge(
                specs.loc[:, ["candidate_id", *missing_metadata]],
                on="candidate_id",
                how="left",
                validate="one_to_one",
            )
        selected = select_candidate(scores)
        ranked = rank_candidates(scores)
        ranked["outer_fold"] = outer_fold.fold_id
        ranked["target_name"] = target_name
        ranked["horizon"] = horizon
        ranked["n_inner_folds"] = int(len(inner_folds))
        ranked["inner_eval_first"] = min(fold.eval_start for fold in inner_folds)
        ranked["inner_eval_last"] = max(fold.eval_end for fold in inner_folds)
        inner_selection_parts.append(ranked)
        inner_prediction_parts.append(outer_inner)

        selected_id = str(selected["candidate_id"])
        selected_spec = next(
            spec for spec in spec_records if str(spec["candidate_id"]) == selected_id
        )
        try:
            outer_prediction = _fit_predict(
                outer_training,
                outer_evaluation,
                target_name=target_name,
                horizon=horizon,
                candidate_spec=selected_spec,
                fit_candidate=fit_candidate,
                predict_candidate=predict_candidate,
            )
        except Exception as exc:
            raise RuntimeError(
                f"outer fit/predict failed for {outer_fold.fold_id}/{selected_id}"
            ) from exc
        cutoffs = outer_fold.horizon_cutoffs(horizon)
        outer_prediction["selection_stage"] = "OUTER_OOF"
        outer_prediction["outer_fold"] = outer_fold.fold_id
        outer_prediction["inner_selected"] = True
        outer_prediction["selection_reason"] = selected["selection_reason"]
        outer_prediction["max_train_target_date_cutoff"] = cutoffs[
            "max_train_target_date"
        ]
        outer_prediction["max_train_origin_date_cutoff"] = cutoffs[
            "max_train_origin_date"
        ]
        oof_parts.append(outer_prediction)

        audit.update(
            {
                "fold_status": "COMPLETE",
                "selected_candidate_id": selected_id,
                "selection_reason": selected["selection_reason"],
                "n_inner_folds": int(len(inner_folds)),
                "n_inner_prediction_rows": int(len(outer_inner)),
            }
        )
        audit_rows.append(audit)

    oof = pd.concat(oof_parts, ignore_index=True) if oof_parts else pd.DataFrame()
    if not oof.empty:
        assert_development_only(oof)
        duplicated = oof.duplicated(list(PAIR_KEY), keep=False)
        if bool(duplicated.any()):
            raise AssertionError("outer OOF rows are not unique by exact pair key")
        reference = validate_pair_keys(
            persistence_reference,
            require_unique=True,
            frame_name="phase04_persistence_reference",
        )
        reference = reference.loc[
            reference["origin_date"].between(
                DEVELOPMENT_START, DEVELOPMENT_END, inclusive="both"
            )
            & reference["target_date"].le(DEVELOPMENT_END)
        ].copy()
        oof = merge_persistence_reference(oof, reference)
        own_support = support_mask(oof, OWN_AVAILABLE)
        paired_support = support_mask(oof, PERSISTENCE_PAIRED)
        oof["support_type"] = np.select(
            [paired_support, own_support],
            [PERSISTENCE_PAIRED, OWN_AVAILABLE],
            default="UNAVAILABLE",
        )
        oof = oof.sort_values(["origin_date", "target_date"], kind="stable")
        oof = oof.reset_index(drop=True)
    inner_selection = (
        pd.concat(inner_selection_parts, ignore_index=True)
        if inner_selection_parts
        else pd.DataFrame()
    )
    inner_predictions = (
        pd.concat(inner_prediction_parts, ignore_index=True)
        if inner_prediction_parts
        else pd.DataFrame()
    )
    return PrequentialResult(
        oof_predictions=oof,
        inner_selection=inner_selection,
        fold_audit=pd.DataFrame(audit_rows),
        inner_predictions=inner_predictions,
    )


run_prequential_oof = run_prequential_selection


__all__ = [
    "FitCandidate",
    "PredictCandidate",
    "PrequentialResult",
    "normalize_candidate_specs",
    "run_prequential_oof",
    "run_prequential_selection",
]
