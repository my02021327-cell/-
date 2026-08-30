"""Deterministic development-only folds and mechanistic candidate selection."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from ...validation.baseline_metrics import compute_metrics, skill_vs_persistence
from ...validation.paired_support import (
    OWN_AVAILABLE,
    PAIR_KEY,
    PERSISTENCE_PAIRED,
    merge_persistence_reference,
    support_mask,
    validate_pair_keys,
)


DEVELOPMENT_START = pd.Timestamp("2018-01-01")
DEVELOPMENT_END = pd.Timestamp("2021-12-31")
OUTER_INITIAL_TRAIN_DAYS = 365
OUTER_EVAL_BLOCK_DAYS = 90
INNER_INITIAL_TRAIN_DAYS = 270
INNER_EVAL_BLOCK_DAYS = 60


@dataclass(frozen=True)
class ExpandingFold:
    """One chronological expanding fold expressed in calendar dates."""

    fold_id: str
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    eval_start: pd.Timestamp
    eval_end: pd.Timestamp

    def __post_init__(self) -> None:
        if not isinstance(self.fold_id, str) or not self.fold_id.strip():
            raise ValueError("fold_id must be a non-empty string")
        dates = tuple(
            _normalize_date(value, name)
            for value, name in (
                (self.train_start, "train_start"),
                (self.train_end, "train_end"),
                (self.eval_start, "eval_start"),
                (self.eval_end, "eval_end"),
            )
        )
        object.__setattr__(self, "train_start", dates[0])
        object.__setattr__(self, "train_end", dates[1])
        object.__setattr__(self, "eval_start", dates[2])
        object.__setattr__(self, "eval_end", dates[3])
        if self.train_start > self.train_end:
            raise ValueError("fold train_start must not exceed train_end")
        if self.train_end >= self.eval_start:
            raise ValueError("training calendar must end before evaluation")
        if self.eval_start > self.eval_end:
            raise ValueError("fold eval_start must not exceed eval_end")
        if self.train_start < DEVELOPMENT_START or self.eval_end > DEVELOPMENT_END:
            raise ValueError("Phase 05 folds must stay inside 2018-2021")

    def horizon_cutoffs(self, horizon: int) -> dict[str, pd.Timestamp]:
        """Return the strict target-maturation cutoffs for one horizon."""

        return training_maturation_cutoffs(self.eval_start, horizon)

    def as_record(self, horizon: int | None = None) -> dict[str, Any]:
        eval_calendar_days = int((self.eval_end - self.eval_start).days + 1)
        nominal_eval_days = (
            OUTER_EVAL_BLOCK_DAYS
            if self.fold_id.startswith("OUTER_")
            else INNER_EVAL_BLOCK_DAYS
        )
        record: dict[str, Any] = {
            "fold_id": self.fold_id,
            "train_start": self.train_start,
            "train_end": self.train_end,
            "eval_start": self.eval_start,
            "eval_end": self.eval_end,
            "eval_calendar_days": eval_calendar_days,
            "partial_fold": eval_calendar_days < nominal_eval_days,
        }
        if horizon is not None:
            horizon_value = _validate_horizon(horizon)
            record.update(self.horizon_cutoffs(horizon_value))
            record["horizon"] = horizon_value
            last_origin = min(
                self.eval_end,
                DEVELOPMENT_END - pd.Timedelta(days=horizon_value),
            )
            if last_origin < self.eval_start:
                record["max_evaluable_origin_date"] = pd.NaT
                record["n_evaluable_origins"] = 0
            else:
                record["max_evaluable_origin_date"] = last_origin
                record["n_evaluable_origins"] = int(
                    (last_origin - self.eval_start).days + 1
                )
        return record


def _validate_horizon(horizon: int) -> int:
    if isinstance(horizon, bool):
        raise TypeError("horizon must be a positive integer")
    try:
        value = int(horizon)
    except (TypeError, ValueError) as exc:
        raise TypeError("horizon must be a positive integer") from exc
    if value <= 0 or value != horizon:
        raise ValueError("horizon must be a positive integer")
    return value


def _normalize_date(value: Any, name: str) -> pd.Timestamp:
    parsed = pd.Timestamp(value)
    if pd.isna(parsed) or parsed.tz is not None or parsed != parsed.normalize():
        raise ValueError(f"{name} must be a timezone-naive midnight date")
    return parsed


def _positive_integer(value: Any, name: str) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be a positive integer")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be a positive integer") from exc
    if parsed <= 0 or parsed != value:
        raise ValueError(f"{name} must be a positive integer")
    return parsed


def training_maturation_cutoffs(
    evaluation_origin: Any,
    horizon: int,
) -> dict[str, pd.Timestamp]:
    """Return the last admissible target and origin dates for fitting.

    For an evaluation whose first origin is ``t0``, a training row is mature
    only when ``target_date < t0``. Therefore an ``h``-day row can have an
    origin no later than ``t0 - h - 1 day``. The returned dates are inclusive
    maxima and are safe to persist directly in fold audit tables.
    """

    eval_origin = _normalize_date(evaluation_origin, "evaluation_origin")
    if not (DEVELOPMENT_START < eval_origin <= DEVELOPMENT_END):
        raise ValueError("evaluation_origin must lie after development starts")
    horizon_value = _validate_horizon(horizon)
    return {
        "max_train_target_date": eval_origin - pd.Timedelta(days=1),
        "max_train_origin_date": eval_origin
        - pd.Timedelta(days=horizon_value + 1),
    }


def validate_development_calendar(
    dates: Iterable[Any] | pd.DatetimeIndex,
    *,
    development_start: Any = DEVELOPMENT_START,
    development_end: Any = DEVELOPMENT_END,
) -> pd.DatetimeIndex:
    """Validate a complete daily calendar containing no post-2021 dates."""

    start = _normalize_date(development_start, "development_start")
    end = _normalize_date(development_end, "development_end")
    if start < DEVELOPMENT_START or end > DEVELOPMENT_END or start > end:
        raise ValueError("Phase 05 calendar must stay inside 2018-2021")
    parsed = pd.DatetimeIndex(pd.to_datetime(list(dates), errors="raise"))
    if parsed.tz is not None or parsed.hasnans:
        raise ValueError("development dates must be timezone-naive and valid")
    if not parsed.equals(parsed.normalize()):
        raise ValueError("development dates must be midnight calendar dates")
    if parsed.has_duplicates or not parsed.is_monotonic_increasing:
        raise ValueError("development dates must be unique and sorted")
    expected = pd.date_range(start, end, freq="D")
    if not parsed.equals(expected):
        raise ValueError("development dates must be the complete requested calendar")
    return parsed


def make_expanding_outer_folds(
    dates: Iterable[Any] | pd.DatetimeIndex | None = None,
    *,
    development_start: Any = DEVELOPMENT_START,
    development_end: Any = DEVELOPMENT_END,
    initial_train_days: int = OUTER_INITIAL_TRAIN_DAYS,
    eval_block_days: int = OUTER_EVAL_BLOCK_DAYS,
) -> tuple[ExpandingFold, ...]:
    """Create expanding 365-day-initial, next-90-day development folds."""

    start = _normalize_date(development_start, "development_start")
    end = _normalize_date(development_end, "development_end")
    if dates is None:
        dates = pd.date_range(start, end, freq="D")
    calendar = validate_development_calendar(
        dates,
        development_start=start,
        development_end=end,
    )
    initial_days = _positive_integer(initial_train_days, "initial_train_days")
    block_days = _positive_integer(eval_block_days, "eval_block_days")
    first_eval = calendar[0] + pd.Timedelta(days=initial_days)
    if first_eval > calendar[-1]:
        return ()

    folds: list[ExpandingFold] = []
    eval_start = first_eval
    fold_number = 1
    while eval_start <= calendar[-1]:
        eval_end = min(
            eval_start + pd.Timedelta(days=block_days - 1),
            calendar[-1],
        )
        folds.append(
            ExpandingFold(
                fold_id=f"OUTER_{fold_number:02d}",
                train_start=calendar[0],
                train_end=eval_start - pd.Timedelta(days=1),
                eval_start=eval_start,
                eval_end=eval_end,
            )
        )
        eval_start = eval_end + pd.Timedelta(days=1)
        fold_number += 1
    return tuple(folds)


def make_inner_expanding_folds(
    outer_eval_start: Any,
    horizon: int,
    dates: Iterable[Any] | pd.DatetimeIndex | None = None,
    *,
    development_start: Any = DEVELOPMENT_START,
    development_end: Any = DEVELOPMENT_END,
    initial_train_days: int = INNER_INITIAL_TRAIN_DAYS,
    eval_block_days: int = INNER_EVAL_BLOCK_DAYS,
) -> tuple[ExpandingFold, ...]:
    """Create fixed 270-day-initial, 60-day inner folds inside outer train."""

    start = _normalize_date(development_start, "development_start")
    end = _normalize_date(development_end, "development_end")
    outer_start = _normalize_date(outer_eval_start, "outer_eval_start")
    horizon_value = _validate_horizon(horizon)
    if not (start < outer_start <= end):
        raise ValueError("outer_eval_start must lie inside development")
    if dates is None:
        dates = pd.date_range(start, end, freq="D")
    calendar = validate_development_calendar(
        dates,
        development_start=start,
        development_end=end,
    )
    initial_days = _positive_integer(initial_train_days, "initial_train_days")
    block_days = _positive_integer(eval_block_days, "eval_block_days")

    # Inner evaluation outcomes must already be mature before the outer
    # evaluation starts: target_date(inner) < outer_eval_start.
    final_inner_origin = outer_start - pd.Timedelta(days=horizon_value + 1)
    first_inner_eval = calendar[0] + pd.Timedelta(days=initial_days)
    if first_inner_eval > final_inner_origin:
        return ()

    folds: list[ExpandingFold] = []
    eval_start = first_inner_eval
    fold_number = 1
    while eval_start <= final_inner_origin:
        eval_end = min(
            eval_start + pd.Timedelta(days=block_days - 1),
            final_inner_origin,
        )
        folds.append(
            ExpandingFold(
                fold_id=f"INNER_{fold_number:02d}",
                train_start=calendar[0],
                train_end=eval_start - pd.Timedelta(days=1),
                eval_start=eval_start,
                eval_end=eval_end,
            )
        )
        eval_start = eval_end + pd.Timedelta(days=1)
        fold_number += 1
    return tuple(folds)


def folds_frame(
    folds: Sequence[ExpandingFold],
    *,
    horizon: int | None = None,
) -> pd.DataFrame:
    """Return fold metadata, including explicit maturation cutoffs."""

    return pd.DataFrame([fold.as_record(horizon) for fold in folds])


def matured_training_mask(
    samples: pd.DataFrame,
    evaluation_origin: Any,
    *,
    horizon: int | None = None,
) -> pd.Series:
    """Select samples with ``target_date < evaluation_origin`` strictly."""

    frame = validate_pair_keys(
        samples,
        require_unique=False,
        frame_name="mechanistic_samples",
    )
    eval_origin = _normalize_date(evaluation_origin, "evaluation_origin")
    if eval_origin > DEVELOPMENT_END:
        raise ValueError("Phase 05 cannot mature targets using post-2021 origins")
    mask = frame["origin_date"].lt(eval_origin) & frame["target_date"].lt(
        eval_origin
    )
    mask &= frame["origin_date"].between(
        DEVELOPMENT_START, DEVELOPMENT_END, inclusive="both"
    )
    mask &= frame["target_date"].le(DEVELOPMENT_END)
    if horizon is not None:
        mask &= frame["horizon"].eq(_validate_horizon(horizon))
    return pd.Series(mask.to_numpy(dtype=bool), index=samples.index)


def evaluation_mask(
    samples: pd.DataFrame,
    fold: ExpandingFold,
    *,
    horizon: int | None = None,
) -> pd.Series:
    """Select fold origins whose target remains within development."""

    frame = validate_pair_keys(
        samples,
        require_unique=False,
        frame_name="mechanistic_samples",
    )
    mask = frame["origin_date"].between(
        fold.eval_start, fold.eval_end, inclusive="both"
    ) & frame["target_date"].le(DEVELOPMENT_END)
    if horizon is not None:
        mask &= frame["horizon"].eq(_validate_horizon(horizon))
    return pd.Series(mask.to_numpy(dtype=bool), index=samples.index)


def split_fold_samples(
    samples: pd.DataFrame,
    fold: ExpandingFold,
    *,
    horizon: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return leakage-safe matured training rows and evaluation rows."""

    train_mask = matured_training_mask(
        samples,
        fold.eval_start,
        horizon=horizon,
    )
    eval_mask = evaluation_mask(samples, fold, horizon=horizon)
    training = samples.loc[train_mask].copy()
    evaluation = samples.loc[eval_mask].copy()
    if not training.empty:
        max_target = pd.to_datetime(training["target_date"]).max()
        if max_target >= fold.eval_start:
            raise AssertionError("training target has not matured before evaluation")
    return training, evaluation


def assert_development_only(frame: pd.DataFrame) -> None:
    """Reject any prediction key that would access 2022/2023 performance."""

    normalized = validate_pair_keys(
        frame,
        require_unique=False,
        frame_name="phase05_frame",
    )
    invalid = (
        normalized["origin_date"].lt(DEVELOPMENT_START)
        | normalized["origin_date"].gt(DEVELOPMENT_END)
        | normalized["target_date"].gt(DEVELOPMENT_END)
    )
    if bool(invalid.any()):
        raise ValueError("Phase 05 frame contains non-development prediction keys")


def score_candidate_predictions(
    predictions: pd.DataFrame,
    persistence_reference: pd.DataFrame,
    *,
    candidate_columns: Sequence[str] = ("candidate_id",),
    smape_epsilon: float = 1.0e-12,
) -> pd.DataFrame:
    """Aggregate inner OOF own-support RMSE and exact-paired persistence skill."""

    if not candidate_columns or len(set(candidate_columns)) != len(candidate_columns):
        raise ValueError("candidate_columns must be non-empty and unique")
    if predictions.empty:
        raise ValueError("candidate predictions must not be empty")
    missing = [column for column in candidate_columns if column not in predictions]
    if missing:
        raise KeyError(f"candidate predictions missing columns: {missing}")
    assert_development_only(predictions)
    key = [*PAIR_KEY, *candidate_columns]
    duplicated = predictions.duplicated(key, keep=False)
    if bool(duplicated.any()):
        raise ValueError("candidate predictions are not unique by pair and candidate")
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
    grouper: str | list[str]
    grouper = candidate_columns[0] if len(candidate_columns) == 1 else list(candidate_columns)
    # The Phase 04 merger intentionally requires one model row per pair key.
    # Inner selection has one row per pair *and candidate*, so attach the
    # reference inside each candidate group rather than weakening that guard.
    paired_parts = [
        merge_persistence_reference(group, reference)
        for _, group in predictions.groupby(grouper, sort=True, dropna=False)
    ]
    paired = pd.concat(paired_parts, ignore_index=True)

    rows: list[dict[str, Any]] = []
    for group_key, group in paired.groupby(grouper, sort=True, dropna=False):
        key_values = (group_key,) if len(candidate_columns) == 1 else tuple(group_key)
        identifiers = dict(zip(candidate_columns, key_values, strict=True))
        own = group.loc[support_mask(group, OWN_AVAILABLE)]
        own_metrics = compute_metrics(
            own["y_true"],
            own["y_pred"],
            y_origin=own["y_origin"] if "y_origin" in own else None,
            smape_epsilon=smape_epsilon,
        )
        common = group.loc[support_mask(group, PERSISTENCE_PAIRED)]
        paired_skill = skill_vs_persistence(
            common["y_true"],
            common["y_pred"],
            common["y_persistence"],
        )
        record: dict[str, Any] = {
            **identifiers,
            "inner_n": own_metrics["n_scored"],
            "own_RMSE": own_metrics["RMSE"],
            "own_MAE": own_metrics["MAE"],
            "own_sMAPE": own_metrics["sMAPE"],
            "paired_n": int(len(common)),
            "paired_Skill_vs_persistence": paired_skill,
        }
        for column in (
            "complexity_rank",
            "model_id",
            "load_basis",
            "k",
            "k_fast",
            "k_slow",
            "tau_res",
        ):
            if column in group.columns:
                values = group[column].drop_duplicates()
                if len(values) > 1:
                    raise ValueError(f"candidate has inconsistent {column}")
                record[column] = values.iloc[0] if len(values) else np.nan
        rows.append(record)
    return pd.DataFrame(rows)


def _resolve_column(
    frame: pd.DataFrame,
    explicit: str | None,
    candidates: Sequence[str],
    role: str,
) -> str:
    if explicit is not None:
        if explicit not in frame:
            raise KeyError(f"candidate table missing {role} column {explicit!r}")
        return explicit
    for candidate in candidates:
        if candidate in frame:
            return candidate
    raise KeyError(f"candidate table has no recognized {role} column")


def select_candidate(
    candidate_scores: pd.DataFrame,
    *,
    candidate_column: str | None = None,
    rmse_column: str | None = None,
    skill_column: str | None = None,
    complexity_column: str | None = None,
    eligible_column: str | None = "eligible",
    rmse_tolerance: float = 1.0e-12,
    skill_tolerance: float = 1.0e-12,
) -> pd.Series:
    """Select by own RMSE, paired skill, simplicity, then stable ID."""

    if candidate_scores.empty:
        raise ValueError("candidate_scores must not be empty")
    candidate_col = _resolve_column(
        candidate_scores,
        candidate_column,
        ("candidate_id", "candidate_model", "model_id"),
        "candidate identifier",
    )
    rmse_col = _resolve_column(
        candidate_scores,
        rmse_column,
        ("own_RMSE", "RMSE_own", "inner_RMSE", "RMSE"),
        "own-support RMSE",
    )
    skill_col = _resolve_column(
        candidate_scores,
        skill_column,
        (
            "paired_Skill_vs_persistence",
            "inner_Skill",
            "Skill_vs_persistence",
            "Skill",
        ),
        "paired skill",
    )
    complexity_col = _resolve_column(
        candidate_scores,
        complexity_column,
        ("complexity_rank", "simplicity_rank"),
        "complexity rank",
    )
    if rmse_tolerance < 0 or skill_tolerance < 0:
        raise ValueError("selection tolerances must be non-negative")

    eligible = candidate_scores.copy()
    if eligible_column and eligible_column in eligible:
        eligible = eligible.loc[eligible[eligible_column].fillna(False).astype(bool)]
    eligible["_rmse"] = pd.to_numeric(eligible[rmse_col], errors="coerce")
    eligible = eligible.loc[np.isfinite(eligible["_rmse"])]
    if eligible.empty:
        raise ValueError("no eligible candidate has finite own-support RMSE")

    min_rmse = float(eligible["_rmse"].min())
    shortlist = eligible.loc[eligible["_rmse"].le(min_rmse + rmse_tolerance)].copy()
    reason = "LOWEST_OWN_SUPPORT_RMSE"

    if len(shortlist) > 1:
        shortlist["_skill"] = pd.to_numeric(shortlist[skill_col], errors="coerce")
        finite_skill = shortlist["_skill"].loc[np.isfinite(shortlist["_skill"])]
        if not finite_skill.empty:
            max_skill = float(finite_skill.max())
            shortlist = shortlist.loc[
                shortlist["_skill"].ge(max_skill - skill_tolerance)
            ].copy()
            reason += "+HIGHER_PAIRED_SKILL"

    if len(shortlist) > 1:
        shortlist["_complexity"] = pd.to_numeric(
            shortlist[complexity_col], errors="coerce"
        ).fillna(np.inf)
        min_complexity = float(shortlist["_complexity"].min())
        shortlist = shortlist.loc[shortlist["_complexity"].eq(min_complexity)]
        reason += "+SIMPLICITY"

    shortlist = shortlist.assign(_candidate_sort=shortlist[candidate_col].astype(str))
    selected = shortlist.sort_values("_candidate_sort", kind="stable").iloc[0].copy()
    selected["selection_reason"] = reason + "+DETERMINISTIC_ID"
    return selected.drop(
        labels=[
            label
            for label in ("_rmse", "_skill", "_complexity", "_candidate_sort")
            if label in selected.index
        ]
    )


def rank_candidates(
    candidate_scores: pd.DataFrame,
    **selection_kwargs: Any,
) -> pd.DataFrame:
    """Return candidate scores with one deterministic selected marker."""

    selected = select_candidate(candidate_scores, **selection_kwargs)
    candidate_column = selection_kwargs.get("candidate_column")
    candidate_col = _resolve_column(
        candidate_scores,
        candidate_column,
        ("candidate_id", "candidate_model", "model_id"),
        "candidate identifier",
    )
    result = candidate_scores.copy()
    result["selected"] = result[candidate_col].astype(str).eq(
        str(selected[candidate_col])
    )
    result["selection_reason"] = np.where(
        result["selected"], selected["selection_reason"], "NOT_SELECTED"
    )
    return result


__all__ = [
    "DEVELOPMENT_END",
    "DEVELOPMENT_START",
    "ExpandingFold",
    "INNER_EVAL_BLOCK_DAYS",
    "INNER_INITIAL_TRAIN_DAYS",
    "OUTER_EVAL_BLOCK_DAYS",
    "OUTER_INITIAL_TRAIN_DAYS",
    "assert_development_only",
    "evaluation_mask",
    "folds_frame",
    "make_expanding_outer_folds",
    "make_inner_expanding_folds",
    "matured_training_mask",
    "rank_candidates",
    "score_candidate_predictions",
    "select_candidate",
    "split_fold_samples",
    "training_maturation_cutoffs",
    "validate_development_calendar",
]
