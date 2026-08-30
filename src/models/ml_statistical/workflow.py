"""Nested expanding-window OOF workflow for Phase 06 exogenous ML."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from ...validation.baseline_metrics import compute_metrics, skill_vs_persistence
from ..mechanistic.fold_selection import (
    ExpandingFold,
    make_expanding_outer_folds,
    make_inner_expanding_folds,
)
from .contracts import (
    AVAILABILITY_TRACKS,
    DEVELOPMENT_END,
    DEVELOPMENT_START,
    FEATURE_SETS,
    HORIZONS,
    MODEL_FAMILIES,
    TRACK_PROVISIONAL,
    TRACK_STRICT,
    build_feature_contract,
    dependency_status,
    features_for,
)
from .estimators import CandidateSpec, candidate_specs, fit_estimator, raw_importance
from .preprocessing import FoldPreprocessor


TARGET_COLUMNS = {
    "CH4_m3d_observed": "CH4_target_h{horizon}",
    "biogas_AB_m3d": "biogas_target_h{horizon}",
}
TARGET_LABELS = {
    "CH4_m3d_observed": "CH4",
    "biogas_AB_m3d": "BIOGAS",
}
MAX_INNER_FOLDS = 2
MIN_TRAIN_TARGETS = 60
MIN_INNER_SCORED = 20


@dataclass
class DevelopmentInputs:
    features: pd.DataFrame
    targets: pd.DataFrame
    persistence: pd.DataFrame
    registry: pd.DataFrame
    feature_contract: pd.DataFrame
    dependencies: pd.DataFrame
    source_hash: str


@dataclass
class OOFResult:
    predictions: pd.DataFrame
    hyperparameters: pd.DataFrame
    preprocessing_audit: pd.DataFrame
    selected_features: pd.DataFrame
    importance_by_fold: pd.DataFrame
    outer_manifest: pd.DataFrame


def combine_oof_results(results: list[OOFResult]) -> OOFResult:
    """Combine independent target/horizon jobs with the same locked contract."""

    if not results:
        raise ValueError("at least one OOF result is required")
    return OOFResult(
        predictions=pd.concat([item.predictions for item in results], ignore_index=True),
        hyperparameters=pd.concat(
            [item.hyperparameters for item in results], ignore_index=True
        ),
        preprocessing_audit=pd.concat(
            [item.preprocessing_audit for item in results], ignore_index=True
        ),
        selected_features=pd.concat(
            [item.selected_features for item in results], ignore_index=True
        ),
        importance_by_fold=pd.concat(
            [item.importance_by_fold for item in results], ignore_index=True
        ),
        outer_manifest=build_outer_manifest(),
    )


def _hash_files(paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.name.encode("utf-8"))
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    return digest.hexdigest()


def _read_development_parquet(path: Path, date_column: str) -> pd.DataFrame:
    """Read only development rows at the storage layer when filters allow it."""

    frame = pd.read_parquet(
        path,
        filters=[
            (date_column, ">=", DEVELOPMENT_START),
            (date_column, "<=", DEVELOPMENT_END),
        ],
    )
    frame[date_column] = pd.to_datetime(frame[date_column], errors="raise").dt.normalize()
    if bool(frame[date_column].gt(DEVELOPMENT_END).any()):
        raise AssertionError("post-2021 data entered Phase 06 development memory")
    return frame.sort_values(date_column, kind="stable").reset_index(drop=True)


def load_development_inputs(project_root: str | Path) -> DevelopmentInputs:
    root = Path(project_root).resolve()
    paths = {
        "features": root / "data/processed/03_feature_base.parquet",
        "targets": root / "data/processed/03_targets_by_horizon.parquet",
        "registry": root / "outputs/03_feature_registry.csv",
        "persistence": root / "outputs/04_persistence_reference.parquet",
        "phase04_version": root / "outputs/04_baseline_version.json",
        "phase05_lock": root / "outputs/05_model_lock.yaml",
    }
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Phase 06 required inputs missing: {missing}")
    features = _read_development_parquet(paths["features"], "date")
    targets = _read_development_parquet(paths["targets"], "date")
    persistence = pd.read_parquet(
        paths["persistence"],
        filters=[
            ("origin_date", ">=", DEVELOPMENT_START),
            ("origin_date", "<=", DEVELOPMENT_END),
            ("target_date", "<=", DEVELOPMENT_END),
        ],
    )
    for column in ("origin_date", "target_date"):
        persistence[column] = pd.to_datetime(
            persistence[column], errors="raise"
        ).dt.normalize()
    if bool(
        persistence["origin_date"].gt(DEVELOPMENT_END).any()
        or persistence["target_date"].gt(DEVELOPMENT_END).any()
    ):
        raise AssertionError("2022/2023 persistence row entered Phase 06")
    registry = pd.read_csv(paths["registry"], encoding="utf-8-sig")
    contract = build_feature_contract(registry, list(features.columns))
    dependencies = dependency_status(root)
    source_hash = _hash_files(list(paths.values()))
    return DevelopmentInputs(
        features=features,
        targets=targets,
        persistence=persistence,
        registry=registry,
        feature_contract=contract,
        dependencies=dependencies,
        source_hash=source_hash,
    )


def build_samples(
    inputs: DevelopmentInputs,
    target_name: str,
    horizon: int,
) -> pd.DataFrame:
    if target_name not in TARGET_COLUMNS:
        raise ValueError(f"unknown target: {target_name}")
    if int(horizon) not in HORIZONS:
        raise ValueError(f"unsupported horizon: {horizon}")
    column = TARGET_COLUMNS[target_name].format(horizon=int(horizon))
    if column not in inputs.targets:
        raise KeyError(f"target table missing {column}")
    base = inputs.features.merge(
        inputs.targets[["date", column]], on="date", how="left", validate="one_to_one"
    )
    base = base.rename(columns={"date": "origin_date", column: "y_true"})
    base["horizon"] = int(horizon)
    base["target_name"] = target_name
    base["target_date"] = base["origin_date"] + pd.to_timedelta(int(horizon), unit="D")
    base = base.loc[base["target_date"].le(DEVELOPMENT_END)].copy()
    reference = inputs.persistence.loc[
        inputs.persistence["target_name"].astype(str).eq(target_name)
        & pd.to_numeric(inputs.persistence["horizon"], errors="coerce").eq(int(horizon)),
        [
            "origin_date",
            "target_date",
            "target_name",
            "horizon",
            "y_origin",
            "y_true",
            "y_persistence",
            "persistence_available",
        ],
    ].rename(columns={"y_true": "reference_y_true"})
    base = base.merge(
        reference,
        on=["origin_date", "target_date", "target_name", "horizon"],
        how="left",
        validate="one_to_one",
    )
    comparable = base["y_true"].notna() & base["reference_y_true"].notna()
    if bool(
        comparable.any()
        and not np.allclose(
            base.loc[comparable, "y_true"],
            base.loc[comparable, "reference_y_true"],
            rtol=0.0,
            atol=1.0e-9,
        )
    ):
        raise ValueError("Phase 03 and immutable Phase 04 target values disagree")
    base = base.drop(columns=["reference_y_true"])
    base["target_available"] = np.isfinite(
        pd.to_numeric(base["y_true"], errors="coerce")
    )
    base["origin_target_available"] = np.isfinite(
        pd.to_numeric(base["y_origin"], errors="coerce")
    )
    base["persistence_available"] = (
        base["persistence_available"].fillna(False).astype(bool)
        & np.isfinite(pd.to_numeric(base["y_persistence"], errors="coerce"))
    )
    if bool(base["origin_date"].gt(DEVELOPMENT_END).any()):
        raise AssertionError("non-development origin in samples")
    return base.sort_values("origin_date", kind="stable").reset_index(drop=True)


def outer_folds() -> tuple[ExpandingFold, ...]:
    return make_expanding_outer_folds(
        pd.date_range(DEVELOPMENT_START, DEVELOPMENT_END, freq="D")
    )


def build_outer_manifest() -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for target_name in TARGET_COLUMNS:
        for horizon in HORIZONS:
            for fold in outer_folds():
                max_origin = fold.eval_start - pd.Timedelta(days=horizon + 1)
                rows.append(
                    {
                        "outer_fold": fold.fold_id,
                        "train_origin_start": DEVELOPMENT_START,
                        "train_origin_end": max_origin,
                        "max_train_target_date": fold.eval_start - pd.Timedelta(days=1),
                        "eval_origin_start": fold.eval_start,
                        "eval_origin_end": min(
                            fold.eval_end,
                            DEVELOPMENT_END - pd.Timedelta(days=horizon),
                        ),
                        "target": target_name,
                        "horizon": horizon,
                    }
                )
    return pd.DataFrame(rows)


def _training_rows(samples: pd.DataFrame, evaluation_start: pd.Timestamp) -> pd.DataFrame:
    truth = pd.to_numeric(samples["y_true"], errors="coerce")
    mask = (
        samples["origin_date"].lt(evaluation_start)
        & samples["target_date"].lt(evaluation_start)
        & np.isfinite(truth)
    )
    return samples.loc[mask].copy()


def _evaluation_rows(samples: pd.DataFrame, fold: ExpandingFold) -> pd.DataFrame:
    return samples.loc[
        samples["origin_date"].between(fold.eval_start, fold.eval_end, inclusive="both")
        & samples["target_date"].le(DEVELOPMENT_END)
    ].copy()


def _predict(estimator: Any, matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    raw = np.asarray(estimator.predict(matrix), dtype=float).reshape(-1)
    negative = np.isfinite(raw) & (raw < 0)
    clipped = np.where(np.isfinite(raw), np.maximum(raw, 0.0), np.nan)
    return raw, clipped, negative


def _candidate_score(rows: list[dict[str, Any]]) -> dict[str, Any]:
    frame = pd.DataFrame(rows)
    available = np.isfinite(pd.to_numeric(frame["y_true"], errors="coerce")) & np.isfinite(
        pd.to_numeric(frame["y_pred"], errors="coerce")
    )
    own = frame.loc[available]
    metrics = compute_metrics(own["y_true"], own["y_pred"])
    paired = own.loc[
        own["persistence_available"].fillna(False).astype(bool)
        & np.isfinite(pd.to_numeric(own["y_persistence"], errors="coerce"))
    ]
    skill = skill_vs_persistence(
        paired["y_true"], paired["y_pred"], paired["y_persistence"]
    )
    return {
        "inner_n": int(metrics["n_scored"]),
        "inner_RMSE": metrics["RMSE"],
        "inner_MAE": metrics["MAE"],
        "inner_Skill": skill,
        "inner_paired_n": int(len(paired)),
    }


def select_inner_candidate(
    samples: pd.DataFrame,
    *,
    model_id: str,
    features: list[str],
    outer_fold: ExpandingFold,
    target_name: str,
    horizon: int,
    feature_set: str,
    availability_track: str,
) -> tuple[CandidateSpec, pd.DataFrame, pd.Timestamp]:
    specs = candidate_specs(model_id)
    inner = make_inner_expanding_folds(
        outer_fold.eval_start,
        horizon,
        pd.date_range(DEVELOPMENT_START, DEVELOPMENT_END, freq="D"),
    )
    inner = inner[-MAX_INNER_FOLDS:]
    prediction_rows: dict[str, list[dict[str, Any]]] = {
        spec.candidate_id: [] for spec in specs
    }
    failures: dict[str, list[str]] = {spec.candidate_id: [] for spec in specs}
    max_validation_target = pd.NaT
    used_inner_ids: list[str] = []
    for fold in inner:
        training = _training_rows(samples, fold.eval_start)
        evaluation = _evaluation_rows(samples, fold)
        if len(training) < MIN_TRAIN_TARGETS or evaluation.empty:
            continue
        preprocessor = FoldPreprocessor(model_id=model_id).fit(
            training,
            features,
            fit_end=training["origin_date"].max(),
        )
        x_train = preprocessor.transform(training)
        x_eval = preprocessor.transform(evaluation)
        y_train = pd.to_numeric(training["y_true"], errors="raise").to_numpy(dtype=float)
        used_inner_ids.append(fold.fold_id)
        observed_eval = evaluation.loc[evaluation["target_available"]]
        if not observed_eval.empty:
            fold_max = observed_eval["target_date"].max()
            max_validation_target = (
                fold_max
                if pd.isna(max_validation_target)
                else max(max_validation_target, fold_max)
            )
        for spec in specs:
            try:
                estimator = fit_estimator(
                    model_id, spec.parameters, x_train, y_train
                )
                _, clipped, _ = _predict(estimator, x_eval)
                for row, prediction in zip(
                    evaluation.itertuples(index=False), clipped, strict=True
                ):
                    prediction_rows[spec.candidate_id].append(
                        {
                            "y_true": row.y_true,
                            "y_pred": prediction,
                            "y_persistence": row.y_persistence,
                            "persistence_available": row.persistence_available,
                        }
                    )
            except Exception as exc:  # candidate failure is explicit, never substituted
                failures[spec.candidate_id].append(
                    f"{fold.fold_id}:{type(exc).__name__}:{exc}"
                )

    audit_rows: list[dict[str, Any]] = []
    for spec in specs:
        values = prediction_rows[spec.candidate_id]
        score = (
            _candidate_score(values)
            if values
            else {
                "inner_n": 0,
                "inner_RMSE": np.nan,
                "inner_MAE": np.nan,
                "inner_Skill": np.nan,
                "inner_paired_n": 0,
            }
        )
        audit_rows.append(
            {
                "outer_fold": outer_fold.fold_id,
                "target": target_name,
                "horizon": horizon,
                "model_id": model_id,
                "feature_set": feature_set,
                "availability_track": availability_track,
                "candidate_id": spec.candidate_id,
                "parameters": spec.parameters_json(),
                "complexity_rank": spec.complexity_rank,
                **score,
                "inner_fold_ids": ";".join(used_inner_ids),
                "max_inner_validation_target_date": max_validation_target,
                "candidate_status": "FAILED" if failures[spec.candidate_id] else "EVALUATED",
                "failure_detail": ";".join(failures[spec.candidate_id]),
                "selected": False,
                "selection_reason": "NOT_SELECTED",
            }
        )
    audit = pd.DataFrame(audit_rows)
    eligible = audit.loc[
        pd.to_numeric(audit["inner_n"], errors="coerce").ge(MIN_INNER_SCORED)
        & np.isfinite(pd.to_numeric(audit["inner_RMSE"], errors="coerce"))
    ].copy()
    if eligible.empty:
        selected = specs[0]
        reason = "PREDECLARED_SIMPLEST_EARLY_FOLD_DEFAULT"
    else:
        eligible["_skill"] = pd.to_numeric(
            eligible["inner_Skill"], errors="coerce"
        ).fillna(-np.inf)
        eligible = eligible.sort_values(
            ["inner_RMSE", "_skill", "complexity_rank", "candidate_id"],
            ascending=[True, False, True, True],
            kind="stable",
        )
        chosen_id = str(eligible.iloc[0]["candidate_id"])
        selected = next(spec for spec in specs if spec.candidate_id == chosen_id)
        reason = "LOWEST_INNER_OWN_RMSE_THEN_SKILL_COMPLEXITY_ID"
    audit.loc[audit["candidate_id"].eq(selected.candidate_id), "selected"] = True
    audit.loc[
        audit["candidate_id"].eq(selected.candidate_id), "selection_reason"
    ] = reason
    if not pd.isna(max_validation_target) and max_validation_target >= outer_fold.eval_start:
        raise AssertionError("inner validation target not mature before outer evaluation")
    return selected, audit, max_validation_target


def _selected_feature_and_importance_rows(
    *,
    preprocessor: FoldPreprocessor,
    estimator: Any,
    model_id: str,
    target_name: str,
    horizon: int,
    feature_set: str,
    availability_track: str,
    outer_fold: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    importance = raw_importance(
        estimator, model_id, preprocessor.output_feature_names
    )
    importance["feature"] = importance["transformed_feature"].str.replace(
        "__missing", "", regex=False
    )
    importance["importance_abs"] = importance["importance"].abs()
    raw_summary = (
        importance.groupby("feature", as_index=False)
        .agg(
            importance=("importance_abs", "sum"),
            signed_importance=("importance", "sum"),
            importance_method=("importance_method", "first"),
        )
    )
    if model_id == "S01_ELASTIC_NET":
        selected_names = set(
            raw_summary.loc[raw_summary["importance"].gt(1.0e-10), "feature"]
        )
        method = "ELASTIC_NET_NONZERO_FOLD_LOCAL"
    else:
        selected_names = set(preprocessor.retained_features)
        method = "LEVEL1_SCREENING_ALL_RETAINED"
    selected_rows = []
    for feature in preprocessor.input_features:
        if feature in preprocessor.dropped_missing:
            reason = "DROPPED_EXTREME_MISSINGNESS"
        elif feature in preprocessor.dropped_constant:
            reason = "DROPPED_CONSTANT"
        elif feature in preprocessor.dropped_duplicate:
            reason = "DROPPED_EXACT_DUPLICATE"
        elif feature in preprocessor.dropped_collinear:
            reason = "DROPPED_COLLINEAR"
        elif feature in selected_names:
            reason = "SELECTED"
        else:
            reason = "INTRINSIC_COEFFICIENT_ZERO"
        selected_rows.append(
            {
                "target": target_name,
                "horizon": horizon,
                "model": model_id,
                "feature_set": feature_set,
                "availability_track": availability_track,
                "outer_fold": outer_fold,
                "feature": feature,
                "selected": feature in selected_names,
                "selection_method": method,
                "selection_reason": reason,
            }
        )
    raw_summary = raw_summary.assign(
        target=target_name,
        horizon=horizon,
        model=model_id,
        feature_set=feature_set,
        availability_track=availability_track,
        outer_fold=outer_fold,
        post_hoc_only=True,
    )
    return pd.DataFrame(selected_rows), raw_summary


def fit_outer_configuration(
    samples: pd.DataFrame,
    *,
    model_id: str,
    features: list[str],
    outer_fold: ExpandingFold,
    target_name: str,
    horizon: int,
    feature_set: str,
    availability_track: str,
    selected_candidate: CandidateSpec | None = None,
    hyperparameter_audit_override: pd.DataFrame | None = None,
    max_inner_target_override: pd.Timestamp | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if selected_candidate is None:
        selected, hyperparameter_audit, max_inner_target = select_inner_candidate(
            samples,
            model_id=model_id,
            features=features,
            outer_fold=outer_fold,
            target_name=target_name,
            horizon=horizon,
            feature_set=feature_set,
            availability_track=availability_track,
        )
    else:
        selected = selected_candidate
        hyperparameter_audit = (
            hyperparameter_audit_override.copy()
            if hyperparameter_audit_override is not None
            else pd.DataFrame()
        )
        max_inner_target = max_inner_target_override
    training = _training_rows(samples, outer_fold.eval_start)
    evaluation = _evaluation_rows(samples, outer_fold)
    if len(training) < MIN_TRAIN_TARGETS or evaluation.empty:
        return (
            pd.DataFrame(),
            hyperparameter_audit,
            pd.DataFrame(),
            pd.DataFrame(),
            pd.DataFrame(),
        )
    max_training_target = training["target_date"].max()
    if max_training_target >= outer_fold.eval_start:
        raise AssertionError("outer training target did not mature")
    fit_end = training["origin_date"].max()
    preprocessor = FoldPreprocessor(model_id=model_id).fit(
        training, features, fit_end=fit_end
    )
    x_train = preprocessor.transform(training)
    x_eval = preprocessor.transform(evaluation)
    estimator = fit_estimator(
        model_id,
        selected.parameters,
        x_train,
        pd.to_numeric(training["y_true"], errors="raise").to_numpy(dtype=float),
    )
    raw, clipped, negative = _predict(estimator, x_eval)
    extrapolation = preprocessor.extrapolation_audit(evaluation).reset_index(drop=True)
    prediction = evaluation[
        [
            "origin_date",
            "target_date",
            "target_name",
            "horizon",
            "y_true",
            "y_origin",
            "target_available",
            "origin_target_available",
            "y_persistence",
            "persistence_available",
        ]
    ].reset_index(drop=True)
    prediction["outer_fold"] = outer_fold.fold_id
    prediction["model_id"] = model_id
    prediction["feature_set"] = feature_set
    prediction["availability_track"] = availability_track
    prediction["selected_candidate_id"] = selected.candidate_id
    prediction["selected_parameters"] = selected.parameters_json()
    prediction["n_features"] = len(preprocessor.output_feature_names)
    prediction["raw_prediction"] = raw
    prediction["clipped_prediction"] = clipped
    prediction["y_pred"] = clipped
    prediction["negative_prediction_flag"] = negative
    prediction["prediction_available"] = np.isfinite(clipped)
    prediction["error"] = prediction["y_pred"] - prediction["y_true"]
    prediction["abs_error"] = prediction["error"].abs()
    prediction["sq_error"] = prediction["error"] ** 2
    prediction["delta_true"] = prediction["y_true"] - prediction["y_origin"]
    prediction["delta_pred"] = prediction["y_pred"] - prediction["y_origin"]
    prediction["persistence_paired"] = (
        prediction["target_available"]
        & prediction["prediction_available"]
        & prediction["persistence_available"]
    )
    prediction["support_type"] = "OWN_AVAILABLE"
    prediction["max_training_target_date"] = max_training_target
    prediction["max_feature_timestamp"] = prediction["origin_date"]
    prediction["feature_source_timestamp_pass"] = True
    prediction = pd.concat([prediction, extrapolation], axis=1)
    audit_values = preprocessor.audit_record()
    for key, value in audit_values.items():
        prediction[key] = value
    configuration_id = (
        f"{model_id}__{availability_track}__{feature_set}__"
        f"{TARGET_LABELS[target_name]}__h{horizon}"
    )
    prediction["configuration_id"] = configuration_id

    preprocessing_audit = pd.DataFrame(
        [
            {
                "outer_fold": outer_fold.fold_id,
                "target": target_name,
                "horizon": horizon,
                "model_id": model_id,
                "feature_set": feature_set,
                "availability_track": availability_track,
                "configuration_id": configuration_id,
                "imputer_fit_end": fit_end,
                "scaler_fit_end": fit_end if preprocessor.scale else pd.NaT,
                "selector_fit_end": fit_end,
                "hyperparameter_validation_target_date": max_inner_target,
                "evaluation_start": outer_fold.eval_start,
                "max_training_target_date": max_training_target,
                "train_only_pass": bool(
                    fit_end < outer_fold.eval_start
                    and max_training_target < outer_fold.eval_start
                    and (pd.isna(max_inner_target) or max_inner_target < outer_fold.eval_start)
                ),
                **audit_values,
            }
        ]
    )
    selected_features, importance = _selected_feature_and_importance_rows(
        preprocessor=preprocessor,
        estimator=estimator,
        model_id=model_id,
        target_name=target_name,
        horizon=horizon,
        feature_set=feature_set,
        availability_track=availability_track,
        outer_fold=outer_fold.fold_id,
    )
    return (
        prediction,
        hyperparameter_audit,
        preprocessing_audit,
        selected_features,
        importance,
    )


def _alias_feature_set(frame: pd.DataFrame, feature_set: str) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    result = frame.copy()
    if "feature_set" in result:
        result["feature_set"] = feature_set
    if "configuration_id" in result:
        result["configuration_id"] = result["configuration_id"].str.replace(
            r"__SET-[A-Z]__", f"__{feature_set}__", regex=True
        )
    return result


def run_nested_oof(
    inputs: DevelopmentInputs,
    *,
    progress: Callable[[str], None] | None = None,
    target_names: tuple[str, ...] | None = None,
    horizons: tuple[int, ...] | None = None,
    model_ids: tuple[str, ...] | None = None,
    availability_tracks: tuple[str, ...] | None = None,
    outer_fold_ids: tuple[str, ...] | None = None,
) -> OOFResult:
    callback = progress or (lambda _: None)
    available = set(
        inputs.dependencies.loc[
            inputs.dependencies["dependency_status"].eq("AVAILABLE"), "model_id"
        ].astype(str)
    )
    predictions: list[pd.DataFrame] = []
    hyperparameters: list[pd.DataFrame] = []
    preprocessing: list[pd.DataFrame] = []
    selected_features: list[pd.DataFrame] = []
    importance: list[pd.DataFrame] = []
    folds = outer_folds()
    if outer_fold_ids is not None:
        requested = set(outer_fold_ids)
        folds = tuple(fold for fold in folds if fold.fold_id in requested)
        if not folds:
            raise ValueError("outer_fold_ids selected no Phase 06 folds")
    target_sequence = target_names or tuple(TARGET_COLUMNS)
    horizon_sequence = horizons or HORIZONS
    model_sequence = model_ids or MODEL_FAMILIES
    track_sequence = availability_tracks or AVAILABILITY_TRACKS
    for target_name in target_sequence:
        for horizon in horizon_sequence:
            samples = build_samples(inputs, target_name, horizon)
            callback(f"START {TARGET_LABELS[target_name]} h{horizon}")
            for model_id in model_sequence:
                if model_id not in available:
                    callback(f"SKIP {model_id}: UNAVAILABLE_DEPENDENCY")
                    continue
                for availability_track in track_sequence:
                    features_by_set = {
                        feature_set: features_for(
                            inputs.feature_contract,
                            horizon=horizon,
                            availability_track=availability_track,
                            feature_set=feature_set,
                        )
                        for feature_set in FEATURE_SETS
                    }
                    # Hyperparameters are selected once per outer fold on the
                    # predeclared richest core set in the same availability
                    # track, then held fixed across A/B/C/E/F.  This preserves
                    # fold-local tuning while making incremental set comparisons
                    # independent of separate tuning luck.
                    anchor_features = features_by_set["SET-F"]
                    for fold in folds:
                        selected, tuning_audit, max_inner_target = select_inner_candidate(
                            samples,
                            model_id=model_id,
                            features=anchor_features,
                            outer_fold=fold,
                            target_name=target_name,
                            horizon=horizon,
                            feature_set="SET-F",
                            availability_track=availability_track,
                        )
                        tuning_audit["tuning_scope"] = (
                            "OUTER_FOLD_TARGET_HORIZON_FAMILY_TRACK_ANCHOR"
                        )
                        tuning_audit["applied_feature_sets"] = ";".join(FEATURE_SETS)
                        hyperparameters.append(tuning_audit)
                        fold_feature_cache: dict[
                            tuple[str, ...],
                            tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame],
                        ] = {}
                        for feature_set in FEATURE_SETS:
                            features = features_by_set[feature_set]
                            if not features:
                                continue
                            cache_key = tuple(features)
                            if cache_key not in fold_feature_cache:
                                result = fit_outer_configuration(
                                    samples,
                                    model_id=model_id,
                                    features=features,
                                    outer_fold=fold,
                                    target_name=target_name,
                                    horizon=horizon,
                                    feature_set=feature_set,
                                    availability_track=availability_track,
                                    selected_candidate=selected,
                                    hyperparameter_audit_override=pd.DataFrame(),
                                    max_inner_target_override=max_inner_target,
                                )
                                fold_feature_cache[cache_key] = (
                                    result[0], result[2], result[3], result[4]
                                )
                            bundle = fold_feature_cache[cache_key]
                            aliased = [
                                _alias_feature_set(frame, feature_set) for frame in bundle
                            ]
                            if not aliased[0].empty:
                                predictions.append(aliased[0])
                            if not aliased[1].empty:
                                preprocessing.append(aliased[1])
                            if not aliased[2].empty:
                                selected_features.append(aliased[2])
                            if not aliased[3].empty:
                                importance.append(aliased[3])
                    callback(
                        f"DONE {TARGET_LABELS[target_name]} h{horizon} "
                        f"{model_id} {availability_track}"
                    )
    result = OOFResult(
        predictions=pd.concat(predictions, ignore_index=True),
        hyperparameters=pd.concat(hyperparameters, ignore_index=True),
        preprocessing_audit=pd.concat(preprocessing, ignore_index=True),
        selected_features=pd.concat(selected_features, ignore_index=True),
        importance_by_fold=pd.concat(importance, ignore_index=True),
        outer_manifest=build_outer_manifest(),
    )
    key = [
        "target_name",
        "horizon",
        "origin_date",
        "target_date",
        "model_id",
        "feature_set",
        "availability_track",
    ]
    if bool(result.predictions.duplicated(key).any()):
        raise AssertionError("Phase 06 OOF configuration prediction key is not unique")
    if bool(
        result.predictions["origin_date"].gt(DEVELOPMENT_END).any()
        or result.predictions["target_date"].gt(DEVELOPMENT_END).any()
    ):
        raise AssertionError("Phase 06 OOF accessed 2022/2023")
    if not bool(result.preprocessing_audit["train_only_pass"].all()):
        raise AssertionError("fold-local preprocessing audit failed")
    return result


def selected_parameter_mode(
    hyperparameters: pd.DataFrame,
    *,
    target: str,
    horizon: int,
    model_id: str,
    feature_set: str,
    availability_track: str,
) -> tuple[str, dict[str, Any]]:
    subset = hyperparameters.loc[
        hyperparameters["target"].astype(str).eq(target)
        & pd.to_numeric(hyperparameters["horizon"], errors="coerce").eq(horizon)
        & hyperparameters["model_id"].astype(str).eq(model_id)
        & hyperparameters["feature_set"].astype(str).eq(feature_set)
        & hyperparameters["availability_track"].astype(str).eq(availability_track)
        & hyperparameters["selected"].fillna(False).astype(bool)
    ].copy()
    if subset.empty:
        raise ValueError("no selected hyperparameter rows for locked configuration")
    counts = subset["candidate_id"].astype(str).value_counts()
    maximum = int(counts.max())
    modes = sorted(counts[counts.eq(maximum)].index.tolist())
    if len(modes) > 1:
        means = (
            subset.loc[subset["candidate_id"].astype(str).isin(modes)]
            .groupby("candidate_id")["inner_RMSE"]
            .mean()
            .sort_values(kind="stable")
        )
        selected_id = str(means.index[0])
    else:
        selected_id = str(modes[0])
    parameters = json.loads(
        subset.loc[subset["candidate_id"].astype(str).eq(selected_id), "parameters"].iloc[0]
    )
    return selected_id, parameters


__all__ = [
    "DevelopmentInputs",
    "MAX_INNER_FOLDS",
    "OOFResult",
    "TARGET_COLUMNS",
    "TARGET_LABELS",
    "build_outer_manifest",
    "build_samples",
    "combine_oof_results",
    "fit_outer_configuration",
    "load_development_inputs",
    "outer_folds",
    "run_nested_oof",
    "select_inner_candidate",
    "selected_parameter_mode",
]
