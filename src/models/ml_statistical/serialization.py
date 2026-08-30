"""Full-development refit and portable Phase 06 model bundle."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .contracts import DEVELOPMENT_END, DEVELOPMENT_START, features_for
from .estimators import fit_estimator, raw_importance
from .preprocessing import FoldPreprocessor
from .workflow import (
    DevelopmentInputs,
    build_samples,
    selected_parameter_mode,
)


@dataclass
class FittedMLModel:
    """One locked target/horizon/configuration with auditable prediction API."""

    preprocessor: FoldPreprocessor
    estimator: Any
    metadata: dict[str, Any]

    def predict(self, frame: pd.DataFrame) -> pd.DataFrame:
        matrix = self.preprocessor.transform(frame)
        raw = np.asarray(self.estimator.predict(matrix), dtype=float).reshape(-1)
        clipped = np.where(np.isfinite(raw), np.maximum(raw, 0.0), np.nan)
        result = pd.DataFrame(
            {
                "raw_prediction": raw,
                "clipped_prediction": clipped,
                "y_pred": clipped,
                "negative_prediction_flag": np.isfinite(raw) & (raw < 0),
                "prediction_available": np.isfinite(clipped),
            },
            index=frame.index,
        )
        return pd.concat(
            [result, self.preprocessor.extrapolation_audit(frame)], axis=1
        )


def _selected_raw_features(
    model_id: str,
    preprocessor: FoldPreprocessor,
    estimator: Any,
) -> list[str]:
    importance = raw_importance(
        estimator, model_id, preprocessor.output_feature_names
    )
    importance["feature"] = importance["transformed_feature"].str.replace(
        "__missing", "", regex=False
    )
    importance["magnitude"] = importance["importance"].abs()
    magnitude = importance.groupby("feature")["magnitude"].sum()
    if model_id == "S01_ELASTIC_NET":
        return sorted(magnitude.index[magnitude.gt(1.0e-10)].tolist())
    return sorted(preprocessor.retained_features)


def refit_locked_models(
    inputs: DevelopmentInputs,
    lock_table: pd.DataFrame,
    hyperparameters: pd.DataFrame,
) -> dict[str, Any]:
    """Fit every retained/locked configuration on development labels only."""

    retained_status = {
        "LOCK_FOR_R7",
        "RETAIN_AS_LINEAR_REFERENCE",
        "RETAIN_AS_SENSITIVITY",
    }
    rows = lock_table.loc[lock_table["R7_status"].isin(retained_status)].copy()
    models: dict[str, FittedMLModel] = {}
    for row in rows.itertuples(index=False):
        target_name = str(row.target_name)
        horizon = int(row.horizon)
        model_id = str(row.model_id)
        feature_set = str(row.feature_set)
        availability_track = str(row.availability_track)
        model_key = (
            f"{model_id}__{availability_track}__{feature_set}__"
            f"{target_name}__h{horizon}"
        )
        samples = build_samples(inputs, target_name, horizon)
        truth = pd.to_numeric(samples["y_true"], errors="coerce")
        training = samples.loc[
            samples["target_date"].le(DEVELOPMENT_END) & np.isfinite(truth)
        ].copy()
        features = features_for(
            inputs.feature_contract,
            horizon=horizon,
            availability_track=availability_track,
            feature_set=feature_set,
        )
        candidate_id, parameters = selected_parameter_mode(
            hyperparameters,
            target=target_name,
            horizon=horizon,
            model_id=model_id,
            feature_set="SET-F",
            availability_track=availability_track,
        )
        preprocessor = FoldPreprocessor(model_id=model_id).fit(
            training,
            features,
            fit_end=training["origin_date"].max(),
        )
        estimator = fit_estimator(
            model_id,
            parameters,
            preprocessor.transform(training),
            pd.to_numeric(training["y_true"], errors="raise").to_numpy(dtype=float),
        )
        selected_features = _selected_raw_features(model_id, preprocessor, estimator)
        metadata = {
            "model_id": model_id,
            "target": target_name,
            "horizon": horizon,
            "feature_set": feature_set,
            "availability_track": availability_track,
            "training_start": DEVELOPMENT_START.strftime("%Y-%m-%d"),
            "training_end": DEVELOPMENT_END.strftime("%Y-%m-%d"),
            "max_training_origin_date": training["origin_date"].max().strftime(
                "%Y-%m-%d"
            ),
            "max_training_target_date": training["target_date"].max().strftime(
                "%Y-%m-%d"
            ),
            "selected_features": selected_features,
            "input_features": features,
            "transformed_features": preprocessor.output_feature_names,
            "preprocessing": preprocessor.audit_record(),
            "hyperparameter_candidate_id": candidate_id,
            "hyperparameters": parameters,
            "hyperparameter_rule": "modal_fold_local_SET_F_anchor_choice_then_mean_inner_RMSE",
            "feature_selection_rule": "registry_plus_fold_local_Level1_and_intrinsic_ElasticNet",
            "source_hash": inputs.source_hash,
            "target_history_use": False,
            "R7_status": str(row.R7_status),
            "final_winner": False,
        }
        models[model_key] = FittedMLModel(
            preprocessor=preprocessor,
            estimator=estimator,
            metadata=metadata,
        )
    return {
        "phase": "06",
        "role": "R6_ML_STATISTICAL_MODELER",
        "selection_data_start": DEVELOPMENT_START.strftime("%Y-%m-%d"),
        "selection_data_end": DEVELOPMENT_END.strftime("%Y-%m-%d"),
        "validation_2022_used": False,
        "final_test_2023_used": False,
        "target_history_in_core": False,
        "source_hash": inputs.source_hash,
        "n_models": len(models),
        "models": models,
    }


def save_bundle(bundle: dict[str, Any], path: str | Path) -> Path:
    from joblib import dump

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    dump(bundle, output, compress=3)
    return output


def load_bundle(path: str | Path) -> dict[str, Any]:
    from joblib import load

    bundle = load(Path(path))
    if not isinstance(bundle, dict) or "models" not in bundle:
        raise ValueError("invalid Phase 06 model bundle")
    return bundle


__all__ = [
    "FittedMLModel",
    "load_bundle",
    "refit_locked_models",
    "save_bundle",
]
