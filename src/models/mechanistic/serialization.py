"""Dependency-light serialization for the locked full-development bundle."""

from __future__ import annotations

import hashlib
from pathlib import Path
import pickle
from typing import Any, Mapping

import pandas as pd


REQUIRED_MODEL_METADATA = (
    "model_id",
    "target",
    "horizon",
    "load_basis",
    "k",
    "kernel_definition",
    "kernel_tail_tolerance",
    "beta_coefficients",
    "training_start",
    "training_end",
    "source_hashes",
    "uses_target_history",
    "preprocessing_rule",
)
LATEST_ALLOWED_TRAINING_END = pd.Timestamp("2021-12-31")


def sha256_file(path: str | Path) -> str:
    """Hash one immutable source artifact without interpreting its rows."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_model_bundle(bundle: Mapping[str, Any]) -> None:
    """Validate R7 hand-off metadata and the sealed development end date."""

    if not isinstance(bundle, Mapping):
        raise TypeError("serialized bundle must be a mapping")
    if bundle.get("phase") != "05":
        raise ValueError("serialized bundle phase must be '05'")
    models = bundle.get("models")
    if not isinstance(models, list) or not models:
        raise ValueError("serialized bundle requires a non-empty models list")
    for position, model in enumerate(models):
        if not isinstance(model, Mapping):
            raise TypeError(f"models[{position}] must be a mapping")
        missing = [key for key in REQUIRED_MODEL_METADATA if key not in model]
        if missing:
            raise KeyError(f"models[{position}] missing metadata: {missing}")
        end = pd.Timestamp(model["training_end"])
        if end > LATEST_ALLOWED_TRAINING_END:
            raise ValueError("full-development artifact may not train beyond 2021-12-31")
        horizon = int(model["horizon"])
        if horizon not in (1, 3, 7, 14, 30):
            raise ValueError(f"unsupported serialized horizon: {horizon}")


def save_model_bundle(bundle: Mapping[str, Any], path: str | Path) -> Path:
    """Validate and atomically pickle a Phase 05 bundle to a ``.joblib`` path."""

    validate_model_bundle(bundle)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with temporary.open("wb") as handle:
        pickle.dump(dict(bundle), handle, protocol=pickle.HIGHEST_PROTOCOL)
    temporary.replace(destination)
    return destination


def load_model_bundle(path: str | Path) -> dict[str, Any]:
    """Load and revalidate a Phase 05 model bundle."""

    with Path(path).open("rb") as handle:
        value = pickle.load(handle)  # noqa: S301 - trusted local research artifact
    validate_model_bundle(value)
    return dict(value)

