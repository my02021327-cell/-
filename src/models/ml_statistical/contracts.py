"""Locked Phase 03 feature-registry interpretation for Phase 06.

This module never creates a new physical feature.  It only selects materialized
Phase 03 columns and assigns them to the two deployment-availability tracks and
the cumulative feature-set progression requested by the Phase 06 contract.
"""

from __future__ import annotations

import ctypes
import importlib
import re
import sys
from pathlib import Path
from typing import Any

import pandas as pd


DEVELOPMENT_START = pd.Timestamp("2018-01-01")
DEVELOPMENT_END = pd.Timestamp("2021-12-31")
HORIZONS = (1, 3, 7, 14, 30)
FEATURE_SETS = ("SET-A", "SET-B", "SET-C", "SET-E", "SET-F")
TRACK_STRICT = "TRACK-S_STRICT"
TRACK_PROVISIONAL = "TRACK-P_PROVISIONAL_DAILY"
AVAILABILITY_TRACKS = (TRACK_STRICT, TRACK_PROVISIONAL)
MODEL_FAMILIES = (
    "S00_RIDGE",
    "S01_ELASTIC_NET",
    "S10_RANDOM_FOREST",
    "S20_XGBOOST",
    "S21_LIGHTGBM",
)

STRICT_AVAILABILITY = {"CONFIRMED_AT_ORIGIN", "PAST_ONLY"}
PROVISIONAL_AVAILABILITY = STRICT_AVAILABILITY | {"DATE_ONLY_PROVISIONAL"}
TARGET_ANCESTOR_TOKENS = ("ch4", "biogas", "target", "prediction", "residual")
FORBIDDEN_CORE_NAMES = {
    "HRT_mass_proxy_d",
    "COD_load_kgd",
    "CH4_lag1",
    "CH4_lag2",
    "CH4_lag7",
    "biogas_lag1",
}
FORBIDDEN_CORE_SUBSTRINGS = (
    "true_hrt",
    "rolling_ch4",
    "rolling_biogas",
    "gas_per_ch4",
)


def bootstrap_local_dependencies(project_root: str | Path | None = None) -> Path:
    """Expose the pre-existing local model runtime and preload macOS OpenMP."""

    root = (
        Path(project_root).resolve()
        if project_root is not None
        else Path(__file__).resolve().parents[3]
    )
    dependency_root = root.parent / ".model_deps"
    if dependency_root.is_dir() and str(dependency_root) not in sys.path:
        sys.path.insert(0, str(dependency_root))
    libomp = dependency_root / "llvm_openmp_runtime" / "lib" / "libomp.dylib"
    if libomp.is_file():
        try:
            ctypes.CDLL(str(libomp), mode=ctypes.RTLD_GLOBAL)
        except OSError:
            # Availability is reported explicitly by dependency_status below.
            pass
    return dependency_root


def dependency_status(project_root: str | Path | None = None) -> pd.DataFrame:
    """Return explicit library availability; no family is silently replaced."""

    bootstrap_local_dependencies(project_root)
    rows: list[dict[str, Any]] = []
    modules = {
        "S00_RIDGE": ("sklearn", "scikit-learn"),
        "S01_ELASTIC_NET": ("sklearn", "scikit-learn"),
        "S10_RANDOM_FOREST": ("sklearn", "scikit-learn"),
        "S20_XGBOOST": ("xgboost", "xgboost"),
        "S21_LIGHTGBM": ("lightgbm", "lightgbm"),
    }
    for model_id, (module_name, library_name) in modules.items():
        try:
            module = importlib.import_module(module_name)
            rows.append(
                {
                    "model_id": model_id,
                    "library": library_name,
                    "dependency_status": "AVAILABLE",
                    "library_version": str(getattr(module, "__version__", "UNKNOWN")),
                    "dependency_detail": "local locked runtime",
                }
            )
        except Exception as exc:  # pragma: no cover - environment-dependent
            rows.append(
                {
                    "model_id": model_id,
                    "library": library_name,
                    "dependency_status": "UNAVAILABLE_DEPENDENCY",
                    "library_version": "",
                    "dependency_detail": f"{type(exc).__name__}: {exc}",
                }
            )
    return pd.DataFrame(rows)


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def _normalized_feature_set(value: Any) -> set[str]:
    raw = str(value or "").strip()
    if not raw or raw.lower() == "nan":
        return set()
    tags = {item.strip() for item in raw.split(";") if item.strip()}
    normalized: set[str] = set()
    for tag in tags:
        if tag == "SET-B_OPTIONAL":
            normalized.add("SET-B")
        elif tag == "SET-C_SOURCE_ATTRIBUTION":
            normalized.add("SET-C")
        elif tag == "SET-F_CALENDAR":
            normalized.add("SET-F")
        else:
            match = re.match(r"^(SET-[A-Z][0-9]?)", tag)
            normalized.add(match.group(1) if match else tag)
    return normalized


def _cumulative_set_allowed(tags: set[str], feature_set: str) -> bool:
    order = {name: index for index, name in enumerate(FEATURE_SETS)}
    if feature_set not in order:
        raise ValueError(f"unknown feature set: {feature_set}")
    allowed = set(FEATURE_SETS[: order[feature_set] + 1])
    # SET-D1/D2 and SET-G are deliberately outside the Phase 06 core path.
    return bool(tags & allowed)


def _has_target_ancestor(row: pd.Series) -> bool:
    text = ";".join(
        str(row.get(column, ""))
        for column in ("raw_ancestors", "source_columns", "formula")
    ).lower()
    return any(token in text for token in TARGET_ANCESTOR_TOKENS)


def _forbidden_name(name: str) -> bool:
    if name in FORBIDDEN_CORE_NAMES:
        return True
    lowered = name.lower()
    return any(token in lowered for token in FORBIDDEN_CORE_SUBSTRINGS)


def build_feature_contract(
    registry: pd.DataFrame,
    materialized_columns: list[str] | tuple[str, ...],
) -> pd.DataFrame:
    """Return one audited row per track/set/materialized registry feature."""

    required = {
        "feature_name",
        "status",
        "availability_status",
        "feature_set",
        "materialized",
        "raw_ancestors",
        "target_dependency",
    }
    missing = sorted(required - set(registry.columns))
    if missing:
        raise KeyError(f"feature registry missing columns: {missing}")
    materialized = set(materialized_columns)
    records: list[dict[str, Any]] = []
    for _, row in registry.iterrows():
        name = str(row["feature_name"])
        tags = _normalized_feature_set(row["feature_set"])
        base_ok = (
            name in materialized
            and _truthy(row["materialized"])
            and not _forbidden_name(name)
            and str(row.get("target_dependency", "NONE")).upper() in {"NONE", "FALSE", "NO"}
            and not _has_target_ancestor(row)
            and not bool(tags & {"SET-D1", "SET-D2", "SET-G"})
        )
        for horizon in HORIZONS:
            allowed_value = str(row.get(f"allowed_h{horizon}", "N")).upper()
            for track in AVAILABILITY_TRACKS:
                if track == TRACK_STRICT:
                    track_ok = (
                        str(row["status"]).upper() == "APPROVED"
                        and str(row["availability_status"]).upper() in STRICT_AVAILABILITY
                        and allowed_value == "Y"
                    )
                else:
                    track_ok = (
                        str(row["status"]).upper() in {"APPROVED", "PROVISIONAL"}
                        and str(row["availability_status"]).upper() in PROVISIONAL_AVAILABILITY
                        and allowed_value in {"Y", "P"}
                    )
                for feature_set in FEATURE_SETS:
                    selected = bool(base_ok and track_ok and _cumulative_set_allowed(tags, feature_set))
                    if selected:
                        reason = "REGISTRY_CONTRACT_SELECTED"
                    elif not base_ok:
                        reason = "CORE_CONTRACT_REJECTED"
                    elif not track_ok:
                        reason = "TRACK_AVAILABILITY_EXCLUDED"
                    else:
                        reason = "NOT_YET_IN_CUMULATIVE_SET"
                    records.append(
                        {
                            "feature_name": name,
                            "horizon": horizon,
                            "availability_track": track,
                            "feature_set": feature_set,
                            "selected_by_registry": selected,
                            "registry_status": str(row["status"]),
                            "availability_status": str(row["availability_status"]),
                            "registry_feature_set": str(row["feature_set"]),
                            "raw_ancestors": str(row.get("raw_ancestors", "")),
                            "target_dependency": str(row.get("target_dependency", "")),
                            "regime_dependency": str(row.get("regime_dependency", "")),
                            "selection_reason": reason,
                        }
                    )
    contract = pd.DataFrame(records)
    selected = contract.loc[contract["selected_by_registry"]]
    if selected.empty:
        raise ValueError("feature contract selected no core features")
    forbidden = selected["feature_name"].map(_forbidden_name)
    target_dependent = selected["raw_ancestors"].str.lower().map(
        lambda text: any(token in text for token in TARGET_ANCESTOR_TOKENS)
    )
    if bool(forbidden.any() or target_dependent.any()):
        raise AssertionError("forbidden core predictor escaped registry filtering")
    return contract


def features_for(
    contract: pd.DataFrame,
    *,
    horizon: int,
    availability_track: str,
    feature_set: str,
) -> list[str]:
    """Return stable feature order for one track/set/horizon contract."""

    subset = contract.loc[
        contract["horizon"].eq(int(horizon))
        & contract["availability_track"].eq(availability_track)
        & contract["feature_set"].eq(feature_set)
        & contract["selected_by_registry"].fillna(False).astype(bool)
    ]
    names = subset["feature_name"].astype(str).drop_duplicates().tolist()
    preferred = ["feed_AB_tpd", "feed_A", "feed_B"]
    ordering = {name: index for index, name in enumerate(preferred)}
    return sorted(names, key=lambda name: (ordering.get(name, len(preferred)), name))


__all__ = [
    "AVAILABILITY_TRACKS",
    "DEVELOPMENT_END",
    "DEVELOPMENT_START",
    "FEATURE_SETS",
    "HORIZONS",
    "MODEL_FAMILIES",
    "TRACK_PROVISIONAL",
    "TRACK_STRICT",
    "bootstrap_local_dependencies",
    "build_feature_contract",
    "dependency_status",
    "features_for",
]
