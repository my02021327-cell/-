"""Fold-local preprocessing with explicit fit-time and feature-count audit."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

import numpy as np
import pandas as pd


LINEAR_MODELS = {"S00_RIDGE", "S01_ELASTIC_NET"}
NATIVE_MISSING_MODELS = {"S20_XGBOOST", "S21_LIGHTGBM"}


def _numeric_frame(frame: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    return frame.loc[:, list(columns)].apply(pd.to_numeric, errors="coerce")


@dataclass
class FoldPreprocessor:
    """A deterministic train-only preprocessing recipe.

    The fitted object is intentionally plain and serializable.  It records
    medians, scaling parameters, missing indicators, dropped columns, and
    training ranges so audit code can prove that evaluation rows never fit any
    transformation.
    """

    model_id: str
    missing_threshold: float = 0.98
    correlation_threshold: float = 0.999999
    input_features: list[str] = field(default_factory=list)
    retained_features: list[str] = field(default_factory=list)
    dropped_missing: list[str] = field(default_factory=list)
    dropped_constant: list[str] = field(default_factory=list)
    dropped_duplicate: list[str] = field(default_factory=list)
    dropped_collinear: list[str] = field(default_factory=list)
    missing_indicator_features: list[str] = field(default_factory=list)
    medians: dict[str, float] = field(default_factory=dict)
    means: dict[str, float] = field(default_factory=dict)
    scales: dict[str, float] = field(default_factory=dict)
    training_min: dict[str, float] = field(default_factory=dict)
    training_max: dict[str, float] = field(default_factory=dict)
    fit_end: pd.Timestamp | None = None

    @property
    def native_missing(self) -> bool:
        return self.model_id in NATIVE_MISSING_MODELS

    @property
    def scale(self) -> bool:
        return self.model_id in LINEAR_MODELS

    @property
    def output_feature_names(self) -> list[str]:
        return self.retained_features + [
            f"{name}__missing" for name in self.missing_indicator_features
        ]

    def fit(
        self,
        frame: pd.DataFrame,
        features: list[str],
        *,
        fit_end: Any,
    ) -> "FoldPreprocessor":
        if not features:
            raise ValueError("preprocessor requires at least one feature")
        if not 0 <= self.missing_threshold < 1:
            raise ValueError("missing_threshold must lie in [0,1)")
        self.input_features = list(dict.fromkeys(features))
        values = _numeric_frame(frame, self.input_features)
        missing_rate = values.isna().mean()
        self.dropped_missing = sorted(
            missing_rate.index[missing_rate.gt(self.missing_threshold)].tolist()
        )
        candidates = [name for name in self.input_features if name not in self.dropped_missing]

        self.dropped_constant = []
        for name in list(candidates):
            finite = values[name].dropna()
            if finite.nunique(dropna=True) <= 1:
                self.dropped_constant.append(name)
        candidates = [name for name in candidates if name not in self.dropped_constant]
        if not candidates:
            raise ValueError("all features removed by missingness/variance screening")

        self.medians = {}
        for name in candidates:
            finite = values[name].dropna()
            self.medians[name] = float(finite.median()) if len(finite) else 0.0
        filled = values[candidates].fillna(self.medians)

        # Exact duplicates are removed in stable registry order.
        retained: list[str] = []
        self.dropped_duplicate = []
        for name in candidates:
            array = filled[name].to_numpy(dtype=float)
            if any(
                np.array_equal(array, filled[other].to_numpy(dtype=float))
                for other in retained
            ):
                self.dropped_duplicate.append(name)
            else:
                retained.append(name)

        # The feed total/split identity is a known exact linear dependency.
        # Stable order keeps feed_AB_tpd and feed_A, dropping feed_B when all
        # three are present.  This is metadata-driven, not target-driven.
        self.dropped_collinear = []
        if {"feed_AB_tpd", "feed_A", "feed_B"}.issubset(retained):
            retained.remove("feed_B")
            self.dropped_collinear.append("feed_B")

        if self.scale and len(retained) > 1:
            matrix = filled[retained].to_numpy(dtype=float)
            corr = np.corrcoef(matrix, rowvar=False)
            for right in range(1, len(retained)):
                if retained[right] in self.dropped_collinear:
                    continue
                for left in range(right):
                    if retained[left] in self.dropped_collinear:
                        continue
                    value = corr[left, right]
                    if np.isfinite(value) and abs(float(value)) >= self.correlation_threshold:
                        self.dropped_collinear.append(retained[right])
                        break
            retained = [name for name in retained if name not in self.dropped_collinear]

        self.retained_features = retained
        self.missing_indicator_features = [
            name for name in retained if bool(values[name].isna().any())
        ]
        self.training_min = {
            name: float(values[name].min(skipna=True)) for name in retained
        }
        self.training_max = {
            name: float(values[name].max(skipna=True)) for name in retained
        }
        self.fit_end = pd.Timestamp(fit_end).normalize()

        transformed = self._unscaled_transform(frame)
        self.means = {}
        self.scales = {}
        for index, name in enumerate(self.output_feature_names):
            column = transformed[:, index]
            finite = column[np.isfinite(column)]
            mean = float(np.mean(finite)) if finite.size else 0.0
            scale = float(np.std(finite, ddof=0)) if finite.size else 1.0
            self.means[name] = mean
            self.scales[name] = scale if np.isfinite(scale) and scale > 0 else 1.0
        return self

    def _unscaled_transform(self, frame: pd.DataFrame) -> np.ndarray:
        if not self.retained_features:
            raise RuntimeError("preprocessor is not fitted")
        values = _numeric_frame(frame, self.retained_features)
        missing = values.isna()
        numeric = values.to_numpy(dtype=float, copy=True)
        if not self.native_missing:
            for index, name in enumerate(self.retained_features):
                mask = ~np.isfinite(numeric[:, index])
                numeric[mask, index] = self.medians[name]
        indicators = [
            missing[name].to_numpy(dtype=float)[:, None]
            for name in self.missing_indicator_features
        ]
        return np.hstack([numeric, *indicators]) if indicators else numeric

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        result = self._unscaled_transform(frame)
        if self.scale:
            for index, name in enumerate(self.output_feature_names):
                result[:, index] = (
                    result[:, index] - self.means[name]
                ) / self.scales[name]
        return result

    def extrapolation_audit(self, frame: pd.DataFrame) -> pd.DataFrame:
        values = _numeric_frame(frame, self.retained_features)
        flags: dict[str, np.ndarray] = {}
        count = np.zeros(len(frame), dtype=int)
        for name in self.retained_features:
            raw = values[name].to_numpy(dtype=float)
            finite = np.isfinite(raw)
            outside = finite & (
                (raw < self.training_min[name]) | (raw > self.training_max[name])
            )
            count += outside.astype(int)
            if name in {"feed_AB_tpd", "acid_TS", "acid_CODcr"}:
                flags[f"{name}_outside_training_range"] = outside
        result = pd.DataFrame(
            {"feature_out_of_training_range_count": count}, index=frame.index
        )
        for name in ("feed_AB_tpd", "acid_TS", "acid_CODcr"):
            result[f"{name}_outside_training_range"] = flags.get(
                f"{name}_outside_training_range", np.zeros(len(frame), dtype=bool)
            )
        return result

    def audit_record(self) -> dict[str, Any]:
        return {
            "n_input_features": len(self.input_features),
            "n_post_status_filter": len(self.input_features),
            "n_post_missing_filter": len(self.input_features) - len(self.dropped_missing),
            "n_post_collinearity": len(self.retained_features),
            "n_selected_final": len(self.output_feature_names),
            "dropped_missing": ";".join(self.dropped_missing),
            "dropped_constant": ";".join(self.dropped_constant),
            "dropped_duplicate": ";".join(self.dropped_duplicate),
            "dropped_collinear": ";".join(self.dropped_collinear),
            "missing_indicators": ";".join(self.missing_indicator_features),
            "native_missing": self.native_missing,
            "scaling": "STANDARD_TRAIN_ONLY" if self.scale else "NONE",
            "imputation": "NATIVE_MISSING" if self.native_missing else "MEDIAN_TRAIN_ONLY",
        }


__all__ = ["FoldPreprocessor", "LINEAR_MODELS", "NATIVE_MISSING_MODELS"]
