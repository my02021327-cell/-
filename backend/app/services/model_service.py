from __future__ import annotations

import json
import math
import uuid
from typing import Any

import joblib
import numpy as np
import pandas as pd

from src.models.mechanistic.kernels import build_first_order_kernel
from src.models.mechanistic.kinetic_state import build_kinetic_state

from ..config import BUNDLE_PATH, PREDICTION_DIR, RELIABILITY_SEED_PATH
from ..repository import Repository
from .dataset_service import DatasetService
from .feature_service import FeatureService


class ForecastError(RuntimeError):
    def __init__(self, code: str, detail: str):
        super().__init__(detail)
        self.code = code
        self.detail = detail


def _value(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


class ModelService:
    MINIMUM_USABLE_HISTORY_ROWS = 20

    def __init__(self, repository: Repository, datasets: DatasetService):
        self.repository = repository
        self.datasets = datasets
        self.features = FeatureService()
        if not BUNDLE_PATH.is_file():
            raise ForecastError("MODEL_NOT_LOADED", f"missing {BUNDLE_PATH}")
        self.bundle = joblib.load(BUNDLE_PATH)
        self.seed = pd.read_parquet(RELIABILITY_SEED_PATH)
        for column in ("origin_date", "target_date"):
            self.seed[column] = pd.to_datetime(self.seed[column]).dt.normalize()

    def _target_series(self, frame: pd.DataFrame, target: str) -> pd.Series:
        daily = frame.set_index("date")[target].apply(pd.to_numeric, errors="coerce")
        calendar = pd.date_range(daily.index.min(), daily.index.max(), freq="D")
        return daily.reindex(calendar)

    def _state(self, frame: pd.DataFrame, k: float) -> tuple[float | None, dict[str, Any]]:
        daily = frame.set_index("date")["feed_AB_tpd"].apply(pd.to_numeric, errors="coerce")
        calendar = pd.date_range(daily.index.min(), daily.index.max(), freq="D")
        daily = daily.reindex(calendar)
        kernel = build_first_order_kernel(k)
        state = build_kinetic_state(daily, kernel)
        last = state.iloc[-1]
        return _value(last["kinetic_state"]), {
            "status": "AVAILABLE" if bool(last["state_available"]) else "UNAVAILABLE_INSUFFICIENT_HISTORY",
            "kernel_max_lag": int(kernel.max_lag), "kernel_coverage": float(last["kernel_coverage"]),
            "minimum_coverage": float(last["minimum_kernel_coverage"]),
        }

    def _weights(
        self, target: str, horizon: int, origin: pd.Timestamp,
        available: list[str], current_observed: bool,
    ) -> tuple[dict[str, float], dict[str, Any]]:
        experts = self.bundle["expert_pool"]
        stats: dict[str, dict[str, Any]] = {}
        truth = self.seed.loc[
            self.seed["target_name"].eq(target) & self.seed["horizon"].eq(horizon)
            & self.seed["target_observed"].fillna(False) & self.seed["target_date"].le(origin)
        ].drop_duplicates(["origin_date", "target_date"])
        target_var = float(np.var(truth["y_true"].dropna())) if len(truth) >= 2 else 1.0
        epsilon = max(1e-12, 1e-8 * target_var)
        mature: dict[str, float] = {}
        for expert in experts:
            rows = self.seed.loc[
                self.seed["target_name"].eq(target) & self.seed["horizon"].eq(horizon)
                & self.seed["expert_id"].eq(expert)
                & self.seed["target_observed"].fillna(False)
                & self.seed["prediction_available"].fillna(False)
                & self.seed["target_date"].le(origin)
            ]
            errors = pd.to_numeric(rows["error"], errors="coerce").dropna().tolist()
            operational = [
                row["error"] for row in self.repository.reliability_errors(
                    target, horizon, origin.strftime("%Y-%m-%d")
                ) if row["expert_id"] == expert
            ]
            errors = pd.Series(errors + operational, dtype=float).dropna()
            mse = float(np.mean(np.square(errors))) if len(errors) else None
            stats[expert] = {"n_matured_errors": len(errors), "MSE": mse, "max_target_date": rows["target_date"].max().strftime("%Y-%m-%d") if len(rows) else None}
            if expert in available and len(errors) >= 20 and mse is not None:
                mature[expert] = mse
        if mature:
            raw = {expert: 1.0 / (mse + epsilon) for expert, mse in mature.items()}
            total = sum(raw.values())
            weights = {expert: float(raw.get(expert, 0.0) / total) for expert in experts}
            mode = "PAST_ONLY_RELIABILITY"
        else:
            priority = ["B10", "M2", "RIDGE", "M1F", "B11"] if current_observed else ["B11", "RIDGE", "M1F"]
            selected = next((expert for expert in priority if expert in available), None)
            if selected is None:
                raise ForecastError("NO_AVAILABLE_EXPERT", f"{target} h{horizon}: no available expert")
            weights = {expert: float(expert == selected) for expert in experts}
            mode = "SAFE_FALLBACK_INSUFFICIENT_HISTORY"
        return weights, {"mode": mode, "epsilon": epsilon, "stats": stats}

    def _expert_predictions(self, frame: pd.DataFrame, target: str, horizon: int) -> tuple[dict[str, float | None], dict[str, Any]]:
        key = f"{target}__h{horizon}"
        if key not in self.bundle["models"]:
            raise ForecastError("MODEL_NOT_LOADED", key)
        config = self.bundle["models"][key]
        target_series = self._target_series(frame, target)
        origin = frame["date"].max()
        current = _value(target_series.reindex([origin]).iloc[0])
        observed = target_series.dropna()
        last = _value(observed.iloc[-1]) if len(observed) else None
        state, state_audit = self._state(frame, float(config["k"]))
        predictions: dict[str, float | None] = {"B10": current, "B11": last}
        if state is None:
            predictions["M1F"] = None; predictions["M2"] = None
        else:
            base = max(0.0, config["m1f"]["beta0"] + config["m1f"]["beta1"] * state)
            current_fit = config["m1f_current"]["beta0"] + config["m1f_current"]["beta1"] * state
            predictions["M1F"] = float(base)
            predictions["M2"] = float(max(0.0, base + (current - current_fit) * np.exp(-horizon / config["tau_res"]))) if current is not None else None
        ridge = config["ridge"]
        latest = frame.loc[frame["date"].eq(origin)].tail(1)
        try:
            raw = ridge["estimator"].predict(ridge["preprocessor"].transform(latest))[0]
            predictions["RIDGE"] = float(max(0.0, raw)) if np.isfinite(raw) else None
        except Exception as exc:
            predictions["RIDGE"] = None
            state_audit["ridge_error"] = f"{type(exc).__name__}: {exc}"
        available = [expert for expert, prediction in predictions.items() if prediction is not None]
        if current is None:
            available = [expert for expert in available if expert not in {"B10", "M2"}]
            predictions["B10"] = None; predictions["M2"] = None
        return predictions, {"available": available, "current_target_observed": current is not None, "last_observation_age_d": int((origin - observed.index.max()).days) if len(observed) else None, "kinetic_state": state_audit}

    def determine_latest_forecastable_origin(
        self, frame: pd.DataFrame
    ) -> tuple[pd.Timestamp, list[dict[str, Any]]]:
        """Return the newest uploaded date that satisfies the deployment contract."""
        ordered = frame.sort_values("date").drop_duplicates("date", keep=False)
        rejected: list[dict[str, Any]] = []
        for candidate in reversed(ordered["date"].tolist()):
            prefix, reasons = self.features.prepare_candidate(
                ordered,
                pd.Timestamp(candidate),
                minimum_history_rows=self.MINIMUM_USABLE_HISTORY_ROWS,
            )
            if not reasons:
                try:
                    for target, horizons in (
                        ("CH4_m3d_observed", range(1, 8)),
                        ("biogas_AB_m3d", (1,)),
                    ):
                        for horizon in horizons:
                            _, availability = self._expert_predictions(prefix, target, horizon)
                            if not availability["available"]:
                                reasons.append(f"NO_EXPERT:{target}:h{horizon}")
                except Exception as exc:
                    reasons.append(f"FEATURE_BUILD_FAILED:{type(exc).__name__}")
            if not reasons:
                return pd.Timestamp(candidate).normalize(), rejected
            rejected.append(
                {
                    "date": pd.Timestamp(candidate).strftime("%Y-%m-%d"),
                    "reasons": reasons,
                }
            )
        code = (
            "INSUFFICIENT_HISTORY"
            if len(ordered) < self.MINIMUM_USABLE_HISTORY_ROWS
            else "NO_FORECASTABLE_ORIGIN"
        )
        raise ForecastError(code, json.dumps(rejected[:10], ensure_ascii=False))

    def unavailable_response(self, dataset_id: str, error: ForecastError) -> dict[str, Any]:
        descriptor = self.datasets.describe(dataset_id)
        response = {
            "dataset_id": dataset_id,
            "latest_data_date": descriptor["latest_data_date"],
            "origin_date": None,
            "forecast_origin_date": None,
            "latest_usable_date": None,
            "status": error.code,
            "reason": error.detail,
            "model": {
                "system_id": self.bundle["system_id"],
                "version": self.bundle["model_version"],
                "supported_horizons": self.bundle["supported_horizons"],
            },
            "methane": {"unit": "m3/day", "predictions": []},
            "biogas": {"unit": "m3/day", "predictions": []},
            "purity_next_day": None,
        }
        self.repository.update_dataset_runtime(
            dataset_id,
            status="READY",
            metadata={
                "forecast_origin_date": None,
                "latest_usable_date": None,
                "forecast_status": error.code,
                "forecast_reason": error.detail,
            },
        )
        return response

    def get_or_create_forecast(self, dataset_id: str, *, refresh: bool = False) -> dict[str, Any]:
        frame = self.datasets.load_clean(dataset_id).sort_values("date")
        try:
            origin, _ = self.determine_latest_forecastable_origin(frame)
        except ForecastError as exc:
            return self.unavailable_response(dataset_id, exc)
        origin_text = origin.strftime("%Y-%m-%d")
        if not refresh:
            cached = self.repository.latest_forecast(
                dataset_id,
                model_version=self.bundle["model_version"],
                origin_date=origin_text,
            )
            # A response stored before the biogas target was published lacks that
            # key. Treat it as stale rather than serving an incomplete contract.
            fresh_contract = cached is not None and "biogas" in cached
            if (
                cached is not None
                and fresh_contract
                and cached.get("latest_data_date") == frame["date"].max().strftime("%Y-%m-%d")
            ):
                self.repository.update_dataset_runtime(
                    dataset_id,
                    status="READY",
                    metadata={
                        "forecast_origin_date": origin_text,
                        "latest_usable_date": origin_text,
                        "forecast_status": "AVAILABLE",
                    },
                )
                return cached
        return self.forecast(dataset_id, persist=True)

    def forecast(self, dataset_id: str, *, persist: bool = True) -> dict[str, Any]:
        full_frame = self.datasets.load_clean(dataset_id).sort_values("date")
        if full_frame.empty:
            raise ForecastError("FEATURE_BUILD_FAILED", "dataset has no rows")
        origin, rejected_origins = self.determine_latest_forecastable_origin(full_frame)
        frame = full_frame.loc[full_frame["date"].le(origin)].copy()
        latest_data_date = full_frame["date"].max().strftime("%Y-%m-%d")
        matured_operational_errors = self.repository.mature_forecast_errors(full_frame)
        audits: list[dict[str, Any]] = []
        methane = []
        biogas_predictions = []
        predictions_by_key: dict[tuple[str, int], float] = {}
        # The bundle carries locked specs for biogas_AB_m3d h1..h7 as well, so the
        # biogas horizons are published from the same A30 V2 run rather than being
        # re-derived anywhere else. Only h1 is contractually required (the origin
        # gate checks it); a later horizon that cannot be produced is omitted
        # instead of failing the whole forecast.
        for target, horizons in (("CH4_m3d_observed", range(1, 8)), ("biogas_AB_m3d", range(1, 8))):
            for horizon in horizons:
                try:
                    expert_predictions, availability = self._expert_predictions(frame, target, horizon)
                    weights, reliability = self._weights(target, horizon, origin, availability["available"], availability["current_target_observed"])
                except ForecastError:
                    if target == "CH4_m3d_observed" or horizon == 1:
                        raise
                    continue
                value = sum(weights[expert] * expert_predictions[expert] for expert in weights if weights[expert] > 0 and expert_predictions[expert] is not None)
                value = float(max(0.0, value))
                predictions_by_key[(target, horizon)] = value
                target_date = (origin + pd.Timedelta(days=horizon)).strftime("%Y-%m-%d")
                audit = {
                    "target": target, "horizon": horizon, "origin_date": origin.strftime("%Y-%m-%d"),
                    "target_date": target_date, "expert_predictions": expert_predictions,
                    "expert_availability": availability, "A30_weights": weights,
                    "A30_prediction": value, "reliability": reliability,
                }
                audits.append(audit)
                if target == "CH4_m3d_observed":
                    methane.append({"horizon": horizon, "target_date": target_date, "value": value})
                else:
                    biogas_predictions.append({"horizon": horizon, "target_date": target_date, "value": value})
        ch4 = predictions_by_key[("CH4_m3d_observed", 1)]
        biogas = predictions_by_key[("biogas_AB_m3d", 1)]
        purity = None
        purity_audit = {"CH4_h1_prediction": ch4, "biogas_h1_prediction": biogas, "formula": "100 * CH4_h1 / biogas_h1"}
        if biogas > 0 and ch4 >= 0 and ch4 <= biogas:
            purity_value = 100.0 * ch4 / biogas
            purity = {"target_date": methane[0]["target_date"], "value_pct": purity_value, "method": "DERIVED_FROM_CH4_AND_BIOGAS_FORECAST"}
            purity_audit.update({"derived_purity": purity_value, "validity": "VALID"})
        else:
            purity = {"target_date": methane[0]["target_date"], "value_pct": None, "method": "DERIVED_FROM_CH4_AND_BIOGAS_FORECAST", "reason": "PHYSICAL_CONSISTENCY_FAILED"}
            purity_audit.update({"derived_purity": None, "validity": "INVALID_PHYSICAL_CONSISTENCY"})
        response = {
            "dataset_id": dataset_id,
            "latest_data_date": latest_data_date,
            "origin_date": origin.strftime("%Y-%m-%d"),
            "forecast_origin_date": origin.strftime("%Y-%m-%d"),
            "latest_usable_date": origin.strftime("%Y-%m-%d"),
            "status": "AVAILABLE",
            "model": {"system_id": self.bundle["system_id"], "version": self.bundle["model_version"], "supported_horizons": self.bundle["supported_horizons"]},
            "methane": {"unit": "m3/day", "predictions": methane},
            "biogas": {"unit": "m3/day", "predictions": biogas_predictions},
            "purity_next_day": purity,
        }
        run_id = uuid.uuid4().hex
        audit_payload = {
            "run_id": run_id,
            "dataset_id": dataset_id,
            "model_version": self.bundle["model_version"],
            "data_authority": "UPLOADED_EXCEL_ONLY",
            "latest_data_date": latest_data_date,
            "forecast_origin_date": origin.strftime("%Y-%m-%d"),
            "rejected_newer_origins": rejected_origins,
            "minimum_usable_history_rows": self.MINIMUM_USABLE_HISTORY_ROWS,
            "forecasts": audits,
            "purity_provenance": purity_audit,
            "matured_operational_errors_added": matured_operational_errors,
            "api_response": response,
        }
        if persist:
            path = PREDICTION_DIR / f"{run_id}.json"
            path.write_text(json.dumps(audit_payload, ensure_ascii=False, indent=2), encoding="utf-8")
            self.repository.save_forecast(run_id, dataset_id, response, audit_payload)
            self.repository.update_dataset_runtime(
                dataset_id,
                status="READY",
                metadata={
                    "forecast_origin_date": origin.strftime("%Y-%m-%d"),
                    "latest_usable_date": origin.strftime("%Y-%m-%d"),
                    "forecast_status": "AVAILABLE",
                },
            )
        return response
