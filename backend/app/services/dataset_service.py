from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..config import PROCESSED_DIR, UPLOAD_DIR
from ..repository import Repository
from .derived_service import add_derived
from .dq_service import run_data_quality
from .excel_service import parse_imputation_log, parse_workbook
from .recommendation_service import build_recommendations, build_trend
from .health_service import (
    build_deltas,
    build_evidence,
    diagnose,
    evaluate_health,
    evaluate_operator_actions,
    evaluate_telemetry_signals,
)


LOGGER = logging.getLogger("uvicorn.error")


def _json_value(value: Any) -> Any:
    if isinstance(value, pd.Timestamp): return value.strftime("%Y-%m-%d")
    if isinstance(value, (np.integer,)): return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    if pd.isna(value): return None
    return value


def frame_records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    return [{key: _json_value(value) for key, value in row.items()} for row in frame.to_dict("records")]


class DatasetService:
    def __init__(self, repository: Repository):
        self.repository = repository

    def upload(self, file_name: str, payload: bytes) -> dict[str, Any]:
        digest = hashlib.sha256(payload).hexdigest()
        dataset_id = digest[:16]
        if existing := self.repository.get_dataset(dataset_id):
            descriptor = self.describe(dataset_id)
            descriptor["processing_mode"] = "REUSED"
            LOGGER.info("dataset reused dataset_id=%s rows=%s", dataset_id, descriptor["row_count"])
            return descriptor
        raw = parse_workbook(payload)
        LOGGER.info(
            "Excel parsed dataset_id=%s sheet=BIOGUARD_INPUT rows=%s first_date=%s latest_date=%s",
            dataset_id, len(raw), raw["date"].min().strftime("%Y-%m-%d"),
            raw["date"].max().strftime("%Y-%m-%d"),
        )
        clean, dq, summary = run_data_quality(raw)
        LOGGER.info(
            "DQ complete dataset_id=%s valid=%s missing=%s suspect=%s invalid=%s",
            dataset_id, summary["valid"], summary["missing"], summary["suspect"], summary["invalid"],
        )
        clean = add_derived(clean)
        uploaded_at = datetime.now(timezone.utc).isoformat()
        upload_path = UPLOAD_DIR / f"{dataset_id}_{Path(file_name).name}"
        upload_path.write_bytes(payload)
        raw_path = PROCESSED_DIR / f"{dataset_id}_raw.parquet"
        clean_path = PROCESSED_DIR / f"{dataset_id}_clean.parquet"
        dq_path = PROCESSED_DIR / f"{dataset_id}_dq.parquet"
        provenance = parse_imputation_log(payload)
        provenance_path = PROCESSED_DIR / f"{dataset_id}_provenance.parquet"
        raw.to_parquet(raw_path, index=False)
        clean.to_parquet(clean_path, index=False)
        dq.to_parquet(dq_path, index=False)
        provenance.to_parquet(provenance_path, index=False)
        LOGGER.info(
            "provenance recorded dataset_id=%s synthesised_cells=%s", dataset_id, len(provenance)
        )
        record = {
            "dataset_id": dataset_id, "file_name": Path(file_name).name,
            "uploaded_at": uploaded_at,
            "first_date": raw["date"].min().strftime("%Y-%m-%d"),
            "latest_date": raw["date"].max().strftime("%Y-%m-%d"),
            "row_count": len(raw), "raw_path": str(raw_path), "clean_path": str(clean_path),
            "dq_path": str(dq_path), "status": "READY",
            "metadata": {
                "provenance_path": str(provenance_path),
                "synthesised_cells": int(len(provenance)),
                "sheet": "BIOGUARD_INPUT",
                "sha256": digest,
                "raw_preserved": True,
                "data_authority": "UPLOADED_EXCEL_ONLY",
                "latest_data_date": raw["date"].max().strftime("%Y-%m-%d"),
                "forecast_origin_date": None,
                "latest_usable_date": None,
                "forecast_status": "NOT_EVALUATED",
                "validation_warnings": (
                    ["MEASUREMENT_DATE_AFTER_SERVER_DATE"]
                    if raw["date"].max().date() > datetime.now(timezone.utc).date()
                    else []
                ),
            },
        }
        self.repository.save_dataset(record, summary)
        descriptor = self.describe(dataset_id)
        descriptor["processing_mode"] = "CREATED"
        LOGGER.info("dataset persisted dataset_id=%s status=READY", dataset_id)
        return descriptor

    def _record(self, dataset_id: str) -> dict[str, Any]:
        record = self.repository.get_dataset(dataset_id)
        if not record:
            raise KeyError("DATASET_NOT_FOUND")
        return record

    def load_clean(self, dataset_id: str) -> pd.DataFrame:
        record = self._record(dataset_id)
        frame = pd.read_parquet(record["clean_path"])
        frame["date"] = pd.to_datetime(frame["date"]).dt.normalize()
        return frame

    def describe(self, dataset_id: str) -> dict[str, Any]:
        record = self._record(dataset_id)
        metadata = record.get("metadata", {})
        return {
            "dataset_id": record["dataset_id"], "file_name": record["file_name"],
            "uploaded_at": record["uploaded_at"], "first_date": record["first_date"],
            "latest_date": record["latest_date"],
            "latest_data_date": metadata.get("latest_data_date", record["latest_date"]),
            "forecast_origin_date": metadata.get("forecast_origin_date"),
            "latest_usable_date": metadata.get("latest_usable_date"),
            "forecast_status": metadata.get("forecast_status", "NOT_EVALUATED"),
            "validation_warnings": metadata.get("validation_warnings", []),
            "data_authority": metadata.get("data_authority", "UPLOADED_EXCEL_ONLY"),
            "row_count": record["row_count"],
            "status": record["status"], "data_quality": self.repository.get_dq_summary(dataset_id),
        }

    def latest(self, dataset_id: str) -> dict[str, Any]:
        frame = self.load_clean(dataset_id)
        latest_date = frame["date"].max().strftime("%Y-%m-%d")
        return self.at_date(dataset_id, latest_date)

    def _dq_statuses(self, dataset_id: str) -> pd.DataFrame:
        record = self._record(dataset_id)
        dq = pd.read_parquet(record["dq_path"])[["date", "field", "dq_status"]]
        dq["date"] = pd.to_datetime(dq["date"]).dt.normalize()
        return dq

    def _dq_for_date(self, dataset_id: str, target: pd.Timestamp) -> dict[str, str]:
        dq = self._dq_statuses(dataset_id)
        rows = dq.loc[dq["date"].eq(target)]
        return {str(row.field): str(row.dq_status) for row in rows.itertuples()}

    def _row_values(self, frame: pd.DataFrame, target: pd.Timestamp) -> dict[str, Any] | None:
        rows = frame.loc[frame["date"].eq(target)]
        if rows.empty:
            return None
        return {key: _json_value(value) for key, value in rows.iloc[-1].items()}

    def diagnose_at(self, dataset_id: str, date: str) -> tuple[dict[str, Any], dict[str, Any], dict[str, str], dict[str, Any] | None]:
        """Run the project health engine for one date.

        The engine also wants the previous direct measurement (for ACID EARLY)
        and the previous health state (for RECOVERY); both come from the
        immediately preceding row of the same uploaded dataset.
        """
        frame = self.load_clean(dataset_id).sort_values("date")
        target = pd.Timestamp(date).normalize()
        values = self._row_values(frame, target)
        if values is None:
            raise KeyError("DATE_NOT_FOUND")
        dq_statuses = self._dq_for_date(dataset_id, target)

        earlier = frame.loc[frame["date"].lt(target), "date"]
        previous_state = None
        deltas = None
        if len(earlier):
            previous_date = earlier.max()
            previous_values = self._row_values(frame, previous_date)
            deltas = build_deltas(values, previous_values)
            previous_diagnosis = diagnose(
                previous_values, self._dq_for_date(dataset_id, previous_date)
            )
            previous_state = previous_diagnosis["health"]["state"]
        diagnosis = diagnose(values, dq_statuses, deltas, previous_state)
        return values, diagnosis, dq_statuses, deltas

    def at_date(self, dataset_id: str, date: str) -> dict[str, Any]:
        values, diagnosis, dq_statuses, deltas = self.diagnose_at(dataset_id, date)
        payload = dict(values)
        payload["dataset_id"] = dataset_id
        payload["measurement_date"] = payload.pop("date")
        evidence = build_evidence(diagnosis, dq_statuses)
        payload["health"] = {**evaluate_health(diagnosis), "evidence": evidence}
        payload["telemetry_signals"] = evaluate_telemetry_signals(diagnosis, values, dq_statuses)
        actions = evaluate_operator_actions(diagnosis, evidence)
        # Practical operator guidance: project conditions set the severity, the
        # recent trend of the same dataset adds context, reference practice adds
        # only the response.
        trend = build_trend(self.history(dataset_id)["rows"], date)
        actions["recommendations"] = build_recommendations(diagnosis, evidence, trend, deltas)
        actions["trend"] = trend
        payload["operator_actions"] = actions
        return payload

    # The observed CH4 formula the project defines; a synthesised input makes the
    # result derived-from-imputed rather than a measurement.
    CH4_INPUT_FIELDS = ("biogas_A_m3d", "biogas_B_m3d", "ch4_purity_A_pct", "ch4_purity_B_pct")

    def _synthesised_cells(self, dataset_id: str) -> set[tuple[pd.Timestamp, str]]:
        """Cells the workbook itself declares as synthesised.

        Datasets stored before provenance was recorded have no parquet, so it is
        derived once from the retained upload rather than being lost.
        """
        record = self._record(dataset_id)
        path = Path(record.get("metadata", {}).get("provenance_path")
                    or PROCESSED_DIR / f"{dataset_id}_provenance.parquet")
        if not path.is_file():
            upload = UPLOAD_DIR / f"{dataset_id}_{record['file_name']}"
            if not upload.is_file():
                return set()
            frame = parse_imputation_log(upload.read_bytes())
            frame.to_parquet(path, index=False)
            self.repository.update_dataset_runtime(
                dataset_id, status=record["status"],
                metadata={"provenance_path": str(path), "synthesised_cells": int(len(frame))},
            )
            LOGGER.info(
                "provenance backfilled dataset_id=%s synthesised_cells=%s", dataset_id, len(frame)
            )
        else:
            frame = pd.read_parquet(path)
        if frame.empty:
            return set()
        frame["date"] = pd.to_datetime(frame["date"]).dt.normalize()
        return set(zip(frame["date"], frame["field"].astype(str)))

    def history(self, dataset_id: str, limit: int | None = None) -> dict[str, Any]:
        frame = self.load_clean(dataset_id).sort_values("date")
        if limit and limit > 0:
            frame = frame.tail(limit)
        synthesised = self._synthesised_cells(dataset_id)
        rows = frame_records(frame)
        for record, date in zip(rows, frame["date"]):
            if record.get("CH4_m3d_observed") is None:
                record["ch4_source"] = "UNAVAILABLE"
                continue
            imputed = [
                field for field in self.CH4_INPUT_FIELDS if (date, field) in synthesised
            ]
            record["ch4_source"] = "DERIVED_FROM_IMPUTED" if imputed else "DERIVED_FROM_OBSERVED"
            record["ch4_imputed_inputs"] = imputed
        return {
            "dataset_id": dataset_id,
            "ch4_formula": "biogas_A * ch4_purity_A / 100 + biogas_B * ch4_purity_B / 100",
            "rows": rows,
        }

    def data_quality(self, dataset_id: str, limit: int = 500) -> dict[str, Any]:
        record = self._record(dataset_id)
        dq = pd.read_parquet(record["dq_path"]).sort_values(["date", "source_row", "field"])
        flagged = dq.loc[~dq["dq_status"].eq("VALID")]
        return {
            "dataset_id": dataset_id, "summary": self.repository.get_dq_summary(dataset_id),
            "flagged": frame_records(flagged.head(limit)), "flagged_total": len(flagged),
        }
