from __future__ import annotations

import logging

from fastapi import APIRouter, File, HTTPException, Query, Request, UploadFile

from ..services.excel_service import WorkbookValidationError
from ..services.model_service import ForecastError


LOGGER = logging.getLogger("uvicorn.error")


def _workbook_error_code(message: str) -> str:
    if message.startswith("INVALID_WORKBOOK"):
        return "WORKBOOK_PARSE_FAILED"
    if message.startswith("MISSING_SHEET"):
        return "BIOGUARD_INPUT_MISSING"
    return "SCHEMA_INVALID"


def router_for(datasets, models) -> APIRouter:
    router = APIRouter(prefix="/api/v1")

    @router.post("/datasets/upload")
    async def upload_dataset(request: Request, file: UploadFile = File(...)):
        filename = file.filename or "upload.xlsx"
        request_content_type = request.headers.get("content-type", "")
        LOGGER.info(
            "request received method=POST path=/api/v1/datasets/upload filename=%s content_type=%s",
            filename,
            request_content_type,
        )
        if not filename.lower().endswith(".xlsx"):
            LOGGER.warning("upload rejected code=INVALID_FILE_TYPE filename=%s", filename)
            raise HTTPException(
                status_code=415,
                detail={"code": "INVALID_FILE_TYPE", "message": "BioGuard official upload format is .xlsx"},
            )
        try:
            payload = await file.read()
            result = datasets.upload(filename, payload)
            LOGGER.info(
                "%s dataset_id=%s rows=%s status=%s",
                "dataset created" if result.get("processing_mode") == "CREATED" else "dataset reused",
                result["dataset_id"], result["row_count"], result["status"],
            )
            # Complete the promised upload->inference flow in one request.
            result["forecast"] = models.get_or_create_forecast(result["dataset_id"])
            LOGGER.info(
                "forecast complete dataset_id=%s status=%s horizons=%s",
                result["dataset_id"], result["forecast"].get("status"),
                len(result["forecast"].get("methane", {}).get("predictions", [])),
            )
            result = {**datasets.describe(result["dataset_id"]), "forecast": result["forecast"]}
            result["latest"] = datasets.latest(result["dataset_id"])
            return result
        except WorkbookValidationError as exc:
            message = str(exc)
            code = _workbook_error_code(message)
            LOGGER.warning("upload rejected code=%s filename=%s detail=%s", code, filename, message)
            raise HTTPException(status_code=422, detail={"code": code, "message": message})
        except ForecastError as exc:
            LOGGER.warning("forecast failed code=%s filename=%s detail=%s", exc.code, filename, exc.detail)
            raise HTTPException(status_code=422, detail={"code": exc.code, "message": exc.detail})

    @router.get("/datasets/{dataset_id}")
    def get_dataset(dataset_id: str):
        try: return datasets.describe(dataset_id)
        except KeyError: raise HTTPException(status_code=404, detail={"code": "DATASET_NOT_FOUND"})

    @router.get("/datasets/{dataset_id}/latest")
    def get_latest(dataset_id: str):
        try: return datasets.latest(dataset_id)
        except KeyError: raise HTTPException(status_code=404, detail={"code": "DATASET_NOT_FOUND"})

    @router.get("/datasets/{dataset_id}/rows/{measurement_date}")
    def get_row(dataset_id: str, measurement_date: str):
        try: return datasets.at_date(dataset_id, measurement_date)
        except KeyError as exc:
            code = str(exc).strip("'")
            raise HTTPException(status_code=404, detail={"code": code})

    @router.get("/datasets/{dataset_id}/history")
    def get_history(dataset_id: str, limit: int | None = Query(None, ge=1, le=10000)):
        try: return datasets.history(dataset_id, limit)
        except KeyError: raise HTTPException(status_code=404, detail={"code": "DATASET_NOT_FOUND"})

    @router.get("/datasets/{dataset_id}/forecast")
    def get_forecast(dataset_id: str, refresh: bool = False):
        try:
            return models.get_or_create_forecast(dataset_id, refresh=refresh)
        except KeyError: raise HTTPException(status_code=404, detail={"code": "DATASET_NOT_FOUND"})
        except ForecastError as exc: raise HTTPException(status_code=422, detail={"code": exc.code, "message": exc.detail})

    @router.get("/datasets/{dataset_id}/data-quality")
    def get_dq(dataset_id: str, limit: int = Query(500, ge=1, le=5000)):
        try: return datasets.data_quality(dataset_id, limit)
        except KeyError: raise HTTPException(status_code=404, detail={"code": "DATASET_NOT_FOUND"})

    return router
