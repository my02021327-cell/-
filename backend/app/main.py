from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from .api.datasets import router_for
from .config import (
    ALLOWED_ORIGINS,
    BUNDLE_PATH,
    ENVIRONMENT,
    FRONTEND_PATH,
    RELIABILITY_SEED_PATH,
    STATIC_DIR,
)
from .repository import Repository
from .services.dataset_service import DatasetService
from .services.model_service import ForecastError, ModelService


repository = Repository()
datasets = DatasetService(repository)
model_error = None
try:
    models = ModelService(repository, datasets)
except Exception as exc:  # explicit degraded startup state
    models = None
    model_error = f"{type(exc).__name__}: {exc}"

app = FastAPI(title="BioGuard V2", version="2.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=list(ALLOWED_ORIGINS),
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)
class RevalidatingStaticFiles(StaticFiles):
    """Force revalidation of the console bundle.

    StaticFiles sends ETag/Last-Modified but no Cache-Control, so a browser is
    free to reuse bioguard_v2.js from heuristic cache without asking. That left
    the console running an old adapter against freshly served HTML. The ETag
    still makes the revalidation a cheap 304.
    """

    def file_response(self, *args, **kwargs):
        response = super().file_response(*args, **kwargs)
        response.headers["Cache-Control"] = "no-cache, must-revalidate"
        return response


app.mount("/static", RevalidatingStaticFiles(directory=STATIC_DIR), name="static")
if models is not None:
    app.include_router(router_for(datasets, models))


@app.get("/api/v1/health")
def health():
    status = "CONNECTED" if models is not None else "MODEL_NOT_LOADED"
    return {
        "status": status, "mode": "BACKEND LIVE" if models is not None else "BACKEND OFFLINE",
        "model_bundle": str(BUNDLE_PATH), "reliability_seed": str(RELIABILITY_SEED_PATH),
        "model_error": model_error,
    }


@app.get("/health", include_in_schema=False)
def service_health():
    return {
        "status": "ok" if models is not None else "degraded",
        "service": "bioguard-backend",
        "environment": ENVIRONMENT,
    }


# The console HTML and its adapter must revalidate on every load. FileResponse
# sends ETag/Last-Modified but no Cache-Control, which lets a browser reuse an
# old document from heuristic cache -- and a stale document paired with a fresh
# adapter (or the reverse) silently renders the wrong console.
NO_CACHE = {"Cache-Control": "no-cache, must-revalidate"}


def _frontend_response():
    if not FRONTEND_PATH.is_file():
        return JSONResponse({"status": "FRONTEND_NOT_FOUND"}, status_code=404)
    return FileResponse(FRONTEND_PATH, media_type="text/html", headers=NO_CACHE)


@app.get("/", include_in_schema=False)
def frontend():
    return _frontend_response()


@app.get("/datasets/{dataset_id}", include_in_schema=False)
def frontend_dataset(dataset_id: str):
    """Client route: the adapter restores this immutable backend dataset by id."""
    return _frontend_response()
