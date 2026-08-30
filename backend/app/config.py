from __future__ import annotations

import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _resolve_path(value: str | None, default: Path) -> Path:
    if not value:
        return default.resolve()
    candidate = Path(value).expanduser()
    return (candidate if candidate.is_absolute() else PROJECT_ROOT / candidate).resolve()


def _database_path(database_url: str | None, storage_root: Path) -> Path:
    if not database_url:
        return storage_root / "bioguard.sqlite3"
    if not database_url.startswith("sqlite:///"):
        raise ValueError("DATABASE_URL currently supports sqlite:/// URLs only")
    value = database_url.removeprefix("sqlite:///")
    return _resolve_path(value, storage_root / "bioguard.sqlite3")


def _split_origins(value: str) -> tuple[str, ...]:
    return tuple(origin.strip().rstrip("/") for origin in value.split(",") if origin.strip())


ENVIRONMENT = os.getenv("ENV", "development").strip().lower()
HOST = os.getenv("HOST", "127.0.0.1").strip()
PORT = int(os.getenv("PORT", "8000"))
FORWARDED_ALLOW_IPS = os.getenv("FORWARDED_ALLOW_IPS", "127.0.0.1").strip()

_default_origins = (
    "null,http://127.0.0.1:8000,http://localhost:8000,http://127.0.0.1:8001"
    if ENVIRONMENT != "production"
    else ""
)
ALLOWED_ORIGINS = _split_origins(os.getenv("ALLOWED_ORIGINS", _default_origins))
if ENVIRONMENT == "production" and "*" in ALLOWED_ORIGINS:
    raise ValueError("ALLOWED_ORIGINS must not contain '*' in production")

STORAGE_ROOT = _resolve_path(os.getenv("DATA_DIR"), PROJECT_ROOT / "backend/storage")
UPLOAD_DIR = STORAGE_ROOT / "uploads"
PROCESSED_DIR = STORAGE_ROOT / "processed"
PREDICTION_DIR = STORAGE_ROOT / "predictions"
DB_PATH = _database_path(os.getenv("DATABASE_URL"), STORAGE_ROOT)
BUNDLE_PATH = _resolve_path(
    os.getenv("MODEL_BUNDLE_PATH"),
    PROJECT_ROOT / "artifacts/deployment/bioguard_a30_v2_h1_h7.joblib",
)
RELIABILITY_SEED_PATH = _resolve_path(
    os.getenv("RELIABILITY_SEED_PATH"),
    PROJECT_ROOT / "artifacts/deployment/a30_reliability_seed.parquet",
)
FRONTEND_PATH = PROJECT_ROOT / "deliverables/yeongcheon_bioguard_multiview_final.html"
STATIC_DIR = PROJECT_ROOT / "backend/static"

for directory in (STORAGE_ROOT, UPLOAD_DIR, PROCESSED_DIR, PREDICTION_DIR, STATIC_DIR):
    directory.mkdir(parents=True, exist_ok=True)
DB_PATH.parent.mkdir(parents=True, exist_ok=True)
