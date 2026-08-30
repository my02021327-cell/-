from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from backend.app.config import (
    ALLOWED_ORIGINS,
    DB_PATH,
    PROJECT_ROOT,
    STORAGE_ROOT,
    _database_path,
    _resolve_path,
    _split_origins,
)
from backend.app.main import app


def test_service_health_and_existing_dataset_client_route():
    client = TestClient(app)
    health = client.get("/health")
    assert health.status_code == 200
    assert health.json()["status"] in {"ok", "degraded"}
    assert health.json()["service"] == "bioguard-backend"

    dataset_page = client.get("/datasets/c3d8d5b85256fb91")
    assert dataset_page.status_code == 200
    assert dataset_page.headers["content-type"].startswith("text/html")


def test_cors_configuration_is_bounded_and_honors_allowed_origin():
    assert "*" not in ALLOWED_ORIGINS
    if not ALLOWED_ORIGINS:
        pytest.skip("No CORS origins are configured for this environment")
    origin = ALLOWED_ORIGINS[0]
    response = TestClient(app).options(
        "/api/v1/datasets/upload",
        headers={
            "Origin": origin,
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "content-type",
        },
    )
    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == origin


def test_portable_path_and_database_configuration_helpers(tmp_path: Path):
    assert PROJECT_ROOT.is_absolute()
    assert STORAGE_ROOT.is_absolute()
    assert DB_PATH.is_absolute()
    assert _resolve_path("runtime/data", tmp_path) == PROJECT_ROOT / "runtime/data"
    assert _database_path("sqlite:///runtime/state.sqlite3", tmp_path) == (
        PROJECT_ROOT / "runtime/state.sqlite3"
    )
    assert _split_origins("http://localhost:3000/, https://example.com") == (
        "http://localhost:3000",
        "https://example.com",
    )
    with pytest.raises(ValueError, match="sqlite"):
        _database_path("postgresql://db/bioguard", tmp_path)
