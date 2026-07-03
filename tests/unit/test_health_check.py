"""
Tests for the detailed health check endpoint.
"""

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from app.core.config import Settings, get_settings
from app.core.dependencies import get_model_service
from app.core.secrets import get_secret_manager


@pytest.fixture
def client(monkeypatch):
    """Builds a fresh app (debug=True, so routes are unprefixed) per test.

    kafka/async_batch are disabled so `get_detailed_health` only runs the
    three checks (model, system, secrets_backend) these tests mock - leaving
    them enabled would also invoke `check_kafka_health`/`check_async_batch_health`,
    which return an unconfigured (and un-dict-like) MagicMock here.

    Patching `app.main.get_settings` only affects `create_app()`'s own call
    (for the app title/debug flag etc); the route-level `Depends(get_settings)`
    in monitoring_routes.py holds a separate reference to the same function,
    so it needs its own override via `app.dependency_overrides`.
    """
    settings = Settings(debug=True, kafka_enabled=False)
    settings.performance.async_batch_enabled = False
    monkeypatch.setattr("app.main.get_settings", lambda: settings)

    from app.main import create_app

    app = create_app()
    app.dependency_overrides[get_settings] = lambda: settings
    return TestClient(app)


@patch("app.services.monitoring_service.HealthChecker")
@pytest.mark.unit
def test_detailed_health_check_healthy(mock_health_checker, client):
    """
    Tests the detailed health check endpoint when all components are healthy.
    """
    mock_model = MagicMock()
    mock_model.is_ready.return_value = True

    mock_secret_manager = MagicMock()
    mock_secret_manager.is_healthy.return_value = True

    mock_checker_instance = mock_health_checker.return_value
    mock_checker_instance.check_model_health.return_value = {
        "status": "healthy",
        "is_ready": True,
        "details": {},
    }
    mock_checker_instance.check_system_health.return_value = {"status": "healthy"}
    mock_checker_instance.check_secrets_backend_health.return_value = {
        "status": "healthy",
        "details": {},
    }

    client.app.dependency_overrides[get_model_service] = lambda: mock_model
    client.app.dependency_overrides[get_secret_manager] = lambda: mock_secret_manager

    response = client.get("/health/details")

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert len(data["dependencies"]) == 3
    for dep in data["dependencies"]:
        assert dep["details"]["status"] == "healthy"


@patch("app.services.monitoring_service.HealthChecker")
@pytest.mark.unit
def test_detailed_health_check_unhealthy_model(mock_health_checker, client):
    """
    Tests the detailed health check endpoint when the model is unhealthy.
    """
    mock_model = MagicMock()
    mock_model.is_ready.return_value = False

    mock_secret_manager = MagicMock()
    mock_secret_manager.is_healthy.return_value = True

    mock_checker_instance = mock_health_checker.return_value
    mock_checker_instance.check_model_health.return_value = {
        "status": "degraded",
        "is_ready": False,
        "details": {},
        "error": "Model not loaded",
    }
    mock_checker_instance.check_system_health.return_value = {"status": "healthy"}
    mock_checker_instance.check_secrets_backend_health.return_value = {
        "status": "healthy",
        "details": {},
    }

    client.app.dependency_overrides[get_model_service] = lambda: mock_model
    client.app.dependency_overrides[get_secret_manager] = lambda: mock_secret_manager

    response = client.get("/health/details")

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "unhealthy"
    model_health = next((d for d in data["dependencies"] if d["component_name"] == "model"), None)
    assert model_health is not None
    assert model_health["details"]["status"] == "degraded"
    assert model_health["details"]["error"] == "Model not loaded"


@patch("app.services.monitoring_service.HealthChecker")
@pytest.mark.unit
def test_detailed_health_check_unhealthy_secrets_backend(mock_health_checker, client):
    """
    Tests the detailed health check endpoint when the secrets backend is unhealthy.
    """
    mock_model = MagicMock()
    mock_model.is_ready.return_value = True

    mock_secret_manager = MagicMock()
    mock_secret_manager.is_healthy.return_value = False

    mock_checker_instance = mock_health_checker.return_value
    mock_checker_instance.check_model_health.return_value = {
        "status": "healthy",
        "is_ready": True,
        "details": {},
    }
    mock_checker_instance.check_system_health.return_value = {"status": "healthy"}
    mock_checker_instance.check_secrets_backend_health.return_value = {
        "status": "unhealthy",
        "details": {},
        "error": "Vault connection failed",
    }

    client.app.dependency_overrides[get_model_service] = lambda: mock_model
    client.app.dependency_overrides[get_secret_manager] = lambda: mock_secret_manager

    response = client.get("/health/details")

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "unhealthy"
    secrets_health = next(
        (d for d in data["dependencies"] if d["component_name"] == "secrets_backend"),
        None,
    )
    assert secrets_health is not None
    assert secrets_health["details"]["status"] == "unhealthy"
    assert secrets_health["details"]["error"] == "Vault connection failed"
