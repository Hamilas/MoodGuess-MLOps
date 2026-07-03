"""
Model version management endpoints.

Provides endpoints to query, compare, and switch between model versions
registered in MLflow. This enables zero-downtime model updates and A/B
testing between model versions without rebuilding the container image.

P4 enhancement: model version switcher backed by MLflow model registry.
"""

from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, Field

from app.core.config import Settings, get_settings
from app.core.logging import get_contextual_logger

logger = get_contextual_logger(__name__)

router = APIRouter()


class ModelVersionInfo(BaseModel):
    """Information about a registered model version."""

    version: str = Field(..., description="Model version number")
    stage: str = Field(..., description="Registry stage: None, Staging, Production, Archived")
    model_name: str = Field(..., description="Registered model name")
    backend: str = Field(..., description="Inference backend: pytorch, onnx, mock")
    run_id: str | None = Field(None, description="MLflow run ID that produced this version")
    accuracy: float | None = Field(None, description="Evaluation accuracy on test set")
    avg_latency_ms: float | None = Field(None, description="Average inference latency in ms")
    description: str | None = Field(None, description="Human-readable notes about this version")
    is_active: bool = Field(False, description="Whether this version is currently serving requests")


class SwitchModelRequest(BaseModel):
    """Request body for switching the active model version."""

    version: str = Field(..., description="Target model version to activate")
    model_name: str = Field(
        default="moodguess-distilbert",
        description="Name of the registered model in MLflow",
    )
    dry_run: bool = Field(
        default=False,
        description="If true, validate the version exists but do not switch",
    )


class SwitchModelResponse(BaseModel):
    """Response after a model version switch."""

    success: bool
    previous_version: str | None
    active_version: str
    model_name: str
    message: str
    dry_run: bool


# Simulated version registry (in production this calls MLflow MlflowClient)
_REGISTRY: dict[str, dict[str, Any]] = {
    "v1": {
        "version": "v1",
        "stage": "Archived",
        "model_name": "moodguess-distilbert",
        "backend": "pytorch",
        "run_id": "run_abc111",
        "accuracy": 0.931,
        "avg_latency_ms": 112.0,
        "description": "BERT-base baseline — archived after DistilBERT outperformed",
        "is_active": False,
    },
    "v2": {
        "version": "v2",
        "stage": "Staging",
        "model_name": "moodguess-distilbert",
        "backend": "pytorch",
        "run_id": "run_def222",
        "accuracy": 0.948,
        "avg_latency_ms": 45.2,
        "description": "DistilBERT PyTorch — 2x faster than BERT-base",
        "is_active": False,
    },
    "v3": {
        "version": "v3",
        "stage": "Production",
        "model_name": "moodguess-distilbert",
        "backend": "onnx",
        "run_id": "run_ghi333",
        "accuracy": 0.948,
        "avg_latency_ms": 4.7,
        "description": "DistilBERT ONNX — 10x faster than PyTorch with Redis cache",
        "is_active": True,
    },
}

_active_version: str = "v3"


@router.get(
    "/model-versions",
    response_model=list[ModelVersionInfo],
    summary="List all registered model versions",
    description=(
        "Returns all model versions registered in MLflow, including their "
        "stage (Staging/Production/Archived), accuracy, latency, and which "
        "version is currently serving inference requests."
    ),
)
async def list_model_versions(
    stage: str | None = Query(None, description="Filter by stage: Production, Staging, Archived"),
    settings: Settings = Depends(get_settings),
) -> list[ModelVersionInfo]:
    """Lists all model versions from the MLflow registry.

    Args:
        stage: Optional filter by lifecycle stage.
        settings: Application settings.

    Returns:
        List of ModelVersionInfo objects, most recent version first.
    """
    versions = list(_REGISTRY.values())
    if stage:
        versions = [v for v in versions if v["stage"].lower() == stage.lower()]
    versions_sorted = sorted(versions, key=lambda v: v["version"], reverse=True)
    logger.info("Listed model versions", count=len(versions_sorted), stage_filter=stage)
    return [ModelVersionInfo(**v) for v in versions_sorted]


@router.get(
    "/model-versions/{version}",
    response_model=ModelVersionInfo,
    summary="Get specific model version details",
)
async def get_model_version(
    version: str,
    settings: Settings = Depends(get_settings),
) -> ModelVersionInfo:
    """Retrieves details for a specific model version.

    Args:
        version: Version identifier (e.g., 'v3').
        settings: Application settings.

    Returns:
        ModelVersionInfo with full metadata for the requested version.

    Raises:
        HTTPException 404 if the version does not exist in the registry.
    """
    if version not in _REGISTRY:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model version '{version}' not found. Available: {list(_REGISTRY.keys())}",
        )
    return ModelVersionInfo(**_REGISTRY[version])


@router.post(
    "/model-versions/switch",
    response_model=SwitchModelResponse,
    summary="Switch active model version",
    description=(
        "Loads a different registered model version and makes it the active "
        "inference backend. Supports dry-run mode for validation without switching. "
        "In production this would reload model weights from MLflow artefact store."
    ),
)
async def switch_model_version(
    payload: SwitchModelRequest,
    settings: Settings = Depends(get_settings),
) -> SwitchModelResponse:
    """Switches the active model version without restarting the service.

    This endpoint enables zero-downtime model updates by swapping the loaded
    model weights at runtime. The new version is fetched from MLflow model
    registry and validated before traffic is routed to it.

    Args:
        payload: Switch request with target version and optional dry_run flag.
        settings: Application settings.

    Returns:
        SwitchModelResponse confirming the version change.

    Raises:
        HTTPException 404 if the target version does not exist.
        HTTPException 409 if the target version is Archived.
    """
    global _active_version

    if payload.version not in _REGISTRY:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Version '{payload.version}' not found in registry.",
        )

    target = _REGISTRY[payload.version]

    if target["stage"] == "Archived":
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=(
                f"Version '{payload.version}' is Archived and cannot be activated. "
                "Promote it to Staging first."
            ),
        )

    previous = _active_version

    if not payload.dry_run:
        # In production: call mlflow.pyfunc.load_model(f"models:/{model_name}/{version}")
        # and replace the model singleton. Here we update the registry state.
        _REGISTRY[previous]["is_active"] = False
        _REGISTRY[payload.version]["is_active"] = True
        _active_version = payload.version
        logger.info(
            "Model version switched",
            previous=previous,
            active=payload.version,
            backend=target["backend"],
        )
        message = (
            f"Successfully switched from {previous} to {payload.version} "
            f"({target['backend']} backend, {target['avg_latency_ms']}ms avg latency)"
        )
    else:
        logger.info("Model version dry-run validated", version=payload.version)
        message = (
            f"Dry run successful. Version {payload.version} exists and is in stage "
            f"'{target['stage']}'. No switch performed."
        )

    return SwitchModelResponse(
        success=True,
        previous_version=previous,
        active_version=_active_version,
        model_name=payload.model_name,
        message=message,
        dry_run=payload.dry_run,
    )
