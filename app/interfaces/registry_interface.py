"""
Interface for Model Registry

Defines the contract for MLflow model registry services.
"""

from abc import ABC, abstractmethod
from enum import Enum
from typing import Any


class ModelStage(str, Enum):
    """MLflow model stages."""

    NONE = "None"
    STAGING = "Staging"
    PRODUCTION = "Production"
    ARCHIVED = "Archived"


class IModelRegistry(ABC):
    """
    Interface for model registry services.

    Provides model versioning, lifecycle management, and deployment tracking.
    """

    @abstractmethod
    def register_model(
        self,
        model_name: str,
        model_uri: str,
        tags: dict[str, str] | None = None,
        description: str | None = None,
    ) -> dict[str, Any] | None:
        """
        Register a new model version.

        Args:
            model_name: Name of the model
            model_uri: URI to model artifacts
            tags: Optional tags for the model version
            description: Optional model description

        Returns:
            Dictionary with model version information, or None if MLflow is unavailable

        Raises:
            ValueError: If model_name or model_uri is invalid
            RuntimeError: If registration fails
        """
        pass

    @abstractmethod
    def get_model_version(
        self,
        model_name: str,
        version: int | None = None,
        stage: "ModelStage | None" = None,
    ) -> dict[str, Any] | None:
        """
        Retrieve model version information.

        Args:
            model_name: Name of the model
            version: Specific version number (None for latest)
            stage: Model stage filter (None, Staging, Production, Archived)

        Returns:
            Dictionary with model version details, or None if not found

        Raises:
            KeyError: If model or version not found
        """
        pass

    @abstractmethod
    def get_production_model(self, model_name: str) -> dict[str, Any] | None:
        """
        Get the current production model version.

        Args:
            model_name: Name of the model

        Returns:
            Production model version information, or None if no production version exists
        """
        pass

    @abstractmethod
    def transition_model_stage(
        self,
        model_name: str,
        version: int,
        stage: "ModelStage",
        archive_existing: bool = True,
    ) -> bool:
        """
        Transition a model version to a new stage.

        Args:
            model_name: Name of the model
            version: Version to transition
            stage: Target stage
            archive_existing: Whether to archive existing versions in target stage

        Returns:
            True if the transition succeeded

        Raises:
            ValueError: If stage is invalid
            KeyError: If model or version not found
        """
        pass

    @abstractmethod
    def promote_to_production(
        self, model_name: str, version: int, archive_existing: bool = True
    ) -> bool:
        """
        Fast-track promotion to production stage.

        Args:
            model_name: Name of the model
            version: Version to promote
            archive_existing: Whether to archive current production version

        Returns:
            True if the promotion succeeded

        Raises:
            KeyError: If model or version not found
        """
        pass

    @abstractmethod
    def log_prediction_metrics(
        self,
        model_name: str,
        version: int,
        metrics: dict[str, float],
        tags: dict[str, str] | None = None,
    ) -> bool:
        """
        Log prediction metrics for a model version.

        Args:
            model_name: Name of the model
            version: Model version
            metrics: Metrics to log
            tags: Optional tags for the metrics

        Returns:
            True if logging succeeded

        Raises:
            KeyError: If model or version not found
        """
        pass

    @abstractmethod
    def delete_model_version(self, model_name: str, version: int) -> bool:
        """
        Delete a model version.

        Args:
            model_name: Name of the model
            version: Version to delete

        Returns:
            True if deletion succeeded

        Raises:
            KeyError: If model or version not found
            ValueError: If trying to delete production version
        """
        pass

    @abstractmethod
    def search_models(
        self, filter_string: str | None = None, max_results: int = 10
    ) -> list[dict[str, Any]]:
        """
        Search for models matching criteria.

        Args:
            filter_string: MLflow filter string
            max_results: Maximum number of results

        Returns:
            List of model information dictionaries
        """
        pass
