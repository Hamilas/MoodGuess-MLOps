"""
Interface for Drift Detector

Defines the contract for model drift detection services.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass
class DriftMetrics:
    """Drift detection metrics"""

    psi_score: float
    ks_statistic: float | None
    chi_squared: float | None
    drift_detected: bool
    confidence_drift: bool
    prediction_drift: bool


class IDriftDetector(ABC):
    """
    Interface for model drift detection services.

    Provides statistical tests to detect data and concept drift.
    """

    @abstractmethod
    def add_prediction(
        self,
        text: str,
        confidence: float,
        prediction: str,
        is_reference: bool = False,
    ) -> None:
        """
        Record a prediction for drift analysis.

        Args:
            text: Input text
            confidence: Prediction confidence score
            prediction: Predicted label
            is_reference: Whether this is reference/baseline data

        Raises:
            ValueError: If inputs are invalid
        """
        pass

    @abstractmethod
    def check_drift(self) -> Any | None:
        """
        Perform drift detection tests.

        Compares current window against baseline using statistical tests.

        Returns:
            Drift metrics containing test results and drift status, or None
            if there isn't enough data yet (see the concrete implementation's
            own metrics type for the exact fields available).

        Raises:
            RuntimeError: If insufficient data for testing
        """
        pass

    @abstractmethod
    def reset_current_window(self) -> None:
        """Reset the current monitoring window."""
        pass

    @abstractmethod
    def update_baseline(self) -> None:
        """Update baseline statistics from current window data."""
        pass

    @abstractmethod
    def get_drift_summary(self) -> dict[str, Any]:
        """
        Get drift detection summary statistics.

        Returns:
            Dictionary with summary metrics and status
        """
        pass

    @abstractmethod
    def export_drift_report(self) -> str | None:
        """
        Generate a drift analysis report.

        Returns:
            The rendered HTML report, or None if generation failed

        Raises:
            RuntimeError: If report generation fails
        """
        pass
