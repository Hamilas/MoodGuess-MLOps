"""
Mock sentiment analysis model for local development and testing.

This module provides a lightweight mock implementation of the sentiment
analysis model that returns realistic-looking results without loading
any actual ML models. Useful for local development, CI, and demo purposes.
"""

import random
import time
from typing import Any, ClassVar, Optional

from app.core.config import Settings, get_settings
from app.core.logging import get_logger
from app.models.base import BaseModelMetrics
from app.utils.exceptions import TextEmptyError

logger = get_logger(__name__)

# Singleton instance
_mock_analyzer_instance: Optional["MockSentimentAnalyzer"] = None


class MockSentimentAnalyzer(BaseModelMetrics):
    """Mock sentiment analyzer for local development.

    Returns pre-computed realistic sentiment predictions without loading
    actual ML models. Simulates realistic inference times.
    """

    MODEL_NAME = "mock-sentiment-model-v1"
    BACKEND = "mock"

    # Pre-computed example outputs (realistic values)
    _POSITIVE_PHRASES: ClassVar[list[str]] = [
        "excellent",
        "amazing",
        "great",
        "fantastic",
        "wonderful",
        "outstanding",
        "superb",
        "brilliant",
        "love",
        "best",
        "perfect",
        "awesome",
        "enjoy",
        "happy",
        "pleased",
        "on time",
        "solid",
        "smooth",
        "efficient",
        "responsive",
        "resolved quickly",
        "works well",
        "worked well",
        "amazingly",
    ]
    _NEGATIVE_PHRASES: ClassVar[list[str]] = [
        "terrible",
        "awful",
        "horrible",
        "worst",
        "hate",
        "disgusting",
        "disappointed",
        "bad",
        "poor",
        "failure",
        "broken",
        "useless",
        "garbage",
        "annoying",
        "frustrating",
        "never responded",
        "no response",
        "unresponsive",
        "ignored",
        "delayed",
        "late",
        "unhelpful",
        "refund",
        "cancelled",
        "complain",
    ]

    def __init__(self, settings: Settings | None = None):
        """Initialize the mock analyzer."""
        super().__init__()
        self.settings = settings or get_settings()
        self._is_loaded = True
        # Mock never caches, but keep this set (not None) so shared metrics
        # code in BaseModelMetrics can safely introspect it.
        self._cached_predict = self._classify_text
        logger.info(
            "Mock sentiment analyzer initialized",
            model=self.MODEL_NAME,
            backend=self.BACKEND,
        )

    def is_ready(self) -> bool:
        """Mock model is always ready."""
        return True

    def _classify_text(self, text: str) -> tuple[str, float]:
        """Classify text sentiment based on keyword heuristics.

        Args:
            text: Input text to classify.

        Returns:
            Tuple of (label, score).
        """
        text_lower = text.lower()

        positive_count = sum(1 for phrase in self._POSITIVE_PHRASES if phrase in text_lower)
        negative_count = sum(1 for phrase in self._NEGATIVE_PHRASES if phrase in text_lower)

        if positive_count > negative_count:
            label = "POSITIVE"
            base_score = 0.75 + (positive_count * 0.05)
        elif negative_count > positive_count:
            label = "NEGATIVE"
            base_score = 0.75 + (negative_count * 0.05)
        else:
            # Random balanced prediction for neutral/unknown text
            if hash(text) % 3 == 0:
                label = "NEGATIVE"
                base_score = 0.55 + (hash(text) % 30) / 100
            else:
                label = "POSITIVE"
                base_score = 0.60 + (hash(text) % 35) / 100

        # Clamp score to valid range
        score = min(0.9998, max(0.5001, base_score))
        return label, round(score, 4)

    def predict(self, text: str) -> dict[str, Any]:
        """Perform mock sentiment analysis.

        Args:
            text: The input text to analyze.

        Returns:
            Prediction result dictionary.

        Raises:
            TextEmptyError: If the input text is empty.
        """
        if not text or not text.strip():
            raise TextEmptyError("Input text cannot be empty")

        # Simulate realistic inference time (5-50ms)
        start_time = time.time()
        sleep_seconds = random.uniform(0.005, 0.05)  # nosec B311 - simulated latency
        time.sleep(sleep_seconds)

        label, score = self._classify_text(text)
        inference_time_ms = (time.time() - start_time) * 1000

        result = {
            "label": label,
            "score": score,
            "inference_time_ms": round(inference_time_ms, 2),
            "model_name": self.MODEL_NAME,
            "text_length": len(text),
            "backend": self.BACKEND,
            "cached": False,
        }

        logger.debug(
            "Mock prediction completed",
            label=label,
            score=score,
            text_length=len(text),
        )

        return result

    def predict_batch(self, texts: list[str]) -> list[dict[str, Any]]:
        """Perform mock batch sentiment analysis.

        Args:
            texts: List of input texts.

        Returns:
            List of prediction result dictionaries.
        """
        return [self.predict(text) for text in texts]

    def get_model_info(self) -> dict[str, Any]:
        """Get mock model information.

        Returns:
            Dictionary with model metadata.
        """
        return {
            "model_name": self.MODEL_NAME,
            "backend": self.BACKEND,
            "version": "1.0.0",
            "description": "Mock sentiment model for local development",
            "supported_languages": ["en"],
            "labels": ["POSITIVE", "NEGATIVE"],
            "is_ready": True,
        }

    def get_metrics(self) -> dict[str, Any]:
        """Get mock model performance metrics.

        Returns:
            Dictionary with performance metrics.
        """
        return {
            "total_predictions": 0,
            "avg_inference_time_ms": 25.0,
            "p95_inference_time_ms": 45.0,
            "cache_hit_rate": 0.0,
            "backend": self.BACKEND,
        }


def get_mock_sentiment_analyzer(
    settings: Settings | None = None,
) -> MockSentimentAnalyzer:
    """Get or create the singleton mock sentiment analyzer.

    Args:
        settings: Optional settings override.

    Returns:
        The singleton MockSentimentAnalyzer instance.
    """
    global _mock_analyzer_instance
    if _mock_analyzer_instance is None:
        _mock_analyzer_instance = MockSentimentAnalyzer(settings=settings)
    return _mock_analyzer_instance


def reset_mock_sentiment_analyzer() -> None:
    """Clear the singleton `MockSentimentAnalyzer` instance.

    Forces the next `get_mock_sentiment_analyzer()` call to construct a fresh
    instance. Used by `ModelFactory.reset_models()` for test isolation.
    """
    global _mock_analyzer_instance
    _mock_analyzer_instance = None
