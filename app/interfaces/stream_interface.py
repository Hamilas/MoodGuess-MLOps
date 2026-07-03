"""
Interface for Stream Processor

Defines the contract for stream processing services with dynamic batching.
"""

from abc import ABC, abstractmethod
from typing import Any


class IStreamProcessor(ABC):
    """
    Interface for stream processing services.

    Provides dynamic batching and stream processing for optimized inference.
    """

    @abstractmethod
    async def predict_async(self, text: str, request_id: str | None = None) -> dict[str, Any]:
        """
        Queue a single prediction for asynchronous processing.

        The processor will batch this with other concurrent requests
        for optimal throughput.

        Args:
            text: Input text to analyze
            request_id: Optional request identifier for tracking

        Returns:
            Prediction result dictionary

        Raises:
            ValueError: If text is invalid
            RuntimeError: If processing fails
        """
        pass

    @abstractmethod
    async def predict_async_batch(self, texts: list[str], batch_id: str) -> list[dict[str, Any]]:
        """
        Process a pre-formed batch asynchronously.

        Args:
            texts: List of texts to process
            batch_id: Batch identifier, used for logging/tracking

        Returns:
            List of prediction results

        Raises:
            ValueError: If texts are invalid
            RuntimeError: If batch processing fails
        """
        pass

    @abstractmethod
    async def shutdown(self) -> None:
        """
        Gracefully shutdown the stream processor.

        Waits for in-flight requests to complete.
        """
        pass

    @abstractmethod
    def get_stats(self) -> dict[str, Any]:
        """
        Get processing statistics.

        Returns:
            Dictionary with metrics like:
                - total_requests: Total requests processed
                - average_batch_size: Average batch size
                - throughput: Requests per second
                - average_latency: Average processing latency
        """
        pass

    @abstractmethod
    def reset_stats(self) -> None:
        """Reset processing statistics."""
        pass
