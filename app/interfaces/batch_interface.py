"""
Interface for Async Batch Service

Defines the contract for asynchronous batch processing services.
"""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from app.api.schemas.responses import AsyncBatchMetricsResponse, BatchPredictionResults
    from app.models.batch_job import BatchJob, Priority


class IAsyncBatchService(ABC):
    """
    Interface for asynchronous batch processing services.

    Provides high-throughput batch processing with priority queues,
    job tracking, and result pagination.
    """

    @abstractmethod
    async def start(self) -> None:
        """
        Start the batch processing service and worker tasks.

        Raises:
            RuntimeError: If service is already running or fails to start
        """
        pass

    @abstractmethod
    async def stop(self) -> None:
        """
        Gracefully stop the batch processing service.

        Waits for in-flight jobs to complete before shutting down.
        """
        pass

    @abstractmethod
    async def submit_batch_job(
        self,
        texts: list[str],
        priority: "str | Priority | None" = "medium",
        max_batch_size: int | None = None,
        timeout_seconds: int | None = None,
    ) -> "BatchJob":
        """
        Submit a batch prediction job.

        Args:
            texts: List of texts to process
            priority: Job priority level
            max_batch_size: Maximum batch size for processing
            timeout_seconds: Job timeout in seconds

        Returns:
            The submitted BatchJob

        Raises:
            ValueError: If inputs are invalid
            RuntimeError: If queue is full or service not running
        """
        pass

    @abstractmethod
    async def get_job_status(self, job_id: str) -> "BatchJob | None":
        """
        Get the status of a batch job.

        Args:
            job_id: Job identifier

        Returns:
            The BatchJob if found, None otherwise
        """
        pass

    @abstractmethod
    async def get_job_results(
        self, job_id: str, page: int = 1, page_size: int = 100
    ) -> "BatchPredictionResults | None":
        """
        Get paginated results for a completed batch job.

        Args:
            job_id: Job identifier
            page: Page number (1-indexed)
            page_size: Results per page

        Returns:
            The paginated results if the job exists, None otherwise

        Raises:
            ValueError: If job is not completed
        """
        pass

    @abstractmethod
    async def cancel_job(self, job_id: str) -> bool:
        """
        Cancel a pending batch job.

        Args:
            job_id: Job identifier

        Returns:
            True if job was cancelled, False if already processing/completed

        Raises:
            KeyError: If job_id not found
        """
        pass

    @abstractmethod
    async def get_batch_metrics(self) -> "AsyncBatchMetricsResponse":
        """
        Get batch processing performance metrics.

        Returns:
            Metrics object containing throughput, queue size, processing
            times, etc.
        """
        pass

    @abstractmethod
    def get_job_queue_status(self) -> dict[str, Any]:
        """
        Get current queue status across all priority levels.

        Returns:
            Dictionary with queue sizes by priority
        """
        pass
