"""
Interface for Anomaly Buffer

Defines the contract for anomaly storage services.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import Any


@dataclass
class AnomalyEntry:
    """Anomaly detection entry"""

    id: str
    timestamp: datetime
    text: str
    prediction: str
    anomaly_score: float
    anomaly_type: str
    metadata: dict[str, Any]

    def is_expired(self, ttl_seconds: int) -> bool:
        """Check if entry has exceeded TTL

        Args:
            ttl_seconds: Time-to-live in seconds

        Returns:
            True if the entry is older than ttl_seconds
        """
        age_seconds = (datetime.now() - self.timestamp).total_seconds()
        return age_seconds > ttl_seconds

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary

        Returns:
            Dictionary representation of the anomaly entry
        """
        return {
            "id": self.id,
            "timestamp": self.timestamp.isoformat(),
            "text": self.text,
            "prediction": self.prediction,
            "anomaly_score": self.anomaly_score,
            "anomaly_type": self.anomaly_type,
            "metadata": self.metadata,
        }


class IAnomalyBuffer(ABC):
    """
    Interface for anomaly buffer services.

    Provides temporary in-memory storage for anomaly detection results.
    """

    @abstractmethod
    def add(self, **kwargs: Any) -> str:
        """
        Add an anomaly entry to the buffer.

        Args:
            **kwargs: Entry fields (text, prediction, anomaly_score,
                anomaly_type, metadata, etc.) — see the concrete
                implementation's entry type for the exact fields accepted.

        Returns:
            Entry ID

        Raises:
            ValueError: If inputs are invalid
        """
        pass

    @abstractmethod
    def get(self, entry_id: str) -> Any | None:
        """
        Retrieve an anomaly entry by ID.

        Args:
            entry_id: Entry identifier

        Returns:
            The anomaly entry if found and not expired, None otherwise
        """
        pass

    @abstractmethod
    def get_all(self) -> list[Any]:
        """
        Get all non-expired anomaly entries.

        Returns:
            List of active anomaly entries
        """
        pass

    @abstractmethod
    def cleanup_expired(self) -> int:
        """
        Remove expired entries from buffer.

        Returns:
            Number of entries removed
        """
        pass

    @abstractmethod
    def get_stats(self) -> dict[str, Any]:
        """
        Get buffer utilization statistics.

        Returns:
            Dictionary containing:
                - total_entries: Current number of entries
                - max_size: Maximum buffer size
                - utilization: Buffer utilization percentage
                - oldest_entry: Timestamp of oldest entry
        """
        pass
