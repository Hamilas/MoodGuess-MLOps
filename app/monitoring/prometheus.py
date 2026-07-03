"""
Prometheus monitoring and metrics collection for the MLOps sentiment analysis service.

This module provides a comprehensive set of Prometheus-compatible metrics for
monitoring the performance, health, and behavior of the sentiment analysis
service. It uses the `prometheus_client` library to define and manage
various metric types, including counters, gauges, and histograms.
"""

import time

from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, Info, generate_latest

try:
    import torch

    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False

from app.core.config import get_settings

# --- Prometheus Metric Definitions ---

# General request metrics
REQUEST_COUNT = Counter(
    "sentiment_requests_total",
    "Total number of sentiment analysis requests",
    ["endpoint", "method", "status_code"],
)
REQUEST_DURATION = Histogram(
    "sentiment_request_duration_seconds",
    "Request duration in seconds",
    ["endpoint", "method"],
)
ACTIVE_REQUESTS = Gauge("sentiment_active_requests", "Number of active requests")

# Model-specific metrics
MODEL_LOADED = Gauge("sentiment_model_loaded", "Indicates if the model is loaded (1) or not (0)")
INFERENCE_DURATION = Histogram("sentiment_inference_duration_seconds", "Model inference duration")
PREDICTION_SCORE = Histogram("sentiment_prediction_score", "Distribution of prediction scores")
TEXT_LENGTH = Histogram("sentiment_text_length_characters", "Distribution of input text lengths")

# System and environment metrics
TORCH_VERSION = Info("sentiment_torch_version", "PyTorch version information")
CUDA_AVAILABLE = Gauge("sentiment_cuda_available", "Indicates if CUDA is available (1) or not (0)")
CUDA_DEVICE_COUNT = Gauge("sentiment_cuda_device_count", "Number of available CUDA devices")
CUDA_MEMORY_ALLOCATED = Gauge("sentiment_cuda_memory_allocated_bytes", "CUDA memory allocated")
CUDA_MEMORY_RESERVED = Gauge("sentiment_cuda_memory_reserved_bytes", "CUDA memory reserved")

# Shadow Mode metrics
SHADOW_PREDICTIONS_TOTAL = Counter(
    "sentiment_shadow_predictions_total",
    "Total number of shadow (dark launch) predictions",
)
SHADOW_PREDICTION_DURATION = Histogram(
    "sentiment_shadow_prediction_duration_seconds",
    "Shadow model inference duration",
)
SHADOW_PREDICTION_MATCHES = Counter(
    "sentiment_shadow_prediction_matches_total",
    "Number of shadow predictions matching primary model",
)
SHADOW_PREDICTION_MISMATCHES = Counter(
    "sentiment_shadow_prediction_mismatches_total",
    "Number of shadow predictions differing from primary model",
)

# Redis cache metrics
REDIS_CACHE_HITS = Counter(
    "sentiment_redis_cache_hits_total",
    "Total number of Redis cache hits",
    ["cache_type"],
)
REDIS_CACHE_MISSES = Counter(
    "sentiment_redis_cache_misses_total",
    "Total number of Redis cache misses",
    ["cache_type"],
)
REDIS_CACHE_ERRORS = Counter(
    "sentiment_redis_cache_errors_total",
    "Total number of Redis cache errors",
    ["operation", "error_type"],
)
REDIS_OPERATION_DURATION = Histogram(
    "sentiment_redis_operation_duration_seconds",
    "Duration of Redis cache operations",
    ["operation"],
)

# Anomaly buffer metrics
ANOMALY_DETECTED_TOTAL = Counter(
    "sentiment_anomaly_detected_total",
    "Total number of anomalies detected",
    ["anomaly_type"],
)
ANOMALY_BUFFER_EVICTIONS_TOTAL = Counter(
    "sentiment_anomaly_buffer_evictions_total",
    "Total number of anomaly buffer evictions",
    ["reason"],
)
ANOMALY_BUFFER_SIZE = Gauge(
    "sentiment_anomaly_buffer_size",
    "Current number of entries in the anomaly buffer",
)


class PrometheusMetrics:
    """Manages the collection and exposure of Prometheus metrics.

    This class centralizes all Prometheus metrics for the application, providing
    a single point of control for initializing, updating, and serving them. It
    includes a caching mechanism to reduce the overhead of generating the
    metrics payload on every scrape from the Prometheus server.
    """

    def __init__(self):
        """Initializes the metrics collectors and sets static metric values."""
        self.settings = get_settings()
        self._initialize_static_metrics()
        self._metrics_cache: bytes | None = None
        self._metrics_cache_ts: float | None = None

    def _initialize_static_metrics(self):
        """Initializes static metrics that do not change during runtime."""
        if _TORCH_AVAILABLE:
            TORCH_VERSION.info({"version": torch.__version__})
            CUDA_AVAILABLE.set(1 if torch.cuda.is_available() else 0)
            CUDA_DEVICE_COUNT.set(torch.cuda.device_count())
        else:
            TORCH_VERSION.info({"version": "not-installed"})
            CUDA_AVAILABLE.set(0)
            CUDA_DEVICE_COUNT.set(0)

    def update_system_metrics(self):
        """Updates dynamic system metrics, such as CUDA memory usage."""
        if _TORCH_AVAILABLE and torch.cuda.is_available():
            CUDA_MEMORY_ALLOCATED.set(torch.cuda.memory_allocated())
            CUDA_MEMORY_RESERVED.set(torch.cuda.memory_reserved())

    def get_metrics(self) -> bytes:
        """Generates and returns the metrics in Prometheus text format.

        This method includes a caching mechanism to avoid regenerating the
        metrics payload on every call, which is important for performance
        under high scrape frequencies.

        Returns:
            A byte string containing the metrics in Prometheus format.
        """
        now = time.time()
        ttl = self.settings.monitoring.metrics_cache_ttl
        if self._metrics_cache and self._metrics_cache_ts and (now - self._metrics_cache_ts) < ttl:
            return self._metrics_cache

        self.update_system_metrics()
        payload: bytes = generate_latest()
        self._metrics_cache = payload
        self._metrics_cache_ts = now
        return payload

    @staticmethod
    def get_metrics_content_type() -> str:
        """Returns the appropriate content type for Prometheus metrics.

        Returns:
            The content type string for Prometheus metrics (`text/plain`).
        """
        content_type: str = CONTENT_TYPE_LATEST
        return content_type


_metrics_instance: PrometheusMetrics | None = None


def get_metrics() -> PrometheusMetrics:
    """Retrieves a singleton instance of the `PrometheusMetrics` class.

    This factory function ensures that only one instance of `PrometheusMetrics`
    is created and used throughout the application, providing a single,
    consistent source for all metrics.

    Returns:
        The singleton instance of `PrometheusMetrics`.
    """
    global _metrics_instance
    if _metrics_instance is None:
        _metrics_instance = PrometheusMetrics()
    return _metrics_instance


# Helper functions for updating metrics from other modules
def record_request(endpoint: str, method: str, status_code: int):
    """Increments the counter for completed requests."""
    REQUEST_COUNT.labels(endpoint=endpoint, method=method, status_code=str(status_code)).inc()


def record_request_duration(endpoint: str, method: str, duration: float):
    """Records the duration of a request."""
    REQUEST_DURATION.labels(endpoint=endpoint, method=method).observe(duration)


def record_inference_duration(duration: float):
    """Records the duration of a model inference operation."""
    INFERENCE_DURATION.observe(duration)


def record_prediction_metrics(score: float, text_length: int):
    """Records metrics specific to a prediction."""
    PREDICTION_SCORE.observe(score)
    TEXT_LENGTH.observe(text_length)


def set_model_status(is_loaded: bool):
    """Sets the gauge for the model's loading status."""
    MODEL_LOADED.set(1 if is_loaded else 0)


def increment_active_requests():
    """Increments the gauge for active requests."""
    ACTIVE_REQUESTS.inc()


def decrement_active_requests():
    """Decrements the gauge for active requests."""
    ACTIVE_REQUESTS.dec()


def record_redis_cache_hit(cache_type: str):
    """Increments the counter for a Redis cache hit."""
    REDIS_CACHE_HITS.labels(cache_type=cache_type).inc()


def record_redis_cache_miss(cache_type: str):
    """Increments the counter for a Redis cache miss."""
    REDIS_CACHE_MISSES.labels(cache_type=cache_type).inc()


def record_redis_cache_error(operation: str, error_type: str):
    """Increments the counter for a Redis cache error."""
    REDIS_CACHE_ERRORS.labels(operation=operation, error_type=error_type).inc()


def record_redis_operation_duration(operation: str, duration: float):
    """Records the duration of a Redis cache operation."""
    REDIS_OPERATION_DURATION.labels(operation=operation).observe(duration)


def record_anomaly_detected(anomaly_type: str):
    """Increments the counter for a detected anomaly."""
    ANOMALY_DETECTED_TOTAL.labels(anomaly_type=anomaly_type).inc()


def record_anomaly_buffer_eviction(reason: str):
    """Increments the counter for an anomaly buffer eviction."""
    ANOMALY_BUFFER_EVICTIONS_TOTAL.labels(reason=reason).inc()


def record_anomaly_buffer_size(size: int):
    """Records the current size of the anomaly buffer."""
    ANOMALY_BUFFER_SIZE.set(size)
