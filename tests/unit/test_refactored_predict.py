"""Tests for the `SentimentAnalyzer.predict()` workflow and its helpers.

This module contains test cases for input validation, text preprocessing,
inference error handling, metrics tracking, and the overall `predict()`
orchestration. `predict()` itself now inlines caching directly via
`_cached_predict`/`_get_cache_info` (a `functools.lru_cache`-backed cache -
see `test_lru_cache.py` for cache-specific coverage), rather than delegating
to separate `_try_get_cached_result`/`_get_cache_key`/`_run_model_inference`
helper methods, which no longer exist.
"""

from unittest.mock import Mock

import pytest

from app.core.config import Settings
from app.models.pytorch_sentiment import SentimentAnalyzer
from app.utils.exceptions import ModelInferenceError, ModelNotLoadedError, TextEmptyError


@pytest.fixture
def mock_settings(monkeypatch):
    """Provides a mocked `Settings` object for testing.

    Args:
        monkeypatch: The pytest `monkeypatch` fixture.

    Returns:
        A mocked `Settings` object.
    """
    settings = Settings(
        model_name="distilbert-base-uncased-finetuned-sst-2-english",
        prediction_cache_max_size=100,
        max_text_length=512,
    )
    monkeypatch.setattr("app.models.pytorch_sentiment.get_settings", lambda: settings)
    return settings


@pytest.fixture
def analyzer_with_mock_pipeline(mock_settings, monkeypatch):
    """Creates a `SentimentAnalyzer` instance with a mocked inference pipeline.

    This fixture allows for testing the analyzer's logic without loading the
    actual machine learning model.

    Args:
        mock_settings: The mocked settings fixture.
        monkeypatch: The pytest `monkeypatch` fixture.

    Returns:
        A `SentimentAnalyzer` instance with a mocked pipeline.
    """
    analyzer = SentimentAnalyzer()

    # Mock pipeline
    def mock_pipeline(text):
        return [{"label": "POSITIVE", "score": 0.95}]

    analyzer._pipeline = mock_pipeline
    analyzer._is_loaded = True

    # Mock get_contextual_logger to avoid dependency
    mock_logger = Mock()
    monkeypatch.setattr(
        "app.models.pytorch_sentiment.get_contextual_logger", lambda *args, **kwargs: mock_logger
    )

    return analyzer


@pytest.mark.unit
class TestValidateInputText:
    """A test suite for the `_validate_input_text` method."""

    def test_validate_with_ready_model(self, analyzer_with_mock_pipeline):
        """Tests that validation passes when the model is ready and the text is valid."""
        analyzer = analyzer_with_mock_pipeline
        mock_logger = Mock()

        # Should not raise
        analyzer._validate_input_text("valid text", mock_logger)

    def test_validate_raises_when_model_not_ready(self, mock_settings):
        """Tests that `ModelNotLoadedError` is raised if the model is not ready."""
        analyzer = SentimentAnalyzer()
        analyzer._is_loaded = False
        mock_logger = Mock()

        with pytest.raises(ModelNotLoadedError):
            analyzer._validate_input_text("text", mock_logger)

    def test_validate_raises_on_empty_text(self, analyzer_with_mock_pipeline):
        """Tests that `TextEmptyError` is raised for empty or whitespace-only text."""
        analyzer = analyzer_with_mock_pipeline
        mock_logger = Mock()

        with pytest.raises(TextEmptyError):
            analyzer._validate_input_text("", mock_logger)

        with pytest.raises(TextEmptyError):
            analyzer._validate_input_text("   ", mock_logger)


@pytest.mark.unit
class TestPreprocessText:
    """A test suite for the (inherited, `BaseModelMetrics`) `_preprocess_text` method.

    `_preprocess_text(text, max_length) -> str` just strips and truncates; it
    takes no logger and returns a single string, not a 3-tuple. Truncation
    logging is the caller's (`predict()`'s) responsibility, not this method's.
    """

    def test_no_truncation_for_short_text(self, analyzer_with_mock_pipeline, mock_settings):
        """Tests that short text is not truncated."""
        analyzer = analyzer_with_mock_pipeline

        text = "Short text"
        processed = analyzer._preprocess_text(text, mock_settings.model.max_text_length)

        assert processed == text

    def test_truncation_for_long_text(self, analyzer_with_mock_pipeline, mock_settings):
        """Tests that long text is correctly truncated to the maximum length."""
        analyzer = analyzer_with_mock_pipeline

        # Create text longer than max_text_length (512)
        long_text = "a" * 600

        processed = analyzer._preprocess_text(long_text, mock_settings.model.max_text_length)

        assert len(processed) == 512

    def test_predict_logs_a_warning_on_truncation(self, analyzer_with_mock_pipeline, monkeypatch):
        """Tests that `predict()` logs a warning when it truncates the input text."""
        analyzer = analyzer_with_mock_pipeline
        mock_logger = Mock()
        monkeypatch.setattr(
            "app.models.pytorch_sentiment.get_contextual_logger",
            lambda *args, **kwargs: mock_logger,
        )

        analyzer.predict("a" * 600)

        assert mock_logger.warning.called


@pytest.mark.unit
class TestModelInference:
    """A test suite for inference success/failure via `predict()`.

    There's no standalone `_run_model_inference` helper anymore - inference
    happens inline in `predict()` via `_cached_predict` -> `_predict_internal`.
    """

    def test_successful_inference(self, analyzer_with_mock_pipeline):
        """Tests a successful prediction end-to-end."""
        analyzer = analyzer_with_mock_pipeline

        result = analyzer.predict("test text")

        assert result["label"] == "POSITIVE"
        assert result["score"] == 0.95
        assert "inference_time_ms" in result
        assert result["inference_time_ms"] >= 0

    def test_inference_error_raises_model_inference_error(self, analyzer_with_mock_pipeline):
        """Tests that a runtime error during inference is wrapped in `ModelInferenceError`."""
        analyzer = analyzer_with_mock_pipeline

        def failing_pipeline(text):
            raise RuntimeError("Pipeline error")

        analyzer._pipeline = failing_pipeline

        with pytest.raises(ModelInferenceError) as exc_info:
            analyzer.predict("text")

        assert "Pipeline error" in str(exc_info.value)


@pytest.mark.unit
class TestMetricsTracking:
    """A test suite for prediction metrics tracked via `_update_metrics`.

    There's no external `MONITORING_AVAILABLE`/Prometheus hook inside the
    model itself anymore (that instrumentation now lives at the middleware
    layer) - the model just tracks prediction count and inference time
    internally, exposed through `get_performance_metrics()`.
    """

    def test_metrics_accumulate_across_predictions(self, analyzer_with_mock_pipeline):
        """Tests that total_predictions increments with each call to predict()."""
        analyzer = analyzer_with_mock_pipeline

        analyzer.predict("text 1")
        analyzer.predict("text 2")

        metrics = analyzer.get_performance_metrics()
        assert metrics["total_predictions"] == 2
        assert metrics["avg_inference_time_ms"] >= 0


@pytest.mark.unit
class TestPredictOrchestration:
    """A test suite for the main `predict()` orchestration method.

    These tests verify that the `predict` method correctly handles the
    overall workflow, including caching and text processing.
    """

    def test_predict_full_workflow(self, analyzer_with_mock_pipeline):
        """Tests the complete prediction workflow through the main `predict` method."""
        analyzer = analyzer_with_mock_pipeline

        result = analyzer.predict("This is a test")

        assert result["label"] == "POSITIVE"
        assert result["score"] == 0.95
        assert result["inference_time_ms"] >= 0

    def test_predict_uses_cache_on_second_call(self, analyzer_with_mock_pipeline):
        """Tests that a second call with the same text hits the cache."""
        analyzer = analyzer_with_mock_pipeline

        analyzer.predict("Same text")
        analyzer.predict("Same text")

        metrics = analyzer.get_performance_metrics()
        assert metrics["cache_hits"] == 1
        assert metrics["cache_misses"] == 1

    def test_predict_handles_whitespace_stripping(self, analyzer_with_mock_pipeline):
        """Tests that leading/trailing whitespace doesn't create a separate cache entry."""
        analyzer = analyzer_with_mock_pipeline

        analyzer.predict("  text  ")
        analyzer.predict("text")

        # Both should use the same cache entry (post-strip, they're identical).
        metrics = analyzer.get_performance_metrics()
        assert metrics["cache_hits"] == 1
