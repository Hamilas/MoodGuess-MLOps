"""Tests for the LRU cache implementation in the sentiment analyzers.

This module contains test cases to verify the correct behavior of the
Least Recently Used (LRU) cache, including item eviction, cache hits and
misses, and statistics reporting.
"""

import pytest

from app.core.config import Settings
from app.models.pytorch_sentiment import SentimentAnalyzer

# Make ONNX import optional
try:
    from app.models.onnx_sentiment import ONNXSentimentAnalyzer

    ONNX_AVAILABLE = True
except ImportError:  # ModuleNotFoundError is itself an ImportError subclass
    ONNXSentimentAnalyzer = None  # type: ignore
    ONNX_AVAILABLE = False


@pytest.fixture
def mock_settings(monkeypatch):
    """Provides a mocked `Settings` object with a small cache size for testing.

    Args:
        monkeypatch: The pytest `monkeypatch` fixture.

    Returns:
        A mocked `Settings` object.
    """
    settings = Settings(
        model_name="distilbert-base-uncased-finetuned-sst-2-english",
        prediction_cache_max_size=10,  # Small cache for testing (10 is the model minimum)
        prediction_cache_enabled=True,  # Cache enabled by default
        max_text_length=512,
    )
    monkeypatch.setattr("app.models.pytorch_sentiment.get_settings", lambda: settings)
    monkeypatch.setattr("app.models.onnx_sentiment.get_settings", lambda: settings, raising=False)
    return settings


@pytest.fixture
def mock_settings_cache_disabled(monkeypatch):
    """Provides a mocked `Settings` object with cache disabled for testing.

    Args:
        monkeypatch: The pytest `monkeypatch` fixture.

    Returns:
        A mocked `Settings` object with cache disabled.
    """
    settings = Settings(
        model_name="distilbert-base-uncased-finetuned-sst-2-english",
        prediction_cache_max_size=1000,
        prediction_cache_enabled=False,  # Cache disabled
        max_text_length=512,
    )
    monkeypatch.setattr("app.models.pytorch_sentiment.get_settings", lambda: settings)
    monkeypatch.setattr("app.models.onnx_sentiment.get_settings", lambda: settings, raising=False)
    return settings


@pytest.mark.unit
@pytest.mark.cache
class TestLRUCache:
    """A test suite for the LRU cache behavior in the sentiment analyzer.

    `SentimentAnalyzer` caches predictions via `functools.lru_cache` wrapping
    `_predict_internal` (set up in `_init_cache`, introspectable through
    `_get_cache_info()` which returns the wrapper's real `CacheInfo`), not a
    hand-rolled `OrderedDict`/hash-key scheme.
    """

    def _mock_pipeline(self, analyzer, monkeypatch):
        monkeypatch.setattr(
            analyzer, "_pipeline", lambda text: [{"label": "POSITIVE", "score": 0.99}]
        )
        monkeypatch.setattr(analyzer, "_is_loaded", True)

    def test_repeated_text_is_a_cache_hit(self, mock_settings, monkeypatch):
        """Tests that predicting the same text twice hits the cache the second time.

        `predict()` itself doesn't return a "cached" key (that's only added by
        the outer Redis-backed `PredictionService` on a distributed-cache hit);
        the model's own in-process lru_cache hit/miss is only observable via
        `get_performance_metrics()`'s hit/miss counters.
        """
        analyzer = SentimentAnalyzer()
        self._mock_pipeline(analyzer, monkeypatch)

        analyzer.predict("test text")
        analyzer.predict("test text")

        metrics = analyzer.get_performance_metrics()
        assert metrics["cache_hits"] == 1
        assert metrics["cache_misses"] == 1

    def test_different_text_is_a_cache_miss(self, mock_settings, monkeypatch):
        """Tests that predicting distinct texts never hits the cache."""
        analyzer = SentimentAnalyzer()
        self._mock_pipeline(analyzer, monkeypatch)

        analyzer.predict("text 1")
        analyzer.predict("text 2")

        metrics = analyzer.get_performance_metrics()
        assert metrics["cache_hits"] == 0
        assert metrics["cache_misses"] == 2

    def test_lru_eviction_bounds_cache_size(self, mock_settings, monkeypatch):
        """Tests that the cache never grows past `prediction_cache_max_size`."""
        analyzer = SentimentAnalyzer()
        self._mock_pipeline(analyzer, monkeypatch)
        max_size = mock_settings.model.prediction_cache_max_size

        # Fill the cache to its max size.
        for i in range(max_size):
            analyzer.predict(f"text {i}")
        assert analyzer._get_cache_info().currsize == max_size

        # One more distinct text should evict the least-recently-used entry
        # rather than grow the cache further.
        analyzer.predict("one more text")
        assert analyzer._get_cache_info().currsize == max_size

    def test_cache_clear(self, mock_settings, monkeypatch):
        """Tests the functionality of clearing the cache."""
        analyzer = SentimentAnalyzer()
        self._mock_pipeline(analyzer, monkeypatch)

        analyzer.predict("text 1")
        analyzer.predict("text 2")
        assert analyzer._get_cache_info().currsize == 2

        analyzer.clear_cache()

        assert analyzer._get_cache_info().currsize == 0

    def test_cache_hit_returns_same_result(self, mock_settings, monkeypatch):
        """Tests that a cache hit returns the same label/score as the original miss."""
        analyzer = SentimentAnalyzer()
        self._mock_pipeline(analyzer, monkeypatch)

        first = analyzer.predict("test text")
        second = analyzer.predict("test text")

        assert first["label"] == second["label"]
        assert first["score"] == second["score"]
        assert analyzer.get_performance_metrics()["cache_hits"] == 1


@pytest.mark.unit
@pytest.mark.cache
class TestCacheStats:
    """A test suite for the cache statistics functionality."""

    def test_get_performance_metrics_reports_cache_stats(self, mock_settings, monkeypatch):
        """Tests that `get_performance_metrics` reports accurate cache size and hit rate."""
        analyzer = SentimentAnalyzer()
        monkeypatch.setattr(
            analyzer, "_pipeline", lambda text: [{"label": "POSITIVE", "score": 0.99}]
        )
        monkeypatch.setattr(analyzer, "_is_loaded", True)

        analyzer.predict("text 1")
        analyzer.predict("text 2")
        analyzer.predict("text 1")  # cache hit

        metrics = analyzer.get_performance_metrics()

        assert metrics["cache_enabled"] is True
        assert metrics["cache_hits"] == 1
        assert metrics["cache_misses"] == 2
        assert metrics["cache_info"]["maxsize"] == mock_settings.model.prediction_cache_max_size


@pytest.mark.unit
@pytest.mark.cache
class TestCacheDisabled:
    """Test suite for disabled cache functionality."""

    def test_predictions_work_without_cache(self, mock_settings_cache_disabled, monkeypatch):
        """Test that predictions work correctly when cache is disabled."""
        analyzer = SentimentAnalyzer()

        # Mock the pipeline to avoid loading actual model
        def mock_predict(text):
            return [{"label": "POSITIVE", "score": 0.99}]

        monkeypatch.setattr(analyzer, "_pipeline", lambda text: mock_predict(text))
        monkeypatch.setattr(analyzer, "_is_loaded", True)

        # Make predictions - should work without cache
        result1 = analyzer.predict("test text 1")
        result2 = analyzer.predict("test text 2")

        assert result1["label"] == "POSITIVE"
        assert result2["label"] == "POSITIVE"
        assert "score" in result1
        assert "score" in result2

    def test_cache_info_returns_mock_when_disabled(self, mock_settings_cache_disabled, monkeypatch):
        """Test that _get_cache_info returns mock cache info when cache is disabled."""
        analyzer = SentimentAnalyzer()

        monkeypatch.setattr(
            analyzer, "_pipeline", lambda text: [{"label": "POSITIVE", "score": 0.99}]
        )
        monkeypatch.setattr(analyzer, "_is_loaded", True)

        cache_info = analyzer._get_cache_info()

        assert cache_info.hits == 0
        assert cache_info.misses == 0
        assert cache_info.maxsize == 0
        assert cache_info.currsize == 0

    def test_get_model_info_shows_cache_disabled(self, mock_settings_cache_disabled, monkeypatch):
        """Test that get_model_info shows cache as disabled."""
        analyzer = SentimentAnalyzer()

        monkeypatch.setattr(
            analyzer, "_pipeline", lambda text: [{"label": "POSITIVE", "score": 0.99}]
        )
        monkeypatch.setattr(analyzer, "_is_loaded", True)

        model_info = analyzer.get_model_info()

        assert model_info["cache_enabled"] is False
        assert model_info["cache_size"] == 0
        assert model_info["cache_maxsize"] == 0

    def test_get_performance_metrics_shows_cache_disabled(
        self, mock_settings_cache_disabled, monkeypatch
    ):
        """Test that performance metrics show cache as disabled."""
        analyzer = SentimentAnalyzer()

        monkeypatch.setattr(
            analyzer, "_pipeline", lambda text: [{"label": "POSITIVE", "score": 0.99}]
        )
        monkeypatch.setattr(analyzer, "_is_loaded", True)

        # Make a prediction
        analyzer.predict("test text")

        metrics = analyzer.get_performance_metrics()

        assert metrics["cache_enabled"] is False
        assert metrics["cache_hits"] == 0
        assert metrics["cache_misses"] >= 0  # Should track misses even when disabled
        assert metrics["cache_hit_rate"] == 0.0
        assert metrics["cache_info"] is not None
        assert metrics["cache_info"]["hits"] == 0

    def test_clear_cache_noop_when_disabled(self, mock_settings_cache_disabled, monkeypatch):
        """Test that clear_cache is a no-op when cache is disabled."""
        analyzer = SentimentAnalyzer()

        monkeypatch.setattr(
            analyzer, "_pipeline", lambda text: [{"label": "POSITIVE", "score": 0.99}]
        )
        monkeypatch.setattr(analyzer, "_is_loaded", True)

        # Should not raise an error
        analyzer.clear_cache()

        # Verify predictions still work
        result = analyzer.predict("test text")
        assert result["label"] == "POSITIVE"

    @pytest.mark.skipif(not ONNX_AVAILABLE, reason="ONNX not available")
    def test_onnx_predictions_work_without_cache(
        self, mock_settings_cache_disabled, monkeypatch, tmp_path
    ):
        """Test that ONNX predictions work correctly when cache is disabled."""

        # __init__ calls _load_model() synchronously, which would otherwise try
        # to parse a real ONNX protobuf file - stub it out so construction
        # succeeds without ever touching onnxruntime/transformers for real.
        monkeypatch.setattr(ONNXSentimentAnalyzer, "_load_model", lambda self: None)

        model_path = tmp_path / "model"
        model_path.mkdir()

        analyzer = ONNXSentimentAnalyzer(str(model_path))

        # Mock the session and tokenizer
        mock_session = type(
            "MockSession",
            (),
            {
                "run": lambda self, output_names, inputs: [[[0.1, 0.9]]],
                "get_providers": lambda self: ["CPUExecutionProvider"],
            },
        )()
        import numpy as np

        mock_tokenizer = type(
            "MockTokenizer",
            (),
            {
                "__call__": lambda self, text, **kwargs: {
                    "input_ids": np.array([[1, 2, 3]]),
                    "attention_mask": np.array([[1, 1, 1]]),
                }
            },
        )()

        monkeypatch.setattr(analyzer, "_session", mock_session)
        monkeypatch.setattr(analyzer, "_tokenizer", mock_tokenizer)
        monkeypatch.setattr(analyzer, "_is_loaded", True)

        # Make predictions - should work without cache
        result = analyzer.predict("test text")

        assert "label" in result
        assert "score" in result

    @pytest.mark.skipif(not ONNX_AVAILABLE, reason="ONNX not available")
    def test_onnx_get_model_info_shows_cache_disabled(
        self, mock_settings_cache_disabled, monkeypatch, tmp_path
    ):
        """Test that ONNX get_model_info shows cache as disabled."""
        monkeypatch.setattr(ONNXSentimentAnalyzer, "_load_model", lambda self: None)

        model_path = tmp_path / "model"
        model_path.mkdir()

        analyzer = ONNXSentimentAnalyzer(str(model_path))

        mock_session = type(
            "MockSession", (), {"get_providers": lambda self: ["CPUExecutionProvider"]}
        )()
        monkeypatch.setattr(analyzer, "_session", mock_session)
        monkeypatch.setattr(analyzer, "_is_loaded", True)

        model_info = analyzer.get_model_info()

        assert model_info["cache_enabled"] is False
        assert model_info["cache_size"] == 0
        assert model_info["cache_maxsize"] == 0
