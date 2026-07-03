"""Tests for the Kafka data models in `app/models/kafka_models.py`."""

import pytest

from app.models.kafka_models import ConsumerMetrics, MessageMetadata, ProcessingResult

pytestmark = pytest.mark.unit


class TestMessageMetadata:
    """Tests for `MessageMetadata`."""

    def test_as_dict_returns_all_fields(self):
        """Tests that `as_dict()` serializes all fields correctly."""
        metadata = MessageMetadata(topic="predictions", partition=2, offset=100, timestamp=123.45)

        assert metadata.as_dict() == {
            "topic": "predictions",
            "partition": 2,
            "offset": 100,
            "timestamp": 123.45,
        }


class TestProcessingResult:
    """Tests for `ProcessingResult`."""

    def test_as_dict_success_includes_result(self):
        """Tests that a successful result includes the `result` payload."""
        result = ProcessingResult(success=True, message_id="msg-1", result={"label": "POSITIVE"})

        assert result.as_dict() == {
            "success": True,
            "message_id": "msg-1",
            "result": {"label": "POSITIVE"},
        }

    def test_as_dict_failure_includes_error(self):
        """Tests that a failed result includes the `error` message."""
        result = ProcessingResult(success=False, message_id="msg-2", error="boom")

        assert result.as_dict() == {
            "success": False,
            "message_id": "msg-2",
            "error": "boom",
        }

    def test_as_dict_omits_none_fields(self):
        """Tests that `result`/`error` are omitted from the dict when `None`."""
        result = ProcessingResult(success=True, message_id="msg-3")

        payload = result.as_dict()

        assert "result" not in payload
        assert "error" not in payload


class TestConsumerMetrics:
    """Tests for `ConsumerMetrics`."""

    def test_defaults(self):
        """Tests that default values are all zero/falsy."""
        metrics = ConsumerMetrics()

        assert metrics.messages_consumed == 0
        assert metrics.messages_processed == 0
        assert metrics.messages_failed == 0
        assert metrics.total_processing_time_ms == 0.0
        assert metrics.throughput_tps == 0.0
        assert metrics.consumer_threads == 0
        assert metrics.running is False
        assert metrics.last_commit_time > 0

    def test_avg_processing_time_ms_with_no_messages(self):
        """Tests that the average is 0.0 when no messages have been consumed."""
        metrics = ConsumerMetrics()

        assert metrics.avg_processing_time_ms() == 0.0

    def test_avg_processing_time_ms_computes_average(self):
        """Tests that the average is computed correctly from totals."""
        metrics = ConsumerMetrics(messages_consumed=4, total_processing_time_ms=100.0)

        assert metrics.avg_processing_time_ms() == 25.0
