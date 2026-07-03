"""Comprehensive input validation tests for the sentiment analysis service.

This module contains extensive test cases for input validation at different
layers of the application, including Pydantic schema validation, model name
security checks, API endpoint handling of invalid data, and various edge cases
to ensure robust error handling and security.
"""

from unittest.mock import Mock

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError as PydanticValidationError

from app.api.routes.predictions import get_prediction_service
from app.api.schemas.requests import TextInput
from app.core.config import Settings, get_settings
from app.main import create_app
from app.utils.exceptions import SettingsValidationError, TextTooLongError


@pytest.mark.unit
class TestTextInputValidation:
    """A test suite for the `TextInput` Pydantic model validation.

    These tests cover various scenarios for the text input field, ensuring
    that validation logic for empty strings, length limits, and whitespace
    stripping is working correctly. `TextInput.text` has a hardcoded
    `Field(min_length=1, max_length=10000)`, checked by Pydantic BEFORE the
    custom `field_validator` runs, so length violations surface Pydantic's
    own `string_too_short`/`string_too_long` messages, not the custom
    "cannot be empty" one - that custom message only fires for
    whitespace-only strings that are non-empty (they pass Field's
    `min_length`, then the field_validator rejects them).
    """

    def test_valid_text_input(self):
        """Tests that a standard, valid text input passes validation."""
        valid_input = TextInput(text="This is a valid text input for sentiment analysis.")
        assert valid_input.text == "This is a valid text input for sentiment analysis."

    def test_empty_string_raises_error(self):
        """Tests that an empty string "" raises a validation error (Field min_length)."""
        with pytest.raises(PydanticValidationError) as exc_info:
            TextInput(text="")

        errors = exc_info.value.errors()
        assert len(errors) > 0
        assert errors[0]["type"] == "string_too_short"

    def test_whitespace_only_raises_error(self):
        """Tests that a string containing only whitespace raises a validation error."""
        with pytest.raises(PydanticValidationError) as exc_info:
            TextInput(text="   \n\t  ")

        assert "cannot be empty" in str(exc_info.value)

    def test_none_raises_error(self):
        """Tests that a `None` value for the text field raises a validation error."""
        with pytest.raises(Exception):  # Pydantic validation error
            TextInput(text=None)

    def test_text_too_long_raises_error(self):
        """Tests that text exceeding the hardcoded 10,000 character max raises an error."""
        long_text = "a" * 10001

        with pytest.raises(PydanticValidationError) as exc_info:
            TextInput(text=long_text)

        errors = exc_info.value.errors()
        assert errors[0]["type"] == "string_too_long"

    def test_text_with_special_characters(self):
        """Tests that text containing special characters and emojis is handled correctly."""
        special_text = "Hello! 🎉 This has émojis & special chars: <script>alert('xss')</script>"
        valid_input = TextInput(text=special_text)

        # Text should be stripped but special characters preserved
        assert valid_input.text == special_text

    def test_text_gets_stripped(self):
        """Tests that leading and trailing whitespace is stripped from the input text."""
        padded_text = "  \t\n  Valid text with padding  \n\t  "
        valid_input = TextInput(text=padded_text)

        assert valid_input.text == "Valid text with padding"

    def test_edge_case_exact_max_length(self):
        """Tests that text with a length exactly equal to the max limit (10,000) passes."""
        exact_length_text = "a" * 10000
        valid_input = TextInput(text=exact_length_text)

        assert valid_input.text == exact_length_text


@pytest.mark.unit
class TestModelValidation:
    """A test suite for model name validation and security checks.

    Model-name-vs-allowed-list validation is enforced at the `Settings` level
    (`Settings._validate_model_in_allowed_list`, raising
    `SettingsValidationError`), not inside `SentimentAnalyzer` - there's no
    code path in `app/models/pytorch_sentiment.py` that raises
    `InvalidModelError` (it's currently unused/dead code), so the real
    security boundary to test is the settings validator.
    """

    def test_invalid_model_name_raises_error(self):
        """Tests that an unauthorized model name fails settings validation."""
        from app.core.config import Settings

        with pytest.raises(SettingsValidationError) as exc_info:
            Settings(
                model={
                    "model_name": "unauthorized-model",
                    "allowed_models": ["model1", "model2"],
                }
            )

        assert "unauthorized-model" in str(exc_info.value)
        assert "allowed_models" in str(exc_info.value)

    def test_valid_model_name_passes(self):
        """Tests that a model name present in `allowed_models` passes validation."""
        from app.core.config import Settings

        settings = Settings(
            model={
                "model_name": "valid-model",
                "allowed_models": ["valid-model"],
            }
        )
        assert settings.model.model_name == "valid-model"


@pytest.mark.unit
class TestAPIEndpointValidation:
    """A test suite for input validation at the API endpoint layer.

    These tests use a `TestClient` to make requests to the API and verify
    that the endpoints correctly handle invalid inputs, such as empty text or
    text that is too long, returning the appropriate HTTP status codes and
    error messages. `Depends(get_prediction_service)` is bound at route
    declaration time, so overriding the dependency requires
    `app.dependency_overrides`, not `unittest.mock.patch` on the module-level
    function (patching after import has no effect on an already-bound
    `Depends(...)`).
    """

    @pytest.fixture
    def app(self, monkeypatch):
        """Creates a fresh FastAPI app instance.

        `debug=False` (routes prefixed with "/api/v1"): Starlette's
        `ServerErrorMiddleware` only invokes a custom `Exception` handler
        (our `global_exception_handler`, which maps `ServiceError` subclasses
        to their real status codes) when `debug` is False - in debug mode it
        always renders a generic 500 traceback page instead, regardless of
        any registered handler.
        """
        settings = Settings(debug=False)
        monkeypatch.setattr("app.main.get_settings", lambda: settings)
        return create_app()

    @pytest.fixture
    def client(self, app):
        """Creates a `TestClient` for the application.

        `raise_server_exceptions=False`: Starlette's `ServerErrorMiddleware`
        unconditionally re-raises the original exception after sending the
        handler's response (so real ASGI servers can log it), which
        `TestClient`'s default `raise_server_exceptions=True` then propagates
        to the caller instead of returning the response - so status-code
        assertions against custom `ServiceError` responses need this off.
        """
        return TestClient(app, raise_server_exceptions=False)

    @pytest.fixture
    def mock_prediction_service(self, app):
        """Overrides the prediction-service dependency with a ready mock.

        Yields:
            A `Mock` object simulating the prediction service.
        """
        service = Mock()
        service.predict.return_value = {
            "label": "POSITIVE",
            "score": 0.95,
            "inference_time_ms": 10.5,
            "model_name": "test-model",
            "text_length": 10,
            "cached": False,
        }
        app.dependency_overrides[get_prediction_service] = lambda: service
        yield service
        app.dependency_overrides.pop(get_prediction_service, None)

    def test_predict_endpoint_empty_text(self, client, mock_prediction_service):
        """Tests that the `/predict` endpoint returns a 422 error for empty text."""
        response = client.post("/api/v1/predict", json={"text": ""})

        assert response.status_code == 422
        assert "validation" in response.json()["error_message"].lower()

    def test_predict_endpoint_missing_text_field(self, client, mock_prediction_service):
        """Tests that the `/predict` endpoint returns a 422 error if the 'text' field is missing."""
        response = client.post("/api/v1/predict", json={})

        assert response.status_code == 422  # Pydantic validation error

    def test_predict_endpoint_text_too_long(self, client, mock_prediction_service):
        """Tests that the `/predict` endpoint returns a 422 error for text over 10,000 chars."""
        long_text = "a" * 10001
        response = client.post("/api/v1/predict", json={"text": long_text})

        assert response.status_code == 422
        assert "validation" in response.json()["error_message"].lower()

    def test_predict_endpoint_valid_text(self, client, mock_prediction_service):
        """Tests that the `/predict` endpoint returns a 200 OK for valid text."""
        response = client.post("/api/v1/predict", json={"text": "This is valid text"})

        assert response.status_code == 200
        data = response.json()
        assert data["label"] == "POSITIVE"
        assert data["score"] == 0.95

    def test_predict_endpoint_model_not_loaded(self, client, app):
        """Tests that the `/predict` endpoint returns a 503 error if the model is not loaded."""
        from app.utils.exceptions import ModelNotLoadedError

        service = Mock()
        service.predict.side_effect = ModelNotLoadedError("test-model")
        app.dependency_overrides[get_prediction_service] = lambda: service

        response = client.post("/api/v1/predict", json={"text": "Test text"})

        assert response.status_code == 503
        assert "MODEL_NOT_LOADED" in response.json()["error_code"]

        app.dependency_overrides.pop(get_prediction_service, None)


@pytest.mark.unit
class TestSecurityInputValidation:
    """A test suite for security-related input validation.

    These tests ensure that potentially malicious inputs, such as XSS or SQL
    injection payloads, are handled safely without being executed or
    sanitized at the input validation layer.
    """

    def test_potential_xss_in_text_input(self):
        """Tests that text containing potential XSS payloads is accepted as-is."""
        xss_text = "<script>alert('XSS')</script>"
        valid_input = TextInput(text=xss_text)

        # Input should be preserved as-is (not escaped at input level)
        # Escaping should happen at output/logging level
        assert valid_input.text == xss_text

    def test_sql_injection_like_content(self):
        """Tests that text resembling SQL injection attacks is accepted as-is."""
        sql_text = "'; DROP TABLE users; --"
        valid_input = TextInput(text=sql_text)

        assert valid_input.text == sql_text

    def test_unicode_and_emoji_handling(self):
        """Tests that Unicode characters and emojis are handled correctly."""
        unicode_text = "Hello 世界! 🌍🚀 Testing üñíçødé"
        valid_input = TextInput(text=unicode_text)

        assert valid_input.text == unicode_text

    def test_very_long_single_word(self):
        """Tests that a single word over the 10,000 char max is correctly rejected."""
        long_word = "a" * 10001

        with pytest.raises(PydanticValidationError):
            TextInput(text=long_word)

    def test_newlines_and_control_characters(self):
        """Tests that newline and other control characters are preserved in the input."""
        text_with_controls = "Line 1\nLine 2\r\nLine 3\tTabbed\x00\x01\x02"
        valid_input = TextInput(text=text_with_controls)

        # Should preserve the text as-is
        assert valid_input.text == text_with_controls


@pytest.mark.unit
class TestEdgeCases:
    """A test suite for various edge cases and boundary conditions.

    These tests cover scenarios like text that becomes empty after stripping
    and validation of diverse international character sets.
    """

    def test_zero_length_after_strip(self):
        """Tests that text that becomes empty after whitespace stripping is rejected."""
        with pytest.raises(PydanticValidationError):
            TextInput(text="   \n\t\r   ")

    def test_international_characters(self):
        """Tests that various international character sets are handled correctly."""
        international_texts = [
            "こんにちは",  # Japanese
            "مرحبا",  # Arabic
            "Здравствуйте",  # Russian
            "नमस्ते",  # Hindi
            "🇺🇸🇬🇧🇫🇷",  # Flag emojis
        ]

        for text in international_texts:
            valid_input = TextInput(text=text)
            assert valid_input.text == text


@pytest.mark.unit
class TestPredictionServiceValidation:
    """A test suite for the `PredictionService` validation logic."""

    @pytest.fixture
    def mock_model(self):
        """Mocks the model strategy."""
        model = Mock()
        model.is_ready.return_value = True
        return model

    @pytest.fixture
    def mock_settings(self):
        """Provides real settings, overriding just `model.max_text_length`."""
        settings = get_settings()
        settings.model.max_text_length = 20
        yield settings
        settings.model.max_text_length = 10000

    def test_text_too_long_raises_error_in_service(self, mock_model, mock_settings):
        """Tests that `PredictionService` raises `TextTooLongError` for oversized input."""
        from app.services.prediction import PredictionService

        service = PredictionService(
            model=mock_model, settings=mock_settings, feature_engineer=Mock()
        )
        long_text = "This text is well over the twenty-character limit."

        with pytest.raises(TextTooLongError) as exc_info:
            service.predict(long_text)

        assert exc_info.value.code == "TEXT_TOO_LONG"
        assert "exceeds" in str(exc_info.value)
        assert str(len(long_text)) in str(exc_info.value)
        assert str(mock_settings.model.max_text_length) in str(exc_info.value)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
