"""Tests for batch_stream module: correctness, error handling, and LiteLLM optional."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from batch_stream_openai import OpenAIRequest, run_batch, stream_batch
from batch_stream_openai.batch_stream import (
    _is_openai_model,
    _is_transient,
)


# ---------------------------------------------------------------------------
# Unit tests: _is_openai_model, _is_transient
# ---------------------------------------------------------------------------

class TestIsOpenaiModel:
    def test_gpt_prefix(self):
        assert _is_openai_model("gpt-4o") is True
        assert _is_openai_model("gpt-5-mini") is True

    def test_o_models(self):
        assert _is_openai_model("o1") is True
        assert _is_openai_model("o3") is True
        assert _is_openai_model("o4") is True

    def test_chatgpt_prefix(self):
        assert _is_openai_model("chatgpt-4") is True

    def test_non_openai_models(self):
        assert _is_openai_model("gemini/gemini-2.0-flash") is False
        assert _is_openai_model("anthropic/claude-3") is False
        assert _is_openai_model("meta/llama-3") is False


class TestIsTransient:
    def test_transient_errors(self):
        assert _is_transient(Exception("connection error")) is True
        assert _is_transient(Exception("Request timed out")) is True
        assert _is_transient(Exception("rate limit exceeded")) is True
        assert _is_transient(Exception("502 Bad Gateway")) is True
        assert _is_transient(Exception("503 Service Unavailable")) is True
        assert _is_transient(Exception("529 Too Many Requests")) is True

    def test_non_transient_errors(self):
        assert _is_transient(Exception("invalid API key")) is False
        assert _is_transient(ValueError("bad request")) is False
        assert _is_transient(Exception("404 Not Found")) is False


# ---------------------------------------------------------------------------
# Integration tests: stream_batch, run_batch with mocked OpenAI
# ---------------------------------------------------------------------------

def _make_chat_response(content: str):
    """Build a mock ChatCompletion response."""
    msg = MagicMock()
    msg.content = content
    choice = MagicMock()
    choice.message = msg
    resp = MagicMock()
    resp.choices = [choice]
    return resp


def _make_responses_api_response(output_text: str):
    """Build a mock Responses API response."""
    resp = MagicMock()
    resp.output_text = output_text
    return resp


@pytest.fixture
def mock_openai_client():
    """Provide a mocked AsyncOpenAI client."""
    client = MagicMock()
    client.chat = MagicMock()
    client.chat.completions = MagicMock()
    client.chat.completions.create = AsyncMock()
    client.responses = MagicMock()
    client.responses.create = AsyncMock()
    return client


class TestStreamBatch:
    @patch("batch_stream_openai.batch_stream._get_async_client")
    def test_stream_batch_success_chat_api(self, mock_get_client, mock_openai_client):
        mock_get_client.return_value = mock_openai_client
        mock_openai_client.chat.completions.create.return_value = _make_chat_response("42")

        requests = [
            OpenAIRequest(system_prompt="You are helpful.", user_prompt="2+2?", model="gpt-4o-mini", api="chat"),
        ]
        results = list(stream_batch(requests))
        assert len(results) == 1
        idx, result = results[0]
        assert idx == 0
        assert result == "42"

    @patch("batch_stream_openai.batch_stream._get_async_client")
    def test_stream_batch_multiple_requests(self, mock_get_client, mock_openai_client):
        mock_get_client.return_value = mock_openai_client
        mock_openai_client.chat.completions.create.side_effect = [
            _make_chat_response("A"),
            _make_chat_response("B"),
            _make_chat_response("C"),
        ]

        requests = [
            OpenAIRequest(system_prompt="S", user_prompt="1", model="gpt-4o", api="chat"),
            OpenAIRequest(system_prompt="S", user_prompt="2", model="gpt-4o", api="chat"),
            OpenAIRequest(system_prompt="S", user_prompt="3", model="gpt-4o", api="chat"),
        ]
        results = list(stream_batch(requests, max_concurrency=3))
        assert len(results) == 3
        indices = {idx for idx, _ in results}
        assert indices == {0, 1, 2}
        texts = {r for _, r in results if isinstance(r, str)}
        assert texts == {"A", "B", "C"}

    @patch("batch_stream_openai.batch_stream._get_async_client")
    def test_stream_batch_empty_requests(self, mock_get_client):
        results = list(stream_batch([]))
        assert results == []
        mock_get_client.assert_not_called()

    @patch("batch_stream_openai.batch_stream._get_async_client")
    def test_stream_batch_failure_yields_exception(self, mock_get_client, mock_openai_client):
        mock_get_client.return_value = mock_openai_client
        mock_openai_client.chat.completions.create.side_effect = ValueError("Invalid request")

        requests = [
            OpenAIRequest(system_prompt="S", user_prompt="P", model="gpt-4o", api="chat"),
        ]
        results = list(stream_batch(requests))
        assert len(results) == 1
        idx, result = results[0]
        assert idx == 0
        assert isinstance(result, ValueError)
        assert str(result) == "Invalid request"

    @patch("batch_stream_openai.batch_stream._get_async_client")
    def test_stream_batch_partial_failure(self, mock_get_client, mock_openai_client):
        mock_get_client.return_value = mock_openai_client
        mock_openai_client.chat.completions.create.side_effect = [
            _make_chat_response("OK"),
            ValueError("Failed"),
            _make_chat_response("Also OK"),
        ]

        requests = [
            OpenAIRequest(system_prompt="S", user_prompt="1", model="gpt-4o", api="chat"),
            OpenAIRequest(system_prompt="S", user_prompt="2", model="gpt-4o", api="chat"),
            OpenAIRequest(system_prompt="S", user_prompt="3", model="gpt-4o", api="chat"),
        ]
        results = list(stream_batch(requests))
        assert len(results) == 3
        by_idx = {idx: r for idx, r in results}
        assert by_idx[0] == "OK"
        assert isinstance(by_idx[1], ValueError)
        assert by_idx[2] == "Also OK"


class TestRunBatch:
    @patch("batch_stream_openai.batch_stream._get_async_client")
    def test_run_batch_ordered_results(self, mock_get_client, mock_openai_client):
        mock_get_client.return_value = mock_openai_client
        mock_openai_client.chat.completions.create.side_effect = [
            _make_chat_response("First"),
            _make_chat_response("Second"),
            _make_chat_response("Third"),
        ]

        requests = [
            OpenAIRequest(system_prompt="S", user_prompt="1", model="gpt-4o", api="chat"),
            OpenAIRequest(system_prompt="S", user_prompt="2", model="gpt-4o", api="chat"),
            OpenAIRequest(system_prompt="S", user_prompt="3", model="gpt-4o", api="chat"),
        ]
        results = run_batch(requests)
        assert results == ["First", "Second", "Third"]

    @patch("batch_stream_openai.batch_stream._get_async_client")
    def test_run_batch_with_failures(self, mock_get_client, mock_openai_client):
        mock_get_client.return_value = mock_openai_client
        mock_openai_client.chat.completions.create.side_effect = [
            _make_chat_response("OK"),
            RuntimeError("Boom"),
            _make_chat_response("OK2"),
        ]

        requests = [
            OpenAIRequest(system_prompt="S", user_prompt="1", model="gpt-4o", api="chat"),
            OpenAIRequest(system_prompt="S", user_prompt="2", model="gpt-4o", api="chat"),
            OpenAIRequest(system_prompt="S", user_prompt="3", model="gpt-4o", api="chat"),
        ]
        results = run_batch(requests)
        assert results[0] == "OK"
        assert isinstance(results[1], RuntimeError)
        assert str(results[1]) == "Boom"
        assert results[2] == "OK2"


# ---------------------------------------------------------------------------
# Responses API
# ---------------------------------------------------------------------------

class TestResponsesApi:
    @patch("batch_stream_openai.batch_stream._get_async_client")
    def test_responses_api_success(self, mock_get_client, mock_openai_client):
        mock_get_client.return_value = mock_openai_client
        mock_openai_client.responses.create.return_value = _make_responses_api_response("Reasoned answer")

        requests = [
            OpenAIRequest(
                system_prompt="Think step by step.",
                user_prompt="What is 3*4?",
                model="gpt-4o",
                api="responses",
            ),
        ]
        results = run_batch(requests)
        assert results == ["Reasoned answer"]
        mock_openai_client.responses.create.assert_called_once()
        mock_openai_client.chat.completions.create.assert_not_called()


# ---------------------------------------------------------------------------
# Retry behavior (transient errors)
# ---------------------------------------------------------------------------

class TestRetryBehavior:
    @patch("batch_stream_openai.batch_stream._get_async_client")
    @patch("batch_stream_openai.batch_stream.asyncio.sleep", new_callable=AsyncMock)
    def test_retries_on_transient_error_then_succeeds(
        self, mock_sleep, mock_get_client, mock_openai_client
    ):
        mock_get_client.return_value = mock_openai_client
        mock_openai_client.chat.completions.create.side_effect = [
            Exception("rate limit exceeded"),
            _make_chat_response("Success after retry"),
        ]

        requests = [
            OpenAIRequest(system_prompt="S", user_prompt="P", model="gpt-4o", api="chat"),
        ]
        results = run_batch(requests)
        assert results == ["Success after retry"]
        assert mock_openai_client.chat.completions.create.call_count == 2
        mock_sleep.assert_called()

    @patch("batch_stream_openai.batch_stream._get_async_client")
    def test_no_retry_on_non_transient_error(self, mock_get_client, mock_openai_client):
        mock_get_client.return_value = mock_openai_client
        mock_openai_client.chat.completions.create.side_effect = ValueError("Invalid API key")

        requests = [
            OpenAIRequest(system_prompt="S", user_prompt="P", model="gpt-4o", api="chat"),
        ]
        results = run_batch(requests)
        assert len(results) == 1
        assert isinstance(results[0], ValueError)
        assert mock_openai_client.chat.completions.create.call_count == 1


# ---------------------------------------------------------------------------
# LiteLLM optional: ImportError when using non-OpenAI model without litellm
# ---------------------------------------------------------------------------

class TestLitellmOptional:
    @patch("batch_stream_openai.batch_stream._get_litellm")
    def test_non_openai_model_without_litellm_yields_helpful_import_error(self, mock_get_litellm):
        """When litellm is not installed, non-OpenAI models yield a helpful ImportError."""
        mock_get_litellm.side_effect = ImportError(
            "LiteLLM is required for non-OpenAI models (e.g. gemini/*). "
            "Install with: pip install batch-stream-openai[litellm]"
        )
        requests = [
            OpenAIRequest(
                system_prompt="S",
                user_prompt="P",
                model="gemini/gemini-2.0-flash",
                api="chat",
            ),
        ]
        results = list(stream_batch(requests))
        assert len(results) == 1
        idx, result = results[0]
        assert idx == 0
        assert isinstance(result, ImportError)
        assert "litellm" in str(result).lower()
        assert "pip install" in str(result).lower()
