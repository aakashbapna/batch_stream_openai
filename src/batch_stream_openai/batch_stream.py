"""Reusable async batch runner for OpenAI API calls.

Provides two entry points:

``stream_batch``
    Fire N requests concurrently, **yield each (index, result) as it completes**.
    Lets the caller write to DB after every single result.

``run_batch``
    Convenience wrapper — collects all ``stream_batch`` results into an
    ordered list.

Both are synchronous wrappers that run a temporary event loop in the calling
thread.  A module-level ``AsyncOpenAI`` client is reused across calls to
keep TCP connections alive.
"""

import asyncio
import logging
from dataclasses import dataclass
from typing import Generator, Literal

from openai import AsyncOpenAI

logger = logging.getLogger(__name__)

_async_client: AsyncOpenAI | None = None

_RETRY_TRANSIENT_MAX = 3
_RETRY_TRANSIENT_DELAY = 2  # seconds

# Transient error strings worth retrying on.
_TRANSIENT_SIGNALS = ("connection error", "timed out", "rate limit", "502", "503", "529")

# Model name prefixes that are routed directly through the OpenAI SDK.
# Everything else (e.g. "gemini/...") is routed through LiteLLM (optional).
_OPENAI_MODEL_PREFIXES = ("gpt-", "o1", "o3", "o4", "chatgpt-")


def _is_openai_model(model: str) -> bool:
    """Return True when *model* should be called via the native OpenAI SDK."""
    return model.startswith(_OPENAI_MODEL_PREFIXES)


def _get_async_client() -> AsyncOpenAI:
    global _async_client
    if _async_client is None:
        _async_client = AsyncOpenAI()
    return _async_client


def _get_litellm():
    """Lazy import of litellm. Raises ImportError if litellm extra not installed."""
    try:
        import litellm
        return litellm
    except ImportError as e:
        raise ImportError(
            "LiteLLM is required for non-OpenAI models (e.g. gemini/*). "
            "Install with: pip install batch-stream-openai[litellm]"
        ) from e


@dataclass(frozen=True)
class OpenAIRequest:
    """A single LLM request specification.

    ``model`` may be any OpenAI model name (e.g. ``"gpt-5-mini"``) or a
    LiteLLM-style model string for other providers (e.g. ``"gemini/gemini-2.0-flash"``).
    Non-OpenAI models are automatically routed through LiteLLM (requires the
    ``[litellm]`` extra) and always use the Chat Completions format regardless
    of the ``api`` field.
    """

    system_prompt: str
    user_prompt: str
    model: str = "gpt-5-mini"
    api: Literal["responses", "chat"] = "responses"
    response_format: dict[str, str] | None = None
    reasoning_effort: Literal["minimal", "low", "medium", "high"] | None = None


# ---------------------------------------------------------------------------
# Internal async helpers
# ---------------------------------------------------------------------------

async def _call_responses_api(client: AsyncOpenAI, req: OpenAIRequest) -> str:
    kwargs: dict = dict(
        model=req.model,
        instructions=req.system_prompt,
        input=req.user_prompt,
    )
    if req.reasoning_effort is not None:
        kwargs["reasoning"] = {"effort": req.reasoning_effort}
    response = await client.responses.create(**kwargs)
    output_text = getattr(response, "output_text", None)
    if isinstance(output_text, str):
        return output_text
    raise ValueError("Unexpected OpenAI Responses API format.")


async def _call_chat_api(client: AsyncOpenAI, req: OpenAIRequest) -> str:
    kwargs: dict = dict(
        model=req.model,
        messages=[
            {"role": "system", "content": req.system_prompt},
            {"role": "user", "content": req.user_prompt},
        ],
    )
    if req.response_format is not None:
        kwargs["response_format"] = req.response_format
    if req.reasoning_effort is not None:
        kwargs["reasoning_effort"] = req.reasoning_effort
    response = await client.chat.completions.create(**kwargs)
    content = response.choices[0].message.content
    if isinstance(content, str):
        return content
    raise ValueError("Unexpected OpenAI Chat API format.")


async def _call_litellm_api(req: OpenAIRequest) -> str:
    """Route the request through LiteLLM (used for non-OpenAI models, e.g. Gemini).

    LiteLLM exposes an OpenAI-compatible ``acompletion`` interface and handles
    provider-specific auth/endpoints automatically via environment variables
    (e.g. ``GEMINI_API_KEY``).  ``reasoning_effort`` is intentionally omitted
    because non-OpenAI models don't support it.
    """
    litellm = _get_litellm()
    kwargs: dict = dict(
        model=req.model,
        messages=[
            {"role": "system", "content": req.system_prompt},
            {"role": "user", "content": req.user_prompt},
        ],
    )
    if req.response_format is not None:
        kwargs["response_format"] = req.response_format
    response = await litellm.acompletion(**kwargs)
    content = response.choices[0].message.content
    if isinstance(content, str):
        return content
    raise ValueError("Unexpected LiteLLM response format.")


async def _call(client: AsyncOpenAI | None, req: OpenAIRequest) -> str:
    if not _is_openai_model(req.model):
        return await _call_litellm_api(req)
    if client is None:
        raise RuntimeError("OpenAI client required for OpenAI models")
    if req.api == "chat":
        return await _call_chat_api(client, req)
    return await _call_responses_api(client, req)


def _is_transient(exc: Exception) -> bool:
    """Return True if the exception looks like a transient network/rate error."""
    msg = str(exc).lower()
    return any(signal in msg for signal in _TRANSIENT_SIGNALS)


async def _call_with_retry(client: AsyncOpenAI | None, req: OpenAIRequest) -> str:
    """Call the OpenAI API with retries on transient failures."""
    last_exc: Exception | None = None
    for attempt in range(1, _RETRY_TRANSIENT_MAX + 1):
        try:
            return await _call(client, req)
        except Exception as exc:
            last_exc = exc
            if attempt < _RETRY_TRANSIENT_MAX and _is_transient(exc):
                logger.warning(
                    "OpenAI transient error (attempt %d/%d): %s — retrying in %ds",
                    attempt, _RETRY_TRANSIENT_MAX, exc, _RETRY_TRANSIENT_DELAY,
                )
                await asyncio.sleep(_RETRY_TRANSIENT_DELAY * attempt)
            else:
                raise
    raise last_exc  # type: ignore[misc]


# ---------------------------------------------------------------------------
# stream_batch — yield each result as it completes
# ---------------------------------------------------------------------------

def stream_batch(
    requests: list[OpenAIRequest],
    *,
    max_concurrency: int = 5,
) -> Generator[tuple[int, str | Exception], None, None]:
    """Fire requests concurrently, **yield** ``(index, result)`` as each one
    finishes.

    Runs the async event loop directly in the calling thread.  Reuses a
    module-level ``AsyncOpenAI`` client across calls for TCP connection
    reuse.

    Individual request failures are caught and yielded as ``Exception``
    values — they never kill the batch.  Transient errors are retried
    automatically.

    Results arrive in **completion order**, *not* input order.
    """
    if not requests:
        return
    if not _is_openai_model(requests[0].model):
        logger.info("Using LiteLLM for model=%s", requests[0].model)
        client = None
    else:
        logger.info("Using OpenAI for model=%s", requests[0].model)
        client = _get_async_client()
    loop = asyncio.new_event_loop()

    async def _run():
        semaphore = asyncio.Semaphore(max_concurrency)

        async def _worker(idx: int, req: OpenAIRequest) -> tuple[int, str | Exception]:
            async with semaphore:
                try:
                    result: str | Exception = await _call_with_retry(client, req)
                except Exception as exc:
                    logger.warning("OpenAI request %d failed: %s", idx, exc)
                    result = exc
            return (idx, result)

        tasks = [
            asyncio.create_task(_worker(i, req))
            for i, req in enumerate(requests)
        ]
        for coro in asyncio.as_completed(tasks):
            yield await coro

    try:
        agen = _run()
        while True:
            try:
                item = loop.run_until_complete(agen.__anext__())
                yield item
            except StopAsyncIteration:
                break
    finally:
        # Drain pending async generator cleanup (httpx connection callbacks)
        # while the loop is still open, then close the loop.  The client
        # itself is NOT closed — it's a module-level singleton reused across
        # calls.
        loop.run_until_complete(loop.shutdown_asyncgens())
        loop.close()


# ---------------------------------------------------------------------------
# run_batch — convenience wrapper over stream_batch
# ---------------------------------------------------------------------------

def run_batch(
    requests: list[OpenAIRequest],
    *,
    max_concurrency: int = 5,
) -> list[str | Exception]:
    """Run requests concurrently, return ordered results.

    Blocks until **all** requests finish.  Good for small batches where you
    don't need incremental progress.
    """
    results: list[str | Exception] = [Exception("Not started")] * len(requests)
    for idx, result in stream_batch(requests, max_concurrency=max_concurrency):
        results[idx] = result
    return results
