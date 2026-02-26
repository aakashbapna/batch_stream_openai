# batch-stream-openai

Async batch runner for OpenAI API calls with streaming support. Fire many requests concurrently and process results as they complete—ideal for batch inference, evals, or any workload where you need to call the OpenAI API (or LiteLLM-compatible providers) at scale.

## Features

- **`stream_batch`**: Fire N requests concurrently and **yield each `(index, result)` as it completes**. Process results incrementally (e.g., write to DB after every single result).
- **`run_batch`**: Convenience wrapper that collects all results into an ordered list.
- **Automatic retries** on transient errors (rate limits, timeouts, 502/503).
- **OpenAI models** (gpt-*, o1, o3, o4, chatgpt-*) use the native OpenAI SDK.
- **Non-OpenAI models** (e.g. `gemini/gemini-2.0-flash`) route through [LiteLLM](https://github.com/BerriAI/litellm) when the optional `[litellm]` extra is installed.

## Installation

```bash
# Core (OpenAI only)
pip install batch-stream-openai

# With LiteLLM support (Gemini, Anthropic, etc.)
pip install batch-stream-openai[litellm]
```

## Quick Start

```python
from batch_stream_openai import OpenAIRequest, stream_batch, run_batch

requests = [
    OpenAIRequest(
        system_prompt="You are a helpful assistant.",
        user_prompt="What is 2+2?",
        model="gpt-4o-mini",
        api="chat",
    ),
    OpenAIRequest(
        system_prompt="You are a helpful assistant.",
        user_prompt="What is the capital of France?",
        model="gpt-4o-mini",
        api="chat",
    ),
]

# Stream results as they complete
for idx, result in stream_batch(requests, max_concurrency=5):
    if isinstance(result, str):
        print(f"Request {idx}: {result[:50]}...")
    else:
        print(f"Request {idx} failed: {result}")

# Or collect all results in order
results = run_batch(requests)
```

## API Reference

### `OpenAIRequest`

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `system_prompt` | `str` | required | System message for the model |
| `user_prompt` | `str` | required | User message |
| `model` | `str` | `"gpt-5-mini"` | Model name (OpenAI or LiteLLM-style, e.g. `gemini/gemini-2.0-flash`) |
| `api` | `"responses"` \| `"chat"` | `"responses"` | `"responses"` for Responses API, `"chat"` for Chat Completions |
| `response_format` | `dict \| None` | `None` | JSON schema for structured output (Chat API) |
| `reasoning_effort` | `str \| None` | `None` | For reasoning models: `"minimal"`, `"low"`, `"medium"`, `"high"` |

### `stream_batch(requests, *, max_concurrency=5)`

Yields `(index, str | Exception)` as each request completes. Results arrive in **completion order**, not input order. Failed requests yield an `Exception` instead of raising.

### `run_batch(requests, *, max_concurrency=5)`

Returns `list[str | Exception]` with results in the same order as `requests`.

## Environment Variables

- **OpenAI**: `OPENAI_API_KEY` (required for OpenAI models)
- **LiteLLM**: Provider-specific keys (e.g. `GEMINI_API_KEY`, `ANTHROPIC_API_KEY`) when using non-OpenAI models with the `[litellm]` extra

## License

Apache 2.0
