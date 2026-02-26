"""Batch streaming client for OpenAI and LiteLLM-compatible APIs."""

from batch_stream_openai.batch_stream import (
    OpenAIRequest,
    run_batch,
    stream_batch,
)

__all__ = ["OpenAIRequest", "run_batch", "stream_batch"]
__version__ = "0.1.0"
