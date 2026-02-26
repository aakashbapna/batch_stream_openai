"""Pytest fixtures and configuration."""

import pytest


@pytest.fixture(autouse=True)
def reset_global_client():
    """Reset the module-level async client between tests to avoid state leakage."""
    import batch_stream_openai.batch_stream as mod
    old_client = mod._async_client
    mod._async_client = None
    yield
    mod._async_client = old_client
