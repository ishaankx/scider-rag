"""
Shared test fixtures.
Provides test client, mock database sessions, and test data.
"""

from collections.abc import AsyncGenerator
from unittest.mock import AsyncMock, MagicMock

import pytest
from httpx import ASGITransport, AsyncClient

from src.config import Settings


@pytest.fixture
def test_settings() -> Settings:
    """Settings configured for testing (no external dependencies)."""
    return Settings(
        app_env="testing",
        database_url="sqlite+aiosqlite:///test.db",
        redis_url="redis://localhost:6379/15",
        qdrant_host="localhost",
        qdrant_port=6333,
        openai_api_key="test-key",
        llm_model="gpt-4o-mini",
        chunk_size=256,
        chunk_overlap=25,
    )


@pytest.fixture
def mock_openai() -> AsyncMock:
    """Mock OpenAI client for unit tests."""
    client = AsyncMock()

    # Mock embeddings
    embedding_response = MagicMock()
    embedding_data = MagicMock()
    embedding_data.embedding = [0.1] * 1536
    embedding_data.index = 0
    embedding_response.data = [embedding_data]
    client.embeddings.create.return_value = embedding_response

    # Mock chat completions
    chat_response = MagicMock()
    choice = MagicMock()
    choice.message.content = '{"answer": "test answer"}'
    choice.message.tool_calls = None
    choice.finish_reason = "stop"
    chat_response.choices = [choice]
    client.chat.completions.create.return_value = chat_response

    return client


@pytest.fixture
def mock_redis() -> AsyncMock:
    """Mock Redis client for unit tests."""
    r = AsyncMock()
    r.get.return_value = None
    r.setex.return_value = True
    r.incr.return_value = 2
    r.ping.return_value = True
    r.pipeline.return_value = r
    r.execute.return_value = [None, 0, True, True]
    return r
