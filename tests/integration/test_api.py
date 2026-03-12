"""
Integration tests for API endpoints.
Tests the HTTP layer with mocked backend services.
"""

import io
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from httpx import ASGITransport, AsyncClient


@pytest.fixture
def mock_services():
    """Patch all external service dependencies."""
    with (
        patch("src.dependencies._engine") as mock_engine,
        patch("src.dependencies._session_factory") as mock_sf,
        patch("src.dependencies._redis_pool") as mock_redis,
        patch("src.dependencies._qdrant_client") as mock_qdrant,
        patch("src.dependencies._openai_client") as mock_openai,
    ):
        # Mock session factory
        mock_session = AsyncMock()
        mock_session.execute.return_value = MagicMock(scalars=MagicMock(return_value=MagicMock(all=MagicMock(return_value=[]))))
        mock_sf.return_value.__aenter__ = AsyncMock(return_value=mock_session)
        mock_sf.return_value.__aexit__ = AsyncMock(return_value=None)

        # Mock Redis
        mock_redis.ping = AsyncMock(return_value=True)
        mock_redis.get = AsyncMock(return_value=None)
        mock_redis.close = AsyncMock()

        # Mock Qdrant
        mock_qdrant.get_collections = AsyncMock(
            return_value=MagicMock(collections=[])
        )
        mock_qdrant.close = AsyncMock()

        yield {
            "engine": mock_engine,
            "session_factory": mock_sf,
            "redis": mock_redis,
            "qdrant": mock_qdrant,
            "openai": mock_openai,
        }


class TestHealthEndpoint:
    @pytest.mark.asyncio
    async def test_health_returns_200(self, mock_services):
        from src.main import app

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get("/api/v1/health")
            # May be degraded due to mocks, but should not 500
            assert response.status_code == 200
            data = response.json()
            assert "status" in data
            assert "services" in data


class TestQueryValidation:
    @pytest.mark.asyncio
    async def test_empty_question_rejected(self, mock_services):
        from src.main import app

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post(
                "/api/v1/query",
                json={"question": ""},
            )
            assert response.status_code == 422  # Validation error

    @pytest.mark.asyncio
    async def test_short_question_rejected(self, mock_services):
        from src.main import app

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post(
                "/api/v1/query",
                json={"question": "ab"},
            )
            assert response.status_code == 422


class TestIngestValidation:
    @pytest.mark.asyncio
    async def test_no_file_rejected(self, mock_services):
        from src.main import app

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post("/api/v1/ingest")
            assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_unsupported_file_type_rejected(self, mock_services):
        from src.main import app

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post(
                "/api/v1/ingest",
                files={"file": ("test.exe", b"binary content", "application/octet-stream")},
            )
            assert response.status_code == 400
