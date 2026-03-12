"""
Dependency injection for FastAPI.
Provides database sessions, Redis connections, and service instances.
"""

from collections.abc import AsyncGenerator

import redis.asyncio as redis
from openai import AsyncOpenAI
from qdrant_client import AsyncQdrantClient
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from src.config import Settings, get_settings


# ── Engine & Session Factory (created once at startup) ──────────────────────

_engine = None
_session_factory = None
_redis_pool = None
_qdrant_client = None
_openai_client = None


async def init_services(settings: Settings) -> None:
    """Initialize all external service connections. Called once at app startup."""
    global _engine, _session_factory, _redis_pool, _qdrant_client, _openai_client

    # PostgreSQL
    _engine = create_async_engine(
        settings.database_url,
        pool_size=20,
        max_overflow=10,
        pool_pre_ping=True,
        echo=False,
    )
    _session_factory = async_sessionmaker(
        bind=_engine, class_=AsyncSession, expire_on_commit=False
    )

    # Redis
    _redis_pool = redis.from_url(
        settings.redis_url,
        decode_responses=True,
        max_connections=20,
    )

    # Qdrant
    _qdrant_client = AsyncQdrantClient(
        host=settings.qdrant_host,
        port=settings.qdrant_port,
    )

    # OpenAI
    _openai_client = AsyncOpenAI(api_key=settings.openai_api_key)


async def shutdown_services() -> None:
    """Close all connections gracefully. Called at app shutdown."""
    global _engine, _redis_pool, _qdrant_client

    if _engine:
        await _engine.dispose()
    if _redis_pool:
        await _redis_pool.close()
    if _qdrant_client:
        await _qdrant_client.close()


# ── FastAPI Dependency Providers ────────────────────────────────────────────

async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """Yields a database session per request. Auto-closes after use."""
    if _session_factory is None:
        raise RuntimeError("Database not initialized. Call init_services() first.")
    async with _session_factory() as session:
        try:
            yield session
        finally:
            await session.close()


def get_redis() -> redis.Redis:
    """Returns the shared Redis connection pool."""
    if _redis_pool is None:
        raise RuntimeError("Redis not initialized. Call init_services() first.")
    return _redis_pool


def get_qdrant() -> AsyncQdrantClient:
    """Returns the shared Qdrant client."""
    if _qdrant_client is None:
        raise RuntimeError("Qdrant not initialized. Call init_services() first.")
    return _qdrant_client


def get_openai() -> AsyncOpenAI:
    """Returns the shared OpenAI client."""
    if _openai_client is None:
        raise RuntimeError("OpenAI not initialized. Call init_services() first.")
    return _openai_client
