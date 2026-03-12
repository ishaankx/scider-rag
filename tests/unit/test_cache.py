"""Unit tests for the caching layer."""

import json
from unittest.mock import AsyncMock

import pytest

from src.storage.cache import QueryCache


@pytest.fixture
def mock_redis():
    r = AsyncMock()
    r.get.return_value = None
    r.setex.return_value = True
    r.incr.return_value = 2
    r.set.return_value = True
    r.delete.return_value = 1
    return r


@pytest.fixture
def cache(mock_redis):
    return QueryCache(mock_redis, default_ttl=3600)


class TestQueryCache:
    @pytest.mark.asyncio
    async def test_cache_miss_returns_none(self, cache, mock_redis):
        mock_redis.get.return_value = None
        result = await cache.get("what is DNA?")
        assert result is None

    @pytest.mark.asyncio
    async def test_cache_hit_returns_data(self, cache, mock_redis):
        # First call gets version, second gets cached data
        cached_data = {"answer": "DNA is...", "sources": []}
        mock_redis.get.side_effect = ["1", json.dumps(cached_data)]

        result = await cache.get("what is DNA?")
        assert result == cached_data

    @pytest.mark.asyncio
    async def test_set_stores_data(self, cache, mock_redis):
        mock_redis.get.return_value = "1"  # version
        data = {"answer": "test", "sources": []}
        await cache.set("question", None, data)
        mock_redis.setex.assert_called_once()

    @pytest.mark.asyncio
    async def test_invalidate_increments_version(self, cache, mock_redis):
        await cache.invalidate_all()
        mock_redis.incr.assert_called_once()

    @pytest.mark.asyncio
    async def test_same_query_different_filters_different_keys(self, cache, mock_redis):
        mock_redis.get.return_value = "1"
        key1 = cache._make_key("test query", None, 1)
        key2 = cache._make_key("test query", {"source_type": "pdf"}, 1)
        assert key1 != key2

    @pytest.mark.asyncio
    async def test_lock_acquire_and_release(self, cache, mock_redis):
        acquired = await cache.acquire_lock("test query")
        assert acquired
        await cache.release_lock("test query")
        mock_redis.delete.assert_called_once()
