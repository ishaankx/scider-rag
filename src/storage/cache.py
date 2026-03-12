"""
Redis-backed caching layer with version-based invalidation.

Cache keys include a data version number. When new documents are ingested,
the version is incremented, causing all old cache entries to miss.
This prevents serving stale results after new data arrives.
"""

import hashlib
import json
import logging

import redis.asyncio as redis

logger = logging.getLogger(__name__)

# Prefix for all cache keys
_CACHE_PREFIX = "rag:cache:"
_VERSION_KEY = "rag:data_version"
_LOCK_PREFIX = "rag:lock:"


class QueryCache:
    """
    Caches query results in Redis with version-based invalidation.
    Prevents cache stampedes using distributed locks.
    """

    def __init__(self, redis_client: redis.Redis, default_ttl: int = 3600):
        self._redis = redis_client
        self._ttl = default_ttl

    def _make_key(self, query: str, filters: dict | None, version: int) -> str:
        """Generate a deterministic cache key from query + filters + version."""
        raw = json.dumps({"q": query.strip().lower(), "f": filters or {}, "v": version})
        digest = hashlib.sha256(raw.encode()).hexdigest()[:32]
        return f"{_CACHE_PREFIX}{digest}"

    async def _get_version(self) -> int:
        """Get current data version. Starts at 1."""
        version = await self._redis.get(_VERSION_KEY)
        return int(version) if version else 1

    async def get(self, query: str, filters: dict | None = None) -> dict | None:
        """
        Retrieve a cached response.
        Returns None on cache miss or if data version has changed.
        """
        try:
            version = await self._get_version()
            key = self._make_key(query, filters, version)
            cached = await self._redis.get(key)
            if cached:
                logger.debug("Cache hit for query: %s", query[:50])
                return json.loads(cached)
        except redis.RedisError as exc:
            logger.warning("Cache read failed: %s", exc)
        return None

    async def set(
        self, query: str, filters: dict | None, response: dict, ttl: int | None = None
    ) -> None:
        """Store a query response in cache."""
        try:
            version = await self._get_version()
            key = self._make_key(query, filters, version)
            await self._redis.setex(
                key,
                ttl or self._ttl,
                json.dumps(response),
            )
            logger.debug("Cached response for query: %s", query[:50])
        except redis.RedisError as exc:
            logger.warning("Cache write failed: %s", exc)

    async def invalidate_all(self) -> None:
        """
        Invalidate all cached results by incrementing the data version.
        Old keys will naturally expire via TTL — no need to delete them.
        """
        try:
            await self._redis.incr(_VERSION_KEY)
            logger.info("Cache invalidated — data version incremented.")
        except redis.RedisError as exc:
            logger.warning("Cache invalidation failed: %s", exc)

    async def acquire_lock(self, query: str, timeout: int = 30) -> bool:
        """
        Acquire a distributed lock for a query to prevent cache stampede.
        Returns True if lock was acquired, False if another request holds it.
        """
        lock_key = f"{_LOCK_PREFIX}{hashlib.sha256(query.encode()).hexdigest()[:16]}"
        try:
            return bool(await self._redis.set(lock_key, "1", nx=True, ex=timeout))
        except redis.RedisError:
            return True  # If Redis fails, allow the request through

    async def release_lock(self, query: str) -> None:
        """Release the distributed lock for a query."""
        lock_key = f"{_LOCK_PREFIX}{hashlib.sha256(query.encode()).hexdigest()[:16]}"
        try:
            await self._redis.delete(lock_key)
        except redis.RedisError:
            pass
