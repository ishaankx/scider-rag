"""
Redis-backed sliding window rate limiter.
Limits requests per IP address within a configurable time window.
"""

import time

import redis.asyncio as redis
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

from src.config import get_settings


class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    Sliding window rate limiter using Redis sorted sets.
    Each IP gets a sorted set keyed by IP, scored by timestamp.
    """

    def __init__(self, app):
        super().__init__(app)

    def _get_redis(self) -> redis.Redis | None:
        """Lazily resolve Redis — not available until after lifespan startup."""
        try:
            from src.dependencies import get_redis
            return get_redis()
        except RuntimeError:
            return None

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        # Skip rate limiting for health checks
        if request.url.path.endswith("/health"):
            return await call_next(request)

        # Resolve Redis lazily (graceful degradation if unavailable)
        redis_client = self._get_redis()
        if redis_client is None:
            return await call_next(request)

        settings = get_settings()
        client_ip = request.client.host if request.client else "unknown"
        key = f"ratelimit:{client_ip}"
        now = time.time()
        window_start = now - 60  # 1-minute sliding window

        try:
            pipe = redis_client.pipeline()
            # Remove entries older than the window
            pipe.zremrangebyscore(key, 0, window_start)
            # Count remaining entries
            pipe.zcard(key)
            # Add current request
            pipe.zadd(key, {f"{now}": now})
            # Set key expiry so it auto-cleans
            pipe.expire(key, 120)
            results = await pipe.execute()

            request_count = results[1]

            if request_count >= settings.rate_limit_per_minute:
                return JSONResponse(
                    status_code=429,
                    content={
                        "error": "Rate limit exceeded",
                        "detail": f"Maximum {settings.rate_limit_per_minute} requests per minute.",
                    },
                )
        except redis.RedisError:
            # If Redis fails, allow the request (graceful degradation)
            pass

        response = await call_next(request)
        return response
