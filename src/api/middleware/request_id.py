"""
Request ID middleware.
Assigns a unique ID to every request for tracing across logs and responses.
"""

import uuid
from contextvars import ContextVar

from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import Response

# Context variable accessible anywhere during a request lifecycle
request_id_ctx: ContextVar[str] = ContextVar("request_id", default="")


class RequestIdMiddleware(BaseHTTPMiddleware):
    """Attaches a unique X-Request-ID header to every request and response."""

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        # Use client-provided ID if present, otherwise generate one
        rid = request.headers.get("X-Request-ID", str(uuid.uuid4()))
        request_id_ctx.set(rid)

        response = await call_next(request)
        response.headers["X-Request-ID"] = rid
        return response
