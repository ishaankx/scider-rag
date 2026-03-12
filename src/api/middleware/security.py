"""
Security middleware for request validation and protection.
- Enforces request body size limits
- Adds security headers to responses
"""

from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

from src.config import get_settings


class SecurityMiddleware(BaseHTTPMiddleware):
    """Adds security headers and enforces body size limits."""

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        settings = get_settings()

        # Enforce request body size limit (for non-file-upload endpoints)
        content_length = request.headers.get("content-length")
        if content_length and int(content_length) > settings.max_file_size_bytes:
            return JSONResponse(
                status_code=413,
                content={
                    "error": "Payload too large",
                    "detail": f"Maximum body size is {settings.max_file_size_mb}MB.",
                },
            )

        response = await call_next(request)

        # Security headers
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"

        return response
