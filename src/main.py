"""
FastAPI application factory.
Creates the app, registers middleware, and mounts API routers.
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import JSONResponse

from src.api.middleware.rate_limit import RateLimitMiddleware
from src.api.middleware.request_id import RequestIdMiddleware
from src.api.middleware.security import SecurityMiddleware
from src.api.v1 import eval as eval_router
from src.api.v1 import health as health_router
from src.api.v1 import ingest as ingest_router
from src.api.v1 import query as query_router
from src.config import get_settings
from src.dependencies import init_services, shutdown_services
from src.storage.init_db import create_tables, ensure_qdrant_collection

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(application: FastAPI):
    """Startup and shutdown lifecycle events."""
    settings = get_settings()

    # Configure logging
    logging.basicConfig(
        level=getattr(logging, settings.log_level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    )
    logger.info("Starting Scider RAG Pipeline — env=%s", settings.app_env)

    # Initialize all external connections
    await init_services(settings)
    logger.info("External services connected.")

    # Create database tables and Qdrant collection
    await create_tables()
    await ensure_qdrant_collection(settings)
    logger.info("Database schema and vector collection ready.")

    yield

    # Graceful shutdown
    logger.info("Shutting down services...")
    await shutdown_services()
    logger.info("Shutdown complete.")


def create_app() -> FastAPI:
    """Build and configure the FastAPI application."""
    settings = get_settings()

    application = FastAPI(
        title="Scider RAG Pipeline",
        description="Multi-agent retrieval and reasoning pipeline over scientific data.",
        version="1.0.0",
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # ── Middleware (applied in reverse order) ──────────────────────────
    # Outermost: Request ID tracking
    application.add_middleware(RequestIdMiddleware)

    # Security headers + body size limit
    application.add_middleware(SecurityMiddleware)

    # Rate limiting — Redis resolved lazily after lifespan startup
    application.add_middleware(RateLimitMiddleware)

    # CORS — restrictive by default, configurable for development
    application.add_middleware(
        CORSMiddleware,
        allow_origins=["*"] if not settings.is_production else [],
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
        expose_headers=["X-Request-ID"],
    )

    # ── Routes ────────────────────────────────────────────────────────
    prefix = "/api/v1"
    application.include_router(health_router.router, prefix=prefix, tags=["health"])
    application.include_router(ingest_router.router, prefix=prefix, tags=["ingestion"])
    application.include_router(query_router.router, prefix=prefix, tags=["query"])
    application.include_router(eval_router.router, prefix=prefix, tags=["evaluation"])

    # ── Global exception handler ──────────────────────────────────────
    @application.exception_handler(Exception)
    async def global_exception_handler(request, exc):
        logger.exception("Unhandled exception: %s", exc)
        # Never leak stack traces to the client in production
        detail = str(exc) if not settings.is_production else "Internal server error"
        return JSONResponse(
            status_code=500,
            content={"error": "Internal server error", "detail": detail},
        )

    return application


# Uvicorn entry point
app = create_app()
