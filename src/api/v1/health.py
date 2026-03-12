"""Health check endpoint with dependency status."""

from fastapi import APIRouter, Depends
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from src.dependencies import get_db, get_qdrant, get_redis

router = APIRouter()


@router.get("/health")
async def health_check(
    db: AsyncSession = Depends(get_db),
):
    """
    Returns service health and the status of each dependency.
    Used by Docker health checks and monitoring.
    """
    checks = {}

    # PostgreSQL
    try:
        await db.execute(text("SELECT 1"))
        checks["postgres"] = "healthy"
    except Exception as exc:
        checks["postgres"] = f"unhealthy: {exc}"

    # Redis
    try:
        redis_client = get_redis()
        await redis_client.ping()
        checks["redis"] = "healthy"
    except Exception as exc:
        checks["redis"] = f"unhealthy: {exc}"

    # Qdrant
    try:
        qdrant = get_qdrant()
        await qdrant.get_collections()
        checks["qdrant"] = "healthy"
    except Exception as exc:
        checks["qdrant"] = f"unhealthy: {exc}"

    all_healthy = all(v == "healthy" for v in checks.values())

    return {
        "status": "healthy" if all_healthy else "degraded",
        "services": checks,
    }
