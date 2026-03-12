"""
Database initialization utilities.
Creates tables from ORM models and ensures the Qdrant collection exists.
"""

from qdrant_client.models import Distance, VectorParams

from src.config import Settings
from src.dependencies import get_qdrant
from src.storage.models import Base


async def create_tables() -> None:
    """Create all database tables if they don't exist."""
    from src.dependencies import _engine

    if _engine is None:
        raise RuntimeError("Database engine not initialized.")

    async with _engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def ensure_qdrant_collection(settings: Settings) -> None:
    """Create the Qdrant vector collection if it doesn't exist."""
    qdrant = get_qdrant()

    collections = await qdrant.get_collections()
    existing_names = [c.name for c in collections.collections]

    if settings.qdrant_collection not in existing_names:
        await qdrant.create_collection(
            collection_name=settings.qdrant_collection,
            vectors_config=VectorParams(
                size=settings.embedding_dimensions,
                distance=Distance.COSINE,
            ),
        )
