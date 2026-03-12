"""
Standalone script to initialize the database schema.
Run with: python -m scripts.init_db
"""

import asyncio

from src.config import get_settings
from src.dependencies import init_services, shutdown_services
from src.storage.init_db import create_tables, ensure_qdrant_collection


async def main():
    settings = get_settings()
    await init_services(settings)

    print("Creating database tables...")
    await create_tables()
    print("Tables created.")

    print("Ensuring Qdrant collection exists...")
    await ensure_qdrant_collection(settings)
    print("Qdrant collection ready.")

    await shutdown_services()
    print("Database initialization complete.")


if __name__ == "__main__":
    asyncio.run(main())
