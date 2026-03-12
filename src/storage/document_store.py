"""
PostgreSQL document and chunk storage operations.
Handles CRUD for documents, chunks, and full-text keyword search.
"""

import hashlib
import uuid

from sqlalchemy import func, select, text
from sqlalchemy.ext.asyncio import AsyncSession

from src.storage.models import Chunk, Document


class DocumentStore:
    """Manages documents and chunks in PostgreSQL."""

    def __init__(self, db: AsyncSession):
        self._db = db

    async def get_document_by_hash(self, content_hash: str) -> Document | None:
        """Find an existing document by its content hash (deduplication)."""
        result = await self._db.execute(
            select(Document).where(Document.content_hash == content_hash)
        )
        return result.scalar_one_or_none()

    async def create_document(
        self,
        title: str,
        source_type: str,
        file_name: str,
        content_bytes: bytes,
        metadata: dict | None = None,
    ) -> Document:
        """
        Create a new document record.
        Returns existing document if content hash matches (dedup).
        """
        content_hash = hashlib.sha256(content_bytes).hexdigest()

        # Check for duplicate
        existing = await self.get_document_by_hash(content_hash)
        if existing:
            raise ValueError(
                f"Document already ingested (id={existing.id}, title='{existing.title}'). "
                "Duplicate content detected."
            )

        doc = Document(
            id=uuid.uuid4(),
            title=title,
            source_type=source_type,
            file_name=file_name,
            content_hash=content_hash,
            metadata_=metadata or {},
        )
        self._db.add(doc)
        await self._db.flush()
        return doc

    async def add_chunks(
        self,
        document_id: uuid.UUID,
        chunks: list[dict],
    ) -> list[Chunk]:
        """
        Bulk insert chunks for a document.
        Each chunk dict: {content, chunk_index, token_count, metadata, embedding_id}
        """
        chunk_objects = []
        for chunk_data in chunks:
            chunk = Chunk(
                id=uuid.uuid4(),
                document_id=document_id,
                content=chunk_data["content"],
                chunk_index=chunk_data["chunk_index"],
                token_count=chunk_data.get("token_count", 0),
                embedding_id=chunk_data.get("embedding_id"),
                metadata_=chunk_data.get("metadata", {}),
            )
            self._db.add(chunk)
            chunk_objects.append(chunk)

        # Update document chunk count
        result = await self._db.execute(
            select(Document).where(Document.id == document_id)
        )
        doc = result.scalar_one()
        doc.chunk_count = len(chunks)

        await self._db.flush()
        return chunk_objects

    async def keyword_search(
        self, query: str, limit: int = 10
    ) -> list[dict]:
        """
        Full-text keyword search across chunk content using PostgreSQL.
        Uses plainto_tsquery for safe query parsing (no injection risk).
        """
        sql = text("""
            SELECT
                c.id AS chunk_id,
                c.content,
                c.metadata AS chunk_metadata,
                d.id AS document_id,
                d.title AS document_title,
                d.source_type,
                ts_rank(
                    to_tsvector('english', c.content),
                    plainto_tsquery('english', :query)
                ) AS rank
            FROM chunks c
            JOIN documents d ON c.document_id = d.id
            WHERE to_tsvector('english', c.content) @@ plainto_tsquery('english', :query)
            ORDER BY rank DESC
            LIMIT :limit
        """)

        result = await self._db.execute(sql, {"query": query, "limit": limit})
        rows = result.mappings().all()

        return [
            {
                "chunk_id": str(row["chunk_id"]),
                "content": row["content"],
                "chunk_metadata": row["chunk_metadata"],
                "document_id": str(row["document_id"]),
                "document_title": row["document_title"],
                "source_type": row["source_type"],
                "relevance_score": float(row["rank"]),
            }
            for row in rows
        ]

    async def get_chunks_by_document(self, document_id: uuid.UUID) -> list[Chunk]:
        """Retrieve all chunks for a document, ordered by index."""
        result = await self._db.execute(
            select(Chunk)
            .where(Chunk.document_id == document_id)
            .order_by(Chunk.chunk_index)
        )
        return list(result.scalars().all())
