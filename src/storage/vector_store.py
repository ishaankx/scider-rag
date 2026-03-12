"""
Qdrant vector store operations.
Handles upserting embeddings and performing similarity searches.
"""

import uuid

from qdrant_client import AsyncQdrantClient
from qdrant_client.models import FieldCondition, Filter, MatchValue, PointStruct, ScoredPoint


class VectorStore:
    """Thin wrapper around Qdrant for vector operations."""

    def __init__(self, client: AsyncQdrantClient, collection_name: str):
        self._client = client
        self._collection = collection_name

    async def upsert_embeddings(
        self,
        embeddings: list[list[float]],
        payloads: list[dict],
        ids: list[str] | None = None,
    ) -> list[str]:
        """
        Insert or update vectors with metadata payloads.
        Returns the list of point IDs.
        """
        if not embeddings:
            return []

        point_ids = ids or [str(uuid.uuid4()) for _ in embeddings]

        points = [
            PointStruct(id=pid, vector=emb, payload=payload)
            for pid, emb, payload in zip(point_ids, embeddings, payloads)
        ]

        # Upsert in batches of 100 to avoid oversized requests
        batch_size = 100
        for i in range(0, len(points), batch_size):
            batch = points[i : i + batch_size]
            await self._client.upsert(
                collection_name=self._collection,
                points=batch,
            )

        return point_ids

    async def search(
        self,
        query_embedding: list[float],
        top_k: int = 10,
        source_type: str | None = None,
        score_threshold: float = 0.0,
    ) -> list[ScoredPoint]:
        """
        Perform similarity search against stored vectors.
        Optionally filter by source_type.
        """
        query_filter = None
        if source_type:
            query_filter = Filter(
                must=[
                    FieldCondition(
                        key="source_type",
                        match=MatchValue(value=source_type),
                    )
                ]
            )

        results = await self._client.search(
            collection_name=self._collection,
            query_vector=query_embedding,
            limit=top_k,
            query_filter=query_filter,
            score_threshold=score_threshold,
        )

        return results

    async def delete_by_document_id(self, document_id: str) -> None:
        """Remove all vectors associated with a document."""
        await self._client.delete(
            collection_name=self._collection,
            points_selector=Filter(
                must=[
                    FieldCondition(
                        key="document_id",
                        match=MatchValue(value=document_id),
                    )
                ]
            ),
        )
