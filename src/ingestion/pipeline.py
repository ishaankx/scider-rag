"""
Ingestion pipeline orchestrator.
Coordinates: file parsing → chunking → embedding → vector storage
             → entity extraction → graph storage → cache invalidation.
"""

import json
import logging
import uuid

import redis.asyncio as redis
from openai import AsyncOpenAI
from qdrant_client import AsyncQdrantClient
from sqlalchemy.ext.asyncio import AsyncSession

from src.config import Settings
from src.ingestion.chunker import chunk_text
from src.ingestion.embeddings import EmbeddingService
from src.ingestion.handlers.base import BaseHandler, ExtractedDocument
from src.ingestion.handlers.csv_handler import CsvHandler, JsonHandler
from src.ingestion.handlers.pdf_handler import PdfHandler
from src.storage.cache import QueryCache
from src.storage.document_store import DocumentStore
from src.storage.graph_store import GraphStore
from src.storage.models import Entity
from src.storage.vector_store import VectorStore

logger = logging.getLogger(__name__)


class IngestionPipeline:
    """
    Full ingestion pipeline from raw bytes to indexed, searchable data.
    Follows Single Responsibility: each step is delegated to a specialist.
    """

    def __init__(
        self,
        db: AsyncSession,
        qdrant: AsyncQdrantClient,
        openai_client: AsyncOpenAI,
        redis_client: redis.Redis,
        settings: Settings,
    ):
        self._db = db
        self._doc_store = DocumentStore(db)
        self._vector_store = VectorStore(qdrant, settings.qdrant_collection)
        self._graph_store = GraphStore(db)
        self._embedding_service = EmbeddingService(openai_client, settings)
        self._cache = QueryCache(redis_client)
        self._openai = openai_client
        self._settings = settings

        # Register all file handlers (Open/Closed: add new handlers here)
        self._handlers: list[BaseHandler] = [
            PdfHandler(openai_client=openai_client, settings=settings),
            CsvHandler(),
            JsonHandler(),
        ]

    def _get_handler(self, source_type: str) -> BaseHandler:
        """Find the handler that supports this source type."""
        for handler in self._handlers:
            if handler.can_handle(source_type):
                return handler
        raise ValueError(f"No handler registered for source type '{source_type}'.")

    async def ingest(
        self,
        file_name: str,
        source_type: str,
        content_bytes: bytes,
    ) -> dict:
        """
        Run the full ingestion pipeline:
        1. Parse file → ExtractedDocument
        2. Store document record in PostgreSQL
        3. Chunk text
        4. Generate embeddings
        5. Store vectors in Qdrant
        6. Extract entities and relationships
        7. Invalidate query cache
        """
        # Step 1: Parse
        handler = self._get_handler(source_type)
        extracted = await handler.extract(content_bytes, file_name)
        logger.info("Parsed '%s': %d chars", file_name, len(extracted.raw_text))

        # Step 2: Store document record (checks for duplicates)
        document = await self._doc_store.create_document(
            title=extracted.title,
            source_type=source_type,
            file_name=file_name,
            content_bytes=content_bytes,
            metadata=extracted.metadata,
        )

        # Step 3: Chunk (strip null bytes — PostgreSQL rejects \x00 in text columns)
        chunks = chunk_text(
            text=extracted.raw_text.replace("\x00", ""),
            chunk_size=self._settings.chunk_size,
            chunk_overlap=self._settings.chunk_overlap,
        )
        logger.info("Created %d chunks for '%s'", len(chunks), file_name)

        if not chunks:
            await self._db.commit()
            return {
                "document_id": document.id,
                "title": extracted.title,
                "chunks_created": 0,
                "entities_extracted": 0,
            }

        # Step 4: Generate embeddings
        chunk_texts = [c["content"] for c in chunks]
        embeddings = await self._embedding_service.embed_texts(chunk_texts)

        # Step 5: Store vectors in Qdrant
        payloads = [
            {
                "document_id": str(document.id),
                "document_title": extracted.title,
                "source_type": source_type,
                "chunk_index": c["chunk_index"],
                "content": c["content"],
                "page_number": c["metadata"].get("page_number"),
            }
            for c in chunks
        ]
        point_ids = await self._vector_store.upsert_embeddings(
            embeddings=embeddings,
            payloads=payloads,
        )

        # Update chunks with embedding IDs and store in PostgreSQL
        for i, chunk in enumerate(chunks):
            chunk["embedding_id"] = point_ids[i]

        await self._doc_store.add_chunks(document.id, chunks)

        # Step 6: Commit document + chunks before entity extraction.
        # Entity extraction opens SELECT queries that trigger SQLAlchemy's
        # autoflush; committing first avoids flush conflicts on pending Chunk objects.
        await self._db.commit()

        # Step 7: Extract entities and relationships (best-effort — never blocks ingest)
        entity_count = 0
        try:
            entity_count = await self._extract_entities(
                extracted.raw_text, document.id
            )
            await self._db.commit()
        except Exception as exc:
            logger.warning(
                "Entity extraction failed for '%s' (non-fatal): %s", file_name, exc
            )
            await self._db.rollback()

        # Step 8: Invalidate cache (new data means old answers may be stale)
        await self._cache.invalidate_all()

        logger.info(
            "Ingestion complete for '%s': %d chunks, %d entities",
            file_name, len(chunks), entity_count,
        )

        return {
            "document_id": document.id,
            "title": extracted.title,
            "chunks_created": len(chunks),
            "entities_extracted": entity_count,
        }

    async def _extract_entities(
        self, text: str, document_id: uuid.UUID
    ) -> int:
        """
        Use LLM to extract named entities and relationships from text.
        Returns the number of entities extracted.
        """
        # Use a truncated sample for entity extraction (first ~3000 chars)
        sample = text[:3000]

        prompt = (
            "Extract named entities and relationships from the following scientific text.\n"
            "Return valid JSON with this exact structure:\n"
            '{"entities": [{"name": "...", "type": "..."}], '
            '"relationships": [{"source": "...", "target": "...", "type": "..."}]}\n\n'
            "Entity types (use whichever fit — this list is not exhaustive):\n"
            "  General:   person, organization, location, concept, method, dataset, publication\n"
            "  AI/ML:     model, algorithm, architecture, framework, benchmark, task\n"
            "  Biology:   gene, protein, compound, disease, organism, pathway\n"
            "  Physics:   particle, qubit, quantum_gate, material, phenomenon\n"
            "  Chemistry: molecule, reaction, catalyst, polymer, element\n\n"
            "Relationship types (use whichever fit — this list is not exhaustive):\n"
            "  authored_by, affiliated_with, uses, implements, outperforms, based_on,\n"
            "  part_of, related_to, applied_to, extends, causes, treats, interacts_with\n\n"
            f"Text:\n{sample}"
        )

        try:
            response = await self._openai.chat.completions.create(
                model=self._settings.llm_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=1024,
                response_format={"type": "json_object"},
            )

            result = json.loads(response.choices[0].message.content)
            entities_data = result.get("entities", [])
            relationships_data = result.get("relationships", [])

        except (json.JSONDecodeError, KeyError, IndexError) as exc:
            logger.warning("Entity extraction failed (parse error): %s", exc)
            return 0
        except Exception as exc:
            logger.warning("Entity extraction failed (API error): %s", exc)
            return 0

        # Store entities
        entity_map: dict[str, Entity] = {}
        for ent in entities_data:
            name = ent.get("name", "").strip()
            etype = ent.get("type", "concept").strip().lower()
            if not name:
                continue

            entity = await self._graph_store.upsert_entity(
                name=name,
                entity_type=etype,
                source_document_id=document_id,
            )
            entity_map[name.lower()] = entity

        # Store relationships
        for rel in relationships_data:
            src_name = rel.get("source", "").strip().lower()
            tgt_name = rel.get("target", "").strip().lower()
            rtype = rel.get("type", "related_to")

            if src_name in entity_map and tgt_name in entity_map:
                await self._graph_store.add_relationship(
                    source_entity_id=entity_map[src_name].id,
                    target_entity_id=entity_map[tgt_name].id,
                    relation_type=rtype,
                )

        return len(entity_map)
