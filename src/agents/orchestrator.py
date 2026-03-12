"""
Agent Orchestrator.
Coordinates the full query pipeline: cache check → retrieval → reasoning → response.
Handles concurrency isolation, caching, and latency tracking.
"""

import asyncio
import logging
import time

import redis.asyncio as redis
from openai import AsyncOpenAI
from qdrant_client import AsyncQdrantClient
from sqlalchemy.ext.asyncio import AsyncSession

from src.agents.base import AgentContext
from src.agents.reasoning import ReasoningAgent
from src.agents.retrieval import RetrievalAgent
from src.agents.tools.calculator import CalculatorTool
from src.agents.tools.code_executor import CodeExecutorTool
from src.agents.tools.graph_traversal import GraphTraversalTool
from src.config import Settings
from src.evaluation.hallucination import HallucinationDetector
from src.ingestion.embeddings import EmbeddingService
from src.storage.cache import QueryCache
from src.storage.document_store import DocumentStore
from src.storage.graph_store import GraphStore
from src.storage.vector_store import VectorStore

logger = logging.getLogger(__name__)

# Semaphore to limit concurrent LLM calls across all requests
_llm_semaphore = asyncio.Semaphore(10)


class AgentOrchestrator:
    """
    Runs the full multi-agent RAG pipeline for a single query.
    Each query gets its own context — no shared mutable state.
    """

    def __init__(
        self,
        db: AsyncSession,
        qdrant: AsyncQdrantClient,
        openai_client: AsyncOpenAI,
        redis_client: redis.Redis,
        settings: Settings,
    ):
        self._settings = settings
        self._cache = QueryCache(redis_client)
        self._openai = openai_client
        self._hallucination_detector = HallucinationDetector(openai_client, settings)

        # Build storage layer
        vector_store = VectorStore(qdrant, settings.qdrant_collection)
        doc_store = DocumentStore(db)
        graph_store = GraphStore(db)
        embedding_service = EmbeddingService(openai_client, settings)

        # Build agents
        self._retrieval_agent = RetrievalAgent(
            openai_client=openai_client,
            vector_store=vector_store,
            document_store=doc_store,
            embedding_service=embedding_service,
            settings=settings,
        )

        # Build tools for reasoning agent
        tools = [
            CalculatorTool(),
            GraphTraversalTool(graph_store),
            CodeExecutorTool(settings),
        ]

        self._reasoning_agent = ReasoningAgent(
            openai_client=openai_client,
            tools=tools,
            settings=settings,
        )

    async def run(
        self,
        question: str,
        filters: dict | None = None,
        max_sources: int = 5,
        check_grounding: bool = False,
    ) -> dict:
        """
        Execute the full pipeline:
        1. Check cache
        2. Run retrieval agent
        3. Run reasoning agent
        4. Cache result
        5. Return structured response
        """
        # Step 0: Check cache (answer/sources only — grounding is never cached)
        cached = await self._cache.get(question, filters)
        if cached:
            logger.info("Cache hit for query: %s", question[:60])
            result = cached
        else:
            # Acquire lock to prevent cache stampede on identical queries
            lock_acquired = await self._cache.acquire_lock(question)

            try:
                # Double-check cache after acquiring lock
                if not lock_acquired:
                    cached = await self._cache.get(question, filters)
                    if cached:
                        result = cached
                    else:
                        result = await self._run_pipeline(question, filters, max_sources)
                        await self._cache.set(question, filters, result)
                else:
                    result = await self._run_pipeline(question, filters, max_sources)
                    await self._cache.set(question, filters, result)
            finally:
                if lock_acquired:
                    await self._cache.release_lock(question)

        # Grounding check always runs when requested — never served from cache
        # (a cached answer may have been produced without this check)
        result = {**result, "grounding": None}
        if check_grounding:
            grounding = await self._hallucination_detector.check(
                answer=result["answer"],
                sources=result["sources"],
            )
            result["grounding"] = {
                "supported_count": grounding["supported_count"],
                "unsupported_count": grounding["unsupported_count"],
                "partial_count": grounding.get("partial_count", 0),
                "confidence": grounding["confidence"],
                "flags": grounding["flags"],
            }

        return result

    async def _run_pipeline(
        self,
        question: str,
        filters: dict | None,
        max_sources: int,
    ) -> dict:
        """Run retrieval + reasoning pipeline and return raw result dict."""
        # Step 1: Create fresh context (isolation — no shared mutable state)
        context = AgentContext(
            question=question,
            filters=filters,
            max_sources=max_sources,
        )

        # Step 2: Retrieval (semaphore limits concurrent LLM calls)
        async with _llm_semaphore:
            retrieval_result = await self._retrieval_agent.execute(context)

        # Step 3: Reasoning (ReAct loop with tools)
        async with _llm_semaphore:
            reasoning_result = await self._reasoning_agent.execute(context)

        sources = self._format_sources(context.retrieved_chunks[:max_sources])
        confidence = self._compute_confidence(context.retrieved_chunks)

        return {
            "answer": reasoning_result.output,
            "sources": sources,
            "retrieval_ms": retrieval_result.latency_ms,
            "reasoning_ms": reasoning_result.latency_ms,
            "confidence": confidence,
            "tools_used": reasoning_result.tool_calls_made,
        }

    def _format_sources(self, chunks: list[dict]) -> list[dict]:
        """Convert internal chunk format to API response format."""
        sources = []
        for chunk in chunks:
            sources.append({
                "document_title": chunk.get("document_title", "Unknown"),
                "chunk_content": chunk.get("content", "")[:500],
                "page_number": chunk.get("page_number"),
                "relevance_score": round(chunk.get("relevance_score", 0), 4),
                "document_id": chunk.get("document_id", ""),
            })
        return sources

    def _compute_confidence(self, chunks: list[dict]) -> float:
        """
        Simple confidence heuristic based on retrieval scores.
        Higher average relevance = higher confidence.
        """
        if not chunks:
            return 0.1  # Very low confidence when no sources found

        scores = [c.get("relevance_score", 0) for c in chunks[:5]]
        avg_score = sum(scores) / len(scores) if scores else 0

        # Normalize to 0-1 range (cosine similarity is already 0-1)
        return round(min(max(avg_score, 0.1), 1.0), 3)
