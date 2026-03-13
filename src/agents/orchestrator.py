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

            # Total pipeline timeout: agents make multiple LLM calls,
            # so allow 3x the per-call timeout as the overall budget.
            total_timeout = self._settings.llm_timeout_seconds * 3

            try:
                if not lock_acquired:
                    # Another request holds the lock — wait for it to populate the cache.
                    # Retry a few times before falling through to run our own pipeline.
                    result = None
                    for _attempt in range(3):
                        await asyncio.sleep(1.0)
                        cached = await self._cache.get(question, filters)
                        if cached:
                            logger.info("Cache populated by another request for: %s", question[:60])
                            result = cached
                            break

                    if result is None:
                        # Lock holder may have failed — run pipeline ourselves
                        result = await asyncio.wait_for(
                            self._run_pipeline(question, filters, max_sources),
                            timeout=total_timeout,
                        )
                        await self._cache.set(question, filters, result)
                else:
                    result = await asyncio.wait_for(
                        self._run_pipeline(question, filters, max_sources),
                        timeout=total_timeout,
                    )
                    await self._cache.set(question, filters, result)
            except asyncio.TimeoutError:
                logger.error("Pipeline timed out after %ds for: %s", total_timeout, question[:60])
                result = {
                    "answer": "The query timed out. Please try a simpler question or retry later.",
                    "sources": [],
                    "retrieval_ms": 0,
                    "reasoning_ms": 0,
                    "confidence": 0.0,
                    "tools_used": [],
                }
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

    async def run_stream(
        self,
        question: str,
        filters: dict | None = None,
        max_sources: int = 5,
        check_grounding: bool = False,
    ):
        """
        Streaming variant of run(). Yields (event_type, data_dict) tuples.
        Provides real-time visibility into every pipeline step.
        Final yield is ("done", full_result_dict).
        """
        # Step 0: Cache check
        yield ("status", {"step": "cache_check", "message": "Checking query cache..."})
        cached = await self._cache.get(question, filters)

        if cached:
            yield ("status", {"step": "cache_check", "message": "Cache hit — returning cached result"})
            result = cached
        else:
            yield ("status", {"step": "cache_check", "message": "Cache miss — running full pipeline"})
            lock_acquired = await self._cache.acquire_lock(question)

            try:
                if not lock_acquired:
                    # Another request holds the lock — wait for cache to be populated
                    result = None
                    for _attempt in range(3):
                        await asyncio.sleep(1.0)
                        cached = await self._cache.get(question, filters)
                        if cached:
                            yield ("status", {"step": "cache_check", "message": "Cache populated by another request"})
                            result = cached
                            break
                else:
                    result = None

                if result is None:
                    pipeline_result = {}
                    async for event in self._run_pipeline_stream(question, filters, max_sources):
                        if event[0] == "_result":
                            pipeline_result = event[1]
                        else:
                            yield event
                    result = pipeline_result
                    await self._cache.set(question, filters, result)
            finally:
                if lock_acquired:
                    await self._cache.release_lock(question)

        # Grounding check (never cached — always fresh when requested)
        result = {**result, "grounding": None}
        if check_grounding:
            yield ("status", {"step": "grounding", "message": "Running hallucination detection..."})
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
            yield ("status", {
                "step": "grounding",
                "message": f"Grounding complete — {grounding['confidence']:.0%} claims supported",
            })

        yield ("done", result)

    async def _run_pipeline_stream(
        self,
        question: str,
        filters: dict | None,
        max_sources: int,
    ):
        """Streaming variant of _run_pipeline. Yields events then final _result."""
        context = AgentContext(
            question=question,
            filters=filters,
            max_sources=max_sources,
        )

        # Retrieval
        yield ("status", {"step": "retrieval", "message": "Planning search strategy..."})
        async with _llm_semaphore:
            retrieval_result = await self._retrieval_agent.execute(context)

        n_chunks = len(context.retrieved_chunks)
        best_score = max(
            (c.get("relevance_score", 0) for c in context.retrieved_chunks), default=0
        )
        yield ("status", {
            "step": "retrieval",
            "message": f"Found {n_chunks} chunks (best score: {best_score:.3f})",
        })

        # Emit sources early so the client can render them immediately
        sources = self._format_sources(context.retrieved_chunks[:max_sources])
        yield ("sources", sources)

        # Reasoning (streaming — emits tool_call and status events)
        reasoning_result = None
        async with _llm_semaphore:
            async for event in self._reasoning_agent.execute_stream(context):
                if event[0] == "_result":
                    reasoning_result = event[1]
                else:
                    yield event

        confidence = self._compute_confidence(context.retrieved_chunks)

        # Emit the answer text as its own event for progressive rendering
        yield ("answer", {"text": reasoning_result.output})

        # Internal result — consumed by run_stream, not forwarded to client
        yield ("_result", {
            "answer": reasoning_result.output,
            "sources": sources,
            "retrieval_ms": retrieval_result.latency_ms,
            "reasoning_ms": reasoning_result.latency_ms,
            "confidence": confidence,
            "tools_used": reasoning_result.tool_calls_made,
        })

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
