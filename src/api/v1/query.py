"""
Query API endpoint.
Accepts a research question, runs the multi-agent RAG pipeline,
and returns a grounded answer with sources and latency breakdown.
"""

import time

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from src.agents.orchestrator import AgentOrchestrator
from src.api.middleware.request_id import request_id_ctx
from src.api.schemas.query import GroundingResult, LatencyBreakdown, QueryRequest, QueryResponse, SourceReference
from src.config import get_settings
from src.dependencies import get_db, get_openai, get_qdrant, get_redis

router = APIRouter()


@router.post("/query", response_model=QueryResponse)
async def query_pipeline(
    request: QueryRequest,
    db: AsyncSession = Depends(get_db),
):
    """
    Submit a research question to the multi-agent RAG pipeline.
    Returns an answer grounded in ingested sources with full attribution.
    """
    start_time = time.perf_counter()
    settings = get_settings()

    orchestrator = AgentOrchestrator(
        db=db,
        qdrant=get_qdrant(),
        openai_client=get_openai(),
        redis_client=get_redis(),
        settings=settings,
    )

    result = await orchestrator.run(
        question=request.question,
        filters=request.filters,
        max_sources=request.max_sources,
        check_grounding=request.check_grounding,
    )

    total_ms = (time.perf_counter() - start_time) * 1000

    sources = [
        SourceReference(
            document_title=s["document_title"],
            chunk_content=s["chunk_content"],
            page_number=s.get("page_number"),
            relevance_score=s["relevance_score"],
            document_id=s["document_id"],
        )
        for s in result["sources"]
    ]

    grounding = None
    if result.get("grounding"):
        g = result["grounding"]
        grounding = GroundingResult(
            supported_count=g["supported_count"],
            unsupported_count=g["unsupported_count"],
            partial_count=g["partial_count"],
            confidence=g["confidence"],
            flags=g["flags"],
        )

    return QueryResponse(
        answer=result["answer"],
        sources=sources,
        latency=LatencyBreakdown(
            retrieval_ms=result["retrieval_ms"],
            reasoning_ms=result["reasoning_ms"],
            total_ms=total_ms,
        ),
        confidence=result["confidence"],
        request_id=request_id_ctx.get(""),
        grounding=grounding,
    )
