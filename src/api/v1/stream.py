"""
Streaming query endpoint using Server-Sent Events (SSE).

Provides real-time visibility into the multi-agent reasoning pipeline —
the client sees each step (cache check, retrieval, tool calls, reasoning)
as it happens instead of waiting for the full answer.

Designed for IDE integration: an editor panel can render the reasoning
trace live while the user continues working.
"""

import json
import logging
import time

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from starlette.responses import StreamingResponse

from src.agents.orchestrator import AgentOrchestrator
from src.api.middleware.request_id import request_id_ctx
from src.api.schemas.query import QueryRequest
from src.config import get_settings
from src.dependencies import get_db, get_openai, get_qdrant, get_redis

logger = logging.getLogger(__name__)

router = APIRouter()


def format_sse_event(event_type: str, data) -> str:
    """Format a Server-Sent Event string per the W3C spec."""
    json_data = json.dumps(data, default=str)
    return f"event: {event_type}\ndata: {json_data}\n\n"


@router.post("/query/stream")
async def query_stream(
    request: QueryRequest,
    db: AsyncSession = Depends(get_db),
):
    """
    Stream the query pipeline as Server-Sent Events.

    Emits the same data as POST /query, but incrementally:

    | Event    | Payload                                | When emitted               |
    |----------|----------------------------------------|----------------------------|
    | status   | {step, message}                        | Each pipeline step         |
    | sources  | [{document_title, relevance_score, …}] | After retrieval completes  |
    | answer   | {text}                                 | After reasoning completes  |
    | done     | Full result (same shape as /query)     | Pipeline finished          |
    | error    | {message, request_id}                  | On failure                 |
    """
    settings = get_settings()
    request_id = request_id_ctx.get("")

    orchestrator = AgentOrchestrator(
        db=db,
        qdrant=get_qdrant(),
        openai_client=get_openai(),
        redis_client=get_redis(),
        settings=settings,
    )

    async def event_generator():
        start_time = time.perf_counter()

        try:
            async for event_type, data in orchestrator.run_stream(
                question=request.question,
                filters=request.filters,
                max_sources=request.max_sources,
                check_grounding=request.check_grounding,
            ):
                if event_type == "done":
                    # Enrich final event with wall-clock timing and request ID
                    total_ms = (time.perf_counter() - start_time) * 1000
                    data["latency"] = {
                        "retrieval_ms": data.pop("retrieval_ms", 0),
                        "reasoning_ms": data.pop("reasoning_ms", 0),
                        "total_ms": total_ms,
                    }
                    data["request_id"] = request_id
                    data.pop("tools_used", None)
                    yield format_sse_event("done", data)
                else:
                    yield format_sse_event(event_type, data)

        except Exception as exc:
            logger.exception("Stream error for query: %s", request.question[:60])
            yield format_sse_event("error", {
                "message": str(exc),
                "request_id": request_id,
            })

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # Prevent nginx/proxy buffering
        },
    )
