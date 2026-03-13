"""
Evaluation API endpoint.
Runs a batch of questions through the pipeline and returns quality metrics.
"""

import logging

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.schemas.eval import EvalRequest, EvalResponse
from src.config import get_settings
from src.dependencies import get_db, get_openai, get_qdrant, get_redis
from src.evaluation.evaluator import PipelineEvaluator

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/eval", response_model=EvalResponse)
async def evaluate_pipeline(
    request: EvalRequest,
    db: AsyncSession = Depends(get_db),
):
    """
    Run evaluation across a set of questions.
    Returns per-question scores and aggregate metrics.
    """
    settings = get_settings()

    evaluator = PipelineEvaluator(
        db=db,
        qdrant=get_qdrant(),
        openai_client=get_openai(),
        redis_client=get_redis(),
        settings=settings,
    )

    try:
        return await evaluator.evaluate(request.questions)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Evaluation failed: %s", exc)
        raise HTTPException(
            status_code=500,
            detail="Evaluation failed. Please check logs for details.",
        ) from exc
