"""
Evaluation API endpoint.
Runs a batch of questions through the pipeline and returns quality metrics.
"""

import time

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.schemas.eval import EvalRequest, EvalResponse
from src.config import get_settings
from src.dependencies import get_db, get_openai, get_qdrant, get_redis
from src.evaluation.evaluator import PipelineEvaluator

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

    return await evaluator.evaluate(request.questions)
