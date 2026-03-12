"""
Pipeline evaluation framework.
Runs a batch of questions through the RAG pipeline and computes quality metrics.
"""

import logging
import time

import redis.asyncio as redis
from openai import AsyncOpenAI
from qdrant_client import AsyncQdrantClient
from sqlalchemy.ext.asyncio import AsyncSession

from src.agents.orchestrator import AgentOrchestrator
from src.api.schemas.eval import (
    EvalQuestionResult,
    EvalResponse,
    EvalSummary,
)
from src.config import Settings
from src.evaluation.hallucination import HallucinationDetector
from src.evaluation.metrics import compute_correctness

logger = logging.getLogger(__name__)


class PipelineEvaluator:
    """Evaluates the RAG pipeline across a set of test questions."""

    def __init__(
        self,
        db: AsyncSession,
        qdrant: AsyncQdrantClient,
        openai_client: AsyncOpenAI,
        redis_client: redis.Redis,
        settings: Settings,
    ):
        self._db = db
        self._qdrant = qdrant
        self._openai = openai_client
        self._redis = redis_client
        self._settings = settings
        self._hallucination_detector = HallucinationDetector(openai_client, settings)

    async def evaluate(self, questions: list) -> EvalResponse:
        """
        Run each question through the pipeline and collect metrics.
        Returns per-question results and aggregate summary.
        """
        results = []

        for q in questions:
            result = await self._evaluate_single(q)
            results.append(result)

        summary = self._compute_summary(results)
        return EvalResponse(summary=summary, results=results)

    async def _evaluate_single(self, question) -> EvalQuestionResult:
        """Evaluate a single question through the full pipeline."""
        start = time.perf_counter()

        orchestrator = AgentOrchestrator(
            db=self._db,
            qdrant=self._qdrant,
            openai_client=self._openai,
            redis_client=self._redis,
            settings=self._settings,
        )

        try:
            result = await orchestrator.run(
                question=question.question,
                max_sources=5,
            )

            answer = result.get("answer", "")
            sources = result.get("sources", [])
            latency_ms = (time.perf_counter() - start) * 1000

            # Run hallucination detection
            hallucination_result = await self._hallucination_detector.check(
                answer=answer,
                sources=sources,
            )

            # Compute correctness if expected answer provided
            correctness = 0.0
            if question.expected_answer:
                correctness = await compute_correctness(
                    answer=answer,
                    expected=question.expected_answer,
                    openai_client=self._openai,
                    settings=self._settings,
                )

            return EvalQuestionResult(
                question=question.question,
                answer=answer,
                expected_answer=question.expected_answer,
                relevance_score=result.get("confidence", 0.0),
                correctness_score=correctness,
                confidence=hallucination_result["confidence"],
                latency_ms=round(latency_ms, 1),
                sources_found=len(sources),
                hallucination_flags=hallucination_result["flags"],
            )

        except Exception as exc:
            logger.error("Evaluation failed for question '%s': %s", question.question[:50], exc)
            return EvalQuestionResult(
                question=question.question,
                answer=f"Error: {exc}",
                expected_answer=question.expected_answer,
                relevance_score=0.0,
                correctness_score=0.0,
                confidence=0.0,
                latency_ms=(time.perf_counter() - start) * 1000,
                sources_found=0,
                hallucination_flags=[f"Pipeline error: {exc}"],
            )

    def _compute_summary(self, results: list[EvalQuestionResult]) -> EvalSummary:
        """Aggregate metrics across all evaluation results."""
        n = len(results)
        if n == 0:
            return EvalSummary(
                total_questions=0,
                avg_relevance=0,
                avg_correctness=0,
                avg_confidence=0,
                avg_latency_ms=0,
                hallucination_rate=0,
                p95_latency_ms=0,
            )

        relevances = [r.relevance_score for r in results]
        correctness_scores = [r.correctness_score for r in results]
        confidences = [r.confidence for r in results]
        latencies = sorted([r.latency_ms for r in results])
        hallucination_count = sum(1 for r in results if r.hallucination_flags)

        p95_idx = int(n * 0.95)
        p95_latency = latencies[min(p95_idx, n - 1)]

        return EvalSummary(
            total_questions=n,
            avg_relevance=round(sum(relevances) / n, 3),
            avg_correctness=round(sum(correctness_scores) / n, 3),
            avg_confidence=round(sum(confidences) / n, 3),
            avg_latency_ms=round(sum(latencies) / n, 1),
            hallucination_rate=round(hallucination_count / n, 3),
            p95_latency_ms=round(p95_latency, 1),
        )
