"""Request and response schemas for the evaluation API."""

from pydantic import BaseModel, Field


class EvalQuestion(BaseModel):
    """A single evaluation question with optional expected answer."""

    question: str
    expected_answer: str | None = None
    expected_source_keywords: list[str] = Field(default_factory=list)


class EvalRequest(BaseModel):
    """Request to run evaluation on a set of questions."""

    questions: list[EvalQuestion] = Field(
        ..., min_length=1, max_length=100
    )


class EvalQuestionResult(BaseModel):
    """Evaluation result for a single question."""

    question: str
    answer: str
    expected_answer: str | None
    relevance_score: float
    correctness_score: float
    confidence: float
    latency_ms: float
    sources_found: int
    hallucination_flags: list[str]


class EvalSummary(BaseModel):
    """Aggregate metrics across all evaluation questions."""

    total_questions: int
    avg_relevance: float
    avg_correctness: float
    avg_confidence: float
    avg_latency_ms: float
    hallucination_rate: float
    p95_latency_ms: float


class EvalResponse(BaseModel):
    """Full evaluation report."""

    summary: EvalSummary
    results: list[EvalQuestionResult]
