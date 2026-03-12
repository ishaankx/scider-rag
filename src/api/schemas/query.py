"""Request and response schemas for the query API."""

from pydantic import BaseModel, Field, field_validator

from src.security.sanitizer import check_sql_injection, sanitize_text


class QueryRequest(BaseModel):
    """Incoming research question."""

    question: str = Field(
        ...,
        min_length=3,
        max_length=2000,
        description="The research question to answer.",
        examples=["How does the transformer attention mechanism work?"],
    )
    filters: dict | None = Field(
        default=None,
        description="Optional filters: source_type, date_range, etc.",
    )
    max_sources: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Maximum number of sources to return.",
    )
    check_grounding: bool = Field(
        default=False,
        description=(
            "Run hallucination detection on the answer. "
            "Adds one LLM call (~1-2s) but flags unsupported claims."
        ),
    )

    @field_validator("question")
    @classmethod
    def clean_question(cls, v: str) -> str:
        cleaned = sanitize_text(v.strip())
        if check_sql_injection(cleaned):
            raise ValueError("Query contains disallowed patterns.")
        return cleaned


class SourceReference(BaseModel):
    """A single source cited in the answer."""

    document_title: str
    chunk_content: str
    page_number: int | None = None
    relevance_score: float
    document_id: str


class LatencyBreakdown(BaseModel):
    """Timing for each pipeline stage in milliseconds."""

    retrieval_ms: float
    reasoning_ms: float
    total_ms: float


class GroundingResult(BaseModel):
    """Result of hallucination detection on the answer."""

    supported_count: int
    unsupported_count: int
    partial_count: int
    confidence: float = Field(ge=0.0, le=1.0)
    flags: list[str]


class QueryResponse(BaseModel):
    """Full response to a research question."""

    answer: str
    sources: list[SourceReference]
    latency: LatencyBreakdown
    confidence: float = Field(ge=0.0, le=1.0)
    request_id: str
    grounding: GroundingResult | None = Field(
        default=None,
        description="Hallucination detection results. Present only when check_grounding=True.",
    )
