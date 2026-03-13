"""Request and response schemas for the ingestion API."""

from pydantic import BaseModel, Field


class IngestResponse(BaseModel):
    """Result of a document ingestion."""

    document_id: str
    title: str
    source_type: str
    chunks_created: int
    entities_extracted: int
    ocr_pages: int = 0
    images_analyzed: int = 0
    message: str = "Document ingested successfully."


class IngestStatusResponse(BaseModel):
    """Status of an ingested document."""

    document_id: str
    title: str
    source_type: str
    chunk_count: int
    ocr_pages: int = 0
    images_analyzed: int = 0
    created_at: str


class DocumentListResponse(BaseModel):
    """List of all ingested documents."""

    documents: list[IngestStatusResponse]
    total: int
