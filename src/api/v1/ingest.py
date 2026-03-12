"""
Ingestion API endpoints.
Handles file uploads, triggers the ingestion pipeline, and returns results.
"""

from fastapi import APIRouter, Depends, HTTPException, UploadFile
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.schemas.ingest import DocumentListResponse, IngestResponse, IngestStatusResponse
from src.config import get_settings
from src.dependencies import get_db, get_openai, get_qdrant, get_redis
from src.ingestion.pipeline import IngestionPipeline
from src.security.sanitizer import sanitize_filename
from src.storage.models import Document

router = APIRouter()

ALLOWED_CONTENT_TYPES = {
    "application/pdf": "pdf",
    "text/csv": "csv",
    "application/json": "json",
    "text/plain": "txt",
}

ALLOWED_EXTENSIONS = {".pdf", ".csv", ".json", ".txt"}


def _validate_extension(filename: str) -> str:
    """Validate file extension and return the source type."""
    ext = "." + filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{ext}'. Allowed: {ALLOWED_EXTENSIONS}",
        )
    return ext.lstrip(".")


@router.post("/ingest", response_model=IngestResponse)
async def ingest_document(
    file: UploadFile,
    db: AsyncSession = Depends(get_db),
):
    """
    Upload and ingest a scientific document.
    Supported formats: PDF, CSV, JSON, TXT.
    """
    settings = get_settings()

    # Validate filename
    if not file.filename:
        raise HTTPException(status_code=400, detail="Filename is required.")

    safe_name = sanitize_filename(file.filename)
    source_type = _validate_extension(safe_name)

    # Read file content with size check
    content_bytes = await file.read()
    if len(content_bytes) == 0:
        raise HTTPException(status_code=400, detail="Empty file.")
    if len(content_bytes) > settings.max_file_size_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Maximum size is {settings.max_file_size_mb}MB.",
        )

    # Run ingestion pipeline
    pipeline = IngestionPipeline(
        db=db,
        qdrant=get_qdrant(),
        openai_client=get_openai(),
        redis_client=get_redis(),
        settings=settings,
    )

    try:
        result = await pipeline.ingest(
            file_name=safe_name,
            source_type=source_type,
            content_bytes=content_bytes,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    return IngestResponse(
        document_id=str(result["document_id"]),
        title=result["title"],
        source_type=source_type,
        chunks_created=result["chunks_created"],
        entities_extracted=result["entities_extracted"],
    )


@router.get("/documents", response_model=DocumentListResponse)
async def list_documents(
    db: AsyncSession = Depends(get_db),
):
    """List all ingested documents."""
    result = await db.execute(
        select(Document).order_by(Document.created_at.desc())
    )
    docs = result.scalars().all()

    return DocumentListResponse(
        documents=[
            IngestStatusResponse(
                document_id=str(d.id),
                title=d.title,
                source_type=d.source_type,
                chunk_count=d.chunk_count,
                created_at=d.created_at.isoformat(),
            )
            for d in docs
        ],
        total=len(docs),
    )
