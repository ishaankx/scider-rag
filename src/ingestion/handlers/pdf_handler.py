"""PDF file handler using PyMuPDF."""

import io
import logging

import fitz  # PyMuPDF

from src.ingestion.handlers.base import BaseHandler, ExtractedDocument

logger = logging.getLogger(__name__)


class PdfHandler(BaseHandler):
    """Extracts text, metadata, and structure from PDF files."""

    def can_handle(self, source_type: str) -> bool:
        return source_type == "pdf"

    async def extract(self, content_bytes: bytes, file_name: str) -> ExtractedDocument:
        try:
            doc = fitz.open(stream=content_bytes, filetype="pdf")
        except Exception as exc:
            raise ValueError(f"Failed to open PDF '{file_name}': {exc}") from exc

        if doc.is_encrypted:
            raise ValueError(f"PDF '{file_name}' is encrypted and cannot be processed.")

        pages_text = []
        page_metadata = []

        page_count = len(doc)
        for page_num in range(page_count):
            page = doc[page_num]
            text = page.get_text("text")

            # Strip null bytes — PostgreSQL rejects them, common in old LaTeX PDFs
            text = text.replace("\x00", "")
            if text.strip():
                pages_text.append(text)
                page_metadata.append({
                    "page_number": page_num + 1,
                    "char_count": len(text),
                })

        doc.close()

        if not pages_text:
            raise ValueError(
                f"PDF '{file_name}' contains no extractable text. "
                "It may be an image-only PDF."
            )

        # Attempt to extract title from first page (first non-empty line)
        title = _extract_title(pages_text[0], file_name)

        full_text = "\n\n".join(pages_text)

        return ExtractedDocument(
            title=title,
            raw_text=full_text,
            metadata={
                "file_name": file_name,
                "page_count": page_count,
                "pages_with_text": len(pages_text),
                "total_chars": len(full_text),
                "pages": page_metadata,
            },
        )


def _extract_title(first_page_text: str, fallback: str) -> str:
    """
    Heuristic: title is typically the first non-empty line on page 1.
    Falls back to the filename without extension.
    """
    for line in first_page_text.split("\n"):
        stripped = line.strip()
        if len(stripped) > 5 and not stripped.startswith("http"):
            # Truncate very long lines (probably not a title)
            return stripped[:200]
    # Fallback to filename without extension
    return fallback.rsplit(".", 1)[0]
