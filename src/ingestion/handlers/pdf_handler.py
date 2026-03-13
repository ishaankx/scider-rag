"""PDF file handler using PyMuPDF with vision-based OCR fallback."""

import logging

import fitz  # PyMuPDF

from src.ingestion.handlers.base import BaseHandler, ExtractedDocument

logger = logging.getLogger(__name__)

# Minimum image dimensions (pixels) to bother analyzing
MIN_IMAGE_WIDTH = 100
MIN_IMAGE_HEIGHT = 100

# Minimum image area to filter out tiny icons/bullets
MIN_IMAGE_AREA = 15_000


class PdfHandler(BaseHandler):
    """
    Extracts text, metadata, and structure from PDF files.

    Supports three extraction modes per page:
    - text_extraction: standard selectable text (fast, free)
    - ocr_vision: scanned/image-only pages via GPT-4o vision (fallback)
    - image_analysis: embedded figures/charts described via vision

    When no OpenAI client is provided, falls back to text-only extraction
    (original behaviour — maintains backward compatibility).
    """

    def __init__(self, openai_client=None, settings=None):
        self._openai = openai_client
        self._settings = settings

    def can_handle(self, source_type: str) -> bool:
        return source_type == "pdf"

    async def extract(self, content_bytes: bytes, file_name: str) -> ExtractedDocument:
        try:
            doc = fitz.open(stream=content_bytes, filetype="pdf")
        except Exception as exc:
            raise ValueError(f"Failed to open PDF '{file_name}': {exc}") from exc

        if doc.is_encrypted:
            raise ValueError(f"PDF '{file_name}' is encrypted and cannot be processed.")

        ocr_enabled = (
            self._openai is not None
            and self._settings is not None
            and self._settings.enable_ocr
        )
        image_analysis_enabled = (
            self._openai is not None
            and self._settings is not None
            and self._settings.enable_image_analysis
        )

        pages_text = []
        page_metadata = []
        ocr_page_count = 0
        images_analyzed = 0
        page_count = len(doc)

        max_ocr = self._settings.max_ocr_pages if self._settings else 50

        for page_num in range(page_count):
            page = doc[page_num]
            text = page.get_text("text")

            # Strip null bytes — PostgreSQL rejects them, common in old LaTeX PDFs
            text = text.replace("\x00", "")
            extraction_method = "text_extraction"

            if text.strip():
                # Page has selectable text — use it directly
                pass
            elif ocr_enabled and ocr_page_count < max_ocr:
                # No selectable text — render page as image and OCR via vision
                text = await self._ocr_page(page, page_num + 1)
                extraction_method = "ocr_vision"
                ocr_page_count += 1
            else:
                # No text and OCR disabled/exhausted — skip page
                continue

            if not text.strip():
                continue

            # Extract and analyze embedded images on this page
            image_descriptions = []
            if image_analysis_enabled:
                image_descriptions = await self._analyze_page_images(
                    doc, page, page_num + 1, text
                )
                images_analyzed += len(image_descriptions)

            # Append image descriptions to page text
            if image_descriptions:
                text += "\n\n" + "\n\n".join(
                    f"[Figure on page {page_num + 1}]: {desc}"
                    for desc in image_descriptions
                )

            pages_text.append(text)
            page_metadata.append({
                "page_number": page_num + 1,
                "char_count": len(text),
                "extraction_method": extraction_method,
                "images_analyzed": len(image_descriptions),
            })

        doc.close()

        if not pages_text:
            raise ValueError(
                f"PDF '{file_name}' contains no extractable text. "
                "It may be an image-only PDF and OCR is disabled."
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
                "ocr_pages": ocr_page_count,
                "images_analyzed": images_analyzed,
                "pages": page_metadata,
            },
        )

    async def _ocr_page(self, page, page_number: int) -> str:
        """Render a PDF page as an image and extract text via vision OCR."""
        from src.ingestion.ocr import ocr_page_image

        # Render at 2x resolution for better OCR quality
        mat = fitz.Matrix(2.0, 2.0)
        pix = page.get_pixmap(matrix=mat)
        image_bytes = pix.tobytes("png")

        logger.info("OCR processing page %d (%dx%d px)", page_number, pix.width, pix.height)

        try:
            text = await ocr_page_image(
                openai_client=self._openai,
                image_bytes=image_bytes,
                model=self._settings.ocr_model,
                page_number=page_number,
            )
            return text
        except Exception as exc:
            logger.warning("OCR failed for page %d: %s", page_number, exc)
            return ""

    async def _analyze_page_images(
        self, doc, page, page_number: int, surrounding_text: str
    ) -> list[str]:
        """Extract embedded images from a page and analyze them via vision."""
        from src.ingestion.ocr import analyze_image

        descriptions = []
        image_list = page.get_images(full=True)

        for img_index, img_info in enumerate(image_list):
            xref = img_info[0]

            try:
                base_image = doc.extract_image(xref)
            except Exception:
                continue

            if not base_image or not base_image.get("image"):
                continue

            width = base_image.get("width", 0)
            height = base_image.get("height", 0)

            # Skip tiny images (icons, bullets, decorative elements)
            if width < MIN_IMAGE_WIDTH or height < MIN_IMAGE_HEIGHT:
                continue
            if width * height < MIN_IMAGE_AREA:
                continue

            image_bytes = base_image["image"]

            # Convert to PNG if not already (vision API accepts PNG/JPEG)
            img_ext = base_image.get("ext", "png")
            if img_ext not in ("png", "jpeg", "jpg"):
                try:
                    pix = fitz.Pixmap(image_bytes)
                    image_bytes = pix.tobytes("png")
                except Exception:
                    continue

            logger.debug(
                "Analyzing image %d on page %d (%dx%d)",
                img_index + 1, page_number, width, height,
            )

            try:
                description = await analyze_image(
                    openai_client=self._openai,
                    image_bytes=image_bytes,
                    model=self._settings.ocr_model,
                    context=surrounding_text[:500],
                )
                if description:
                    descriptions.append(description)
            except Exception as exc:
                logger.warning(
                    "Image analysis failed for image %d on page %d: %s",
                    img_index + 1, page_number, exc,
                )

        return descriptions


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
