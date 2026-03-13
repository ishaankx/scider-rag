"""Tests for vision-based OCR and the enhanced PDF handler."""

import base64
from dataclasses import dataclass
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.ingestion.ocr import analyze_image, ocr_page_image


def _mock_vision_response(text: str):
    """Create a mock OpenAI chat completion response."""
    choice = MagicMock()
    choice.message.content = text
    response = MagicMock()
    response.choices = [choice]
    return response


class TestOcrPageImage:
    """Tests for vision OCR text extraction."""

    @pytest.mark.asyncio
    async def test_returns_extracted_text(self):
        client = AsyncMock()
        client.chat.completions.create.return_value = _mock_vision_response(
            "Extracted text from scanned page."
        )

        result = await ocr_page_image(
            openai_client=client,
            image_bytes=b"\x89PNG\r\n\x1a\n" + b"\x00" * 100,
            model="gpt-4o-mini",
            page_number=1,
        )

        assert result == "Extracted text from scanned page."
        client.chat.completions.create.assert_called_once()

    @pytest.mark.asyncio
    async def test_sends_base64_encoded_image(self):
        client = AsyncMock()
        client.chat.completions.create.return_value = _mock_vision_response("text")
        image_bytes = b"\x89PNG" + b"\x00" * 50

        await ocr_page_image(client, image_bytes, "gpt-4o-mini", 1)

        call_args = client.chat.completions.create.call_args
        messages = call_args.kwargs["messages"]
        image_content = messages[0]["content"][1]
        expected_b64 = base64.b64encode(image_bytes).decode("utf-8")
        assert expected_b64 in image_content["image_url"]["url"]

    @pytest.mark.asyncio
    async def test_uses_high_detail(self):
        client = AsyncMock()
        client.chat.completions.create.return_value = _mock_vision_response("text")

        await ocr_page_image(client, b"img", "gpt-4o-mini", 1)

        call_args = client.chat.completions.create.call_args
        messages = call_args.kwargs["messages"]
        image_content = messages[0]["content"][1]
        assert image_content["image_url"]["detail"] == "high"

    @pytest.mark.asyncio
    async def test_strips_whitespace(self):
        client = AsyncMock()
        client.chat.completions.create.return_value = _mock_vision_response(
            "  text with whitespace  \n"
        )

        result = await ocr_page_image(client, b"img", "gpt-4o-mini", 1)
        assert result == "text with whitespace"

    @pytest.mark.asyncio
    async def test_handles_empty_response(self):
        client = AsyncMock()
        client.chat.completions.create.return_value = _mock_vision_response("")

        result = await ocr_page_image(client, b"img", "gpt-4o-mini", 1)
        assert result == ""

    @pytest.mark.asyncio
    async def test_passes_correct_model(self):
        client = AsyncMock()
        client.chat.completions.create.return_value = _mock_vision_response("text")

        await ocr_page_image(client, b"img", "gpt-4o", 1)

        call_args = client.chat.completions.create.call_args
        assert call_args.kwargs["model"] == "gpt-4o"


class TestAnalyzeImage:
    """Tests for vision-based image/figure analysis."""

    @pytest.mark.asyncio
    async def test_returns_description(self):
        client = AsyncMock()
        client.chat.completions.create.return_value = _mock_vision_response(
            "Bar chart showing accuracy vs model size."
        )

        result = await analyze_image(client, b"img", "gpt-4o-mini")
        assert "Bar chart" in result

    @pytest.mark.asyncio
    async def test_includes_context_in_prompt(self):
        client = AsyncMock()
        client.chat.completions.create.return_value = _mock_vision_response("desc")

        await analyze_image(client, b"img", "gpt-4o-mini", context="Table 3 shows results")

        call_args = client.chat.completions.create.call_args
        messages = call_args.kwargs["messages"]
        text_content = messages[0]["content"][0]["text"]
        assert "Table 3 shows results" in text_content

    @pytest.mark.asyncio
    async def test_truncates_long_context(self):
        client = AsyncMock()
        client.chat.completions.create.return_value = _mock_vision_response("desc")

        long_context = "x" * 1000
        await analyze_image(client, b"img", "gpt-4o-mini", context=long_context)

        call_args = client.chat.completions.create.call_args
        messages = call_args.kwargs["messages"]
        text_content = messages[0]["content"][0]["text"]
        # Context should be truncated to 500 chars
        assert "x" * 500 in text_content
        assert "x" * 600 not in text_content


class TestPdfHandlerBackwardCompatibility:
    """Verify PdfHandler still works without OCR (no OpenAI client)."""

    @pytest.mark.asyncio
    async def test_text_pdf_works_without_openai(self):
        """Original behaviour: text PDFs work when no OpenAI client is provided."""
        from src.ingestion.handlers.pdf_handler import PdfHandler

        handler = PdfHandler()  # No openai_client, no settings
        assert handler.can_handle("pdf")

    @pytest.mark.asyncio
    async def test_handler_with_none_clients(self):
        """PdfHandler(None, None) should behave like the original handler."""
        from src.ingestion.handlers.pdf_handler import PdfHandler

        handler = PdfHandler(openai_client=None, settings=None)
        assert handler.can_handle("pdf")
        assert not handler.can_handle("csv")


class TestPdfHandlerOcrIntegration:
    """Tests for OCR and image analysis integration in PdfHandler."""

    def _make_settings(self, **overrides):
        """Create a mock settings object with OCR defaults."""
        settings = MagicMock()
        settings.enable_ocr = overrides.get("enable_ocr", True)
        settings.enable_image_analysis = overrides.get("enable_image_analysis", True)
        settings.ocr_model = overrides.get("ocr_model", "gpt-4o-mini")
        settings.max_ocr_pages = overrides.get("max_ocr_pages", 50)
        return settings

    @pytest.mark.asyncio
    async def test_ocr_disabled_skips_vision(self):
        """When enable_ocr=False, image-only pages are skipped (not OCR'd)."""
        from src.ingestion.handlers.pdf_handler import PdfHandler

        settings = self._make_settings(enable_ocr=False)
        client = AsyncMock()
        handler = PdfHandler(openai_client=client, settings=settings)

        # Mock a PDF with one image-only page — should raise ValueError
        with patch("src.ingestion.handlers.pdf_handler.fitz") as mock_fitz:
            mock_doc = MagicMock()
            mock_doc.is_encrypted = False
            mock_doc.__len__ = lambda self: 1

            mock_page = MagicMock()
            mock_page.get_text.return_value = ""  # No text → image-only
            mock_doc.__getitem__ = lambda self, idx: mock_page

            mock_fitz.open.return_value = mock_doc

            with pytest.raises(ValueError, match="no extractable text"):
                await handler.extract(b"fake_pdf_bytes", "scan.pdf")

        # OpenAI should never be called
        client.chat.completions.create.assert_not_called()

    @pytest.mark.asyncio
    async def test_max_ocr_pages_respected(self):
        """OCR stops after max_ocr_pages even if more pages need it."""
        from src.ingestion.handlers.pdf_handler import PdfHandler

        settings = self._make_settings(max_ocr_pages=2, enable_image_analysis=False)
        client = AsyncMock()
        client.chat.completions.create.return_value = _mock_vision_response("OCR text")
        handler = PdfHandler(openai_client=client, settings=settings)

        with patch("src.ingestion.handlers.pdf_handler.fitz") as mock_fitz:
            mock_doc = MagicMock()
            mock_doc.is_encrypted = False

            # 5 pages, all image-only
            pages = []
            for _ in range(5):
                p = MagicMock()
                p.get_text.return_value = ""
                p.get_pixmap.return_value = MagicMock(
                    width=800, height=600, tobytes=lambda fmt: b"png_bytes"
                )
                pages.append(p)

            mock_doc.__len__ = lambda self: 5
            mock_doc.__getitem__ = lambda self, idx: pages[idx]
            mock_fitz.open.return_value = mock_doc
            mock_fitz.Matrix.return_value = MagicMock()

            result = await handler.extract(b"fake_pdf", "scan.pdf")

            # Only 2 pages should have been OCR'd
            assert result.metadata["ocr_pages"] == 2
            assert result.metadata["pages_with_text"] == 2

    @pytest.mark.asyncio
    async def test_metadata_tracks_extraction_method(self):
        """Each page's metadata should record how text was extracted."""
        from src.ingestion.handlers.pdf_handler import PdfHandler

        settings = self._make_settings(enable_image_analysis=False)
        client = AsyncMock()
        client.chat.completions.create.return_value = _mock_vision_response("OCR result")
        handler = PdfHandler(openai_client=client, settings=settings)

        with patch("src.ingestion.handlers.pdf_handler.fitz") as mock_fitz:
            mock_doc = MagicMock()
            mock_doc.is_encrypted = False

            # Page 0: has text, Page 1: image-only
            text_page = MagicMock()
            text_page.get_text.return_value = "Normal selectable text content here"
            text_page.get_images.return_value = []

            ocr_page = MagicMock()
            ocr_page.get_text.return_value = ""
            ocr_page.get_pixmap.return_value = MagicMock(
                width=800, height=600, tobytes=lambda fmt: b"png_bytes"
            )

            mock_doc.__len__ = lambda self: 2
            mock_doc.__getitem__ = lambda self, idx: [text_page, ocr_page][idx]
            mock_fitz.open.return_value = mock_doc
            mock_fitz.Matrix.return_value = MagicMock()

            result = await handler.extract(b"fake_pdf", "mixed.pdf")

            pages = result.metadata["pages"]
            assert pages[0]["extraction_method"] == "text_extraction"
            assert pages[1]["extraction_method"] == "ocr_vision"

    @pytest.mark.asyncio
    async def test_image_analysis_appends_descriptions(self):
        """Embedded images should be analyzed and descriptions appended to text."""
        from src.ingestion.handlers.pdf_handler import PdfHandler

        settings = self._make_settings(enable_image_analysis=True)
        client = AsyncMock()
        client.chat.completions.create.return_value = _mock_vision_response(
            "A scatter plot showing correlation between X and Y."
        )
        handler = PdfHandler(openai_client=client, settings=settings)

        with patch("src.ingestion.handlers.pdf_handler.fitz") as mock_fitz:
            mock_doc = MagicMock()
            mock_doc.is_encrypted = False

            page = MagicMock()
            page.get_text.return_value = "Some text on the page."
            # One large enough image
            page.get_images.return_value = [(42, 0, 0, 0, 0, 0, 0)]
            mock_doc.extract_image.return_value = {
                "image": b"fake_image_data",
                "ext": "png",
                "width": 400,
                "height": 300,
            }

            mock_doc.__len__ = lambda self: 1
            mock_doc.__getitem__ = lambda self, idx: page
            mock_fitz.open.return_value = mock_doc

            result = await handler.extract(b"fake_pdf", "paper.pdf")

            assert "[Figure on page 1]" in result.raw_text
            assert "scatter plot" in result.raw_text
            assert result.metadata["images_analyzed"] == 1

    @pytest.mark.asyncio
    async def test_small_images_skipped(self):
        """Images below minimum dimensions should not be analyzed."""
        from src.ingestion.handlers.pdf_handler import PdfHandler

        settings = self._make_settings(enable_image_analysis=True)
        client = AsyncMock()
        handler = PdfHandler(openai_client=client, settings=settings)

        with patch("src.ingestion.handlers.pdf_handler.fitz") as mock_fitz:
            mock_doc = MagicMock()
            mock_doc.is_encrypted = False

            page = MagicMock()
            page.get_text.return_value = "Text content here."
            page.get_images.return_value = [(42, 0, 0, 0, 0, 0, 0)]
            # Image too small (icon/bullet)
            mock_doc.extract_image.return_value = {
                "image": b"tiny",
                "ext": "png",
                "width": 16,
                "height": 16,
            }

            mock_doc.__len__ = lambda self: 1
            mock_doc.__getitem__ = lambda self, idx: page
            mock_fitz.open.return_value = mock_doc

            result = await handler.extract(b"fake_pdf", "paper.pdf")

            # No images should have been analyzed
            assert result.metadata["images_analyzed"] == 0
            # OpenAI vision should not have been called
            client.chat.completions.create.assert_not_called()

    @pytest.mark.asyncio
    async def test_ocr_failure_gracefully_skips_page(self):
        """If OCR fails for a page, it should log and continue, not crash."""
        from src.ingestion.handlers.pdf_handler import PdfHandler

        settings = self._make_settings(enable_image_analysis=False)
        client = AsyncMock()
        client.chat.completions.create.side_effect = Exception("API error")
        handler = PdfHandler(openai_client=client, settings=settings)

        with patch("src.ingestion.handlers.pdf_handler.fitz") as mock_fitz:
            mock_doc = MagicMock()
            mock_doc.is_encrypted = False

            # Page 0: text, Page 1: image-only (OCR will fail)
            text_page = MagicMock()
            text_page.get_text.return_value = "Normal text here."
            text_page.get_images.return_value = []

            ocr_page = MagicMock()
            ocr_page.get_text.return_value = ""
            ocr_page.get_pixmap.return_value = MagicMock(
                width=800, height=600, tobytes=lambda fmt: b"png"
            )

            mock_doc.__len__ = lambda self: 2
            mock_doc.__getitem__ = lambda self, idx: [text_page, ocr_page][idx]
            mock_fitz.open.return_value = mock_doc
            mock_fitz.Matrix.return_value = MagicMock()

            # Should not crash — OCR page is skipped
            result = await handler.extract(b"fake_pdf", "mixed.pdf")
            assert result.metadata["pages_with_text"] == 1
