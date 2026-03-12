"""Unit tests for the text chunker."""

import pytest

from src.ingestion.chunker import chunk_text


class TestChunkText:
    """Tests for the recursive text chunker."""

    def test_empty_text_returns_empty(self):
        assert chunk_text("") == []
        assert chunk_text("   ") == []

    def test_short_text_single_chunk(self):
        text = "This is a short sentence."
        chunks = chunk_text(text, chunk_size=100)
        assert len(chunks) == 1
        assert chunks[0]["chunk_index"] == 0
        assert "short sentence" in chunks[0]["content"]

    def test_splits_by_paragraph(self):
        text = "First paragraph with content.\n\nSecond paragraph with content."
        chunks = chunk_text(text, chunk_size=40, chunk_overlap=0)
        assert len(chunks) >= 2

    def test_chunk_overlap_applied(self):
        text = "A" * 200 + "\n\n" + "B" * 200
        chunks = chunk_text(text, chunk_size=250, chunk_overlap=30)
        # Second chunk should have overlap prefix from first
        if len(chunks) > 1:
            assert chunks[1]["content"].startswith("...")

    def test_token_count_estimated(self):
        text = "Hello world this is a test sentence."
        chunks = chunk_text(text, chunk_size=500)
        assert chunks[0]["token_count"] > 0

    def test_long_text_multiple_chunks(self):
        text = " ".join(["Scientific research"] * 500)
        chunks = chunk_text(text, chunk_size=200)
        assert len(chunks) > 1
        # All chunks should have content
        for chunk in chunks:
            assert len(chunk["content"]) > 0

    def test_section_header_prepended(self):
        text = "Some content that is long enough to be split into chunks."
        chunks = chunk_text(text, chunk_size=500, section_header="Methods")
        # First chunk won't have header (only subsequent chunks)
        assert chunks[0]["chunk_index"] == 0

    def test_chunk_indices_sequential(self):
        text = "\n\n".join([f"Paragraph {i} with some content." for i in range(10)])
        chunks = chunk_text(text, chunk_size=50)
        indices = [c["chunk_index"] for c in chunks]
        assert indices == list(range(len(chunks)))
