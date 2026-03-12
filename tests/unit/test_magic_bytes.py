"""Unit tests for magic byte validation in file uploads."""

import pytest

try:
    from fastapi import HTTPException
    from src.api.v1.ingest import _validate_magic_bytes

    HAS_FASTAPI = True
except ImportError:
    HAS_FASTAPI = False


@pytest.mark.skipif(not HAS_FASTAPI, reason="FastAPI not installed locally")
class TestMagicByteValidation:
    def test_valid_pdf(self):
        """Real PDF content starts with %PDF."""
        content = b"%PDF-1.4 some pdf content..."
        _validate_magic_bytes(content, "paper.pdf")  # Should not raise

    def test_invalid_pdf_rejected(self):
        """Non-PDF content with .pdf extension is rejected."""
        content = b"<html><body>not a pdf</body></html>"
        with pytest.raises(HTTPException) as exc_info:
            _validate_magic_bytes(content, "malicious.pdf")
        assert exc_info.value.status_code == 400
        assert "spoofed" in exc_info.value.detail.lower()

    def test_valid_json_object(self):
        """JSON starting with { is accepted."""
        content = b'{"key": "value"}'
        _validate_magic_bytes(content, "data.json")

    def test_valid_json_array(self):
        """JSON starting with [ is accepted."""
        content = b'[1, 2, 3]'
        _validate_magic_bytes(content, "data.json")

    def test_invalid_json_rejected(self):
        """Non-JSON content with .json extension is rejected."""
        content = b"this is not json at all"
        with pytest.raises(HTTPException) as exc_info:
            _validate_magic_bytes(content, "data.json")
        assert exc_info.value.status_code == 400

    def test_csv_skipped(self):
        """CSV has no magic bytes — validation is skipped."""
        content = b"col1,col2\nval1,val2"
        _validate_magic_bytes(content, "data.csv")  # Should not raise

    def test_txt_skipped(self):
        """TXT has no magic bytes — validation is skipped."""
        content = b"just some plain text"
        _validate_magic_bytes(content, "notes.txt")  # Should not raise

    def test_json_with_leading_whitespace(self):
        """JSON with leading whitespace is still accepted."""
        content = b"  \n  {\"key\": \"value\"}"
        _validate_magic_bytes(content, "data.json")
