"""Unit tests for input sanitization."""

import pytest

from src.security.sanitizer import (
    check_sql_injection,
    sanitize_filename,
    sanitize_metadata,
    sanitize_text,
)


class TestSanitizeText:
    def test_strips_html(self):
        assert sanitize_text("<script>alert('xss')</script>hello") == "alert('xss')hello"

    def test_strips_whitespace(self):
        assert sanitize_text("  hello world  ") == "hello world"

    def test_enforces_max_length(self):
        long_text = "a" * 5000
        result = sanitize_text(long_text, max_length=100)
        assert len(result) == 100

    def test_empty_string(self):
        assert sanitize_text("") == ""

    def test_preserves_normal_text(self):
        text = "What is the effect of CRISPR on gene therapy?"
        assert sanitize_text(text) == text


class TestSanitizeFilename:
    def test_removes_path_traversal(self):
        assert "/" not in sanitize_filename("../../etc/passwd")
        assert "\\" not in sanitize_filename("..\\..\\windows\\system32")

    def test_removes_special_chars(self):
        result = sanitize_filename("my file (1).pdf")
        assert all(c.isalnum() or c in "._-" for c in result)

    def test_prevents_hidden_files(self):
        result = sanitize_filename(".hidden_file")
        assert not result.startswith(".")

    def test_empty_filename(self):
        assert sanitize_filename("") == "unnamed"

    def test_normal_filename_preserved(self):
        assert sanitize_filename("research_paper.pdf") == "research_paper.pdf"


class TestSqlInjection:
    def test_detects_drop_table(self):
        assert check_sql_injection("DROP TABLE users") is True

    def test_detects_union_select(self):
        assert check_sql_injection("1 UNION SELECT * FROM passwords") is True

    def test_detects_comment_injection(self):
        assert check_sql_injection("admin' --") is True

    def test_normal_text_passes(self):
        assert check_sql_injection("What is photosynthesis?") is False


class TestSanitizeMetadata:
    def test_sanitizes_nested_dict(self):
        meta = {"title": "<b>Bold Title</b>", "nested": {"key": "<script>bad</script>"}}
        result = sanitize_metadata(meta)
        assert "<b>" not in result["title"]
        assert "<script>" not in result["nested"]["key"]

    def test_preserves_non_string_values(self):
        meta = {"count": 42, "ratio": 3.14, "tags": ["a", "b"]}
        result = sanitize_metadata(meta)
        assert result["count"] == 42
        assert result["ratio"] == 3.14
