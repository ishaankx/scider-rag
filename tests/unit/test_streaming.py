"""Tests for SSE streaming query endpoint."""

import json

import pytest

from src.api.v1.stream import format_sse_event


class TestSSEFormatting:
    """Tests for Server-Sent Event formatting (W3C spec compliance)."""

    def test_status_event_format(self):
        result = format_sse_event("status", {"step": "retrieval", "message": "Searching..."})
        lines = result.strip().split("\n")
        assert lines[0] == "event: status"
        assert lines[1].startswith("data: ")
        data = json.loads(lines[1][6:])
        assert data["step"] == "retrieval"
        assert data["message"] == "Searching..."

    def test_sources_event_format(self):
        sources = [{"document_title": "Paper A", "relevance_score": 0.92}]
        result = format_sse_event("sources", sources)
        lines = result.strip().split("\n")
        assert lines[0] == "event: sources"
        data = json.loads(lines[1][6:])
        assert isinstance(data, list)
        assert data[0]["document_title"] == "Paper A"

    def test_answer_event_format(self):
        result = format_sse_event("answer", {"text": "CRISPR works by targeting specific DNA sequences."})
        lines = result.strip().split("\n")
        assert lines[0] == "event: answer"
        data = json.loads(lines[1][6:])
        assert data["text"] == "CRISPR works by targeting specific DNA sequences."

    def test_done_event_contains_full_result(self):
        payload = {
            "answer": "CRISPR works by...",
            "sources": [],
            "confidence": 0.85,
            "latency": {"retrieval_ms": 100, "reasoning_ms": 200, "total_ms": 300},
            "request_id": "abc-123",
        }
        result = format_sse_event("done", payload)
        data = json.loads(result.strip().split("\n")[1][6:])
        assert data["answer"] == "CRISPR works by..."
        assert data["confidence"] == 0.85
        assert data["request_id"] == "abc-123"

    def test_error_event_format(self):
        result = format_sse_event("error", {"message": "Pipeline failed", "request_id": "xyz"})
        lines = result.strip().split("\n")
        assert lines[0] == "event: error"
        data = json.loads(lines[1][6:])
        assert data["message"] == "Pipeline failed"

    def test_event_terminates_with_double_newline(self):
        """SSE spec requires events to end with \\n\\n."""
        result = format_sse_event("status", {"step": "test"})
        assert result.endswith("\n\n")

    def test_data_is_valid_json(self):
        result = format_sse_event("status", {"key": "value", "num": 42, "nested": {"a": 1}})
        data_line = result.strip().split("\n")[1]
        assert data_line.startswith("data: ")
        parsed = json.loads(data_line[6:])
        assert parsed["key"] == "value"
        assert parsed["num"] == 42
        assert parsed["nested"]["a"] == 1

    def test_special_characters_escaped_in_json(self):
        result = format_sse_event("status", {"message": 'line1\nline2\t"quoted"'})
        data_line = result.strip().split("\n")[1]
        parsed = json.loads(data_line[6:])
        assert parsed["message"] == 'line1\nline2\t"quoted"'


class TestStreamEventContract:
    """Tests documenting the expected SSE event type contract."""

    VALID_EVENT_TYPES = {"status", "sources", "answer", "done", "error"}

    @pytest.mark.parametrize("event_type", sorted(VALID_EVENT_TYPES))
    def test_all_event_types_produce_valid_sse(self, event_type):
        result = format_sse_event(event_type, {"test": True})
        assert result.startswith(f"event: {event_type}\n")
        assert result.endswith("\n\n")
        # Data line is valid JSON
        data_line = result.strip().split("\n")[1]
        json.loads(data_line[6:])

    def test_status_event_has_step_and_message_fields(self):
        """Status events must always contain step and message."""
        event_data = {"step": "retrieval", "message": "Searching..."}
        result = format_sse_event("status", event_data)
        parsed = json.loads(result.strip().split("\n")[1][6:])
        assert "step" in parsed
        assert "message" in parsed
