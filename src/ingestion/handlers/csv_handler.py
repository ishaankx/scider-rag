"""CSV and JSON file handler. Converts structured data into searchable text."""

import csv
import io
import json
import logging

from src.ingestion.handlers.base import BaseHandler, ExtractedDocument

logger = logging.getLogger(__name__)

# Maximum rows to process (guard against extremely large files)
MAX_ROWS = 50_000


class CsvHandler(BaseHandler):
    """Handles CSV files: extracts schema, statistics, and row data."""

    def can_handle(self, source_type: str) -> bool:
        return source_type == "csv"

    async def extract(self, content_bytes: bytes, file_name: str) -> ExtractedDocument:
        try:
            text_content = content_bytes.decode("utf-8")
        except UnicodeDecodeError:
            text_content = content_bytes.decode("latin-1")

        reader = csv.DictReader(io.StringIO(text_content))
        if not reader.fieldnames:
            raise ValueError(f"CSV '{file_name}' has no headers.")

        columns = list(reader.fieldnames)
        rows = []
        for i, row in enumerate(reader):
            if i >= MAX_ROWS:
                logger.warning("CSV '%s' truncated at %d rows.", file_name, MAX_ROWS)
                break
            rows.append(row)

        if not rows:
            raise ValueError(f"CSV '{file_name}' contains no data rows.")

        # Build natural-language description for embedding
        description = _describe_csv(file_name, columns, rows)

        return ExtractedDocument(
            title=file_name.rsplit(".", 1)[0],
            raw_text=description,
            metadata={
                "file_name": file_name,
                "columns": columns,
                "row_count": len(rows),
                "column_count": len(columns),
            },
            structured_data=rows,
        )


class JsonHandler(BaseHandler):
    """Handles JSON files: flattens structure into searchable text."""

    def can_handle(self, source_type: str) -> bool:
        return source_type in ("json", "txt")

    async def extract(self, content_bytes: bytes, file_name: str) -> ExtractedDocument:
        try:
            text_content = content_bytes.decode("utf-8")
        except UnicodeDecodeError:
            text_content = content_bytes.decode("latin-1")

        # For plain text files, treat as raw text
        if file_name.endswith(".txt"):
            return ExtractedDocument(
                title=file_name.rsplit(".", 1)[0],
                raw_text=text_content,
                metadata={"file_name": file_name, "char_count": len(text_content)},
            )

        # Parse JSON
        try:
            data = json.loads(text_content)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON in '{file_name}': {exc}") from exc

        if isinstance(data, list):
            rows = data[:MAX_ROWS]
            description = _describe_json_array(file_name, rows)
            structured = rows if all(isinstance(r, dict) for r in rows) else None
        elif isinstance(data, dict):
            description = _describe_json_object(file_name, data)
            structured = [data]
        else:
            raise ValueError(f"JSON '{file_name}' must be an object or array.")

        return ExtractedDocument(
            title=file_name.rsplit(".", 1)[0],
            raw_text=description,
            metadata={
                "file_name": file_name,
                "type": "array" if isinstance(data, list) else "object",
                "item_count": len(data) if isinstance(data, list) else 1,
            },
            structured_data=structured,
        )


# ── Helper functions ─────────────────────────────────────────────────────────


def _describe_csv(file_name: str, columns: list[str], rows: list[dict]) -> str:
    """Convert CSV data into natural-language text suitable for embedding."""
    parts = [f"Dataset: {file_name}", f"Columns: {', '.join(columns)}"]
    parts.append(f"Total rows: {len(rows)}")

    # Sample first few rows as natural language
    for i, row in enumerate(rows[:20]):
        row_desc = "; ".join(f"{k}: {v}" for k, v in row.items() if v)
        parts.append(f"Row {i + 1}: {row_desc}")

    if len(rows) > 20:
        parts.append(f"... and {len(rows) - 20} more rows")

    return "\n".join(parts)


def _describe_json_array(file_name: str, items: list) -> str:
    """Convert a JSON array into readable text."""
    parts = [f"Dataset: {file_name}", f"Items: {len(items)}"]
    for i, item in enumerate(items[:20]):
        if isinstance(item, dict):
            row_desc = "; ".join(f"{k}: {v}" for k, v in item.items())
        else:
            row_desc = str(item)
        parts.append(f"Item {i + 1}: {row_desc}")
    return "\n".join(parts)


def _describe_json_object(file_name: str, obj: dict) -> str:
    """Convert a single JSON object into readable text."""
    parts = [f"Document: {file_name}"]
    for key, value in obj.items():
        if isinstance(value, (list, dict)):
            parts.append(f"{key}: {json.dumps(value, default=str)[:500]}")
        else:
            parts.append(f"{key}: {value}")
    return "\n".join(parts)
