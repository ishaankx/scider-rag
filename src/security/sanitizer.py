"""
Input sanitization utilities.
Prevents XSS, strips dangerous content, and enforces length limits.
"""

import re

import bleach


# Characters that could be used for SQL injection in raw contexts
_SQL_INJECTION_PATTERN = re.compile(
    r"(--|;|\b(DROP|ALTER|DELETE|INSERT|UPDATE|EXEC|UNION|SELECT)\b)",
    re.IGNORECASE,
)

# Maximum lengths for different input types
MAX_QUERY_LENGTH = 2000
MAX_FILENAME_LENGTH = 255
MAX_METADATA_VALUE_LENGTH = 1000


def sanitize_text(text: str, max_length: int = MAX_QUERY_LENGTH) -> str:
    """
    Clean user-provided text input.
    - Strips HTML tags
    - Enforces max length
    - Strips leading/trailing whitespace
    """
    if not text:
        return ""

    # Strip HTML tags (prevents XSS if text is ever rendered)
    cleaned = bleach.clean(text, tags=[], strip=True)

    # Trim whitespace
    cleaned = cleaned.strip()

    # Enforce length limit
    if len(cleaned) > max_length:
        cleaned = cleaned[:max_length]

    return cleaned


def sanitize_filename(filename: str) -> str:
    """
    Sanitize an uploaded filename.
    - Removes path traversal characters
    - Allows only alphanumeric, dots, hyphens, underscores
    """
    if not filename:
        return "unnamed"

    # Remove any directory components
    filename = filename.replace("\\", "/").split("/")[-1]

    # Keep only safe characters
    safe = re.sub(r"[^a-zA-Z0-9._-]", "_", filename)

    # Prevent hidden files
    safe = safe.lstrip(".")

    # Enforce length
    if len(safe) > MAX_FILENAME_LENGTH:
        safe = safe[:MAX_FILENAME_LENGTH]

    return safe or "unnamed"


def check_sql_injection(text: str) -> bool:
    """
    Returns True if the text contains suspicious SQL patterns.
    This is a defense-in-depth check — SQLAlchemy parameterized queries
    are the primary defense. This catches obvious attempts early.
    """
    return bool(_SQL_INJECTION_PATTERN.search(text))


def sanitize_metadata(metadata: dict) -> dict:
    """Recursively sanitize all string values in a metadata dict."""
    cleaned = {}
    for key, value in metadata.items():
        safe_key = sanitize_text(str(key), max_length=100)
        if isinstance(value, str):
            cleaned[safe_key] = sanitize_text(value, max_length=MAX_METADATA_VALUE_LENGTH)
        elif isinstance(value, dict):
            cleaned[safe_key] = sanitize_metadata(value)
        elif isinstance(value, list):
            cleaned[safe_key] = [
                sanitize_text(str(v), max_length=MAX_METADATA_VALUE_LENGTH)
                if isinstance(v, str) else v
                for v in value
            ]
        else:
            cleaned[safe_key] = value
    return cleaned
