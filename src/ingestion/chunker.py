"""
Recursive text chunker with configurable size and overlap.
Splits by paragraph → sentence → word boundaries, preserving context.
"""

import re


def chunk_text(
    text: str,
    chunk_size: int = 512,
    chunk_overlap: int = 50,
    section_header: str | None = None,
) -> list[dict]:
    """
    Split text into overlapping chunks of roughly `chunk_size` characters.

    Strategy: Try splitting by double newlines (paragraphs) first,
    then single newlines, then sentences, then words.

    Returns list of dicts: {content, chunk_index, token_count, metadata}
    """
    if not text or not text.strip():
        return []

    # Separators in order of preference
    separators = ["\n\n", "\n", ". ", " "]
    raw_chunks = _recursive_split(text.strip(), separators, chunk_size)

    # Merge very small chunks with their neighbors
    merged = _merge_small_chunks(raw_chunks, min_size=50, max_size=chunk_size)

    # Apply overlap: prepend tail of previous chunk to current chunk
    results = []
    for i, chunk_text_content in enumerate(merged):
        content = chunk_text_content.strip()
        if not content:
            continue

        # Prepend section header for context if available
        if section_header and i > 0:
            content = f"[{section_header}] {content}"

        # Apply overlap from previous chunk
        if i > 0 and chunk_overlap > 0:
            prev_tail = merged[i - 1].strip()[-chunk_overlap:]
            content = f"...{prev_tail} {content}"

        results.append({
            "content": content,
            "chunk_index": i,
            "token_count": _estimate_tokens(content),
            "metadata": {},
        })

    return results


def _recursive_split(text: str, separators: list[str], max_size: int) -> list[str]:
    """Recursively split text using progressively finer separators."""
    if len(text) <= max_size:
        return [text]

    # Try each separator
    for sep in separators:
        parts = text.split(sep)
        if len(parts) > 1:
            chunks = []
            current = ""
            for part in parts:
                candidate = f"{current}{sep}{part}" if current else part
                if len(candidate) <= max_size:
                    current = candidate
                else:
                    if current:
                        chunks.append(current)
                    # If single part exceeds max, try next separator
                    if len(part) > max_size:
                        remaining_seps = separators[separators.index(sep) + 1 :]
                        if remaining_seps:
                            chunks.extend(_recursive_split(part, remaining_seps, max_size))
                        else:
                            # Last resort: hard cut
                            for j in range(0, len(part), max_size):
                                chunks.append(part[j : j + max_size])
                        current = ""
                    else:
                        current = part
            if current:
                chunks.append(current)
            return chunks

    # No separator worked — hard split by character count
    return [text[i : i + max_size] for i in range(0, len(text), max_size)]


def _merge_small_chunks(chunks: list[str], min_size: int, max_size: int) -> list[str]:
    """Merge chunks that are too small with their next neighbor."""
    if not chunks:
        return []

    merged = []
    buffer = ""

    for chunk in chunks:
        if buffer:
            combined = f"{buffer} {chunk}"
            if len(combined) <= max_size:
                buffer = combined
            else:
                merged.append(buffer)
                buffer = chunk
        elif len(chunk) < min_size:
            buffer = chunk
        else:
            merged.append(chunk)

    if buffer:
        merged.append(buffer)

    return merged


def _estimate_tokens(text: str) -> int:
    """Rough token estimate: ~4 chars per token for English text."""
    return max(1, len(text) // 4)
