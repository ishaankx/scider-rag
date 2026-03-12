"""Abstract base class for file format handlers."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class ExtractedDocument:
    """Result of parsing a file. Uniform format regardless of source type."""

    title: str
    raw_text: str
    metadata: dict = field(default_factory=dict)
    # Structured rows for tabular data (CSV/JSON)
    structured_data: list[dict] | None = None


class BaseHandler(ABC):
    """Interface for file format handlers. One handler per source type."""

    @abstractmethod
    def can_handle(self, source_type: str) -> bool:
        """Return True if this handler supports the given source type."""
        ...

    @abstractmethod
    async def extract(self, content_bytes: bytes, file_name: str) -> ExtractedDocument:
        """Parse raw file bytes into an ExtractedDocument."""
        ...
