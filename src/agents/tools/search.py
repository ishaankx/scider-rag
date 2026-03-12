"""
Search tools: vector similarity search and keyword search.
These are the primary retrieval mechanisms for the RAG pipeline.
"""

from src.agents.tools.base import BaseTool, ToolResult
from src.ingestion.embeddings import EmbeddingService
from src.storage.document_store import DocumentStore
from src.storage.vector_store import VectorStore


class VectorSearchTool(BaseTool):
    """Semantic similarity search using vector embeddings."""

    def __init__(
        self,
        vector_store: VectorStore,
        embedding_service: EmbeddingService,
        top_k: int = 10,
    ):
        self._vector_store = vector_store
        self._embedding_service = embedding_service
        self._top_k = top_k

    @property
    def name(self) -> str:
        return "vector_search"

    @property
    def description(self) -> str:
        return (
            "Search for relevant scientific text passages using semantic similarity. "
            "Best for finding conceptually related content even if exact keywords differ. "
            "Input a natural language query describing what you're looking for."
        )

    @property
    def parameters_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query in natural language.",
                },
                "top_k": {
                    "type": "integer",
                    "description": "Number of results to return (default 10).",
                },
                "source_type": {
                    "type": "string",
                    "description": "Filter by source type: pdf, csv, json, txt.",
                    "enum": ["pdf", "csv", "json", "txt"],
                },
            },
            "required": ["query"],
        }

    async def execute(self, **params) -> ToolResult:
        query = params.get("query", "")
        top_k = params.get("top_k", self._top_k)
        source_type = params.get("source_type")

        if not query.strip():
            return ToolResult(output="Empty query.", success=False, error="Query cannot be empty.")

        try:
            query_embedding = await self._embedding_service.embed_query(query)

            results = await self._vector_store.search(
                query_embedding=query_embedding,
                top_k=top_k,
                source_type=source_type,
            )

            if not results:
                return ToolResult(output="No relevant documents found.")

            formatted = []
            for r in results:
                payload = r.payload or {}
                formatted.append(
                    f"[Score: {r.score:.3f}] "
                    f"(Doc: {payload.get('document_title', 'Unknown')}) "
                    f"(ID: {payload.get('document_id', '')}) "
                    f"{payload.get('content', '')[:500]}"
                )

            output = "\n\n".join(formatted)
            return ToolResult(output=output)

        except Exception as exc:
            return ToolResult(
                output="Search failed.",
                success=False,
                error=str(exc),
            )


class KeywordSearchTool(BaseTool):
    """Full-text keyword search in PostgreSQL."""

    def __init__(self, document_store: DocumentStore):
        self._doc_store = document_store

    @property
    def name(self) -> str:
        return "keyword_search"

    @property
    def description(self) -> str:
        return (
            "Search for documents containing specific keywords or phrases. "
            "Best for finding exact terms, names, or technical identifiers. "
            "Uses full-text search with ranking."
        )

    @property
    def parameters_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Keywords or phrases to search for.",
                },
                "limit": {
                    "type": "integer",
                    "description": "Max results to return (default 10).",
                },
            },
            "required": ["query"],
        }

    async def execute(self, **params) -> ToolResult:
        query = params.get("query", "")
        limit = params.get("limit", 10)

        if not query.strip():
            return ToolResult(output="Empty query.", success=False, error="Query cannot be empty.")

        try:
            results = await self._doc_store.keyword_search(query, limit=limit)

            if not results:
                return ToolResult(output="No keyword matches found.")

            formatted = []
            for r in results:
                formatted.append(
                    f"[Rank: {r['relevance_score']:.3f}] "
                    f"(Doc: {r['document_title']}) "
                    f"(ID: {r.get('document_id', '')}) "
                    f"{r['content'][:500]}"
                )

            return ToolResult(output="\n\n".join(formatted))

        except Exception as exc:
            return ToolResult(
                output="Keyword search failed.",
                success=False,
                error=str(exc),
            )
