"""
Retrieval Agent.
Decides the best search strategy for a query, executes searches,
and returns ranked, deduplicated results.
"""

import json
import logging
import time

from openai import AsyncOpenAI

from src.agents.base import AgentContext, AgentResult, BaseAgent
from src.agents.tools.search import KeywordSearchTool, VectorSearchTool
from src.config import Settings
from src.ingestion.embeddings import EmbeddingService
from src.storage.document_store import DocumentStore
from src.storage.vector_store import VectorStore

logger = logging.getLogger(__name__)

STRATEGY_PROMPT = """You are a search strategy planner for a scientific knowledge base.
Given a user's research question, decide the best retrieval strategy.

Available strategies:
- "vector": Semantic similarity search. Best for conceptual or broad questions.
- "keyword": Full-text keyword search. Best for specific terms, names, identifiers.
- "hybrid": Both vector and keyword search combined. Best for complex queries.

Respond with valid JSON:
{"strategy": "vector|keyword|hybrid", "search_queries": ["query1", "query2"], "reasoning": "why this strategy"}

Generate 1-3 search queries optimized for retrieval. Rephrase the user's question
if needed to improve search recall. Keep queries focused and specific.
"""


class RetrievalAgent(BaseAgent):
    """
    Plans and executes the retrieval strategy.
    Uses LLM to decide between vector, keyword, or hybrid search.
    """

    def __init__(
        self,
        openai_client: AsyncOpenAI,
        vector_store: VectorStore,
        document_store: DocumentStore,
        embedding_service: EmbeddingService,
        settings: Settings,
    ):
        self._openai = openai_client
        self._vector_search = VectorSearchTool(
            vector_store, embedding_service, top_k=settings.retrieval_top_k
        )
        self._keyword_search = KeywordSearchTool(document_store)
        self._settings = settings

    @property
    def name(self) -> str:
        return "retrieval_agent"

    async def execute(self, context: AgentContext) -> AgentResult:
        start = time.perf_counter()

        # Step 1: Ask LLM for search strategy
        strategy = await self._plan_strategy(context.question)
        logger.info(
            "Retrieval strategy: %s with %d queries",
            strategy["strategy"],
            len(strategy["search_queries"]),
        )

        # Step 2: Execute searches based on strategy
        all_results = []
        tools_used = []

        for query in strategy["search_queries"]:
            if strategy["strategy"] in ("vector", "hybrid"):
                result = await self._vector_search.execute(
                    query=query,
                    top_k=self._settings.retrieval_top_k,
                )
                if result.success:
                    all_results.append(("vector", query, result.output))
                    tools_used.append("vector_search")

            if strategy["strategy"] in ("keyword", "hybrid"):
                result = await self._keyword_search.execute(
                    query=query,
                    limit=self._settings.retrieval_top_k,
                )
                if result.success:
                    all_results.append(("keyword", query, result.output))
                    tools_used.append("keyword_search")

        # Step 3: Parse and deduplicate results
        chunks = self._parse_results(all_results)

        # If no results, try a broader fallback search
        if not chunks and strategy["strategy"] != "hybrid":
            logger.info("No results found, falling back to hybrid search.")
            vector_result = await self._vector_search.execute(query=context.question)
            keyword_result = await self._keyword_search.execute(query=context.question)

            if vector_result.success:
                all_results.append(("vector", context.question, vector_result.output))
            if keyword_result.success:
                all_results.append(("keyword", context.question, keyword_result.output))

            chunks = self._parse_results(all_results)
            tools_used.append("fallback_hybrid")

        elapsed_ms = (time.perf_counter() - start) * 1000
        context.retrieved_chunks = chunks

        return AgentResult(
            output=chunks,
            sources=chunks,
            tool_calls_made=tools_used,
            latency_ms=elapsed_ms,
            metadata={"strategy": strategy},
        )

    async def _plan_strategy(self, question: str) -> dict:
        """Use LLM to decide retrieval strategy."""
        try:
            response = await self._openai.chat.completions.create(
                model=self._settings.llm_model,
                messages=[
                    {"role": "system", "content": STRATEGY_PROMPT},
                    {"role": "user", "content": question},
                ],
                temperature=0.0,
                max_tokens=256,
                response_format={"type": "json_object"},
            )

            result = json.loads(response.choices[0].message.content)

            # Validate strategy
            valid_strategies = {"vector", "keyword", "hybrid"}
            if result.get("strategy") not in valid_strategies:
                result["strategy"] = "hybrid"

            if not result.get("search_queries"):
                result["search_queries"] = [question]

            return result

        except Exception as exc:
            logger.warning("Strategy planning failed, defaulting to hybrid: %s", exc)
            return {
                "strategy": "hybrid",
                "search_queries": [question],
                "reasoning": "Fallback due to planning error.",
            }

    def _parse_results(self, raw_results: list[tuple]) -> list[dict]:
        """Parse raw search outputs into structured chunks, deduplicating by content."""
        seen_content = set()
        chunks = []

        for source_type, query, output in raw_results:
            if not output or output in ("No relevant documents found.", "No keyword matches found."):
                continue

            for block in output.split("\n\n"):
                block = block.strip()
                if not block:
                    continue

                # Extract score and content from formatted output
                score = 0.0
                content = block
                doc_title = "Unknown"
                doc_id = ""

                if block.startswith("[Score:") or block.startswith("[Rank:"):
                    try:
                        score_str = block.split("]")[0].split(":")[1].strip()
                        score = float(score_str)
                    except (IndexError, ValueError):
                        pass

                    # Extract document title
                    if "(Doc:" in block:
                        try:
                            doc_title = block.split("(Doc:")[1].split(")")[0].strip()
                        except IndexError:
                            pass

                    # Extract document ID
                    if "(ID:" in block:
                        try:
                            doc_id = block.split("(ID:")[1].split(")")[0].strip()
                        except IndexError:
                            pass

                    # Get content after the last metadata prefix
                    parts = block.split(") ", 1)
                    if len(parts) > 1:
                        # Skip past all (Key: value) prefixes
                        remainder = parts[1]
                        while remainder.startswith("(") and ") " in remainder:
                            remainder = remainder.split(") ", 1)[1]
                        content = remainder

                # Dedup by content prefix (first 200 chars)
                dedup_key = content[:200].strip().lower()
                if dedup_key in seen_content:
                    continue
                seen_content.add(dedup_key)

                chunks.append({
                    "content": content,
                    "document_title": doc_title,
                    "document_id": doc_id,
                    "relevance_score": score,
                    "retrieval_method": source_type,
                    "search_query": query,
                })

        # Sort by score descending
        chunks.sort(key=lambda c: c["relevance_score"], reverse=True)
        return chunks
