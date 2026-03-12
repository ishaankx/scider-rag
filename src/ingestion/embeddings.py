"""
Embedding generation using OpenAI API.
Handles batching, retries, and rate limit backoff.
"""

import logging

from openai import AsyncOpenAI
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from src.config import Settings

logger = logging.getLogger(__name__)

# OpenAI batch limit for embeddings
MAX_BATCH_SIZE = 2048


class EmbeddingService:
    """Generates text embeddings via OpenAI API with batching and retries."""

    def __init__(self, openai_client: AsyncOpenAI, settings: Settings):
        self._client = openai_client
        self._model = settings.embedding_model
        self._dimensions = settings.embedding_dimensions

    @retry(
        retry=retry_if_exception_type(Exception),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=30),
        reraise=True,
    )
    async def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed a single batch with retries on failure."""
        response = await self._client.embeddings.create(
            input=texts,
            model=self._model,
            dimensions=self._dimensions,
        )
        # Return embeddings in input order
        sorted_data = sorted(response.data, key=lambda x: x.index)
        return [item.embedding for item in sorted_data]

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """
        Generate embeddings for a list of texts.
        Automatically batches large inputs to stay within API limits.
        """
        if not texts:
            return []

        # Clean inputs — OpenAI rejects empty strings
        cleaned = [t.strip() or " " for t in texts]

        all_embeddings = []
        for i in range(0, len(cleaned), MAX_BATCH_SIZE):
            batch = cleaned[i : i + MAX_BATCH_SIZE]
            logger.debug("Embedding batch %d-%d of %d texts", i, i + len(batch), len(cleaned))
            embeddings = await self._embed_batch(batch)
            all_embeddings.extend(embeddings)

        return all_embeddings

    async def embed_query(self, query: str) -> list[float]:
        """Embed a single query string. Convenience wrapper."""
        results = await self.embed_texts([query])
        return results[0]
