"""
Hallucination detection.
Verifies that claims in the answer are grounded in the retrieved sources.
Uses a separate LLM call to cross-reference answer against source material.
"""

import json
import logging

from openai import AsyncOpenAI

from src.config import Settings

logger = logging.getLogger(__name__)

GROUNDING_PROMPT = """You are a fact-checking assistant. Your job is to verify whether
each claim in the ANSWER is supported by the provided SOURCES.

For each distinct claim in the answer, determine if it is:
- "supported": clearly backed by information in the sources
- "unsupported": not found in any source (possible hallucination)
- "partial": partially supported but includes details not in sources

Respond with valid JSON:
{
  "claims": [
    {"claim": "...", "status": "supported|unsupported|partial", "source_ref": "which source supports it or null"}
  ],
  "overall_grounded": true/false,
  "confidence": 0.0-1.0
}
"""


class HallucinationDetector:
    """Detects unsupported claims in RAG pipeline answers."""

    def __init__(self, openai_client: AsyncOpenAI, settings: Settings):
        self._openai = openai_client
        self._settings = settings

    async def check(
        self,
        answer: str,
        sources: list[dict],
    ) -> dict:
        """
        Verify answer grounding against sources.

        Returns:
            {
                "claims": [...],
                "supported_count": int,
                "unsupported_count": int,
                "confidence": float,
                "flags": [str]
            }
        """
        if not answer or not sources:
            return {
                "claims": [],
                "supported_count": 0,
                "unsupported_count": 0,
                "confidence": 0.0,
                "flags": ["No answer or sources to check."],
            }

        # Format sources for the prompt
        sources_text = "\n\n".join(
            f"[Source {i+1}]: {s.get('content', s.get('chunk_content', ''))[:600]}"
            for i, s in enumerate(sources)
        )

        try:
            response = await self._openai.chat.completions.create(
                model=self._settings.llm_model,
                messages=[
                    {"role": "system", "content": GROUNDING_PROMPT},
                    {
                        "role": "user",
                        "content": (
                            f"SOURCES:\n{sources_text}\n\n"
                            f"ANSWER:\n{answer}"
                        ),
                    },
                ],
                temperature=0.0,
                max_tokens=1024,
                response_format={"type": "json_object"},
            )

            result = json.loads(response.choices[0].message.content)
            claims = result.get("claims", [])

            supported = sum(1 for c in claims if c.get("status") == "supported")
            unsupported = sum(1 for c in claims if c.get("status") == "unsupported")
            partial = sum(1 for c in claims if c.get("status") == "partial")
            total = len(claims) if claims else 1

            confidence = result.get("confidence", supported / total)

            flags = []
            for claim in claims:
                if claim.get("status") in ("unsupported", "partial"):
                    flags.append(
                        f"[{claim['status']}] {claim.get('claim', 'Unknown claim')}"
                    )

            return {
                "claims": claims,
                "supported_count": supported,
                "unsupported_count": unsupported,
                "partial_count": partial,
                "confidence": round(confidence, 3),
                "flags": flags,
            }

        except (json.JSONDecodeError, KeyError) as exc:
            logger.warning("Hallucination check parse error: %s", exc)
            return {
                "claims": [],
                "supported_count": 0,
                "unsupported_count": 0,
                "confidence": 0.5,
                "flags": [f"Hallucination check failed: {exc}"],
            }
        except Exception as exc:
            logger.warning("Hallucination check API error: %s", exc)
            return {
                "claims": [],
                "supported_count": 0,
                "unsupported_count": 0,
                "confidence": 0.5,
                "flags": [f"Hallucination check unavailable: {exc}"],
            }
