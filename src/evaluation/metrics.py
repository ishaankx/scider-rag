"""
Evaluation metrics.
Uses LLM-as-judge for correctness scoring when expected answers are provided.
"""

import json
import logging

from openai import AsyncOpenAI

from src.config import Settings

logger = logging.getLogger(__name__)

JUDGE_PROMPT = """You are an evaluation judge. Compare the GENERATED answer
against the EXPECTED answer for a scientific research question.

Score the generated answer on a scale of 0.0 to 1.0:
- 1.0: Perfectly matches the expected answer in meaning
- 0.7-0.9: Mostly correct with minor differences
- 0.4-0.6: Partially correct, missing key information
- 0.1-0.3: Mostly wrong but has some relevant content
- 0.0: Completely wrong or irrelevant

Respond with valid JSON: {"score": 0.0-1.0, "reasoning": "brief explanation"}
"""


async def compute_correctness(
    answer: str,
    expected: str,
    openai_client: AsyncOpenAI,
    settings: Settings,
) -> float:
    """
    Use LLM-as-judge to score answer correctness against expected answer.
    Returns a float between 0.0 and 1.0.
    """
    try:
        response = await openai_client.chat.completions.create(
            model=settings.llm_model,
            messages=[
                {"role": "system", "content": JUDGE_PROMPT},
                {
                    "role": "user",
                    "content": (
                        f"EXPECTED ANSWER:\n{expected}\n\n"
                        f"GENERATED ANSWER:\n{answer}"
                    ),
                },
            ],
            temperature=0.0,
            max_tokens=256,
            response_format={"type": "json_object"},
        )

        result = json.loads(response.choices[0].message.content)
        score = float(result.get("score", 0.0))
        return round(min(max(score, 0.0), 1.0), 3)

    except Exception as exc:
        logger.warning("Correctness scoring failed: %s", exc)
        return 0.0
