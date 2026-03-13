"""
Vision-based OCR and image analysis using OpenAI GPT-4o.

Handles three scenarios:
1. Scanned/image-only PDF pages → OCR text extraction via vision
2. Embedded figures/charts in PDFs → descriptive text generation
3. Post-OCR cleanup → the vision model inherently produces clean text
"""

import base64
import logging

from openai import AsyncOpenAI

logger = logging.getLogger(__name__)

# Prompts tuned for scientific documents
OCR_PROMPT = (
    "Extract ALL text from this scanned document page. "
    "Preserve the original structure (headings, paragraphs, lists, tables). "
    "Format mathematical equations in LaTeX notation where possible. "
    "Fix obvious OCR artifacts but do not add or remove content. "
    "Return only the extracted text, no commentary."
)

IMAGE_ANALYSIS_PROMPT = (
    "Describe this scientific figure, chart, or diagram in detail. "
    "Include: type of visualization, axis labels, data trends, key values, "
    "legends, annotations, and any conclusions that can be drawn. "
    "If it contains a table, reproduce the data in a readable text format. "
    "Be precise and quantitative where possible."
)


async def ocr_page_image(
    openai_client: AsyncOpenAI,
    image_bytes: bytes,
    model: str = "gpt-4o-mini",
    page_number: int = 1,
) -> str:
    """
    Extract text from a scanned PDF page image using vision.

    Args:
        openai_client: Async OpenAI client.
        image_bytes: PNG bytes of the rendered page.
        model: Vision-capable model to use.
        page_number: Page number (for logging).

    Returns:
        Extracted text from the page.
    """
    b64 = base64.b64encode(image_bytes).decode("utf-8")

    response = await openai_client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": OCR_PROMPT},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{b64}",
                            "detail": "high",
                        },
                    },
                ],
            }
        ],
        max_tokens=4096,
        temperature=0.0,
    )

    text = response.choices[0].message.content or ""
    logger.debug("OCR page %d: extracted %d chars", page_number, len(text))
    return text.strip()


async def analyze_image(
    openai_client: AsyncOpenAI,
    image_bytes: bytes,
    model: str = "gpt-4o-mini",
    context: str = "",
) -> str:
    """
    Generate a detailed description of a scientific figure or chart.

    Args:
        openai_client: Async OpenAI client.
        image_bytes: PNG/JPEG bytes of the extracted image.
        model: Vision-capable model to use.
        context: Optional surrounding text for better interpretation.

    Returns:
        Descriptive text about the image content.
    """
    b64 = base64.b64encode(image_bytes).decode("utf-8")

    prompt = IMAGE_ANALYSIS_PROMPT
    if context:
        prompt += f"\n\nSurrounding document context:\n{context[:500]}"

    response = await openai_client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{b64}",
                            "detail": "high",
                        },
                    },
                ],
            }
        ],
        max_tokens=2048,
        temperature=0.0,
    )

    text = response.choices[0].message.content or ""
    logger.debug("Image analysis: %d chars", len(text))
    return text.strip()
