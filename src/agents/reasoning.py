"""
Reasoning Agent.
Synthesizes a grounded answer using retrieved context and tools.
Follows the ReAct pattern: Reason → Act (call tool) → Observe → ... → Answer.
"""

import json
import logging
import time

from openai import AsyncOpenAI

from src.agents.base import AgentContext, AgentResult, BaseAgent
from src.agents.tools.base import BaseTool, ToolResult
from src.config import Settings

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a scientific research assistant. Your job is to answer
research questions accurately using ONLY the provided sources and tools.

CRITICAL RULES:
1. Base your answer ONLY on the provided sources. Do NOT use prior knowledge.
2. Cite sources using [1], [2], etc. corresponding to the source numbers below.
3. If the sources don't contain enough information, say so explicitly.
4. If sources conflict, acknowledge the discrepancy.
5. Use tools (calculator, graph_traverse, code_executor) when computation or
   relationship exploration would improve your answer.
6. Keep answers concise but thorough. Use bullet points for clarity.

You have access to the following tools — call them when they would help:
{tool_descriptions}
"""


class ReasoningAgent(BaseAgent):
    """
    Produces the final answer using ReAct-style tool use.
    Makes up to N iterations of tool calls before producing a final answer.
    """

    def __init__(
        self,
        openai_client: AsyncOpenAI,
        tools: list[BaseTool],
        settings: Settings,
    ):
        self._openai = openai_client
        self._tools = {t.name: t for t in tools}
        self._settings = settings
        self._max_iterations = settings.agent_max_iterations

    @property
    def name(self) -> str:
        return "reasoning_agent"

    async def execute(self, context: AgentContext) -> AgentResult:
        start = time.perf_counter()
        tools_used = []

        # Build the context message with retrieved sources
        sources_text = self._format_sources(context.retrieved_chunks)

        # Build system prompt with tool descriptions
        tool_desc = "\n".join(
            f"- {t.name}: {t.description}" for t in self._tools.values()
        )
        system = SYSTEM_PROMPT.format(tool_descriptions=tool_desc)

        # Build OpenAI tools list for function calling
        openai_tools = [t.to_openai_tool() for t in self._tools.values()]

        messages = [
            {"role": "system", "content": system},
            {
                "role": "user",
                "content": (
                    f"Question: {context.question}\n\n"
                    f"Retrieved Sources:\n{sources_text}\n\n"
                    "Please answer the question using the sources above. "
                    "Use tools if you need to compute something or explore entity relationships."
                ),
            },
        ]

        # ReAct loop: let the LLM call tools iteratively
        final_answer = ""
        for iteration in range(self._max_iterations):
            response = await self._openai.chat.completions.create(
                model=self._settings.llm_model,
                messages=messages,
                tools=openai_tools if openai_tools else None,
                temperature=self._settings.llm_temperature,
                max_tokens=self._settings.llm_max_tokens,
            )

            choice = response.choices[0]

            # If the model produced a final text answer (no tool calls)
            if choice.finish_reason == "stop" or not choice.message.tool_calls:
                final_answer = choice.message.content or ""
                break

            # Process tool calls
            messages.append(choice.message)

            for tool_call in choice.message.tool_calls:
                func_name = tool_call.function.name
                tools_used.append(func_name)

                try:
                    func_args = json.loads(tool_call.function.arguments)
                except json.JSONDecodeError:
                    func_args = {}

                # Execute the tool
                tool = self._tools.get(func_name)
                if tool:
                    logger.info("Reasoning agent calling tool: %s(%s)", func_name, func_args)
                    result = await tool.execute(**func_args)
                    tool_output = result.output if result.success else f"Error: {result.error}"
                else:
                    tool_output = f"Unknown tool: {func_name}"

                # Append tool result to conversation
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": tool_output[:2000],  # Truncate long outputs
                })

                context.tool_results.append({
                    "tool": func_name,
                    "args": func_args,
                    "output": tool_output[:500],
                    "success": result.success if tool else False,
                })
        else:
            # Hit max iterations — force a final answer
            messages.append({
                "role": "user",
                "content": "Please provide your final answer now based on everything above.",
            })
            response = await self._openai.chat.completions.create(
                model=self._settings.llm_model,
                messages=messages,
                temperature=self._settings.llm_temperature,
                max_tokens=self._settings.llm_max_tokens,
            )
            final_answer = response.choices[0].message.content or ""

        elapsed_ms = (time.perf_counter() - start) * 1000

        return AgentResult(
            output=final_answer,
            sources=context.retrieved_chunks,
            tool_calls_made=tools_used,
            latency_ms=elapsed_ms,
        )

    def _format_sources(self, chunks: list[dict]) -> str:
        """Format retrieved chunks as numbered sources for the LLM."""
        if not chunks:
            return "(No sources retrieved)"

        parts = []
        for i, chunk in enumerate(chunks, 1):
            title = chunk.get("document_title", "Unknown")
            content = chunk.get("content", "")[:800]
            score = chunk.get("relevance_score", 0)
            parts.append(f"[{i}] (Source: {title}, Relevance: {score:.2f})\n{content}")

        return "\n\n".join(parts)
