"""Abstract base class for agent tools."""

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class ToolResult:
    """Uniform result from any tool execution."""

    output: str
    success: bool = True
    error: str | None = None


class BaseTool(ABC):
    """
    Interface for tools that agents can invoke.
    Each tool is described in natural language so the LLM can decide when to use it.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique tool name used in function calls."""
        ...

    @property
    @abstractmethod
    def description(self) -> str:
        """Natural-language description of what this tool does."""
        ...

    @property
    @abstractmethod
    def parameters_schema(self) -> dict:
        """JSON Schema for the tool's parameters."""
        ...

    @abstractmethod
    async def execute(self, **params) -> ToolResult:
        """Run the tool with the given parameters."""
        ...

    def to_openai_tool(self) -> dict:
        """Convert to OpenAI function-calling format."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters_schema,
            },
        }
