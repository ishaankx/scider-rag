"""
Abstract base classes for agents and their execution context.
All agents follow the same interface (Liskov Substitution Principle).
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class AgentContext:
    """Shared context passed between agents during a query."""

    question: str
    filters: dict | None = None
    max_sources: int = 5
    retrieved_chunks: list[dict] = field(default_factory=list)
    tool_results: list[dict] = field(default_factory=list)
    agent_messages: list[dict] = field(default_factory=list)


@dataclass
class AgentResult:
    """Uniform result from any agent."""

    output: Any
    sources: list[dict] = field(default_factory=list)
    tool_calls_made: list[str] = field(default_factory=list)
    latency_ms: float = 0.0
    metadata: dict = field(default_factory=dict)


class BaseAgent(ABC):
    """
    Interface that all agents implement.
    Each agent has a single responsibility (SRP) and can be swapped
    independently (Dependency Inversion).
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable agent name for logging and tracing."""
        ...

    @abstractmethod
    async def execute(self, context: AgentContext) -> AgentResult:
        """Run the agent's logic and return a result."""
        ...
