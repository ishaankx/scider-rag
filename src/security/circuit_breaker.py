"""
Lightweight circuit breaker for external API calls.
Prevents cascading failures by short-circuiting calls to unhealthy services.

States:
  CLOSED  → requests flow normally; failures are counted
  OPEN    → requests are rejected immediately; after reset_timeout, move to HALF_OPEN
  HALF_OPEN → one probe request is allowed; success → CLOSED, failure → OPEN
"""

import asyncio
import logging
import time
from enum import Enum

logger = logging.getLogger(__name__)


class CircuitState(Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitBreakerOpen(Exception):
    """Raised when a call is rejected because the circuit is open."""

    def __init__(self, service: str, retry_after: float):
        self.service = service
        self.retry_after = retry_after
        super().__init__(
            f"Circuit breaker OPEN for '{service}'. Retry after {retry_after:.0f}s."
        )


class CircuitBreaker:
    """
    Async-safe circuit breaker.

    Args:
        service_name: Label for logging (e.g. "openai").
        failure_threshold: Consecutive failures before opening the circuit.
        reset_timeout: Seconds to wait in OPEN state before probing.
    """

    def __init__(
        self,
        service_name: str = "external",
        failure_threshold: int = 5,
        reset_timeout: float = 30.0,
    ):
        self._service = service_name
        self._failure_threshold = failure_threshold
        self._reset_timeout = reset_timeout

        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._last_failure_time: float = 0
        self._lock = asyncio.Lock()

    @property
    def state(self) -> CircuitState:
        return self._state

    async def __aenter__(self):
        await self._before_call()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if exc_type is None:
            await self._on_success()
        else:
            await self._on_failure()
        return False  # Don't suppress the exception

    async def _before_call(self) -> None:
        async with self._lock:
            if self._state == CircuitState.CLOSED:
                return

            if self._state == CircuitState.OPEN:
                elapsed = time.monotonic() - self._last_failure_time
                if elapsed >= self._reset_timeout:
                    logger.info(
                        "Circuit breaker %s: OPEN → HALF_OPEN (probing)", self._service
                    )
                    self._state = CircuitState.HALF_OPEN
                    return
                raise CircuitBreakerOpen(
                    self._service, self._reset_timeout - elapsed
                )

            # HALF_OPEN: allow the probe request through

    async def _on_success(self) -> None:
        async with self._lock:
            if self._state == CircuitState.HALF_OPEN:
                logger.info(
                    "Circuit breaker %s: HALF_OPEN → CLOSED (probe succeeded)",
                    self._service,
                )
            self._state = CircuitState.CLOSED
            self._failure_count = 0

    async def _on_failure(self) -> None:
        async with self._lock:
            self._failure_count += 1
            self._last_failure_time = time.monotonic()

            if self._state == CircuitState.HALF_OPEN:
                self._state = CircuitState.OPEN
                logger.warning(
                    "Circuit breaker %s: HALF_OPEN → OPEN (probe failed)",
                    self._service,
                )
            elif self._failure_count >= self._failure_threshold:
                self._state = CircuitState.OPEN
                logger.warning(
                    "Circuit breaker %s: CLOSED → OPEN after %d consecutive failures",
                    self._service,
                    self._failure_count,
                )
