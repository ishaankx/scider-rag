"""Unit tests for the circuit breaker."""

import pytest

from src.security.circuit_breaker import CircuitBreaker, CircuitBreakerOpen, CircuitState


class TestCircuitBreaker:
    @pytest.fixture
    def breaker(self):
        return CircuitBreaker(
            service_name="test_service",
            failure_threshold=3,
            reset_timeout=0.1,  # Short timeout for tests
        )

    async def test_starts_closed(self, breaker):
        assert breaker.state == CircuitState.CLOSED

    async def test_stays_closed_on_success(self, breaker):
        async with breaker:
            pass
        assert breaker.state == CircuitState.CLOSED

    async def test_opens_after_threshold_failures(self, breaker):
        for _ in range(3):
            with pytest.raises(ValueError):
                async with breaker:
                    raise ValueError("simulated failure")

        assert breaker.state == CircuitState.OPEN

    async def test_open_circuit_rejects_calls(self, breaker):
        # Trip the breaker
        for _ in range(3):
            with pytest.raises(ValueError):
                async with breaker:
                    raise ValueError("simulated")

        # Next call should be rejected immediately
        with pytest.raises(CircuitBreakerOpen) as exc_info:
            async with breaker:
                pass

        assert "test_service" in str(exc_info.value)

    async def test_half_open_after_timeout(self, breaker):
        import asyncio

        # Trip the breaker
        for _ in range(3):
            with pytest.raises(ValueError):
                async with breaker:
                    raise ValueError("simulated")

        assert breaker.state == CircuitState.OPEN

        # Wait for reset timeout
        await asyncio.sleep(0.15)

        # Should allow a probe request (transitions to HALF_OPEN)
        async with breaker:
            pass

        assert breaker.state == CircuitState.CLOSED

    async def test_half_open_failure_reopens(self, breaker):
        import asyncio

        # Trip the breaker
        for _ in range(3):
            with pytest.raises(ValueError):
                async with breaker:
                    raise ValueError("simulated")

        await asyncio.sleep(0.15)

        # Probe fails → back to OPEN
        with pytest.raises(ValueError):
            async with breaker:
                raise ValueError("probe failed")

        assert breaker.state == CircuitState.OPEN

    async def test_success_resets_failure_count(self, breaker):
        # 2 failures (below threshold)
        for _ in range(2):
            with pytest.raises(ValueError):
                async with breaker:
                    raise ValueError("simulated")

        # 1 success resets counter
        async with breaker:
            pass

        assert breaker.state == CircuitState.CLOSED

        # 2 more failures should NOT open (counter was reset)
        for _ in range(2):
            with pytest.raises(ValueError):
                async with breaker:
                    raise ValueError("simulated")

        assert breaker.state == CircuitState.CLOSED
