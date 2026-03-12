"""Unit tests for the calculator tool."""

import pytest

from src.agents.tools.calculator import CalculatorTool


@pytest.fixture
def calculator():
    return CalculatorTool()


class TestCalculatorTool:
    @pytest.mark.asyncio
    async def test_basic_arithmetic(self, calculator):
        result = await calculator.execute(expression="2 + 3")
        assert result.success
        assert result.output == "5"

    @pytest.mark.asyncio
    async def test_multiplication(self, calculator):
        result = await calculator.execute(expression="7 * 8")
        assert result.success
        assert result.output == "56"

    @pytest.mark.asyncio
    async def test_division(self, calculator):
        result = await calculator.execute(expression="10 / 3")
        assert result.success
        assert float(result.output) == pytest.approx(3.333, rel=1e-2)

    @pytest.mark.asyncio
    async def test_sqrt(self, calculator):
        result = await calculator.execute(expression="sqrt(144)")
        assert result.success
        assert float(result.output) == 12.0

    @pytest.mark.asyncio
    async def test_complex_expression(self, calculator):
        result = await calculator.execute(expression="sqrt(9) + 2 ** 3")
        assert result.success
        assert float(result.output) == 11.0

    @pytest.mark.asyncio
    async def test_division_by_zero(self, calculator):
        result = await calculator.execute(expression="1 / 0")
        assert not result.success

    @pytest.mark.asyncio
    async def test_rejects_large_exponent(self, calculator):
        result = await calculator.execute(expression="2 ** 10000")
        assert not result.success

    @pytest.mark.asyncio
    async def test_empty_expression(self, calculator):
        result = await calculator.execute(expression="")
        assert not result.success

    @pytest.mark.asyncio
    async def test_rejects_dangerous_code(self, calculator):
        result = await calculator.execute(expression="__import__('os').system('ls')")
        assert not result.success
