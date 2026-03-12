"""
Calculator tool for safe mathematical expression evaluation.
Uses ast.literal_eval approach — no eval() or exec() allowed.
"""

import ast
import math
import operator

from src.agents.tools.base import BaseTool, ToolResult

# Allowed operators for safe expression evaluation
_SAFE_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}

# Safe math functions the calculator can use
_SAFE_FUNCTIONS = {
    "abs": abs,
    "round": round,
    "min": min,
    "max": max,
    "sum": sum,
    "sqrt": math.sqrt,
    "log": math.log,
    "log10": math.log10,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "pi": math.pi,
    "e": math.e,
}

# Guard against absurdly large exponents
MAX_EXPONENT = 1000


class CalculatorTool(BaseTool):
    """Safely evaluates mathematical expressions without using eval()."""

    @property
    def name(self) -> str:
        return "calculator"

    @property
    def description(self) -> str:
        return (
            "Evaluate mathematical expressions safely. Supports: +, -, *, /, **, "
            "sqrt(), log(), sin(), cos(), tan(), abs(), round(), min(), max(). "
            "Use this for any numerical computation needed to answer the question."
        )

    @property
    def parameters_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "Mathematical expression to evaluate, e.g., 'sqrt(144) + 3 * 2'",
                },
            },
            "required": ["expression"],
        }

    async def execute(self, **params) -> ToolResult:
        expression = params.get("expression", "").strip()

        if not expression:
            return ToolResult(output="", success=False, error="Empty expression.")

        try:
            tree = ast.parse(expression, mode="eval")
            result = _safe_eval_node(tree.body)
            return ToolResult(output=str(result))
        except (ValueError, TypeError, ZeroDivisionError) as exc:
            return ToolResult(output="", success=False, error=f"Math error: {exc}")
        except Exception as exc:
            return ToolResult(output="", success=False, error=f"Cannot evaluate: {exc}")


def _safe_eval_node(node: ast.AST) -> float | int:
    """Recursively evaluate an AST node using only allowed operations."""

    # Number literal
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value

    # Named constant (pi, e)
    if isinstance(node, ast.Name) and node.id in _SAFE_FUNCTIONS:
        val = _SAFE_FUNCTIONS[node.id]
        if isinstance(val, (int, float)):
            return val
        raise ValueError(f"'{node.id}' is a function, not a constant. Use {node.id}(...).")

    # Unary operator (-x, +x)
    if isinstance(node, ast.UnaryOp) and type(node.op) in _SAFE_OPERATORS:
        operand = _safe_eval_node(node.operand)
        return _SAFE_OPERATORS[type(node.op)](operand)

    # Binary operator (x + y, x ** y)
    if isinstance(node, ast.BinOp) and type(node.op) in _SAFE_OPERATORS:
        left = _safe_eval_node(node.left)
        right = _safe_eval_node(node.right)

        # Guard against absurd exponents
        if isinstance(node.op, ast.Pow) and isinstance(right, (int, float)):
            if abs(right) > MAX_EXPONENT:
                raise ValueError(f"Exponent {right} exceeds maximum of {MAX_EXPONENT}.")

        return _SAFE_OPERATORS[type(node.op)](left, right)

    # Function call: sqrt(x), log(x), etc.
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
        func_name = node.func.id
        if func_name not in _SAFE_FUNCTIONS:
            raise ValueError(f"Function '{func_name}' is not allowed.")

        func = _SAFE_FUNCTIONS[func_name]
        if not callable(func):
            raise ValueError(f"'{func_name}' is not callable.")

        args = [_safe_eval_node(arg) for arg in node.args]
        return func(*args)

    raise ValueError(f"Unsupported expression node: {ast.dump(node)}")
