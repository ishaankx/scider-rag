"""
Sandboxed Python code execution tool.
Runs untrusted code in a restricted subprocess with strict limits.
"""

import asyncio
import logging
import os
import tempfile

from src.agents.tools.base import BaseTool, ToolResult
from src.config import Settings

logger = logging.getLogger(__name__)

# Modules that are safe to import in the sandbox
ALLOWED_IMPORTS = {
    "math", "statistics", "collections", "itertools",
    "functools", "decimal", "fractions", "datetime",
    "json", "re", "string", "textwrap",
}

# Modules explicitly blocked
BLOCKED_IMPORTS = {
    "os", "sys", "subprocess", "shutil", "pathlib",
    "socket", "http", "urllib", "requests", "httpx",
    "importlib", "ctypes", "signal", "threading",
    "multiprocessing", "pickle", "shelve",
}


class CodeExecutorTool(BaseTool):
    """Executes Python code in a sandboxed subprocess."""

    def __init__(self, settings: Settings):
        self._timeout = settings.sandbox_timeout_seconds
        self._max_memory_mb = settings.sandbox_max_memory_mb

    @property
    def name(self) -> str:
        return "code_executor"

    @property
    def description(self) -> str:
        return (
            "Execute Python code for data analysis and computation. "
            "The code runs in a sandbox with limited imports: math, statistics, "
            "collections, datetime, json, re. "
            "No file system, network, or OS access. "
            "Use print() to output results."
        )

    @property
    def parameters_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "Python code to execute. Use print() for output.",
                },
            },
            "required": ["code"],
        }

    async def execute(self, **params) -> ToolResult:
        code = params.get("code", "").strip()

        if not code:
            return ToolResult(output="", success=False, error="No code provided.")

        # Static analysis: check for blocked imports
        violation = _check_imports(code)
        if violation:
            return ToolResult(
                output="",
                success=False,
                error=f"Blocked import: '{violation}'. Only allowed: {sorted(ALLOWED_IMPORTS)}",
            )

        # Check for dangerous patterns
        danger = _check_dangerous_patterns(code)
        if danger:
            return ToolResult(output="", success=False, error=f"Blocked: {danger}")

        # Write code to temp file and execute in subprocess
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".py", delete=False
            ) as f:
                # Wrap with resource limits
                wrapper = _build_wrapper(code, self._max_memory_mb, self._timeout)
                f.write(wrapper)
                f.flush()
                temp_path = f.name

            try:
                process = await asyncio.create_subprocess_exec(
                    "python3", temp_path,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    # Prevent child process from inheriting env vars with secrets
                    env=_safe_env(),
                )

                stdout, stderr = await asyncio.wait_for(
                    process.communicate(),
                    timeout=self._timeout,
                )

                output = stdout.decode("utf-8", errors="replace").strip()
                errors = stderr.decode("utf-8", errors="replace").strip()

                # Truncate excessive output
                if len(output) > 5000:
                    output = output[:5000] + "\n... (output truncated)"

                if process.returncode != 0:
                    return ToolResult(
                        output=output,
                        success=False,
                        error=errors[:1000] if errors else "Code execution failed.",
                    )

                return ToolResult(output=output or "(no output)")

            except asyncio.TimeoutError:
                process.kill()
                try:
                    await asyncio.wait_for(process.wait(), timeout=2)
                except asyncio.TimeoutError:
                    pass  # Process refused to die — OS will reap it
                return ToolResult(
                    output="",
                    success=False,
                    error=f"Execution timed out after {self._timeout} seconds.",
                )
        finally:
            # Clean up temp file
            try:
                os.unlink(temp_path)
            except OSError:
                pass


def _check_imports(code: str) -> str | None:
    """Check imports against the allowlist. Only ALLOWED_IMPORTS may be used."""
    for line in code.split("\n"):
        stripped = line.strip()
        if stripped.startswith("import ") or stripped.startswith("from "):
            # Explicitly blocked modules (fast reject)
            for blocked in BLOCKED_IMPORTS:
                if blocked in stripped:
                    return blocked

            # Allowlist enforcement: extract module name and verify
            module = _extract_module_name(stripped)
            if module and module not in ALLOWED_IMPORTS:
                return module
    return None


def _extract_module_name(statement: str) -> str | None:
    """Extract the top-level module name from an import statement."""
    # "from foo.bar import baz" → "foo"
    # "import foo" → "foo"
    # "import foo.bar" → "foo"
    statement = statement.strip()
    if statement.startswith("from "):
        parts = statement[5:].split()
        if parts:
            return parts[0].split(".")[0]
    elif statement.startswith("import "):
        parts = statement[7:].split(",")[0].split()
        if parts:
            return parts[0].split(".")[0]
    return None


def _check_dangerous_patterns(code: str) -> str | None:
    """Check for obviously dangerous patterns."""
    patterns = {
        "exec(": "Dynamic code execution not allowed",
        "eval(": "Dynamic evaluation not allowed",
        "__import__": "Dynamic imports not allowed",
        "globals()": "Accessing globals not allowed",
        "locals()": "Accessing locals not allowed",
        "open(": "File access not allowed",
        "compile(": "Code compilation not allowed",
    }
    for pattern, message in patterns.items():
        if pattern in code:
            return message
    return None


def _build_wrapper(user_code: str, max_memory_mb: int, cpu_timeout: int = 10) -> str:
    """Wrap user code with resource limits."""
    return f"""
import resource
import sys

# Set memory limit
memory_bytes = {max_memory_mb} * 1024 * 1024
resource.setrlimit(resource.RLIMIT_AS, (memory_bytes, memory_bytes))

# Restrict CPU time (synced with sandbox_timeout_seconds)
resource.setrlimit(resource.RLIMIT_CPU, ({cpu_timeout}, {cpu_timeout}))

# Run user code
try:
{_indent(user_code, 4)}
except MemoryError:
    print("Error: Out of memory", file=sys.stderr)
    sys.exit(1)
except Exception as e:
    print(f"Error: {{type(e).__name__}}: {{e}}", file=sys.stderr)
    sys.exit(1)
"""


def _indent(code: str, spaces: int) -> str:
    """Indent every line of code by N spaces."""
    prefix = " " * spaces
    return "\n".join(prefix + line for line in code.split("\n"))


def _safe_env() -> dict:
    """Minimal environment for the sandbox subprocess."""
    return {
        "PATH": "/usr/bin:/usr/local/bin",
        "HOME": "/tmp",
        "LANG": "en_US.UTF-8",
    }
