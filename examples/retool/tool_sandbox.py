# Modified for COPC.
"""Restricted Python subprocesses and tool dispatch for ReTool."""

from __future__ import annotations

import ast
import asyncio
import gc
import os
import re
import tempfile
from contextlib import contextmanager
from copy import deepcopy
from typing import Any

import psutil

from slime.utils.retool_protocol import RETOOL_TOOLS

TOOL_CONFIGS = {
    "max_turns": int(os.environ.get("MAX_TURNS", 16)),
    "max_tool_calls": int(os.environ.get("MAX_TOOL_CALLS", 16)),
    "tool_concurrency": 128,  # maximum concurrent subprocesses
    "python_timeout": 30,  # seconds per Python tool call
    "python_memory_limit": "4GB",
    "python_cpu_limit": 1,
    # Parent-process RSS thresholds in MiB; excludes tool subprocesses.
    "max_memory_usage": 65536,  # 64 GiB admission threshold
    "cleanup_threshold": 32768,
    "aggressive_cleanup_threshold": 16384,
    "force_cleanup_threshold": 49152,
}

SEMAPHORE = asyncio.Semaphore(TOOL_CONFIGS["tool_concurrency"])


def get_memory_usage() -> float:
    """Return the current process's resident memory in MiB."""
    process = psutil.Process()
    return process.memory_info().rss / 1024 / 1024


def cleanup_memory():
    gc.collect()


def aggressive_cleanup_memory():
    """Run garbage collection and invoke available numerical-library cache hooks."""
    for _ in range(3):
        gc.collect()

    import sys

    if hasattr(sys, "modules"):
        modules_to_clear = ["numpy", "pandas", "matplotlib", "scipy"]
        for module_name in modules_to_clear:
            if module_name in sys.modules:
                module = sys.modules[module_name]
                if hasattr(module, "clear_cache"):
                    module.clear_cache()


def check_and_cleanup_memory():
    """Choose a cleanup level from the parent process's resident memory."""
    current_memory = get_memory_usage()

    if current_memory > TOOL_CONFIGS["force_cleanup_threshold"]:
        aggressive_cleanup_memory()
        return f"Warning: High memory usage ({current_memory:.1f}MB), performed aggressive cleanup"
    elif current_memory > TOOL_CONFIGS["cleanup_threshold"]:
        cleanup_memory()
        return f"Info: Memory usage ({current_memory:.1f}MB), performed cleanup"
    elif current_memory > TOOL_CONFIGS["aggressive_cleanup_threshold"]:
        gc.collect()
        return f"Info: Memory usage ({current_memory:.1f}MB), performed light cleanup"

    return None


class PythonSandbox:
    """Restricted execution backend; deploy workers with external isolation."""

    def __init__(self, timeout: int = 10, memory_limit: str = "100MB"):
        self.timeout = timeout
        self.memory_limit = memory_limit
        self.allowed_modules = {
            "math",
            "random",
            "datetime",
            "collections",
            "itertools",
            "functools",
            "operator",
            "statistics",
            "decimal",
            "fractions",
            # These packages must be installed in the execution image.
            "numpy",
            "sympy",
            "scipy",
        }

    def _check_code_safety(self, code: str) -> tuple[bool, str]:
        """Reject disallowed imports and operations; worker isolation is still required."""
        if not isinstance(code, str):
            return False, "Code must be a string"
        try:
            tree = ast.parse(code)
        except (SyntaxError, ValueError) as error:
            return False, f"Invalid Python code: {error}"

        # Block library file I/O and native-code loaders.
        blocked_modules = {
            "numpy.ctypeslib", "numpy.f2py", "numpy.distutils", "numpy.testing",
            "scipy.io", "scipy.datasets", "sympy.external",
            "sympy.utilities.autowrap", "sympy.utilities.codegen",
        }
        blocked_members = {
            "load", "save", "loadtxt", "savetxt", "savez", "savez_compressed",
            "fromfile", "tofile", "memmap", "DataSource", "load_library",
            "ctypes", "ctypeslib", "f2py", "autowrap", "ufuncify", "codegen",
            "dump", "dumps", "to_pickle", "read_pickle",
        }
        for node in ast.walk(tree):
            modules = []
            if isinstance(node, ast.Import):
                modules = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                if node.level:
                    return False, "Relative imports are not allowed"
                modules = [node.module or ""]
                # Import aliases must not bypass blocked library entry points.
                for alias in node.names:
                    qualified_name = f"{node.module}.{alias.name}"
                    if alias.name in blocked_members or qualified_name in blocked_modules:
                        return False, f"Use of '{alias.name}' is not allowed"
            for module in modules:
                if module.split(".")[0] not in self.allowed_modules:
                    return False, f"Import of '{module}' is not allowed"
                if any(part.startswith("_") for part in module.split(".")) or any(
                    module == blocked or module.startswith(blocked + ".")
                    for blocked in blocked_modules
                ):
                    return False, f"Import of '{module}' is not allowed"
            if isinstance(node, ast.Attribute) and (
                node.attr in blocked_members or node.attr.startswith("_")
                or node.attr in {"io", "datasets", "external", "testing", "distutils"}
            ):
                return False, f"Use of '{node.attr}' is not allowed"
            if isinstance(node, ast.Name) and node.id in blocked_members:
                return False, f"Use of '{node.id}' is not allowed"

        dangerous_patterns = [
            r"import\s+os",
            r"import\s+sys",
            r"import\s+subprocess",
            r"import\s+shutil",
            r"import\s+glob",
            r"import\s+pathlib",
            r"__import__",
            r"\beval\s*\(",
            r"\bexec\s*\(",
            r"\bopen\s*\(",
            r"\bfile\s*\(",
            r"\binput\s*\(",
            r"\braw_input\s*\(",
            r"\bcompile\s*\(",
            r"\bexecfile\s*\(",
            r"\bgetattr\s*\(",
            r"\bsetattr\s*\(",
            r"\bdelattr\s*\(",
            r"\bhasattr\s*\(",
            r"\bglobals\s*\(",
            r"\blocals\s*\(",
            r"\bvars\s*\(",
            r"\bdir\s*\(",
            r"\btype\s*\(",
            r"\bisinstance\s*\(",
            r"\bissubclass\s*\(",
            r"\bsuper\s*\(",
            r"\bproperty\s*\(",
            r"\bstaticmethod\s*\(",
            r"\bclassmethod\s*\(",
            r"__\w+__",
        ]

        for pattern in dangerous_patterns:
            if re.search(pattern, code, re.IGNORECASE):
                return False, f"Code contains dangerous pattern: {pattern}"

        return True, "Code is safe"

    @contextmanager
    def _create_safe_environment(self):
        """Create a temporary working directory for a stateless execution."""
        temp_dir = tempfile.mkdtemp(prefix="python_sandbox_")

        try:
            script_path = os.path.join(temp_dir, "code.py")

            env = os.environ.copy()
            env["PYTHONPATH"] = temp_dir
            env["PYTHONUNBUFFERED"] = "1"

            yield script_path, env, temp_dir

        finally:
            try:
                import shutil

                shutil.rmtree(temp_dir)
            except Exception:
                pass

    async def execute_code(self, code: str) -> str:
        """Execute Python code in sandbox with safety checks"""
        current_memory = get_memory_usage()
        if current_memory > TOOL_CONFIGS["max_memory_usage"]:
            aggressive_cleanup_memory()
            return "Error: Memory usage too high, please try again"

        is_safe, message = self._check_code_safety(code)
        if not is_safe:
            return f"Error: {message}"

        indented_code = "\n".join("    " + line for line in code.split("\n"))

        wrapped_code = f"""import sys
import traceback
from io import StringIO
import resource

# Set memory limit (4GB)
try:
    resource.setrlimit(resource.RLIMIT_AS, (4 * 1024 * 1024 * 1024, -1))
except Exception:
    pass

# Redirect stdout and stderr
old_stdout = sys.stdout
old_stderr = sys.stderr
stdout_capture = StringIO()
stderr_capture = StringIO()
sys.stdout = stdout_capture
sys.stderr = stderr_capture

try:
    # User code
{indented_code}
    
    # Get output
    stdout_output = stdout_capture.getvalue()
    stderr_output = stderr_capture.getvalue()
    
    # Restore standard output
    sys.stdout = old_stdout
    sys.stderr = old_stderr
    
    # Return result
    result = ""
    if stdout_output:
        result += f"Output:\\n{{stdout_output}}"
    if stderr_output:
        result += f"\\nErrors:\\n{{stderr_output}}"
    
    print(result)
    
except ImportError as e:
    sys.stdout = old_stdout
    sys.stderr = old_stderr
    print(f"Error: Required Python dependency unavailable in the sandbox: {{e}}")
except Exception as e:
    # Restore standard output
    sys.stdout = old_stdout
    sys.stderr = old_stderr
    
    # Return error information
    error_msg = f"Error: {{str(e)}}\\nTraceback:\\n{{traceback.format_exc()}}"
    print(error_msg)"""

        with self._create_safe_environment() as (script_path, env, temp_dir):
            with open(script_path, "w") as f:
                f.write(wrapped_code)

            process = None
            spawn_task = None
            try:
                # Shield spawn so cancellation cannot orphan an untracked child.
                spawn_task = asyncio.ensure_future(asyncio.create_subprocess_exec(
                    "python3", script_path,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    env=env,
                    cwd=temp_dir,
                ))
                process = await asyncio.shield(spawn_task)
                stdout_bytes, stderr_bytes = await asyncio.wait_for(
                    process.communicate(), timeout=self.timeout
                )
                stdout = stdout_bytes.decode(errors="replace") if stdout_bytes else ""
                stderr = stderr_bytes.decode(errors="replace") if stderr_bytes else ""

                if process.returncode == 0:
                    result = stdout.strip()
                else:
                    result = f"Error: Process exited with code {process.returncode}\n{stderr}"

            except asyncio.TimeoutError:
                await self._terminate_process(process)
                result = f"Error: Code execution timed out after {self.timeout} seconds"
            except asyncio.CancelledError:
                if process is None and spawn_task is not None:
                    try:
                        process = await asyncio.shield(spawn_task)
                    except Exception:
                        pass
                if process is not None:
                    await asyncio.shield(self._terminate_process(process))
                raise
            except Exception as error:
                if process is not None:
                    await self._terminate_process(process)
                result = f"Error: Failed to execute code: {error}"

            cleanup_message = check_and_cleanup_memory()
            if cleanup_message:
                print(f"Memory cleanup: {cleanup_message}")

            return result


    @staticmethod
    async def _terminate_process(process):
        """Kill and reap a child while draining its output pipes."""
        if process is None:
            return
        if process.returncode is None:
            try:
                process.kill()
            except ProcessLookupError:
                pass
        await process.communicate()


class ToolRegistry:
    """Dispatch registered ReTool calls to the Python executor."""

    def __init__(self):
        self.tools = {}
        self.python_sandbox = PythonSandbox(
            timeout=TOOL_CONFIGS["python_timeout"], memory_limit=TOOL_CONFIGS["python_memory_limit"]
        )
        self._register_default_tools()

    def _register_default_tools(self):
        for specification in RETOOL_TOOLS:
            self.register_tool(specification["function"]["name"], deepcopy(specification))

    def register_tool(self, name: str, tool_spec: dict[str, Any]):
        self.tools[name] = tool_spec

    def get_tool_specs(self) -> list[dict[str, Any]]:
        return list(self.tools.values())

    async def execute_tool(self, tool_name: str, arguments: dict[str, Any]) -> str:
        """Execute a tool call; the rollout caller must already hold SEMAPHORE."""
        if tool_name not in self.tools:
            return f"Error: Tool '{tool_name}' not found"

        if tool_name == "code_interpreter":
            return await self._execute_python(arguments)
        else:
            return f"Error: Tool '{tool_name}' not implemented"

    async def _execute_python(self, arguments: dict[str, Any]) -> str:
        if not isinstance(arguments, dict):
            return "Error: Tool arguments must be an object"
        code = arguments.get("code", "")
        if not isinstance(code, str):
            return "Error: Code must be a string"
        if not code.strip():
            return "Error: No code provided"

        result = await self.python_sandbox.execute_code(code)
        return result


tool_registry = ToolRegistry()
