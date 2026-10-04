"""Subprocess-isolated Python execution for the ``run_python`` tool.

This gives the model a place to compute things it cannot do in its head.

.. warning::
   This is **not** a security sandbox.  It runs a real interpreter with a
   real filesystem and, if installed, real site-packages.  It adds process
   isolation (``-I``), a wall-clock timeout, no stdin and bounded output so
   that a runaway snippet cannot wedge the UI -- it does *not* stop code from
   touching your machine.  Only run this on a machine you trust the model with.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

__all__ = ["run_python"]

DEFAULT_TIMEOUT_S = 20
DEFAULT_MAX_OUTPUT = 8_000


def run_python(
    code: str,
    *,
    cwd: Path | str,
    timeout: int = DEFAULT_TIMEOUT_S,
    max_output: int = DEFAULT_MAX_OUTPUT,
    python: str | None = None,
) -> str:
    """Run *code* in a fresh interpreter rooted at *cwd* and return the output.

    ``-I`` (isolated mode) implies ``-E -P -s``: environment variables, the
    current directory and user site-packages are all kept out of the child so
    a snippet cannot influence how it is loaded or pick up local config.
    """
    if not code or not code.strip():
        return "Error: no code was supplied."

    workdir = Path(cwd)
    if not workdir.is_dir():
        return f"Error: working directory does not exist: {workdir}"

    interpreter = python or sys.executable
    command = [interpreter, "-I", "-c", code]

    try:
        proc = subprocess.run(  # noqa: S603 - fixed argv, no shell
            command,
            cwd=str(workdir),
            stdin=subprocess.DEVNULL,
            capture_output=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return f"Error: execution timed out after {timeout} seconds."
    except OSError as exc:
        return f"Error: could not start the interpreter: {exc}"

    return _format(proc.stdout, proc.stderr, proc.returncode, max_output)


def _format(stdout: str, stderr: str, returncode: int, max_output: int) -> str:
    parts: list[str] = []

    if stdout:
        parts.append(stdout.rstrip("\n"))
    if stderr:
        parts.append("[stderr]\n" + stderr.rstrip("\n"))

    if not parts:
        body = "(no output)"
        if returncode:
            body += f"\n[exit code {returncode}]"
        return body

    text = "\n".join(parts)
    if returncode:
        text += f"\n[exit code {returncode}]"

    if len(text) > max_output:
        text = text[:max_output] + f"\n... [truncated at {max_output} characters]"
    return text
