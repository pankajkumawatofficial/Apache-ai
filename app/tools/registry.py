"""Where LangChain meets Apache's capabilities.

This is the only module that imports :mod:`langchain.tools`.  Each ``@tool``
is a thin adapter over a plain-Python implementation in a sibling module, so
the substantive logic stays independently testable while the agent still sees
ordinary, well-described LangChain tools.

Tool descriptions are written for the model, not the reader: they are what the
model uses to decide whether to call a tool at all.
"""

from __future__ import annotations

import datetime
from pathlib import Path
from typing import Any

from ..config import Settings, settings as default_settings
from ..rag import DocumentStore, format_hits
from .calculator import CalcError, safe_eval
from .control import ControlError, open_app, open_file, open_url
from .files import FileAccessError, list_files, read_file, write_file
from .sandbox import run_python

__all__ = ["build_tools"]


def build_tools(
    store: DocumentStore,
    workspace: Path,
    settings: Settings | None = None,
    *,
    include_web: bool = True,
) -> list[Any]:
    """Build the tool list handed to ``create_agent``.

    Parameters
    ----------
    store:
        The live document index; ``search_documents`` reads it at call time so
        newly ingested documents are visible without rebuilding the agent.
    workspace:
        Root directory that every file tool is confined to.
    include_web:
        Set ``False`` to drop ``web_search`` (offline machines, or when the
        ``ddgs`` package is unavailable).
    """
    from langchain.tools import tool

    cfg = settings or default_settings
    root = Path(workspace).resolve()

    @tool
    def calculator(expression: str) -> str:
        """Evaluate an arithmetic expression and return the number.

        Use this for any calculation beyond trivial mental math. Supports
        + - * / // % **, parentheses, and functions such as sqrt, sin, cos,
        log, exp, abs, round, min, max, factorial, and the constants pi and e.
        Example: "sqrt(2) * 100" or "2 ** 10".
        """
        try:
            return f"{safe_eval(expression):.12g}"
        except CalcError as exc:
            return f"Calculator error: {exc}"
        except Exception as exc:  # noqa: BLE001 - never crash the agent loop
            return f"Calculator error: {type(exc).__name__}: {exc}"

    @tool
    def run_python_code(code: str) -> str:
        """Execute a Python script in an isolated subprocess and return its output.

        Use this for anything a calculator cannot do: data processing, string
        manipulation, statistics, or working through a multi-step problem.
        The script runs with only the standard library and prints to stdout.
        There is no network and no input; return the answer you need from print().
        """
        try:
            return run_python(
                code,
                cwd=root,
                timeout=cfg.sandbox_timeout_s,
                max_output=cfg.sandbox_max_output,
            )
        except Exception as exc:  # noqa: BLE001
            return f"Sandbox error: {type(exc).__name__}: {exc}"

    @tool
    def list_files(subdirectory: str = ".") -> str:
        """List the files in the assistant's workspace, optionally under a subdirectory.

        Use this before reading or writing files so you know what exists.
        Pass "." for the workspace root, or a relative folder name.
        """
        try:
            return list_files(root, subdirectory)
        except FileAccessError as exc:
            return f"Error: {exc}"

    @tool
    def read_file_tool(path: str) -> str:
        """Read a text file from the workspace and return its contents.

        The path is relative to the workspace root, e.g. "notes/todo.txt".
        Only files in the workspace are readable.
        """
        try:
            return read_file(root, path)
        except FileAccessError as exc:
            return f"Error: {exc}"

    @tool
    def write_file_tool(path: str, content: str) -> str:
        """Create or overwrite a text file in the workspace.

        The path is relative to the workspace root; parent folders are created
        automatically. Returns a confirmation of how much was written.
        """
        try:
            return write_file(root, path, content)
        except FileAccessError as exc:
            return f"Error: {exc}"

    @tool
    def search_documents(query: str) -> str:
        """Search the user's uploaded documents and return the most relevant passages.

        Use this whenever the question is about something the user has uploaded
        (reports, notes, PDFs, articles). The result includes the source file
        name for each passage, so cite it in your answer.
        """
        hits = store.search(query, k=cfg.retriever_k)
        if not hits:
            return "No documents have been uploaded yet, or nothing matched."
        return format_hits(hits)

    @tool
    def current_datetime() -> str:
        """Return today's date and the current local time.

        Use this for questions about what day it is, how long until an event,
        or anything else that depends on the current date.
        """
        now = datetime.datetime.now().astimezone()
        return now.strftime("%A, %d %B %Y %H:%M:%S %Z (UTC%z)")

    @tool
    def open_url_tool(url: str) -> str:
        """Open a website in the browser.

        Use this when the user wants to watch or read something online:
        "open YouTube", "go to gmail", "open github.com". Pass the address
        or just the site's name -- a known service is recognised by name
        and a bare host gets https:// added. Only http and https open.

        To start a program installed on this computer, use open_app
        instead; this one is for pages, not applications.
        """
        try:
            return open_url(url)
        except ControlError as exc:
            return f"Error: {exc}"

    @tool
    def open_app_tool(name: str) -> str:
        """Launch a program installed on this computer.

        Use this whenever the user asks to open, start or launch an
        application: "open Spotify", "start Notepad", "launch Calculator".
        Pass the program's name -- matching ignores case and most of the
        wording, so "chrome" finds Google Chrome. When nothing of that name
        is installed, the service's own website opens instead, so a name
        that turns out to be a site still works.

        This is the tool for "open <something>"; open_url is for a specific
        page the user wants to read.
        """
        try:
            return open_app(name)
        except ControlError as exc:
            return f"Error: {exc}"

    @tool
    def open_file_tool(path: str) -> str:
        """Open a file or folder with the program Windows associates with it.

        The path may be absolute ("C:\\Users\\me\\Videos") or relative to the
        assistant's workspace; pass "." for the workspace root. Folders open in
        File Explorer, documents in their own program. Refuses to run
        programs -- this reaches things, it does not start applications.
        """
        try:
            return open_file(path, root)
        except ControlError as exc:
            return f"Error: {exc}"

    tools: list[Any] = [
        calculator,
        run_python_code,
        list_files,
        read_file_tool,
        write_file_tool,
        search_documents,
        current_datetime,
        open_url_tool,
        open_app_tool,
        open_file_tool,
    ]

    if include_web:
        web = _build_web_search_tool(cfg)
        if web is not None:
            tools.append(web)

    return tools


def _build_web_search_tool(cfg: Settings):
    """Build ``web_search`` only if ``ddgs`` imports; otherwise skip it."""
    try:
        from langchain.tools import tool

        from .websearch import web_search as _web_search
    except Exception:  # noqa: BLE001 - missing package or blocked dependency
        return None

    @tool
    def web_search(query: str) -> str:
        """Search the public web and return short summaries of the top results.

        Use only for current events, recent facts, or anything you are unsure
        about. Do not call it for arithmetic, files, or uploaded documents.
        """
        try:
            return _web_search(query, max_results=cfg.web_search_results)
        except Exception as exc:  # noqa: BLE001 - network tool, must not crash
            return f"Web search failed: {type(exc).__name__}: {exc}"

    return web_search

