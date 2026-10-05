"""Tests for the assembled application -- the layer the pure suites cannot reach.

Run with:  python -m tests.test_stack

:mod:`tests.test_pure` and :mod:`tests.test_rag` deliberately import nothing
from LangChain, Gradio, NumPy or the audio stack, so they stay runnable on a
machine where that native code is blocked. The cost is that wiring bugs -- a
helper deleted from one module but still imported by another, a component
constructed with a bad argument -- are invisible to them.

That gap is not hypothetical: these tests exist because a dead-code cleanup
removed ``agent.supported_tool_names`` while ``ui._tools_status_text`` still
imported it. The broad ``except`` around that import meant nothing crashed;
the UI just silently rendered "Tools unavailable". Both existing suites
passed throughout.

If the ML stack is unavailable the suite SKIPS rather than fails, so it stays
runnable everywhere.
"""

from __future__ import annotations

import sys
import tempfile
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

FAILURES: list[str] = []
SKIPPED: list[str] = []


def check(label: str, condition: bool, detail: str = "") -> None:
    if condition:
        print(f"  ok   {label}")
    else:
        FAILURES.append(label)
        print(f"  FAIL {label}  {detail}")


def _short(value: object, limit: int = 200) -> str:
    return " ".join(str(value).split())[:limit]


#: The toolset the agent is contractually offered.
EXPECTED_TOOLS = {
    "calculator",
    "run_python_code",
    "list_files",
    "read_file_tool",
    "write_file_tool",
    "search_documents",
    "current_datetime",
    "web_search",
}

#: Components without which the UI is not the app described in the README.
EXPECTED_COMPONENTS = {"chatbot", "timer", "uploadbutton", "audio"}


def _load_stack():
    """Import the UI-facing stack, or explain why it cannot be loaded."""
    try:
        from app.core import get_assistant
        from app.tools.registry import build_tools
        from app import ui
    except Exception as exc:  # noqa: BLE001 - blocked or missing natives
        return None, None, None, _short(exc)
    return get_assistant, build_tools, ui, ""


def test_assistant() -> None:
    get_assistant, _, _, _ = _load_stack()
    assistant = get_assistant()
    check("assistant singleton constructs", assistant is not None)
    check("assistant exposes a model name",
          bool(getattr(assistant, "model", "")), repr(getattr(assistant, "model", None)))
    check("assistant workspace is under the project",
          str(getattr(assistant, "workspace", "")).endswith("workspace"),
          repr(getattr(assistant, "workspace", None)))

    snap = assistant.snapshot()
    check("snapshot is a 4-tuple", isinstance(snap, tuple) and len(snap) == 4,
          f"len={len(snap) if isinstance(snap, tuple) else type(snap)}")
    messages, _activity, _audio, status = snap
    check("snapshot starts with a welcome message",
          isinstance(messages, list) and len(messages) >= 1,
          f"messages={len(messages) if isinstance(messages, list) else '?'}")
    check("snapshot status is a string", isinstance(status, str), repr(status))


def test_tool_registry() -> None:
    get_assistant, build_tools, _, _ = _load_stack()
    assistant = get_assistant()
    tools = build_tools(assistant.store, assistant.workspace, assistant.settings)
    check("tools construct", bool(tools), f"count={len(tools)}")

    names = {getattr(tool, "name", "") for tool in tools}
    missing = EXPECTED_TOOLS - names
    check("all eight contract tools registered", not missing,
          f"missing={sorted(missing)} extra={sorted(names - EXPECTED_TOOLS)}")

    # A tool without a description will not be called by the model.
    undescribed = [getattr(t, "name", "?") for t in tools
                   if not getattr(t, "description", None)]
    check("every tool carries a description", not undescribed,
          f"undescribed={undescribed}")


def test_rag_end_to_end() -> None:
    get_assistant, _, _, _ = _load_stack()
    assistant = get_assistant()

    with tempfile.TemporaryDirectory() as tmp:
        doc = Path(tmp) / "apache_notes.txt"
        doc.write_text(
            "Apache is a wake word based AI assistant. "
            "It uses Ollama for the model and Gradio for the interface. "
            "The code sandbox runs python in isolated mode with a timeout.",
            encoding="utf-8",
        )
        summary = assistant.store.ingest([doc])
        check("store accepts an uploaded document", bool(summary), _short(summary))
        check("store reports a non-empty chunk count",
              assistant.store.chunk_count > 0,
              f"chunks={assistant.store.chunk_count}")

        hits = assistant.store.search("what wake word does it use")
        check("search returns hits", bool(hits), f"hits={len(hits)}")
        if hits:
            check("search ranks the ingested document first",
                  hits[0].source == "apache_notes.txt",
                  f"top={hits[0].source}")
            check("hit carries a numeric score",
                  isinstance(hits[0].score, float), repr(getattr(hits[0], "score", None)))


def test_gradio_ui() -> None:
    _, _, ui, _ = _load_stack()
    get_assistant, _, _, _ = _load_stack()
    assistant = get_assistant()

    demo = ui.build_demo(assistant)
    check("build_demo returns Blocks", type(demo).__name__ == "Blocks",
          type(demo).__name__)
    check("blocks carry the app title", getattr(demo, "title", None) == "Apache",
          repr(getattr(demo, "title", None)))

    blocks = getattr(demo, "blocks", None) or {}
    check("blocks contain components", len(blocks) > 0, f"count={len(blocks)}")

    types = {type(component).__name__ for component in blocks.values()}
    # Match on class names rather than Gradio's lowercase type strings so a
    # rename in either direction is still caught.
    joined = " ".join(types).lower()
    missing = [name for name in EXPECTED_COMPONENTS if name not in joined]
    check("chatbot, timer, upload button and audio are all present", not missing,
          f"missing={missing}")

    handlers = getattr(demo, "fns", None)
    check("event handlers are registered",
          handlers is not None and len(handlers) > 0,
          f"count={len(handlers) if handlers is not None else 'n/a'}")


def test_status_helpers() -> None:
    """Regression guard for the deleted-`supported_tool_names` bug."""
    _, _, ui, _ = _load_stack()
    get_assistant, _, _, _ = _load_stack()
    assistant = get_assistant()

    for name in ("_settings_status_text", "_tools_status_text", "_docs_status_text"):
        helper = getattr(ui, name, None)
        check(f"ui exposes {name}", callable(helper), type(helper).__name__)
        if not callable(helper):
            continue
        try:
            text = helper(assistant)
        except Exception as exc:  # noqa: BLE001
            check(f"{name} runs", False, _short(exc))
            continue
        check(f"{name} runs", isinstance(text, str), type(text).__name__)

    try:
        tools_text = ui._tools_status_text(assistant)
    except Exception as exc:  # noqa: BLE001
        tools_text = f"ERROR {_short(exc)}"

    check("tools status lists the registry by name",
          "unavailable" not in tools_text.lower() and "calculator" in tools_text,
          _short(tools_text, 240))


def test_snapshot_rendering() -> None:
    _, _, ui, _ = _load_stack()
    get_assistant, _, _, _ = _load_stack()
    assistant = get_assistant()

    rendered = ui._render(assistant.snapshot())
    check("_render returns four values",
          isinstance(rendered, tuple) and len(rendered) == 4,
          f"len={len(rendered) if isinstance(rendered, tuple) else type(rendered)}")


def test_agent_turn_degrades_cleanly() -> None:
    """A turn must terminate readably with or without a model available."""
    get_assistant, _, _, _ = _load_stack()
    assistant = get_assistant()

    before = len(assistant.snapshot()[0])
    last = None
    raised = None
    try:
        for snap in assistant.submit("hello"):
            last = snap
    except Exception as exc:  # noqa: BLE001
        raised = exc

    check("agent turn does not raise", raised is None, _short(raised) if raised else "")
    check("agent turn terminates with a snapshot", last is not None)

    if last is not None:
        messages, _activity, _audio, status = last
        check("turn appends messages", len(messages) > before,
              f"{before} -> {len(messages)}")
        check("turn leaves a non-empty reply",
              bool(messages and messages[-1].get("content", "").strip()),
              _short(messages[-1].get("content", "") if messages else ""))
        check("turn settles on a terminal status",
              isinstance(status, str) and bool(status.strip()), repr(status))


def main() -> int:
    print("Apache stack tests")
    print("-" * 64)

    _g, _b, _ui, reason = _load_stack()
    if _ui is None:
        print(f"  SKIP the ML/UI stack could not be loaded: {reason}")
        print()
        print("Skipped (native stack unavailable) -- this is expected where")
        print("Smart App Control or a code-integrity policy blocks PyPI wheels.")
        print("Run python run.py --check for the diagnosis.")
        return 0

    for suite in (
        test_assistant,
        test_tool_registry,
        test_rag_end_to_end,
        test_gradio_ui,
        test_status_helpers,
        test_snapshot_rendering,
        test_agent_turn_degrades_cleanly,
    ):
        print(f"-- {suite.__name__}")
        try:
            suite()
        except Exception:  # noqa: BLE001
            traceback.print_exc()
            FAILURES.append(f"{suite.__name__} crashed")
        print()

    if FAILURES:
        print(f"{len(FAILURES)} FAILURE(S):")
        for name in FAILURES:
            print(f"  - {name}")
        return 1
    print("All stack tests passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
