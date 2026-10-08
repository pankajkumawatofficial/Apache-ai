"""Agent construction and streaming.

LangChain 1.x's ``create_agent`` builds a LangGraph state graph whose nodes
are named ``"model"`` and ``"tools"``.  This module drives it with a combined
``stream_mode=["updates", "messages"]`` so the UI can show tokens as they
arrive *and* see which tools fired, while a ``MemorySaver`` checkpointer keeps
conversation history per session.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

from .config import Settings, settings as default_settings
from .llm import build_chat_model
from .rag import DocumentStore
from .tools.registry import build_tools

__all__ = ["AgentEvent", "AgentRunner"]

_MODEL_NODE = "model"
_TOOLS_NODE = "tools"


@dataclass(frozen=True)
class AgentEvent:
    """One step of a turn, as the UI should render it.

    ``kind`` is one of:

    ``text``
        ``text`` is the assistant's answer accumulated so far (replace).
    ``tool_call``
        ``text`` is ``name(args)`` the model asked for.
    ``tool``
        ``text`` is a short description of what a tool returned.
    ``final``
        ``text`` is the definitive answer; overrides any earlier ``text``.
    ``error``
        ``text`` is a human-readable failure.
    """

    kind: str
    text: str


def _content_to_text(content: Any) -> str:
    """Normalise LangChain content (str, or content blocks) to plain text."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, (list, tuple)):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                if block.get("type") in ("text", "output_text"):
                    parts.append(str(block.get("text", "")))
                elif "text" in block:
                    parts.append(str(block["text"]))
        return "".join(parts)
    return str(content)


def _messages_of(update: Any) -> list[Any]:
    if not isinstance(update, dict):
        return []
    messages = update.get("messages")
    if not messages:
        return []
    return list(messages)


def _format_tool_args(args: Any) -> str:
    if not isinstance(args, dict) or not args:
        return ""
    rendered = []
    for key, value in args.items():
        text = value if isinstance(value, str) else json.dumps(value, default=str)
        if len(text) > 60:
            text = text[:60] + "..."
        rendered.append(f"{key}={text}")
    return ", ".join(rendered)


def _summarise(body: str, limit: int = 160) -> str:
    body = " ".join(body.split())
    if len(body) > limit:
        body = body[:limit] + "..."
    return body or "(no result)"


def _tool_result_summary(message: Any, limit: int = 160) -> str:
    body = _content_to_text(getattr(message, "content", "")).strip()
    return _summarise(body, limit)


#: The tool call a small model writes out instead of emitting one: the tool
#: name, then its arguments as JSON. The runtime has already decided this is
#: ordinary prose by the time it arrives, so it would be spoken aloud --
#: "play_media_tool open brace quote request quote colon ..." read out to
#: whoever asked for a song.
_TEXT_TOOL_CALL = re.compile(r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*\n?\s*(\{.*\})\s*$", re.S)

#: The same intent with the arguments never written down: the name on its
#: own, sometimes quoted. "play a song", "put on a song", "start some music"
#: and "i want to hear music" all came back as the single word "play_media",
#: which was spoken aloud and started nothing -- so this shape is treated as
#: a call too, and the only argument available for it is the request itself.
_BARE_TOOL_CALL = re.compile(
    r"^\s*[\"'`]*([A-Za-z_][A-Za-z0-9_]*)[\"'`.]*\s*$"
)

#: How the system prompt spells a tool that LangChain registered differently.
_TOOL_ALIASES = {"run_python": "run_python_code"}


def _find_tool(name: str, tools: list[Any]) -> Any | None:
    """Resolve *name* against *tools*, accepting the prompt's spelling.

    The prompt teaches ``open_app`` and ``play_media`` while the registry
    registers ``open_app_tool`` and ``play_media_tool``. A call the model
    writes out by hand uses the name it was taught, so the shorter form has
    to reach the same tool or the call is lost.
    """
    wanted = _TOOL_ALIASES.get(name, name)
    for tool in tools:
        registered = getattr(tool, "name", "")
        if registered == wanted:
            return tool
        if registered.endswith("_tool") and registered[: -len("_tool")] == wanted:
            return tool
    return None


def _call_schema(tool: Any) -> dict | None:
    """The JSON schema LangChain showed the model for this tool's arguments.

    ``BaseTool.args`` is only the properties mapping, so it cannot say which
    of them are optional; ``tool_call_schema`` is the whole object, the one
    the model had to satisfy when it wrote the call out.
    """
    builder = getattr(getattr(tool, "tool_call_schema", None),
                      "model_json_schema", None)
    if not callable(builder):
        return None
    try:
        built = builder()
    except Exception:  # noqa: BLE001 - a hand-rolled tool, fall back below
        return None
    return built if isinstance(built, dict) else None


def _sentence_arguments(tool: Any, sentence: str) -> dict | None:
    """Arguments for a tool the model named without supplying any.

    The user's own request is the only thing to hand over, so it goes to the
    tool's one string parameter: "play a song" becomes
    ``play_media(request="play a song")``. A tool needing more than one
    argument has nothing to fill the second one with, and one needing none
    at all is called as written.
    """
    schema = _call_schema(tool)
    if schema is None:
        properties = getattr(tool, "args", None)
        if not isinstance(properties, dict) or not properties:
            return None
        # Only the properties arrived, so required is what has no default --
        # which is how the flat mapping distinguishes the two.
        schema = {
            "properties": properties,
            "required": [key for key, spec in properties.items()
                         if not (isinstance(spec, dict) and "default" in spec)],
        }

    required = list(schema.get("required") or [])
    if len(required) > 1:
        return None
    if not required:
        return {}
    properties = schema.get("properties") or {}
    if (properties.get(required[0]) or {}).get("type") != "string":
        return None
    return {required[0]: sentence}


def _run_text_tool_call(
    text: str, tools: list[Any], user_text: str | None = None
) -> tuple[str, dict, str] | None:
    """Run a tool call the model wrote out as text, or ``None``.

    Returns ``(name, arguments, result)`` when *text* is recognisably a
    call to one of *tools*. Anything else is a normal reply and is left
    alone -- the shapes being matched are specific enough that prose has no
    reason to fit them.
    """
    match = _TEXT_TOOL_CALL.match(text or "")
    if match:
        tool = _find_tool(match.group(1), tools)
        if tool is None:
            return None
        try:
            args = json.loads(match.group(2))
        except Exception:  # noqa: BLE001 - not a call, just a sentence
            return None
        if not isinstance(args, dict):
            return None
    else:
        bare = _BARE_TOOL_CALL.match(text or "")
        if not bare:
            return None
        tool = _find_tool(bare.group(1), tools)
        if tool is None:
            return None
        if not (user_text or "").strip():
            return None
        args = _sentence_arguments(tool, user_text.strip())
        if args is None:
            return None

    # _find_tool only returns a tool whose registered name matched, so this
    # is never empty -- the getattr keeps a hand-rolled tool from being None.
    name = getattr(tool, "name", "") or "tool"
    try:
        result = tool.invoke(args)
    except Exception as exc:  # noqa: BLE001 - report it, never raise it
        return name, args, f"Error: {type(exc).__name__}: {exc}"
    return name, args, _content_to_text(result)


class AgentRunner:
    """Owns the compiled agent(s) and the checkpointer that holds history."""

    def __init__(
        self,
        store: DocumentStore,
        workspace: Path,
        settings: Settings | None = None,
    ) -> None:
        from langgraph.checkpoint.memory import MemorySaver

        self.settings = settings or default_settings
        self.store = store
        self.workspace = Path(workspace)
        # One checkpointer for the process, so rebuilding the agent when
        # settings change does not throw the conversation away.
        self._checkpointer = MemorySaver()
        self._agents: dict[tuple, Any] = {}
        self._tools: list[Any] | None = None
        self._tools_revision: int = -1

    # -- construction ----------------------------------------------------
    @property
    def tools(self) -> list[Any]:
        """Rebuild the tool list only when the document index changes."""
        revision = self.store.revision
        if self._tools is None or revision != self._tools_revision:
            self._tools = build_tools(self.store, self.workspace, self.settings)
            self._tools_revision = revision
        return self._tools

    def _agent(
        self,
        model: str,
        temperature: float,
        system_prompt: str,
        reasoning: bool | None = None,
    ):
        from langchain.agents import create_agent

        # Refresh the tool list first so the cache key reflects it.
        self.tools
        # reasoning belongs in the key: ChatOllama captures it at construction,
        # so a key without it would reuse an agent built with the old mode.
        key = (model, round(temperature, 3), system_prompt, self._tools_revision, reasoning)

        cached = self._agents.get(key)
        if cached is not None:
            return cached

        agent = create_agent(
            build_chat_model(model=model, temperature=temperature, reasoning=reasoning),
            tools=self._tools or [],
            system_prompt=system_prompt or None,
            checkpointer=self._checkpointer,
        )
        self._agents[key] = agent
        # Drop stale entries so changing settings does not leak graph objects.
        if len(self._agents) > 8:
            for stale in list(self._agents)[: len(self._agents) - 8]:
                self._agents.pop(stale, None)
        return agent

    def reset_session(self, session_id: str) -> None:
        """Forget a session's checkpointed history."""
        try:
            self._checkpointer.delete_thread(session_id)
        except Exception:  # noqa: BLE001 - MemorySaver supports this, be safe
            pass

    # -- streaming -------------------------------------------------------
    def stream(
        self,
        session_id: str,
        user_text: str,
        *,
        model: str,
        temperature: float,
        system_prompt: str,
        reasoning: bool | None = None,
    ) -> Iterator[AgentEvent]:
        """Run one turn, yielding events in the order the UI should show them."""
        from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage

        try:
            graph = self._agent(model, temperature, system_prompt, reasoning)
        except Exception as exc:
            yield AgentEvent("error", f"Could not build the agent: {_detail(exc)}")
            return

        config = {
            "configurable": {"thread_id": session_id},
            "recursion_limit": self.settings.recursion_limit,
        }
        payload = {"messages": [HumanMessage(content=user_text)]}

        streamed = ""          # tokens seen from the model node this turn
        final_text: str | None = None

        try:
            iterator = graph.stream(
                payload,
                config,
                stream_mode=["updates", "messages"],
            )
            for mode, data in iterator:
                if mode == "messages":
                    chunk, metadata = data
                    if not isinstance(chunk, AIMessageChunk):
                        continue
                    if metadata.get("langgraph_node") != _MODEL_NODE:
                        continue
                    piece = _content_to_text(chunk.content)
                    if piece:
                        streamed += piece
                        yield AgentEvent("text", streamed)
                    continue

                # mode == "updates"
                if not isinstance(data, dict):
                    continue
                for node, update in data.items():
                    if node == _TOOLS_NODE:
                        for message in _messages_of(update):
                            name = getattr(message, "name", "") or "tool"
                            yield AgentEvent(
                                "tool",
                                f"{name}: {_tool_result_summary(message)}",
                            )
                    elif node == _MODEL_NODE:
                        messages = _messages_of(update)
                        if not messages:
                            continue
                        message = messages[-1]
                        tool_calls = getattr(message, "tool_calls", None) or []
                        if tool_calls:
                            for call in tool_calls:
                                if not isinstance(call, dict):
                                    continue
                                yield AgentEvent(
                                    "tool_call",
                                    f"{call.get('name', '?')}"
                                    f"({_format_tool_args(call.get('args'))})",
                                )
                            # Discard any preamble the model emitted alongside
                            # the tool call: the real answer comes next.
                            streamed = ""
                        elif isinstance(message, (AIMessage, AIMessageChunk)):
                            text = _content_to_text(message.content)
                            if not text:
                                continue
                            # Still a call, whatever the transport decided:
                            # run it rather than read it out loud.
                            recovered = _run_text_tool_call(
                                text, self.tools, user_text
                            )
                            if recovered is None:
                                final_text = text
                                yield AgentEvent("final", text)
                                continue
                            name, args, result = recovered
                            yield AgentEvent(
                                "tool_call", f"{name}({_format_tool_args(args)})"
                            )
                            yield AgentEvent("tool", f"{name}: {_summarise(result)}")
                            final_text = result
                            yield AgentEvent("final", result)

        except Exception as exc:
            yield AgentEvent("error", _detail(exc))
            return

        if final_text is None and streamed:
            yield AgentEvent("final", streamed)
        elif final_text is None:
            yield AgentEvent(
                "final",
                "I could not produce a reply. Check that Ollama is running and "
                "the selected model is pulled.",
            )


def _detail(exc: BaseException) -> str:
    """Flatten an exception into one readable line for the chat."""
    message = str(exc).strip() or type(exc).__name__
    text = " ".join(message.split())
    lowered = text.lower()
    if (
        ("model" in lowered and "not found" in lowered)
        or "status code: 404" in lowered
    ):
        return (
            f"{text} Ollama does not have the selected model. "
            "Run `ollama list`, then `ollama pull <model>` or choose an "
            "installed model in the Model tab."
        )
    if len(text) > 400:
        text = text[:400] + "..."
    return text
