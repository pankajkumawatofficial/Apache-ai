"""Agent construction and streaming.

LangChain 1.x's ``create_agent`` builds a LangGraph state graph whose nodes
are named ``"model"`` and ``"tools"``.  This module drives it with a combined
``stream_mode=["updates", "messages"]`` so the UI can show tokens as they
arrive *and* see which tools fired, while a ``MemorySaver`` checkpointer keeps
conversation history per session.
"""

from __future__ import annotations

import json
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


def _tool_result_summary(message: Any, limit: int = 160) -> str:
    body = _content_to_text(getattr(message, "content", "")).strip()
    body = " ".join(body.split())
    if len(body) > limit:
        body = body[:limit] + "..."
    return body or "(no result)"


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

    def _agent(self, model: str, temperature: float, system_prompt: str):
        from langchain.agents import create_agent

        # Refresh the tool list first so the cache key reflects it.
        self.tools
        key = (model, round(temperature, 3), system_prompt, self._tools_revision)

        cached = self._agents.get(key)
        if cached is not None:
            return cached

        agent = create_agent(
            build_chat_model(model=model, temperature=temperature),
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
    ) -> Iterator[AgentEvent]:
        """Run one turn, yielding events in the order the UI should show them."""
        from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage

        try:
            graph = self._agent(model, temperature, system_prompt)
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
                            if text:
                                final_text = text
                                yield AgentEvent("final", text)

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
