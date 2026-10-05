"""The application's single source of truth.

Gradio has two ways in (typed messages, and the wake-word listener) and several
ways out (chat transcript, activity log, TTS clip, status line).  Rather than
letting those paths keep separate copies of the conversation, everything lives
here behind one lock, and the UI simply renders :meth:`Assistant.snapshot`.

The voice path runs in the listener's own thread and blocks on
:meth:`Assistant.submit_blocking`; the text path iterates
:meth:`Assistant.submit` directly.  Both drive the same turn machinery, so a
spoken question and a typed one behave identically.
"""

from __future__ import annotations

import threading
import time
import uuid
from collections.abc import Callable, Iterable, Iterator
from pathlib import Path
from typing import Any

from .agent import AgentEvent, AgentRunner
from .config import AUDIO, WORKSPACE, Settings, settings as default_settings
from .rag import DocumentStore

__all__ = ["Assistant", "Snapshot", "VoicePrompt"]

#: ``(messages, activity, audio_path, status)``
Snapshot = tuple[list[dict[str, str]], str, str | None, str]

#: What the wake-word listener passes in after stripping the wake word.
VoicePrompt = str

_MAX_ACTIVITY_LINES = 200
_MAX_AUDIO_CLIPS = 20


class Assistant:
    """Owns the agent, the document index and the visible conversation."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or default_settings
        self.workspace = Path(WORKSPACE)
        self.store = DocumentStore(self.settings)
        self.runner = AgentRunner(self.store, self.workspace, self.settings)

        self.session_id = uuid.uuid4().hex
        self._lock = threading.RLock()
        self._history: list[dict[str, str]] = welcome_history()
        self._activity: list[str] = []
        self._audio: str | None = None
        self._status = "Ready."
        self._busy = False
        self._turn = 0
        #: Set by the UI to abandon the turn in flight. Checked between agent
        #: events; the agent yields per token, so this takes effect in about
        #: one token rather than after the whole turn.
        self._stop_requested = False

        # UI-controlled preferences.
        self.speak_replies: bool = True
        self.voice: str = self.settings.tts_voice
        self.model: str = self.settings.model
        self.temperature: float = self.settings.temperature
        self.system_prompt: str = self.settings.system_prompt
        #: Chain-of-thought before answering. Off by default: measured 6.9x
        #: faster end to end on this CPU-only machine with no loss of accuracy.
        self.reasoning: bool = bool(self.settings.reasoning)

        #: Called as ``(text, seconds)`` just before audio is handed to the UI,
        #: so the listener can mute and stop hearing itself talk.
        self.on_reply_audio: Callable[[str, float], None] | None = None

    # ------------------------------------------------------------------
    # Reads
    # ------------------------------------------------------------------
    def snapshot(self) -> Snapshot:
        """A consistent view of everything the UI renders."""
        with self._lock:
            return (
                [dict(m) for m in self._history],
                "\n".join(self._activity),
                self._audio,
                self._status,
            )

    @property
    def busy(self) -> bool:
        with self._lock:
            return self._busy

    def document_status(self) -> dict[str, Any]:
        return self.store.status()

    def request_stop(self) -> bool:
        """Abandon the turn in flight.

        Returns True if there was something to stop. The generator running the
        turn notices at its next yield, tears down cleanly and releases the
        busy flag, so the UI becomes usable again immediately.
        """
        with self._lock:
            if not self._busy:
                return False
            self._stop_requested = True
            self._status = "Stopping…"
            return True

    @property
    def stop_requested(self) -> bool:
        with self._lock:
            return self._stop_requested

    # ------------------------------------------------------------------
    # UI actions
    # ------------------------------------------------------------------
    def reset(self) -> Snapshot:
        """Start a fresh conversation, leaving the document index alone."""
        with self._lock:
            self._history = welcome_history()
            self._activity.clear()
            self._audio = None
            self._status = "Conversation cleared."
            self._turn = 0
        self.runner.reset_session(self.session_id)
        return self.snapshot()

    def configure(
        self,
        *,
        model: str | None = None,
        temperature: float | None = None,
        system_prompt: str | None = None,
        speak_replies: bool | None = None,
        voice: str | None = None,
        reasoning: bool | None = None,
    ) -> Snapshot:
        with self._lock:
            if model:
                self.model = model
            if temperature is not None:
                self.temperature = float(temperature)
            if system_prompt is not None:
                self.system_prompt = system_prompt
            if speak_replies is not None:
                self.speak_replies = bool(speak_replies)
            if voice:
                self.voice = voice
            if reasoning is not None:
                self.reasoning = bool(reasoning)
            self._status = "Settings updated."
        return self.snapshot()

    def ingest(self, paths: Iterable[str | Path]) -> Snapshot:
        """Index uploaded documents. Safe to call while idle."""
        materialised = [p for p in (paths or []) if p]
        with self._lock:
            self._status = f"Indexing {len(materialised)} file(s)..."
        summary = self.store.ingest(materialised)
        self.log(summary)
        with self._lock:
            self._status = summary
        return self.snapshot()

    def clear_documents(self) -> Snapshot:
        summary = self.store.clear()
        self.log(summary)
        with self._lock:
            self._status = summary
        return self.snapshot()

    # ------------------------------------------------------------------
    # Turns
    # ------------------------------------------------------------------
    def submit(self, text: str) -> Iterator[Snapshot]:
        """Run one turn, yielding a snapshot after every visible change.

        Consuming this drives the UI.  The generator always releases the busy
        flag, even if the caller abandons it (Gradio's stop button does).
        """
        text = (text or "").strip()
        if not text:
            yield self.snapshot()
            return

        claimed = self._begin_turn(text)
        if not claimed:
            yield self.snapshot()
            return

        events = self.runner.stream(
            self.session_id,
            text,
            model=self.model,
            temperature=self.temperature,
            system_prompt=self.system_prompt,
            reasoning=self.reasoning,
        )
        try:
            yield self.snapshot()
            stopped = False
            for event in events:
                if self.stop_requested:
                    stopped = True
                    break
                if self._apply(event):
                    yield self.snapshot()

            if stopped:
                # Do not synthesise speech for a reply the user abandoned.
                self._abort_turn("Stopped.")
                yield self.snapshot()
            else:
                self._finish_turn()
                yield self.snapshot()
        except GeneratorExit:
            # The consumer went away mid-stream: leave a usable transcript.
            self._abort_turn("Stopped.")
            raise
        except Exception as exc:  # noqa: BLE001 - never wedge the UI
            self._abort_turn(f"Unexpected error: {type(exc).__name__}: {exc}")
            yield self.snapshot()
        finally:
            # Releasing the inner generator promptly frees the HTTP stream to
            # Ollama instead of waiting for it to be garbage collected.
            events.close()
            self._release()

    def submit_blocking(self, text: str) -> Snapshot:
        """Run a turn to completion (the voice path) and return the result."""
        final = self.snapshot()
        for final in self.submit(text):
            pass
        return final

    # ------------------------------------------------------------------
    # Turn machinery
    # ------------------------------------------------------------------
    def _begin_turn(self, text: str) -> bool:
        with self._lock:
            if self._busy:
                self._status = "Still working on the previous request."
                return False
            self._busy = True
            self._stop_requested = False
            self._turn += 1
            self._history.append({"role": "user", "content": text})
            self._history.append({"role": "assistant", "content": ""})
            self._status = "Thinking..."
            self._trim()
            return True

    def _apply(self, event: AgentEvent) -> bool:
        """Fold one agent event into state. Returns True if the UI should redraw."""
        with self._lock:
            if event.kind == "text":
                self._set_reply(event.text)
                return True

            if event.kind == "final":
                self._set_reply(event.text)
                self._status = "Answered."
                return True

            if event.kind == "tool_call":
                self._activity.append(f"→ {event.text}")
                self._status = f"Using {event.text.split('(')[0]}..."
                self._trim_activity()
                return True

            if event.kind == "tool":
                self._activity.append(f"  {event.text}")
                self._trim_activity()
                return True

            if event.kind == "error":
                self._set_reply(f"⚠ {event.text}")
                self._status = "Error."
                return True

            return False

    def _set_reply(self, text: str) -> None:
        if len(self._history) >= 2 and self._history[-1]["role"] == "assistant":
            self._history[-1]["content"] = text
        else:  # pragma: no cover - _begin_turn always leaves a placeholder
            self._history.append({"role": "assistant", "content": text})

    def _finish_turn(self) -> None:
        reply = ""
        errored = False
        with self._lock:
            if self._history and self._history[-1]["role"] == "assistant":
                reply = self._history[-1]["content"]
            errored = reply.startswith("⚠")
            if not reply:
                self._history[-1]["content"] = "(no reply)"
            if self._status in ("Thinking...", "Answered.") or self._status.startswith("Using"):
                self._status = "Ready."

        if not self.speak_replies or errored or not reply.strip():
            return
        self._synthesize(reply)

    def _abort_turn(self, message: str) -> None:
        with self._lock:
            if self._history and self._history[-1]["role"] == "assistant":
                if not self._history[-1]["content"]:
                    self._history[-1]["content"] = f"⚠ {message}"
            self._status = message

    def _release(self) -> None:
        with self._lock:
            self._busy = False

    # ------------------------------------------------------------------
    # Voice output
    # ------------------------------------------------------------------
    def _synthesize(self, reply: str) -> None:
        from .voice.tts import TTSUnavailable, estimate_duration_s, synthesize

        clip = AUDIO / f"reply-{self._turn}-{uuid.uuid4().hex[:8]}.mp3"
        try:
            path = synthesize(reply, clip, voice=self.voice, rate=self.settings.tts_rate)
        except TTSUnavailable as exc:
            self.log(f"Voice output unavailable: {exc}")
            return
        except Exception as exc:  # noqa: BLE001 - audio must never break a turn
            self.log(f"Voice output failed: {type(exc).__name__}: {exc}")
            return

        duration = estimate_duration_s(reply)
        self._prune_audio()
        with self._lock:
            self._audio = str(path)
        if self.on_reply_audio is not None:
            try:
                self.on_reply_audio(str(path), duration)
            except Exception:  # noqa: BLE001 - mic control is best effort
                pass

    def _prune_audio(self) -> None:
        try:
            clips = sorted(AUDIO.glob("reply-*.mp3"), key=lambda p: p.stat().st_mtime)
            for stale in clips[:-_MAX_AUDIO_CLIPS]:
                stale.unlink(missing_ok=True)
        except OSError:
            pass

    # ------------------------------------------------------------------
    # State helpers
    # ------------------------------------------------------------------
    def log(self, line: str) -> None:
        """Append a timestamped line to the activity panel."""
        stamp = time.strftime("%H:%M:%S")
        with self._lock:
            self._activity.append(f"[{stamp}] {line}")
            self._trim_activity()

    def _trim_activity(self) -> None:
        if len(self._activity) > _MAX_ACTIVITY_LINES:
            del self._activity[: len(self._activity) - _MAX_ACTIVITY_LINES]

    def _trim(self) -> None:
        limit = self.settings.max_display_messages
        if len(self._history) > limit:
            del self._history[: len(self._history) - limit]


def welcome_history() -> list[dict[str, str]]:
    """The transcript shown before the first message."""
    return [
        {
            "role": "assistant",
            "content": (
                "Apache ready. Type a message, or say **\"Apache\"** followed by "
                "your question to talk to me."
            ),
        }
    ]


_ASSISTANT: Assistant | None = None
_ASSISTANT_LOCK = threading.Lock()


def get_assistant(settings: Settings | None = None) -> Assistant:
    """Process-wide singleton shared by the text and voice input paths."""
    global _ASSISTANT  # noqa: PLW0603 - deliberate application singleton
    if _ASSISTANT is None:
        with _ASSISTANT_LOCK:
            if _ASSISTANT is None:
                _ASSISTANT = Assistant(settings)
    return _ASSISTANT
