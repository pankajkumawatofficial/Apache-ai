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
from .text import plain_math, stream_cut

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
        #: Clips waiting to be played, each paired with the duration estimated
        #: for it. One slot was not enough: a greeting, an idle check-in and
        #: an answer can each finish synthesising inside the same 0.6 s tick,
        #: and every write overwrote the clip before the browser had seen it.
        #: That is how a reply ended up never spoken.
        self._audio_queue: list[tuple[str, float]] = []
        #: When the clip now playing has had time to finish. Keeps playback
        #: sequential instead of the next clip cutting the previous one off.
        self._audio_free_at = 0.0
        self._status = "Ready."
        self._busy = False
        self._turn = 0
        #: How much of the reply in flight the voice has already been given.
        #: Speech starts while the answer is still typing, so the synthesiser
        #: must know where it stopped -- and, if an event replaces the text
        #: under it, how much of what is on screen it has already said.
        self._spoken = ""
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

        # Presence. _last_input is the clock the idle check-in watches; every
        # interaction restarts it. Read and written under the same lock as the
        # rest of the state so a reply in flight can never be talked over.
        self._last_input = time.monotonic()
        self._prompts_started = False
        self._stop_prompts = threading.Event()

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
        self.note_input()
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
        self.note_input()
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
        self.note_input()
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
        self.note_input()
        materialised = [p for p in (paths or []) if p]
        with self._lock:
            self._status = f"Indexing {len(materialised)} file(s)..."
        summary = self.store.ingest(materialised)
        self.log(summary)
        with self._lock:
            self._status = summary
        return self.snapshot()

    def clear_documents(self) -> Snapshot:
        self.note_input()
        summary = self.store.clear()
        self.log(summary)
        with self._lock:
            self._status = summary
        return self.snapshot()

    # ------------------------------------------------------------------
    # Presence: startup greeting and idle check-in
    # ------------------------------------------------------------------
    def note_input(self) -> None:
        """Record that the user did something, restarting the idle clock."""
        with self._lock:
            self._last_input = time.monotonic()

    def say(self, text: str) -> None:
        """Speak *text* without joining the conversation.

        An announcement is not a turn: it must not touch the transcript, must
        not claim the busy flag, and must never delay whatever the caller was
        doing. Speech is synthesised on its own thread and reaches the browser
        through the timer that already watches for the audio path changing.
        """
        text = (text or "").strip()
        if not text or not self.speak_replies:
            return
        threading.Thread(
            target=self._synthesize,
            args=(text,),
            name="apache-announce",
            daemon=True,
        ).start()

    def start_prompts(self) -> None:
        """Greet on startup, then check in whenever input goes quiet.

        Called once by the launcher. Tests deliberately do not call it: they
        would otherwise open a network TTS request and leave a thread running
        after the suite had finished.
        """
        with self._lock:
            if self._prompts_started:
                return
            self._prompts_started = True
            self._last_input = time.monotonic()
        self._stop_prompts.clear()
        self.say(self.settings.greeting)

        period = self.settings.idle_prompt_s
        if period > 0:
            threading.Thread(
                target=self._idle_loop,
                args=(float(period),),
                name="apache-idle",
                daemon=True,
            ).start()

    def stop_prompts(self) -> None:
        """Halt the greeting and idle threads (shutdown and tests)."""
        self._stop_prompts.set()

    def _idle_loop(self, period: float) -> None:
        """Say the check-in line every *period* seconds of genuine silence."""
        while not self._stop_prompts.wait(1.0):
            with self._lock:
                if self._busy:
                    # Answering is not the user going away. Holding the clock
                    # here is what stops Apache talking over its own reply.
                    self._last_input = time.monotonic()
                    continue
                if time.monotonic() - self._last_input < period:
                    continue
                # Re-arm before speaking, so a TTS failure cannot spin.
                self._last_input = time.monotonic()
            self.say(self.settings.idle_prompt)

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

        # Anything typed or spoken is the user being present.
        self.note_input()
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
                # The abandoned reply itself is never spoken; the stop is
                # confirmed out loud so an interrupted turn is not silent.
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
            self._spoken = ""
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
            elif event.kind == "final":
                self._set_reply(event.text)
                self._status = "Answered."
            elif event.kind == "tool_call":
                self._activity.append(f"→ {event.text}")
                self._status = f"Using {event.text.split('(')[0]}..."
                self._trim_activity()
            elif event.kind == "tool":
                self._activity.append(f"  {event.text}")
                self._trim_activity()
            elif event.kind == "error":
                self._set_reply(f"⚠ {event.text}")
                self._status = "Error."
            else:
                return False

        # Outside the lock, at the moment the text lands: speech has to begin
        # with the typing, so it cannot wait behind a state update the UI is
        # waiting for, and it cannot wait for the turn to end. Errors are
        # spoken whole by _finish_turn instead -- a message that replaces the
        # reply rather than extending it would be cut in the middle.
        if event.kind in ("text", "final"):
            self._stream_speech(flush=event.kind == "final")
        return True

    def _set_reply(self, text: str) -> None:
        # Sanitised here so the transcript and the synthesiser see one text.
        clean = plain_math(text)
        if len(self._history) >= 2 and self._history[-1]["role"] == "assistant":
            self._history[-1]["content"] = clean
        else:  # pragma: no cover - _begin_turn always leaves a placeholder
            self._history.append({"role": "assistant", "content": clean})

    def _finish_turn(self) -> None:
        reply = ""
        with self._lock:
            if self._history and self._history[-1]["role"] == "assistant":
                reply = self._history[-1]["content"]
            if not reply:
                self._history[-1]["content"] = "(no reply)"
            if self._status in ("Thinking...", "Answered.") or self._status.startswith("Using"):
                self._status = "Ready."

        # Every reply is spoken, errors included. A failure that is only
        # written down looks identical, from the user's side of the screen,
        # to Apache deciding not to talk -- and that is the one thing they
        # cannot diagnose. The warning sign is stripped by the synthesiser
        # rather than read aloud as nothing.
        if not self.speak_replies or not reply.strip():
            return
        # Only what is left: most of it was already queued while it was
        # still being typed, and saying the whole thing a second time is
        # worse than saying nothing.
        self._stream_speech(flush=True)

    def _abort_turn(self, message: str) -> None:
        with self._lock:
            if self._history and self._history[-1]["role"] == "assistant":
                if not self._history[-1]["content"]:
                    self._history[-1]["content"] = f"⚠ {message}"
            self._status = message
        # Speech now starts while the answer is still typing, so a stopped
        # answer has clips queued behind it. They are the words the user just
        # silenced, and leaving them there meant the stop was confirmed out
        # loud by the rest of the reply they had cut off.
        self._drop_unsaid_clips()
        # An interrupted turn used to end in silence, which is the least
        # helpful possible outcome: nothing on screen, nothing to hear.
        if message:
            self.say(message)

    def _release(self) -> None:
        with self._lock:
            self._busy = False

    # ------------------------------------------------------------------
    # Voice output
    # ------------------------------------------------------------------
    def _stream_speech(self, *, flush: bool = False) -> None:
        """Hand the voice whatever of the reply has finished arriving.

        The answer reaches the screen a token at a time, and the speaker used
        to wait for all of it -- so a reply you could read was one you could
        not yet hear. Each chunk complete enough to say is queued the moment
        it lands, and because the queue plays clips in order the speaker
        trails the transcript instead of racing it.

        ``flush`` marks the end of the turn: whatever is still unsaid goes
        out as it stands, sentence boundary or not.
        """
        if not self.speak_replies:
            return

        with self._lock:
            if not self._history or self._history[-1]["role"] != "assistant":
                return
            text = self._history[-1]["content"]
            # The reply normally only grows, so everything said is a prefix
            # of it. plain_math rewriting a half-written formula is the
            # exception; there the mark is held where it was rather than
            # walked back, because repeating a sentence already heard is
            # worse than losing one clause of it.
            shared = min(len(self._spoken), len(text))
            tail = text[shared:]
            cut = len(tail) if flush else stream_cut(tail)
            if cut <= 0:
                return
            chunk = tail[:cut].strip()
            self._spoken = text[: shared + cut]

        # Outside the lock: synthesis runs on its own thread, and this is
        # called for every token that completes a sentence.
        if chunk:
            self._synthesize(chunk)

    def _drop_unsaid_clips(self) -> None:
        """Forget this turn's clips the browser has not been handed yet.

        Speech beginning with the typing is what puts them there: stopping a
        reply halfway leaves the rest of it queued. The clips are named for
        the turn that asked for them, so this only ever reaches the ones the
        current turn produced -- a greeting or an idle check-in already
        waiting its turn is left alone.
        """
        with self._lock:
            prefix = f"reply-{self._turn}-"
            keep: list[tuple[str, float]] = []
            dropped: list[str] = []
            for path, duration in self._audio_queue:
                if Path(path).name.startswith(prefix):
                    dropped.append(path)
                else:
                    keep.append((path, duration))
            self._audio_queue[:] = keep

        for path in dropped:
            try:
                Path(path).unlink(missing_ok=True)
            except OSError:  # pragma: no cover - a file already gone
                pass

    def _synthesize(self, reply: str) -> None:
        from .voice.tts import (
            TTSUnavailable,
            clip_extension,
            estimate_duration_s,
            synthesize,
        )

        # The suffix follows the engine: a Piper WAV named .mp3 would fail to
        # decode in the browser, which looks exactly like Apache not speaking.
        clip = AUDIO / f"reply-{self._turn}-{uuid.uuid4().hex[:8]}{clip_extension()}"
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
            # Queued, not assigned: next_audio() releases it once whatever is
            # playing has had time to finish.
            self._audio_queue.append((str(path), duration))
        if self.on_reply_audio is not None:
            try:
                self.on_reply_audio(str(path), duration)
            except Exception:  # noqa: BLE001 - mic control is best effort
                pass

    def next_audio(self) -> str | None:
        """Return the next clip to play, or ``None`` when it is not yet time.

        Every reply and every announcement lands in one queue, and they can
        finish synthesising within a single tick -- the greeting, an idle
        check-in and an answer all at once. A single slot let each overwrite
        the last before the browser had seen it, which is how a response ended
        up never spoken at all.

        Clips are released on their estimated duration, so the next one waits
        for the current one rather than cutting it off.
        """
        with self._lock:
            now = time.monotonic()
            if now < self._audio_free_at or not self._audio_queue:
                return None
            path, duration = self._audio_queue.pop(0)
            # A floor so a very short clip is not replaced by the next tick.
            self._audio_free_at = now + max(duration, 0.5)
            self._audio = path
            return path

    def pending_audio(self) -> int:
        """How many clips are queued and not yet handed out."""
        with self._lock:
            return len(self._audio_queue)

    def _prune_audio(self) -> None:
        """Delete old clips, never one still owed to the browser.

        The queue and the clip currently playing are exactly the files that
        must survive: trimming one of those is a reply that was written,
        queued, and then quietly made inaudible -- indistinguishable from
        Apache failing to speak.
        """
        try:
            with self._lock:
                live = {path for path, _ in self._audio_queue}
                if self._audio:
                    live.add(self._audio)
            clips = sorted(AUDIO.glob("reply-*.*"), key=lambda p: p.stat().st_mtime)
            for stale in clips[:-_MAX_AUDIO_CLIPS]:
                if str(stale) in live:
                    continue
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
