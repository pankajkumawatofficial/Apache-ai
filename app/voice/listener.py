"""Continuous microphone listening with wake-word detection.

A ``sounddevice`` callback drops each block of samples on a list; a worker
thread runs them through the energy VAD, transcribes anything that looks like
speech, and -- if the transcript opens with the wake word -- hands the
remaining instruction to the assistant.

The listener deliberately owns no conversation state: it only calls
``on_command(text)`` and reports its own status for the UI to display.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from collections.abc import Callable
from typing import Any

from ..config import Settings, settings as default_settings
from .vad import EnergyVAD
from .wake import match_wake

__all__ = ["WakeListener", "WakeOnlyPrompt"]

#: What the assistant is told when the wake word arrives with no instruction.
WAKE_ONLY_PROMPT = (
    "I just said your wake word but gave no instruction. "
    "Reply with a brief greeting and ask what you can help with."
)


class WakeListener:
    """Microphone -> VAD -> STT -> wake-word filter -> ``on_command``."""

    #: Give up after this many consecutive recogniser failures (no network).
    max_stt_failures = 5

    def __init__(
        self,
        on_command: Callable[[str], None],
        settings: Settings | None = None,
        on_error: Callable[[str], None] | None = None,
    ) -> None:
        self.settings = settings or default_settings
        self.on_command = on_command
        self.on_error = on_error

        self._lock = threading.RLock()
        self._pending: deque[Any] = deque()
        self._thread: threading.Thread | None = None
        self._stream: Any = None
        self._vad: EnergyVAD | None = None
        self._running = False
        self._muted_until = 0.0

        self._state = "stopped"
        self._message = "Microphone is off."
        self._transcript = ""
        self._rate = 0
        self._device_name = ""
        self._stt_failures = 0

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    @property
    def running(self) -> bool:
        return self._running

    def status(self) -> dict[str, Any]:
        with self._lock:
            return {
                "state": self._state,
                "message": self._message,
                "transcript": self._transcript,
                "sample_rate": self._rate,
                "device": self._device_name,
                "muted": time.time() < self._muted_until,
                "wake_word": self.settings.wake_word,
            }

    def start(self) -> str:
        """Open the microphone and start the worker. Returns a status line."""
        with self._lock:
            if self._running:
                return self._message

        try:
            import sounddevice as sd
        except Exception as exc:  # noqa: BLE001 - blocked or missing native lib
            return self._fail(f"Audio input unavailable: {exc}")

        try:
            device_index = sd.default.device[0]
            info = sd.query_devices(device_index, "input")
            rate = int(round(float(info.get("default_samplerate", 16000)) or 16000))
            device_name = str(info.get("name", "default input"))
        except Exception as exc:  # noqa: BLE001 - no microphone at all
            return self._fail(f"No microphone found: {exc}")

        vad = EnergyVAD(
            sample_rate=rate,
            frame_ms=self.settings.frame_ms,
            min_energy=self.settings.vad_min_energy,
            multiplier=self.settings.vad_multiplier,
            silence_end_ms=self.settings.silence_end_ms,
            min_utterance_ms=self.settings.utterance_min_ms,
            max_utterance_s=self.settings.utterance_max_s,
        )

        try:
            stream = sd.InputStream(
                samplerate=rate,
                channels=1,
                dtype="float32",
                blocksize=max(64, int(round(rate * self.settings.frame_ms / 1000))),
                callback=self._on_audio,
            )
            stream.start()
        except Exception as exc:  # noqa: BLE001 - device busy or unsupported
            return self._fail(f"Could not open the microphone: {exc}")

        with self._lock:
            self._stream = stream
            self._vad = vad
            self._rate = rate
            self._device_name = device_name
            self._pending.clear()
            self._running = True
            self._state = "listening"
            self._message = (
                f"Listening on {device_name} @ {rate} Hz. "
                f'Say "{self.settings.wake_word}" to ask something.'
            )

        self._thread = threading.Thread(
            target=self._worker, name="apache-listener", daemon=True
        )
        self._thread.start()
        return self._message

    def stop(self) -> str:
        """Close the microphone and stop the worker."""
        with self._lock:
            if not self._running:
                return self._message or "Microphone is off."
            self._running = False
            stream, self._stream = self._stream, None

        try:
            if stream is not None:
                stream.stop()
                stream.close()
        except Exception:  # noqa: BLE001 - already closed
            pass

        thread, self._thread = self._thread, None
        # Never join ourselves: the worker reports an error by calling stop().
        if (
            thread is not None
            and thread.is_alive()
            and thread is not threading.current_thread()
        ):
            thread.join(timeout=2.0)

        with self._lock:
            self._pending.clear()
            self._state = "stopped"
            self._message = "Microphone is off."
        return self._message

    def mute_for(self, seconds: float) -> None:
        """Ignore audio for a while -- used so we never transcribe ourselves."""
        if seconds <= 0:
            return
        with self._lock:
            self._muted_until = max(self._muted_until, time.time() + seconds)

    def unmute(self) -> None:
        with self._lock:
            self._muted_until = 0.0

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _fail(self, message: str) -> str:
        """Startup failure: there is no stream and no thread to tear down."""
        with self._lock:
            self._running = False
            self._state = "error"
            self._message = message
        self._notify_error(message)
        return message

    def _report_error(self, message: str) -> None:
        """Runtime failure: the microphone stays open, only the status changes.

        Using :meth:`_fail` here would set ``_running`` false without closing
        the PortAudio stream, leaking the device.
        """
        with self._lock:
            if self._running:
                self._state = "error"
                self._message = message
        self._notify_error(message)

    def _notify_error(self, message: str) -> None:
        if self.on_error is not None:
            try:
                self.on_error(message)
            except Exception:  # noqa: BLE001 - callbacks must not kill us
                pass

    def _on_audio(self, indata: Any, frames: int, time_info: Any, status: Any) -> None:
        """PortAudio callback. Must never raise."""
        try:
            block = indata.copy().reshape(-1)
        except Exception:  # noqa: BLE001 - pragma: no cover
            return
        with self._lock:
            self._pending.append(block)
            # Bound memory if the worker stalls for any reason.
            while len(self._pending) > 64:
                self._pending.popleft()

    def _worker(self) -> None:
        while True:
            with self._lock:
                if not self._running:
                    return
                blocks = list(self._pending)
                self._pending.clear()
                vad = self._vad

            if not blocks or vad is None:
                time.sleep(0.02)
                continue

            utterances = []
            try:
                for block in blocks:
                    utterances.extend(vad.feed(block))
            except Exception:  # noqa: BLE001 - a bad block must not kill audio
                continue

            if not utterances:
                continue

            with self._lock:
                muted = time.time() < self._muted_until
            if muted:
                # We are mid-playback; transcribing our own voice would loop.
                continue

            for utterance in utterances:
                if not self._running:
                    return
                self._handle_utterance(utterance)

    def _handle_utterance(self, utterance: Any) -> None:
        from . import stt

        with self._lock:
            self._state = "transcribing"

        try:
            transcript = stt.transcribe(
                utterance.samples,
                utterance.sample_rate,
                language=self.settings.stt_language,
            )
        except stt.STTUnavailable as exc:
            self._stt_failure(str(exc))
            return

        with self._lock:
            self._stt_failures = 0
            if not transcript:
                self._state = "listening"
                self._message = "Heard something, but could not make out words."
                return
            self._transcript = transcript

        matched, command = match_wake(transcript, self.settings.wake_aliases)

        with self._lock:
            if not matched:
                self._state = "listening"
                self._message = f'Heard "{transcript}" (no wake word).'
                return

            self._state = "command"
            if command:
                self._message = f'Command: "{command}"'
            else:
                self._message = "Wake word detected."

        payload = command.strip() or WAKE_ONLY_PROMPT
        try:
            self.on_command(payload)
        except Exception as exc:  # noqa: BLE001 - handler errors stay local
            self._report_error(f"Could not handle the command: {exc}")
            return

        with self._lock:
            self._state = "listening"
            self._message = (
                f'Heard "{transcript}". '
                f'Say "{self.settings.wake_word}" again for the next question.'
            )

    def _stt_failure(self, message: str) -> None:
        """Report a recogniser failure, and give up if they never stop.

        A permanently unreachable endpoint must not leave the microphone
        open and churning forever.
        """
        with self._lock:
            self._stt_failures += 1
            consecutive = self._stt_failures

        self._report_error(message)

        if consecutive >= self.max_stt_failures:
            self._report_error(
                f"Speech recognition failed {consecutive} times in a row; "
                "stopping the microphone. Typing still works."
            )
            self.stop()
