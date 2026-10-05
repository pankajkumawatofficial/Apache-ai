"""Continuous microphone listening with wake-word detection.

Three stages, deliberately kept apart so no stage can stall capture:

1. **PortAudio callback** -- copies each block onto a deque and returns. It
   never does I/O and never raises.
2. **Capture worker** -- drains that deque through the energy VAD. Pure CPU,
   microseconds per block, so the microphone is always drained in time.
3. **Pipeline worker** -- recognises completed utterances and runs the matched
   command. This is the only stage that touches the network or the model, and
   it can block for a whole agent turn (tens of seconds on CPU), which is
   exactly why it must not be the thread that drains the microphone.

Before this split the pipeline ran inline in the capture worker, so a single
spoken question stalled audio consumption: the 64-block ring filled, the
oldest audio was silently dropped, and whatever the user said while Apache
answered was lost. Worse, the VAD kept being fed the text-to-speech playback,
which walked the adaptive noise floor up until ordinary speech no longer
crossed the gate -- capture degraded more with every reply.

The listener owns no conversation state: it only calls ``on_command(text)``
and reports its own status for the UI to display.
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

__all__ = ["WakeListener", "WAKE_ONLY_PROMPT"]

#: What the assistant is told when the wake word arrives with no instruction.
WAKE_ONLY_PROMPT = (
    "I just said your wake word but gave no instruction. "
    "Reply with a brief greeting and ask what you can help with."
)

#: PortAudio blocks held while the capture worker catches up. At the default
#: 30 ms frame this is ~1.9 s of audio -- a ceiling, not a target; the worker
#: normally empties it every 20 ms.
_MAX_PENDING_BLOCKS = 64

#: Completed utterances held while the recogniser is busy. Small on purpose:
#: speech captured *during* an answer is stale by the time we could act on it,
#: so we would rather discard it and say so than replay it minutes later.
_MAX_UTTERANCES = 4


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
        self._utterances: deque[Any] = deque(maxlen=_MAX_UTTERANCES)
        self._capture_thread: threading.Thread | None = None
        self._pipeline_thread: threading.Thread | None = None
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

        # Diagnostics. Silent loss is the hardest kind of failure to debug, so
        # every counter that would otherwise be invisible is surfaced in the UI.
        self._dropped_blocks = 0
        self._xruns = 0
        self._ignored_utterances = 0
        self._capture_errors = 0
        self._level = 0.0

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
                # Capture health.
                "level": self._level,
                "threshold": self._vad.threshold if self._vad is not None else 0.0,
                "dropped_blocks": self._dropped_blocks,
                "xruns": self._xruns,
                "ignored": self._ignored_utterances,
                "capture_errors": self._capture_errors,
                "queued": len(self._utterances),
            }

    def start(self) -> str:
        """Open the microphone and start both workers. Returns a status line."""
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
            self._utterances.clear()
            self._dropped_blocks = 0
            self._xruns = 0
            self._ignored_utterances = 0
            self._capture_errors = 0
            self._level = 0.0
            self._stt_failures = 0
            self._running = True
            self._state = "listening"
            self._message = (
                f"Listening on {device_name} @ {rate} Hz. "
                f'Say "{self.settings.wake_word}" to ask something.'
            )

        self._capture_thread = threading.Thread(
            target=self._capture_loop, name="apache-capture", daemon=True
        )
        self._pipeline_thread = threading.Thread(
            target=self._pipeline_loop, name="apache-pipeline", daemon=True
        )
        self._capture_thread.start()
        self._pipeline_thread.start()
        return self._message

    def stop(self) -> str:
        """Close the microphone and stop both workers."""
        with self._lock:
            if not self._running:
                return self._message or "Microphone is off."
            self._running = False
            stream, self._stream = self._stream, None
            capture, self._capture_thread = self._capture_thread, None
            pipeline, self._pipeline_thread = self._pipeline_thread, None

        try:
            if stream is not None:
                stream.stop()
                stream.close()
        except Exception:  # noqa: BLE001 - already closed
            pass

        current = threading.current_thread()
        for thread in (capture, pipeline):
            # Never join ourselves: the pipeline reports recogniser failures by
            # calling stop() from its own thread.
            if (
                thread is not None
                and thread.is_alive()
                and thread is not current
            ):
                thread.join(timeout=2.0)

        with self._lock:
            self._pending.clear()
            self._utterances.clear()
            self._state = "stopped"
            self._message = "Microphone is off."
        return self._message

    def mute_for(self, seconds: float) -> None:
        """Ignore audio for a while -- used so we never transcribe ourselves."""
        if seconds <= 0:
            return
        with self._lock:
            self._muted_until = max(self._muted_until, time.time() + seconds)
            # Anything already segmented was captured before playback started;
            # it is about to become stale anyway, so drop it rather than have
            # the pipeline answer something we are currently saying out loud.
            self._ignored_utterances += len(self._utterances)
            self._utterances.clear()

    def unmute(self) -> None:
        with self._lock:
            self._muted_until = 0.0

    # ------------------------------------------------------------------
    # Stage 1: PortAudio callback
    # ------------------------------------------------------------------
    def _on_audio(self, indata: Any, frames: int, time_info: Any, status: Any) -> None:
        """PortAudio callback. Must never raise, must never block."""
        try:
            if status:
                # PortAudio raises this flag on overflow/underflow -- an xrun
                # means real samples were lost, which is precisely the symptom
                # of a consumer that fell behind. Count it instead of hiding it.
                with self._lock:
                    self._xruns += 1
            block = indata.copy().reshape(-1)
        except Exception:  # noqa: BLE001 - pragma: no cover
            return
        with self._lock:
            self._pending.append(block)
            while len(self._pending) > _MAX_PENDING_BLOCKS:
                self._pending.popleft()
                self._dropped_blocks += 1

    # ------------------------------------------------------------------
    # Stage 2: capture worker (drains the microphone, never blocks)
    # ------------------------------------------------------------------
    def _capture_loop(self) -> None:
        while True:
            with self._lock:
                if not self._running:
                    return
                blocks = list(self._pending)
                self._pending.clear()
                vad = self._vad
                muted = time.time() < self._muted_until

            if not blocks or vad is None:
                time.sleep(0.02)
                continue

            if muted:
                # Do NOT feed the VAD while the reply plays. Its noise floor
                # adapts from quiet frames, so letting loud TTS through would
                # walk the threshold up until the next real utterance never
                # trips it -- capture degrades with every reply.
                vad.reset()
                with self._lock:
                    self._level = 0.0
                continue

            utterances = []
            try:
                for block in blocks:
                    self._note_level(block)
                    try:
                        utterances.extend(vad.feed(block))
                    except Exception:  # noqa: BLE001 - one bad block, not all
                        with self._lock:
                            self._capture_errors += 1
            except Exception:  # noqa: BLE001 - a bad block must not kill audio
                continue

            if not utterances:
                continue

            with self._lock:
                for utterance in utterances:
                    self._utterances.append(utterance)

    def _note_level(self, block: Any) -> None:
        """Track the loudest recent block so the UI can show a live meter.

        Entirely self-contained: it must never raise, because it runs inside
        the capture loop and an exception there used to abort the whole batch
        -- audio was consumed and never reached the VAD.
        """
        try:
            if len(block) == 0:
                return
            total = 0.0
            for sample in block:
                total += float(sample) * float(sample)
            rms = (total / len(block)) ** 0.5
        except Exception:  # noqa: BLE001 - display only, never fatal
            return
        try:
            with self._lock:
                # Decay rather than snap, so a single loud frame does not pin
                # the meter at full scale for the rest of the utterance.
                self._level = max(rms, self._level * 0.9)
        except Exception:  # noqa: BLE001 - pragma: no cover
            pass

    # ------------------------------------------------------------------
    # Stage 3: pipeline (recognition + the agent turn)
    # ------------------------------------------------------------------
    def _pipeline_loop(self) -> None:
        while True:
            with self._lock:
                if not self._running:
                    return
                utterance = self._utterances.popleft() if self._utterances else None

            if utterance is None:
                time.sleep(0.03)
                continue

            self._handle_utterance(utterance)

            # Whatever arrived while we were recognising or answering is stale.
            # Replaying a command two minutes after it was spoken is worse than
            # dropping it, so drop it -- visibly.
            with self._lock:
                stale = len(self._utterances)
                if stale:
                    self._ignored_utterances += stale
                    self._utterances.clear()

    def _handle_utterance(self, utterance: Any) -> None:
        from . import stt

        with self._lock:
            if not self._running:
                return
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
            self._message = (
                f'Command: "{command}" — answering now. '
                "Anything said until the reply finishes is ignored."
            )

        payload = command.strip() or WAKE_ONLY_PROMPT
        try:
            self.on_command(payload)
        except Exception as exc:  # noqa: BLE001 - handler errors stay local
            self._report_error(f"Could not handle the command: {exc}")
            return

        with self._lock:
            if not self._running:
                return
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

    # ------------------------------------------------------------------
    # Errors
    # ------------------------------------------------------------------
    def _fail(self, message: str) -> str:
        """Startup failure: there is no stream and no worker to tear down."""
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
