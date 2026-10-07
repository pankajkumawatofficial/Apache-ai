"""Speech to text.

Offline by default: utterances go to :mod:`faster_whisper`, which runs a local
model and never touches the network. That removes a round trip from every
query, removes the throttling Google applies to a burst of recognitions, and
keeps working with the cable unplugged. ``APACHE_OFFLINE=0`` uses Google's
free web endpoint through :mod:`speech_recognition` instead, which is more
accurate on noisy input but needs the network.

Utterances arrive from the VAD as float samples. The local path passes them to
Whisper as a NumPy array rather than a file: ``transcribe()`` skips its
``decode_audio`` step for an array, which is both faster and the only way to
avoid a PyAV incompatibility (``av.open`` no longer accepts the
``metadata_errors`` argument faster-whisper passes, and would raise
``TypeError`` on every call).
"""

from __future__ import annotations

import sys
import threading
from typing import Any

from ..config import settings

__all__ = ["STTUnavailable", "to_pcm16", "transcribe", "warm_up"]

#: 800 samples -- the same floor the online path has always used.
_MIN_SAMPLES = 800

#: Whisper's own input rate. Anything else is resampled on the way in.
_WHISPER_RATE = 16_000


class STTUnavailable(RuntimeError):
    """Raised when the recogniser failed (usually no network, or no model)."""


def to_pcm16(samples: Any) -> bytes:
    """Convert float samples in [-1, 1] to little-endian signed 16-bit PCM.

    Uses NumPy when it is importable (it always is alongside ``sounddevice``)
    and falls back to the standard library so this stays testable in isolation.
    """
    try:
        import numpy as np
    except ImportError:
        import array as _array

        packed = _array.array("h")
        for sample in samples:
            value = float(sample)
            value = -1.0 if value < -1.0 else (1.0 if value > 1.0 else value)
            packed.append(int(value * 32767.0))
        if sys.byteorder != "little":
            packed.byteswap()
        return packed.tobytes()

    array = np.asarray(samples, dtype=np.float32).reshape(-1)
    np.clip(array, -1.0, 1.0, out=array)
    return (array * 32767.0).astype("<i2").tobytes()


# --------------------------------------------------------------------------
# Offline: faster-whisper
# --------------------------------------------------------------------------
_MODEL: Any = None
_MODEL_LOCK = threading.Lock()


def _whisper_model():
    """Load the local model once and keep it; loading per query costs seconds."""
    global _MODEL
    if _MODEL is not None:
        return _MODEL

    with _MODEL_LOCK:
        if _MODEL is not None:
            return _MODEL
        try:
            from faster_whisper import WhisperModel
        except Exception as exc:  # noqa: BLE001 - blocked or missing wheel
            raise STTUnavailable(f"offline recognition unavailable: {exc}") from exc
        try:
            _MODEL = WhisperModel(
                settings.whisper_model,
                device="cpu",
                compute_type="int8",
                download_root=str(settings.whisper_dir),
            )
        except Exception as exc:  # noqa: BLE001 - missing or partial model
            raise STTUnavailable(
                f"could not load {settings.whisper_model}: {exc}"
            ) from exc
    return _MODEL


def _whisper_audio(samples: Any, sample_rate: int):
    """Float32 mono at 16 kHz -- what Whisper's feature extractor expects.

    The capture rate is whatever the microphone's default is (often 44.1 or
    48 kHz), so this resamples on the way in. Linear interpolation is enough
    here: Whisper is trained on speech that has already been band-limited, and
    the alternative is pulling in SciPy for one step.
    """
    import numpy as np

    array = np.asarray(samples, dtype=np.float32).reshape(-1)
    np.clip(array, -1.0, 1.0, out=array)
    if sample_rate == _WHISPER_RATE or array.size < 2:
        return array

    duration = array.size / float(sample_rate)
    target = max(1, int(round(duration * _WHISPER_RATE)))
    source = np.linspace(0.0, duration, num=array.size, endpoint=False)
    at = np.arange(target, dtype=np.float64) / float(_WHISPER_RATE)
    return np.interp(at, source, array).astype(np.float32)


def _whisper_language(language: str) -> str:
    """Whisper wants ``en``, not ``en-US``."""
    return (language or "en").split("-", 1)[0].split("_", 1)[0].lower() or "en"


def _transcribe_local(samples: Any, sample_rate: int, language: str) -> str:
    model = _whisper_model()
    audio = _whisper_audio(samples, sample_rate)
    try:
        # An ndarray is passed straight through: no decode, no PyAV.
        segments, _info = model.transcribe(
            audio,
            language=_whisper_language(language),
            # Greedy decoding (1) takes the first word that fits each frame,
            # which is how a noisy utterance came back as "45" for
            # "spotify". Whisper's own default width scores whole candidates
            # before committing; it is the cheaper half of the fix, the
            # larger model above being the other.
            beam_size=max(1, int(settings.whisper_beam)),
            vad_filter=False,
            # A local model with no history is more predictable, and this is
            # one utterance at a time rather than a transcript.
            condition_on_previous_text=False,
        )
        text = " ".join(segment.text for segment in segments).strip()
    except Exception as exc:  # noqa: BLE001 - engine internals, never fatal
        raise STTUnavailable(
            f"offline recognition failed: {type(exc).__name__}: {exc}"
        ) from exc
    return text


# --------------------------------------------------------------------------
# Online: Google's free web endpoint
# --------------------------------------------------------------------------
def _recognizer():
    import speech_recognition as sr

    # Reused: constructing one per utterance re-allocates state for no benefit.
    # The recogniser's own energy threshold is irrelevant here -- we segment
    # the audio ourselves and hand recogniser a finished utterance.
    global _RECOGNIZER  # noqa: PLW0603 - deliberate module-level cache
    if _RECOGNIZER is None:
        _RECOGNIZER = sr.Recognizer()
    return _RECOGNIZER


_RECOGNIZER: Any = None


def _transcribe_google(samples: Any, sample_rate: int, language: str) -> str:
    pcm = to_pcm16(samples)
    if len(pcm) < 2 * _MIN_SAMPLES:
        return ""

    try:
        import speech_recognition as sr
    except Exception as exc:  # noqa: BLE001 - blocked/native import
        raise STTUnavailable(f"speech recognition unavailable: {exc}") from exc

    audio = sr.AudioData(pcm, sample_rate, 2)
    try:
        result = _recognizer().recognize_google(
            audio, language=language, show_all=False
        )
    except sr.UnknownValueError:
        return ""
    except sr.RequestError as exc:
        raise STTUnavailable(f"Google speech recognition failed: {exc}") from exc
    except Exception as exc:  # noqa: BLE001 - flac encoder, DNS, proxy, ...
        raise STTUnavailable(
            f"speech recognition failed: {type(exc).__name__}: {exc}"
        ) from exc

    return str(result or "").strip()


# --------------------------------------------------------------------------
def warm_up() -> bool:
    """Load the local model so the first query does not pay for it.

    Returns False when there is nothing to warm up (online mode, or the model
    is missing) -- warming up is best effort and never fails a launch.
    """
    if not settings.offline:
        return False
    try:
        _whisper_model()
    except STTUnavailable:
        return False
    return True


def transcribe(samples: Any, sample_rate: int, language: str = "en-US") -> str:
    """Recognise one utterance.

    Returns ``""`` when nothing was audible, and raises
    :class:`STTUnavailable` when the engine failed -- the caller decides
    whether that is fatal.
    """
    if sample_rate <= 0:
        raise STTUnavailable("invalid sample rate")
    if samples is None or len(samples) < _MIN_SAMPLES:
        return ""

    if settings.offline:
        return _transcribe_local(samples, sample_rate, language)
    return _transcribe_google(samples, sample_rate, language)
