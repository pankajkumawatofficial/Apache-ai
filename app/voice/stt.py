"""Speech to text via Google's free web endpoint, through SpeechRecognition.

Utterances arrive from the VAD as float samples; they are converted to 16-bit
PCM and handed to :class:`speech_recognition.AudioData`, which encodes FLAC
using the encoder it bundles for the current platform.
"""

from __future__ import annotations

import sys
from typing import Any

__all__ = ["STTUnavailable", "to_pcm16", "transcribe"]

_MIN_PCM_BYTES = 2 * 800          # <0.25 s of mono 16-bit at 16 kHz


class STTUnavailable(RuntimeError):
    """Raised when the recogniser could not be reached (usually no network)."""


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


def transcribe(samples: Any, sample_rate: int, language: str = "en-US") -> str:
    """Recognise one utterance.

    Returns ``""`` when nothing was audible, and raises
    :class:`STTUnavailable` when the network endpoint cannot be reached --
    the caller decides whether that is fatal.
    """
    pcm = to_pcm16(samples)
    if len(pcm) < _MIN_PCM_BYTES:
        return ""
    if sample_rate <= 0:
        raise STTUnavailable("invalid sample rate")

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
