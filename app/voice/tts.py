"""Text-to-speech for Apache's spoken replies.

Offline by default: replies are synthesised by :mod:`piper` from a local ONNX
voice, which takes milliseconds once the model is loaded and touches no
network at all. ``APACHE_OFFLINE=0`` switches to Microsoft Edge neural voices
via :mod:`edge_tts`, which sound better but cost a network round trip for
every single reply -- the reason a spoken answer was slow to arrive, and on a
bad connection, never arrived.

``speakable_text`` lives here because the reply path -- not the voice path --
is where a Markdown table would otherwise get read out as pipes.
"""

from __future__ import annotations

import asyncio
import re
import tempfile
import threading
import wave
from pathlib import Path
from typing import Any

from ..config import MODELS, settings
from .wake import clean_for_speech

__all__ = [
    "TTSUnavailable",
    "available_voices",
    "clip_extension",
    "estimate_duration_s",
    "resolve_model",
    "speakable_text",
    "synthesize",
    "warm_up",
]

DEFAULT_VOICE = "en-US-AndrewMultilingualNeural"

#: Where downloaded Piper voices live.
PIPER_DIR = MODELS / "piper"

#: Loaded on first use and kept: Piper takes seconds to load and milliseconds
#: to speak, so reloading it per reply would undo the whole point. Keyed by
#: the ONNX path, so changing voice costs one load rather than one per reply.
_PIPER_VOICES: dict[str, Any] = {}
#: The greeting, an idle check-in and a first reply can all reach the voice
#: at once, so loading is serialised rather than raced.
_PIPER_LOCK = threading.Lock()


class TTSUnavailable(RuntimeError):
    """Raised when synthesis could not be completed (usually no network)."""


def speakable_text(text: str) -> str:
    """Prepare an assistant reply for synthesis.

    Drops the speaker attribution some models add, and caps very long replies
    so a rambling answer does not hold the microphone mute for a minute.
    """
    cleaned = clean_for_speech(text)
    if not cleaned:
        return ""

    # "Assistant: ..." / "Apache: ..." prefixes.
    cleaned = re.sub(r"(?i)^\s*(assistant|apache)\s*:\s*", "", cleaned)

    # Markers are silent to a synthesiser and only ever sound like a stumble.
    # Errors are spoken too -- every reply is spoken -- so the warning sign
    # that opens one has to go before the sentence it decorates is read out.
    cleaned = re.sub(r"[⚠✓✔✗✖★☆→←↑↓●■]+", " ", cleaned)
    cleaned = re.sub(r"(?i)^\s*(error|warning|notice)\s*[:\-]\s*", "", cleaned)

    max_chars = 1_500
    if len(cleaned) > max_chars:
        cut = cleaned[:max_chars]
        # Prefer to stop on a sentence boundary.
        stop = max(cut.rfind(". "), cut.rfind("! "), cut.rfind("? "))
        if stop > max_chars // 2:
            cut = cut[: stop + 1]
        cleaned = cut + " Answer continues in the chat."

    return cleaned.strip()


def estimate_duration_s(text: str) -> float:
    """Rough spoken length, used to mute the mic so we do not hear ourselves."""
    if not text:
        return 0.0
    # ~150 words/minute, ~5.5 characters per word.
    words = max(1, len(text) / 5.5)
    return round(words / 150.0 * 60.0, 2)


def clip_extension() -> str:
    """Suffix matching the current engine: ``.wav`` offline, ``.mp3`` online.

    The reply path asks for this so a clip is named for what is inside it --
    a Piper WAV wearing an .mp3 extension fails to decode in the browser,
    which looks exactly like Apache choosing not to speak.
    """
    return ".wav" if settings.offline else ".mp3"


async def _save(text: str, voice: str, rate: str, path: Path) -> None:
    import edge_tts  # imported lazily so text-only runs never touch the network stack

    communicate = edge_tts.Communicate(text, voice=voice, rate=rate)
    await communicate.save(str(path))


def available_voices() -> list[str]:
    """Piper voices already on disk, for the Voice dropdown.

    Offline, the dropdown used to list edge-tts names that the engine then
    ignored -- so "speak in a different voice" was a control that did
    nothing. This is what it should have been offering.
    """
    try:
        return sorted(p.stem for p in PIPER_DIR.glob("*.onnx"))
    except OSError:  # pragma: no cover - unreadable models directory
        return []


def resolve_model(voice: str | None = None) -> Path:
    """Pick the ONNX to speak with, honouring a voice chosen in the UI.

    Accepts a bare name ("en_US-lessac-medium"), a filename, or a path, and
    compares names case-insensitively because that is how people type them --
    the filesystem's own rules are not something a dropdown should expose.
    Anything that is not on disk falls back to the configured model: a stale
    dropdown value must degrade to "sounds like the usual Apache", never to
    silence.
    """
    wanted = (voice or "").strip()
    if wanted:
        stem = Path(wanted).name
        if stem.lower().endswith(".onnx"):
            stem = stem[: -len(".onnx")]
        for candidate in (PIPER_DIR / f"{stem}.onnx", Path(wanted)):
            if candidate.exists():
                return candidate
        lowered = stem.lower()
        for candidate in PIPER_DIR.glob("*.onnx"):
            if candidate.stem.lower() == lowered:
                return candidate
    return Path(settings.piper_voice)


def _piper_voice(model: Path | None = None):
    """Load a local voice once and keep it in memory.

    Double-checked under a lock: the greeting, an idle check-in and a first
    reply can all arrive together, and two threads loading 100 MB at once is
    how the first spoken line ends up several seconds late.
    """
    target = resolve_model(None) if model is None else Path(model)
    key = str(target)
    cached = _PIPER_VOICES.get(key)
    if cached is not None:
        return cached

    with _PIPER_LOCK:
        cached = _PIPER_VOICES.get(key)
        if cached is not None:
            return cached

        from piper import PiperVoice  # optional dependency, imported on use

        if not target.exists():
            raise TTSUnavailable(
                f"no offline voice at {target}. Fetch it once following the README, "
                "or set APACHE_OFFLINE=0 to go back to edge-tts."
            )
        try:
            loaded = PiperVoice.load(str(target))
        except Exception as exc:  # noqa: BLE001 - a bad download must not wedge
            raise TTSUnavailable(f"could not load {target.name}: {exc}") from exc
        _PIPER_VOICES[key] = loaded
    return loaded


def warm_up() -> bool:
    """Load the offline voice and run one tiny synthesis.

    Loading plus the first inference costs several seconds, because
    onnxruntime plans the graph on that call and never again. Paying it
    before the greeting turns the greeting into a sub-second clip instead of
    five seconds of silence after the page opens. Best effort: warming up is
    never allowed to fail the launch.
    """
    if not settings.offline:
        return False
    try:
        _piper_voice()
    except TTSUnavailable:
        return False
    try:
        target = Path(tempfile.gettempdir()) / "apache-tts-warmup.wav"
        _piper_speech("hi", target, "+0%")
        target.unlink(missing_ok=True)
    except Exception:  # noqa: BLE001 - best effort by design
        return False
    return True


def _length_scale(rate: str) -> float:
    """Turn an edge-style ``+20%`` into Piper's length scale.

    Piper's scale measures duration rather than speed, so saying the same
    words in 120% of the time is a scale of 1/1.2 -- faster means smaller.
    """
    match = re.match(r"^\s*([+-]?\d+(?:\.\d+)?)\s*%", str(rate))
    if not match:
        return 1.0
    try:
        speed = 1.0 + float(match.group(1)) / 100.0
    except ValueError:
        return 1.0
    if speed <= 0.05:
        return 1.0
    return 1.0 / speed


def _piper_speech(
    spoken: str, path: Path, rate: str, voice: str | None = None
) -> None:
    """Synthesise locally. No network, and the model stays loaded."""
    engine = _piper_voice(resolve_model(voice))
    # Both knobs decide how human this sounds: length scale is pace, noise
    # scale is how much the voice may vary between runs. Flat prosody at a
    # metronome pace is exactly what reads as a machine reading text.
    from piper import SynthesisConfig

    config = SynthesisConfig(
        length_scale=_length_scale(rate),
        noise_scale=float(settings.tts_noise_scale),
    )
    try:
        with wave.open(str(path), "wb") as handle:
            engine.synthesize_wav(spoken, handle, syn_config=config)
    except Exception as exc:  # noqa: BLE001 - audio must never break a turn
        raise TTSUnavailable(f"offline synthesis failed: {exc}") from exc


def _edge_speech(spoken: str, path: Path, voice: str, rate: str) -> None:
    try:
        # edge_tts owns the event loop; run it in isolation so we never
        # collide with whatever loop the caller (Gradio) is already using.
        asyncio.run(_save(spoken, voice, rate, path))
    except Exception as exc:  # network, DNS, revoked endpoint, bad voice name
        raise TTSUnavailable(str(exc)) from exc


def synthesize(
    text: str,
    out_path: Path | str,
    *,
    voice: str = DEFAULT_VOICE,
    rate: str = "+0%",
) -> Path:
    """Write *text* to *out_path* and return the path.

    Raises :class:`TTSUnavailable` on failure so the caller can fall back to
    a silent reply rather than crashing the turn.
    """
    spoken = speakable_text(text)
    if not spoken:
        raise TTSUnavailable("nothing to speak")

    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if settings.offline:
        _piper_speech(spoken, path, rate, voice)
    else:
        _edge_speech(spoken, path, voice, rate)

    if not path.exists() or path.stat().st_size == 0:
        raise TTSUnavailable("synthesis produced no audio")
    return path
