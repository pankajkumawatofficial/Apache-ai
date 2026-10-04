"""Text-to-speech for Apache's spoken replies.

Microsoft Edge neural voices via :mod:`edge_tts`.  They need the network, but
they sound dramatically better than the built-in SAPI5 voices, which is the
whole point of a spoken assistant.

``clear_for_speech`` lives here because the reply path -- not the voice path --
is where a Markdown table would otherwise get read out as pipes.
"""

from __future__ import annotations

import asyncio
import re
from pathlib import Path

from .wake import clean_for_speech

__all__ = ["synthesize", "speakable_text", "estimate_duration_s", "TTSUnavailable"]

DEFAULT_VOICE = "en-US-AndrewMultilingualNeural"


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


async def _save(text: str, voice: str, rate: str, path: Path) -> None:
    import edge_tts  # imported lazily so text-only runs never touch the network stack

    communicate = edge_tts.Communicate(text, voice=voice, rate=rate)
    await communicate.save(str(path))


def synthesize(
    text: str,
    out_path: Path | str,
    *,
    voice: str = DEFAULT_VOICE,
    rate: str = "+0%",
) -> Path:
    """Write *text* to *out_path* as an MP3 and return the path.

    Raises :class:`TTSUnavailable` on failure so the caller can fall back to
    a silent reply rather than crashing the turn.
    """
    spoken = speakable_text(text)
    if not spoken:
        raise TTSUnavailable("nothing to speak")

    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    try:
        # edge_tts owns the event loop; run it in isolation so we never
        # collide with whatever loop the caller (Gradio) is already using.
        asyncio.run(_save(spoken, voice, rate, path))
    except Exception as exc:  # network, DNS, revoked endpoint, bad voice name
        raise TTSUnavailable(str(exc)) from exc

    if not path.exists() or path.stat().st_size == 0:
        raise TTSUnavailable("synthesis produced no audio")
    return path
