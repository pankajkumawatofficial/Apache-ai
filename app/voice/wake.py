"""Wake-word matching for the continuous listener.

Transcripts arrive from Google's speech API already lower-cased most of the
time, but not always, and it regularly mis-hears the wake word -- hence the
list of aliases.  Matching is case-insensitive, punctuation-tolerant and only
accepts the wake word at the very start of an utterance, so a sentence that
merely *mentions* Apache does not fire the assistant.
"""

from __future__ import annotations

import re

__all__ = ["match_wake", "resolve_command", "clean_for_speech"]

# Trailing punctuation/whitespace consumed along with the wake word itself.
_TAIL = r"[\s,.;:!?\-—'\"`]*"


def _compile(aliases: list[str]) -> re.Pattern[str]:
    parts = []
    for alias in aliases:
        # Allow optional whitespace inside multi-word aliases ("a patch").
        words = [re.escape(w) for w in str(alias).split() if w]
        if not words:
            continue
        parts.append(r"\s+".join(words))
    if not parts:
        # Fall back to the configured wake word so a misconfiguration still
        # behaves like a normal wake word rather than matching everything.
        parts = [re.escape("apache")]
    body = "|".join(sorted(parts, key=len, reverse=True))
    # Anchored: the wake word must open the utterance.  The lookahead stops
    # "apathology" or "apachee" from matching "apache".
    return re.compile(rf"^\s*(?:{body})(?![a-z0-9]){_TAIL}", re.IGNORECASE)


def match_wake(text: str, aliases: list[str]) -> tuple[bool, str]:
    """Split *text* on a leading wake word.

    Returns ``(matched, command)``.  When matched, *command* is the utterance
    with the wake word and its trailing punctuation removed.  An utterance
    that is only the wake word yields an empty command.
    """
    if not text or not text.strip():
        return False, ""

    pattern = _compile(aliases)
    match = pattern.match(text)
    if match is None:
        return False, ""

    command = text[match.end():].strip()
    return True, command


def resolve_command(
    text: str, aliases: list[str], wake_required: bool = True
) -> tuple[bool, str]:
    """Decide what, if anything, from *text* should be run.

    Returns ``(act, command)`` -- an utterance to run, or an utterance to
    ignore -- because the listener only has those two options and confusing
    them is expensive either way.

    With *wake_required* the wake word gates everything, and a transcript
    without it is dropped. That is what a television, a neighbour or someone
    in the next room sounds like from inside a microphone.

    Without it every complete utterance is a command, but a leading wake word
    is still stripped, so "Apache, what time is it" and "what time is it"
    both arrive as "what time is it" and behave identically.
    """
    matched, stripped = match_wake(text, aliases)
    if matched:
        return True, stripped
    if wake_required or not text or not text.strip():
        return False, ""
    return True, text.strip()


def clean_for_speech(text: str) -> str:
    """Strip formatting that sounds wrong when read aloud.

    The system prompt already asks the model for plain prose, but small models
    drift, so this is the belt to that prompt's braces.
    """
    if not text:
        return ""

    out = str(text)

    # Code fences -> keep the code, drop the fence markers and language tag.
    out = re.sub(r"```[a-zA-Z0-9_+-]*\n?", " ", out)
    out = out.replace("```", " ")

    # Headings, list markers, block quotes, horizontal rules.
    out = re.sub(r"(?m)^\s{0,3}#{1,6}\s*", "", out)
    out = re.sub(r"(?m)^\s{0,3}>\s?", "", out)
    out = re.sub(r"(?m)^\s*[-*+]\s+", "", out)
    out = re.sub(r"(?m)^\s*\d+\.\s+", "", out)
    out = re.sub(r"(?m)^\s*[-*_]{3,}\s*$", "", out)

    # Inline emphasis and links: keep the visible text.
    out = re.sub(r"!\[([^\]]*)\]\([^)]*\)", r"\1", out)   # images
    out = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", out)     # links
    out = re.sub(r"`([^`]+)`", r"\1", out)                 # inline code
    out = re.sub(r"(\*\*|__)(.+?)\1", r"\2", out)          # bold
    out = re.sub(r"(\*|_)(.+?)\1", r"\2", out)             # italic

    # Table separators and pipes read badly.
    out = re.sub(r"(?m)^\s*\|?[\s:|-]+\|\s*$", " ", out)
    out = out.replace("|", ". ")

    out = re.sub(r"[ \t]+", " ", out)
    out = re.sub(r"\n{2,}", "\n", out)
    return out.strip()
