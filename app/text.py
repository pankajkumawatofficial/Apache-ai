"""Turning model markup into words that read and speak the same way.

The system prompt in :mod:`app.config` tells the model never to write LaTeX,
because every reply is passed to a synthesiser. A 1.7-billion-parameter model
disobeys often enough to be heard -- ``$1739 \\times 42$`` reaches the speaker
as dollar signs and a backslash -- so :func:`plain_math` is the belt to that
prompt's braces.

Only the standard library is imported here. That keeps it testable on machines
where the ML and audio stacks cannot be installed, which is the contract
``tests/test_pure.py`` is built on.
"""

from __future__ import annotations

import re

__all__ = ["plain_math", "stream_cut"]

#: LaTeX a small model emits anyway, with the words it should become. A bare
#: command is worse than missing information because the synthesiser spells it
#: out loud, so the vocabulary is deliberately small and spoken-word shaped.
_MATH_WORDS = {
    "cdot": "times",
    "times": "times",
    "div": "divided by",
    "pm": "plus or minus",
    "leq": "less than or equal to",
    "le": "less than or equal to",
    "geq": "greater than or equal to",
    "ge": "greater than or equal to",
    "neq": "not equal to",
    "approx": "about",
    "infty": "infinity",
    "sum": "sum",
    "int": "integral",
    "pi": "pi",
}

#: Fenced and inline code must not be rewritten: ``\t`` in a Python sample is
#: a tab escape, not a maths command.
_CODE = re.compile(r"(```.*?```|`[^`\n]+`)", re.S)

#: Markup between the dollars gives the game away: ``$a^2$`` and
#: ``$1739 \times 42$`` are formulas, while ``$5`` is a price and ``$5 and
#: $10`` is prose. Losing a currency symbol would just be a different bug, so
#: the check stays conservative -- and runs before the commands are rewritten,
#: while the backslashes that identify a formula are still there.
_MATHY = re.compile(r"[\\^_]")


def plain_math(text: str) -> str:
    """Turn maths markup into words so the transcript and the voice agree.

    Cheap, always on, and a no-op on text that carries no markup at all. The
    rewrite is applied to the model's cumulative output on every token rather
    than appended to what came before, so a half-written command corrects
    itself as the rest of it arrives.
    """
    if "\\" not in text and "$" not in text:
        return text

    def prose(chunk: str) -> str:
        def dollars(match: re.Match[str]) -> str:
            inner = match.group(1)
            # Markup, or a lone token, is a formula. Words following a price
            # are prose, and keep the sign that makes them readable as money.
            if _MATHY.search(inner) or " " not in inner.strip():
                return inner
            return match.group(0)

        # Delimiters come off first: this is the one point where the
        # backslash identifying a formula is still in place.
        out = chunk
        out = re.sub(r"\$\$(.+?)\$\$", dollars, out, flags=re.S)
        out = re.sub(r"\$(.+?)\$", dollars, out, flags=re.S)

        # Structure next, while the braces are still intact.
        out = re.sub(r"\\frac\s*\{([^{}]*)\}\s*\{([^{}]*)\}", r"\1 over \2", out)
        out = re.sub(r"\\sqrt\s*\{([^{}]*)\}", r"square root of \1", out)
        out = re.sub(
            r"\\(?:text|mathrm|mathbf|mathit|operatorname|emph)\s*\{([^{}]*)\}",
            r"\1",
            out,
        )
        # Everything else falls back to the command's own name.
        out = re.sub(
            r"\\([a-zA-Z]+)",
            lambda m: _MATH_WORDS.get(m.group(1), m.group(1)),
            out,
        )
        # Grouping left over from superscripts and subscripts.
        out = re.sub(r"\^\{([^{}]*)\}", r" to the power \1", out)
        out = re.sub(r"_\{([^{}]*)\}", r" \1", out)
        out = re.sub(r"\^(\w)", r" to the power \1", out)
        return re.sub(r"[ \t]{2,}", " ", out)

    # Only prose is rewritten; code spans keep their escapes exactly as sent.
    parts = _CODE.split(text)
    for i in range(0, len(parts), 2):
        parts[i] = prose(parts[i])
    return "".join(parts)


#: Characters that mean something has been said whole. A newline counts
#: because a reply's paragraphs and list items arrive line by line, and
#: waiting for a full stop inside a bulleted list would hold the first item
#: until the last one landed.
_SENTENCE_END = ".!?…\n"

#: Without any punctuation at all -- a model writing one long sentence, or
#: a run of technical text -- the speaker waits no longer than this before
#: cutting at the last space it passed, so a paragraph still reaches the
#: ear while it is still being written.
_STREAM_LIMIT = 140
#: The shortest cut worth making at a space. Without a floor, "the" would
#: be handed over as its own clip, one word at a time.
_STREAM_FLOOR = 60


def _is_boundary(text: str, index: int) -> bool:
    """Is the punctuation at *index* the end of something said whole?"""
    char = text[index]
    # A line break ends a line by whatever follows it -- "- one\n- two" is
    # two lines, and waiting for a full stop would hold the first until the
    # last. "!?…" end a sentence on their own too.
    if char != ".":
        return True
    after = text[index + 1 : index + 2]
    # "3.14", "v2.0", "run.py": a full stop followed by more of the same
    # token belongs to the token, not to the sentence.
    if after and not after.isspace():
        return False
    head = text[:index].rsplit(None, 1)
    token = head[-1].lstrip("(\"'") if head else ""
    letters = token.replace(".", "")
    # "e.g.", "i.e.", "U.S.": a full stop after one or two letters ends an
    # abbreviation, and the sentence it sits in is still coming.
    return not (0 < len(letters) <= 2 and letters.isalpha())


def stream_cut(text: str) -> int:
    """How much of *text* is complete enough to be said out loud now.

    A reply arrives a token at a time, and the voice is asked for a chunk as
    soon as one is ready rather than at the end: half a sentence, an open
    code fence or a truncated link all read as nonsense through a speaker.
    This returns the length of the leading run that ends on a sentence
    boundary, and 0 while there is nothing safe to say yet -- so the caller
    can speak the front of the reply and hold the rest back for the next
    call, without either of them guessing where a thought stops.
    """
    if not text:
        return 0

    # Never stop inside a code fence. An unclosed one is a half-finished
    # block the model is still writing, and cutting there would read out a
    # fragment of code as if it were prose.
    if text.count("```") % 2:
        opener = text.find("```")
        if opener <= 0:
            return 0
        text = text[:opener]

    cut = 0
    for i, char in enumerate(text):
        if char in _SENTENCE_END and _is_boundary(text, i):
            cut = i + 1
    if cut:
        return cut

    if len(text) >= _STREAM_LIMIT:
        cut = text.rfind(" ", _STREAM_FLOOR, len(text))
        if cut > 0:
            return cut + 1
    return 0
