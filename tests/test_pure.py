"""Unit tests for Apache's dependency-free logic.

Run with:  python -m tests.test_pure

These import only the standard library plus Apache's own pure modules, so they
pass even where the ML/audio stack cannot be installed.
"""

from __future__ import annotations

import math
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import settings                                        # noqa: E402
from app.tools.calculator import CalcError, safe_eval          # noqa: E402
from app.tools.files import (                                  # noqa: E402
    FileAccessError,
    list_files,
    read_file,
    resolve_inside,
    write_file,
)
from app.tools.sandbox import run_python                       # noqa: E402
from app.voice.vad import EnergyVAD                            # noqa: E402
from app.voice.tts import estimate_duration_s, speakable_text  # noqa: E402
from app.voice.wake import clean_for_speech, match_wake        # noqa: E402

FAILURES: list[str] = []


def check(label: str, condition: bool, detail: str = "") -> None:
    if condition:
        print(f"  ok   {label}")
    else:
        FAILURES.append(label)
        print(f"  FAIL {label}  {detail}")


def expect_error(label: str, expression: str, exc_type=CalcError) -> None:
    try:
        result = safe_eval(expression)
    except exc_type:
        print(f"  ok   {label} (raised as expected)")
        return
    except Exception as exc:  # noqa: BLE001
        FAILURES.append(label)
        print(f"  FAIL {label}  wrong exception {type(exc).__name__}: {exc}")
        return
    FAILURES.append(label)
    print(f"  FAIL {label}  expected {exc_type.__name__}, got {type(result).__name__}")


# --------------------------------------------------------------------------
def test_calculator() -> None:
    print("calculator")
    check("addition", safe_eval("2 + 2") == 4)
    check("precedence", safe_eval("2 + 3 * 4") == 14)
    check("float division", abs(safe_eval("7 / 2") - 3.5) < 1e-12)
    check("floor division", safe_eval("7 // 2") == 3)
    check("modulo", safe_eval("7 % 3") == 1)
    check("power", safe_eval("2 ** 10") == 1024)
    check("unary minus", safe_eval("-5 + 10") == 5)
    check("parens", safe_eval("(2 + 3) * 4") == 20)
    check("constant pi", abs(safe_eval("pi") - math.pi) < 1e-12)
    check("sqrt", safe_eval("sqrt(16)") == 4)
    check("nested call", safe_eval("sqrt(abs(-25))") == 5)
    check("log", abs(safe_eval("log(e)") - 1) < 1e-12)
    check("min/max", safe_eval("max(1, 9, 3)") == 9)
    check("round with ndigits", safe_eval("round(3.14159, 2)") == 3.14)
    check("factorial", safe_eval("factorial(10)") == 3_628_800)
    check("comparison", safe_eval("3 < 4") is True)
    check("chained comparison", safe_eval("1 < 2 < 3") is True)
    check("list for sum", safe_eval("sum([1, 2, 3, 4])") == 10)
    check("scientific notation", safe_eval("1.5e3") == 1500.0)

    expect_error("empty rejected", "   ")
    expect_error("garbage rejected", "not a maths expression")
    expect_error("name rejected", "__import__")
    expect_error("attribute access rejected", "(1).__class__")
    expect_error("subscript rejected", "[1,2,3][0]")
    expect_error("unknown function", "open('/etc/passwd')")
    expect_error("unknown name", "some_undefined_var")
    expect_error("string literal rejected", "'hello'")
    expect_error("division by zero", "1 / 0")
    expect_error("huge exponent", "9 ** 99999")
    expect_error("huge factorial", "factorial(10**9)")
    expect_error("oversized expression", "1 + " * 300 + "1")
    expect_error("lambda rejected", "(lambda: 1)()")
    expect_error("comprehension rejected", "[x for x in range(3)]")


# --------------------------------------------------------------------------
def test_files() -> None:
    print("files")
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)

        check("round trip", write_file(root, "a/b.txt", "hello") .startswith("Wrote 5"))
        check("read back", read_file(root, "a/b.txt") == "hello")
        check("list shows file", "a/b.txt" in list_files(root))
        check("resolve relative", resolve_inside(root, "a/b.txt").is_absolute())

        for bad in ("../outside.txt", "..\\outside.txt", "/etc/passwd", "a/../../x"):
            try:
                resolve_inside(root, bad)
            except FileAccessError:
                check(f"blocks escape: {bad}", True)
            else:
                # Absolute paths outside the root are the ones that matter.
                resolved = resolve_inside(root, bad)
                inside = str(resolved).startswith(str(root.resolve()))
                check(f"blocks escape: {bad}", inside, f"resolved to {resolved}")

        check("missing file errors", _raises(lambda: read_file(root, "nope.txt")))
        check("directory read errors", _raises(lambda: read_file(root, "a")))

        (root / "bin.png").write_bytes(b"\x89PNG")
        check("binary refused", _raises(lambda: read_file(root, "bin.png")))

        big = "x" * 10
        write_file(root, "big.txt", big)
        check("read bounded", len(read_file(root, "big.txt", max_bytes=5)) >= 5)


def _raises(fn) -> bool:
    try:
        fn()
    except FileAccessError:
        return True
    except Exception:  # noqa: BLE001
        return False
    return False


# --------------------------------------------------------------------------
def test_wake_word() -> None:
    print("wake word")
    aliases = ["apache", "a patch", "a path"]

    matched, cmd = match_wake("Apache, what time is it", aliases)
    check("strips wake word", matched and cmd == "what time is it", f"{matched!r} {cmd!r}")

    matched, cmd = match_wake("apache what's 2 plus 2", aliases)
    check("lowercase match", matched and cmd == "what's 2 plus 2", repr(cmd))

    matched, cmd = match_wake("APACHE!", aliases)
    check("wake word alone", matched and cmd == "", repr(cmd))

    matched, cmd = match_wake("  apache: list my files", aliases)
    check("leading space + colon", matched and cmd == "list my files", repr(cmd))

    matched, _ = match_wake("what is the apache capital of gcse history", aliases)
    check("ignores wake word mid-sentence", not matched)

    matched, _ = match_wake("apathology is the study of disease", aliases)
    check("rejects apathology", not matched)

    matched, _ = match_wake("the apache helicopter is loud", aliases)
    check("ignores non-leading mention", not matched)

    matched, _ = match_wake("", aliases)
    check("ignores empty", not matched)

    matched, cmd = match_wake("a patch of grass", aliases)
    check("multi-word alias", matched and cmd == "of grass", repr(cmd))

    # Utterance that is only punctuation after the wake word.
    matched, cmd = match_wake("apache...", aliases)
    check("trailing dots consumed", matched and cmd == "", repr(cmd))


def test_speech_cleanup() -> None:
    print("speech cleanup")
    check("strips fences", "```" not in clean_for_speech("```python\nprint(1)\n```"))
    check("keeps code body", "print(1)" in clean_for_speech("```python\nprint(1)\n```"))
    check("strips headings", clean_for_speech("## Title") == "Title")
    check("strips bullets", clean_for_speech("- one\n- two") == "one\ntwo")
    check("keeps link text", clean_for_speech("[docs](http://x)") == "docs")
    check("strips bold", clean_for_speech("**bold**") == "bold")
    check("strips inline code", clean_for_speech("`x = 1`") == "x = 1")

    spoken = speakable_text("Assistant: The answer is 42.")
    check("drops attribution", spoken.startswith("The answer"), repr(spoken))

    long_reply = "This is sentence one. " * 200
    check("caps long reply", len(speakable_text(long_reply)) < 1_800)
    check("estimate sane", 0 < estimate_duration_s("hello there") < 5)
    check("estimate empty", estimate_duration_s("") == 0.0)


# --------------------------------------------------------------------------
def test_vad() -> None:
    print("vad")
    sr = 16_000
    vad = EnergyVAD(sample_rate=sr, frame_ms=30, min_utterance_ms=100)

    silence = [0.0] * (sr // 10)          # 100 ms
    check("silence produces nothing", vad.feed(silence) == [])

    loud = [0.4 * math.sin(2 * math.pi * 440 * i / sr) for i in range(sr // 2)]  # 500 ms
    check("speech alone does not close", vad.feed(loud) == [])
    check("gate marked active", vad.listening)

    # Must exceed silence_end_ms (1500 ms) to close the gate. The default is
    # deliberately long: a shorter endpoint chops a question on a thinking
    # pause, and the fragment that survives is sent off as if it were the
    # whole thought.
    trailing = [0.0] * (sr * 2)             # 2000 ms of quiet
    done = vad.feed(trailing)
    check("silence closes utterance", len(done) == 1, f"got {len(done)}")
    if done:
        check("utterance has samples", len(done[0]) > 0)
        check("duration recorded", done[0].duration_s > 0.05)
        # The clip now carries context at both ends. Google's recogniser
        # segments on silence: a window cut hard at the first and last voiced
        # sample comes back as an EMPTY transcript even when every word is
        # audible. Trailing silence is still trimmed, just not all of it --
        # here that is 0.1s lead + 0.5s voice + 0.69s closing run, minus a
        # 0.39s trim, so ~0.9s. The old all-or-nothing trim gave ~0.6s and
        # the recogniser answered with nothing.
        check("keeps context at both ends",
              0.80 < done[0].duration_s < 1.05,
              f"{done[0].duration_s:.3f}s (expect ~0.9s)")
        check("still trims the closing run", done[0].duration_s < 1.20,
              f"{done[0].duration_s:.3f}s")
    check("gate released", not vad.listening)

    # Regression: the pre-roll and the retained tail pad the clip, so the
    # minimum-duration floor must be judged on VOICED audio alone. Otherwise a
    # 250 ms mouth click wrapped in 150 ms of room tone reads as 700 ms and
    # gets sent to the recogniser, which is the noise the floor exists for.
    click_vad = EnergyVAD(sample_rate=sr, frame_ms=30,
                          min_utterance_ms=350)
    click_vad.feed([0.0] * (sr * 3 // 20))              # 150 ms quiet (pre-roll)
    click_vad.feed([0.5] * (sr // 4))                   # 250 ms burst
    rejected = click_vad.feed([0.0] * (sr * 2))         # 2000 ms quiet to close
    check("mouth click still rejected",
          len(rejected) == 0, f"got {len(rejected)}")
    # Being rejected must be observable: a real word that the gate clipped
    # lands in the same place as a mouth click, and without a count the two
    # are indistinguishable to the user.
    check("discarded utterances are counted",
          click_vad.rejected_count >= 1,
          f"count={click_vad.rejected_count}")

    # Regression: the noise floor may only learn from quiet frames. If it
    # adapted during speech, a long monologue would walk its own threshold up
    # until the gate silenced itself part-way through.
    vad_long = EnergyVAD(sample_rate=sr, frame_ms=30, min_utterance_ms=100)
    threshold_before = vad_long.threshold
    speech_15s = [0.25 * math.sin(2 * math.pi * 220 * i / sr) for i in range(sr * 15)]
    closed_early = []
    for i in range(0, len(speech_15s), 1024):
        closed_early.extend(vad_long.feed(speech_15s[i:i + 1024]))
    check("monologue does not self-silence", closed_early == [],
          f"closed {len(closed_early)} times")
    check("noise floor does not ratchet during speech",
          abs(vad_long.threshold - threshold_before) < 1e-9,
          f"{threshold_before:.6f} -> {vad_long.threshold:.6f}")
    check("monologue still gated open", vad_long.listening)

    # The max-duration cap must still split very long speech deliberately.
    capped = EnergyVAD(sample_rate=sr, frame_ms=30,
                       min_utterance_ms=100, max_utterance_s=1.0)
    pieces = []
    long_speech = [0.25 * math.sin(2 * math.pi * 220 * i / sr) for i in range(sr * 3)]
    for i in range(0, len(long_speech), 1024):
        pieces.extend(capped.feed(long_speech[i:i + 1024]))
    pieces.extend(capped.flush())
    check("max duration splits speech", len(pieces) >= 2, f"got {len(pieces)}")

    # Tiny transient below min duration must be discarded.
    tick = [0.9] * int(sr * 0.02)         # 20 ms click
    quiet = [0.0] * (sr // 3)
    check("short transient discarded", vad.feed(tick + quiet) == [])

    # Feed a long stream in odd-sized chunks; segmentation must not care.
    vad2 = EnergyVAD(sample_rate=sr, frame_ms=30, min_utterance_ms=100)
    stream = silence + loud + trailing + silence
    total = []
    for i in range(0, len(stream), 977):
        total.extend(vad2.feed(stream[i:i + 977]))
    check("chunk-size invariant", len(total) == 1, f"got {len(total)}")

    vad2.reset()
    check("reset clears state", not vad2.listening and vad2.feed(silence) == [])


def _gate_frame(rms: float, samples: int = 480) -> list[float]:
    """One frame whose RMS is exactly *rms*. The VAD measures nothing else."""
    return [rms, -rms] * (samples // 2) + [rms] * (samples % 2)


def _gate_track(parts: list[tuple[float, float]]) -> list[float]:
    """Build a signal from ``(seconds, rms)`` spans, cut into 30 ms frames."""
    signal: list[float] = []
    for seconds, rms in parts:
        total = int(round(seconds * 16_000))
        for start in range(0, total, 480):
            signal.extend(_gate_frame(rms, min(480, total - start)))
    return signal


def _gate_run(signal: list[float], multiplier: float):
    """Feed a signal through the production VAD configuration."""
    vad = EnergyVAD(
        sample_rate=16_000,
        frame_ms=30,
        min_energy=settings.vad_min_energy,
        multiplier=multiplier,
        silence_end_ms=settings.silence_end_ms,
        min_utterance_ms=settings.utterance_min_ms,
        max_utterance_s=settings.utterance_max_s,
        pre_roll_ms=settings.vad_pre_roll_ms,
        keep_tail_ms=settings.vad_keep_tail_ms,
    )
    utterances: list = []
    for i in range(0, len(signal), 480):
        utterances.extend(vad.feed(signal[i:i + 480]))
    utterances.extend(vad.flush())
    return utterances, vad


def test_gate_multiplier() -> None:
    """A spoken question must survive the gate without the room getting in."""
    print("gate multiplier")

    # Rebuilt from a live panel: room 0.0073, gate 0.0283 at the old
    # multiplier of 3.5 -- 3.9x ambient, which an ordinary voice does not
    # sustain across a whole sentence. Speech only partly cleared it, the
    # question fragmented, and most of it was discarded as too short. These
    # spans are that room in synthetic form; only the RMS matters to a VAD.
    room = 0.008
    spans = [
        (2.0, room),           # lead-in while the noise floor settles
        (0.6, 0.030),          # "apache", spoken with emphasis
        (0.15, 0.009),         # natural inter-word gap
        (2.25, 0.019),         # the question, conversational level
        (0.15, 0.009),
        (0.85, 0.021),         # the last clause
        (3.0, room),           # trailing room so the endpoint can close
    ]
    spoken = sum(sec for sec, rms in spans if rms > 0.012)
    signal = _gate_track(spans)

    utterances, vad = _gate_run(signal, settings.vad_multiplier)
    kept = sum(u.duration_s for u in utterances)
    check("the configured gate keeps the question in one piece",
          len(utterances) == 1, f"got {len(utterances)}")
    check("nothing is discarded", vad.rejected_count == 0,
          f"rejected={vad.rejected_count}")
    check("the whole question is retained", kept >= spoken * 0.9,
          f"kept {kept:.2f}s of {spoken:.2f}s spoken")

    # What this replaced, kept in the suite so a regression is visible as the
    # exact loss the user reported rather than as an abstract threshold.
    old, _old_vad = _gate_run(signal, 3.5)
    check("the old multiplier demonstrably lost the question",
          sum(u.duration_s for u in old) < spoken * 0.6,
          f"kept {sum(u.duration_s for u in old):.2f}s of {spoken:.2f}s")

    # Lowering the gate must not have turned it into a hole: the room alone
    # still has to stay quiet, and must not even be counted as a discard.
    quiet, quiet_vad = _gate_run(_gate_track([(8.0, room)]),
                                 settings.vad_multiplier)
    check("pure room tone trips nothing", len(quiet) == 0,
          f"got {len(quiet)}")
    check("pure room tone is not counted as a discard",
          quiet_vad.rejected_count == 0, f"rejected={quiet_vad.rejected_count}")


def test_sandbox() -> None:
    print("sandbox")
    with tempfile.TemporaryDirectory() as tmp:
        out = run_python("print('hello from sandbox')", cwd=tmp)
        check("stdout captured", "hello from sandbox" in out, out)

        out = run_python("raise ValueError('boom')", cwd=tmp)
        check("stderr captured", "ValueError" in out and "[stderr]" in out, out)

        out = run_python("", cwd=tmp)
        check("empty rejected", out.startswith("Error:"), out)

        out = run_python("print('x')", cwd=tmp, timeout=1)
        check("success exit has no code tag", "[exit code" not in out, out)

        out = run_python(
            "import time\ntime.sleep(5)", cwd=tmp, timeout=1
        )
        check("timeout enforced", "timed out" in out, out)

        out = run_python("print('y' * 50000)", cwd=tmp, max_output=100)
        check("output truncated", "truncated" in out and len(out) < 400, out[:120])

        out = run_python("import sys; print(sys.argv)", cwd=tmp)
        check("isolated: no args leaked", "[]" in out or "['-c']" in out, out)

        out = run_python("print(1)", cwd=str(Path(tmp) / "missing"))
        check("missing cwd handled", out.startswith("Error:"), out)

        out = run_python("import os; print('APACHE_X' in os.environ)", cwd=tmp)
        check("env passed through is harmless", "True" in out or "False" in out, out)


# --------------------------------------------------------------------------
def main() -> int:
    for suite in (
        test_calculator,
        test_files,
        test_wake_word,
        test_speech_cleanup,
        test_vad,
        test_gate_multiplier,
        test_sandbox,
    ):
        try:
            suite()
        except Exception as exc:  # noqa: BLE001
            import traceback

            traceback.print_exc()
            FAILURES.append(f"{suite.__name__} crashed: {exc}")
        print()

    if FAILURES:
        print(f"{len(FAILURES)} FAILURE(S):")
        for name in FAILURES:
            print(f"  - {name}")
        return 1
    print("All pure-logic tests passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
