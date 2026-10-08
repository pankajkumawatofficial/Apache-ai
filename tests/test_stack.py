"""Tests for the assembled application -- the layer the pure suites cannot reach.

Run with:  python -m tests.test_stack

:mod:`tests.test_pure` and :mod:`tests.test_rag` deliberately import nothing
from LangChain, Gradio, NumPy or the audio stack, so they stay runnable on a
machine where that native code is blocked. The cost is that wiring bugs -- a
helper deleted from one module but still imported by another, a component
constructed with a bad argument -- are invisible to them.

That gap is not hypothetical: these tests exist because a dead-code cleanup
removed ``agent.supported_tool_names`` while ``ui._tools_status_text`` still
imported it. The broad ``except`` around that import meant nothing crashed;
the UI just silently rendered "Tools unavailable". Both existing suites
passed throughout.

If the ML stack is unavailable the suite SKIPS rather than fails, so it stays
runnable everywhere.
"""

from __future__ import annotations

import sys
import tempfile
import threading
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

FAILURES: list[str] = []
SKIPPED: list[str] = []


def check(label: str, condition: bool, detail: str = "") -> None:
    if condition:
        print(f"  ok   {label}")
    else:
        FAILURES.append(label)
        print(f"  FAIL {label}  {detail}")


def _short(value: object, limit: int = 200) -> str:
    return " ".join(str(value).split())[:limit]


#: The toolset the agent is contractually offered.
EXPECTED_TOOLS = {
    "calculator",
    "run_python_code",
    "list_files",
    "read_file_tool",
    "write_file_tool",
    "search_documents",
    "current_datetime",
    "open_url_tool",
    "open_app_tool",
    "play_media_tool",
    "open_file_tool",
    "web_search",
}

#: Components without which the UI is not the app described in the README.
EXPECTED_COMPONENTS = {"chatbot", "timer", "uploadbutton", "audio"}


def _load_stack():
    """Import the UI-facing stack, or explain why it cannot be loaded."""
    try:
        from app.core import get_assistant
        from app.tools.registry import build_tools
        from app import ui
    except Exception as exc:  # noqa: BLE001 - blocked or missing natives
        return None, None, None, _short(exc)
    return get_assistant, build_tools, ui, ""


def test_assistant() -> None:
    get_assistant, _, _, _ = _load_stack()
    assistant = get_assistant()
    check("assistant singleton constructs", assistant is not None)
    check("assistant exposes a model name",
          bool(getattr(assistant, "model", "")), repr(getattr(assistant, "model", None)))
    check("assistant workspace is under the project",
          str(getattr(assistant, "workspace", "")).endswith("workspace"),
          repr(getattr(assistant, "workspace", None)))

    snap = assistant.snapshot()
    check("snapshot is a 4-tuple", isinstance(snap, tuple) and len(snap) == 4,
          f"len={len(snap) if isinstance(snap, tuple) else type(snap)}")
    messages, _activity, _audio, status = snap
    check("snapshot starts with a welcome message",
          isinstance(messages, list) and len(messages) >= 1,
          f"messages={len(messages) if isinstance(messages, list) else '?'}")
    check("snapshot status is a string", isinstance(status, str), repr(status))


def test_tool_registry() -> None:
    get_assistant, build_tools, _, _ = _load_stack()
    assistant = get_assistant()
    tools = build_tools(assistant.store, assistant.workspace, assistant.settings)
    check("tools construct", bool(tools), f"count={len(tools)}")

    names = {getattr(tool, "name", "") for tool in tools}
    missing = EXPECTED_TOOLS - names
    check("every contract tool registered", not missing,
          f"missing={sorted(missing)} extra={sorted(names - EXPECTED_TOOLS)}")

    # A tool without a description will not be called by the model.
    undescribed = [getattr(t, "name", "?") for t in tools
                   if not getattr(t, "description", None)]
    check("every tool carries a description", not undescribed,
          f"undescribed={undescribed}")

    # The bare-name recovery fills a tool's one argument from the user's own
    # request, which means reading the schema the model was actually shown.
    # BaseTool.args is only the properties mapping and leaves the defaults
    # out, so reading it instead makes every parameter look mandatory.
    from app.agent import _sentence_arguments

    player = next((t for t in tools if getattr(t, "name", "") == "play_media_tool"),
                  None)
    check("the player tool is built", player is not None, "")
    if player is not None:
        check("a real tool's one argument can be filled in",
              _sentence_arguments(player, "a song") == {"request": "a song"},
              repr(_sentence_arguments(player, "a song")))
    folder = next((t for t in tools if getattr(t, "name", "") == "list_files"), None)
    if folder is not None:
        check("a defaulted argument is not demanded",
              _sentence_arguments(folder, "a song") == {},
              repr(_sentence_arguments(folder, "a song")))


def test_text_tool_call() -> None:
    """A tool call written out as prose is still a tool call.

    qwen3 formats some calls itself -- the tool name, then its arguments as
    JSON -- and the runtime has already decided by then that it is looking
    at prose. Left alone, Apache reads the whole thing aloud. It also leaves
    the arguments off entirely: "play a song", "put on a song" and "start
    some music" each answered with the bare word "play_media".
    """
    print("text tool calls")
    from app.agent import _run_text_tool_call

    ran: list[dict] = []

    class _Tool:
        name = "play_media_tool"
        # BaseTool.args is the properties mapping, not the whole schema --
        # the shape the real tools on this list describe themselves with.
        args = {"request": {"title": "Request", "type": "string"}}

        def invoke(self, args):
            ran.append(args)
            return "Playing Bohemian Rhapsody."

    tool = _Tool()

    name, args, result = _run_text_tool_call(
        'play_media_tool\n{"request": "bohemian rhapsody"}', [tool])
    check("a name and JSON on separate lines is run",
          name == "play_media_tool"
          and args == {"request": "bohemian rhapsody"}
          and "Playing" in result, f"{name=} {args=} {result=}")
    check("and the tool really ran",
          ran == [{"request": "bohemian rhapsody"}], repr(ran))

    name, _args, _result = _run_text_tool_call(
        'play_media_tool {"request": "despacito"}', [tool])
    check("the same on one line", name == "play_media_tool", repr(name))

    check("the prompt's own spelling of the tool reaches it",
          _run_text_tool_call(
              'play_media\n{"request": "despacito"}', [tool]) is not None, "")

    ran.clear()
    name, args, _result = _run_text_tool_call("play_media", [tool], "a song")
    check("a bare tool name runs with the request as its argument",
          name == "play_media_tool"
          and args == {"request": "a song"}
          and ran == [{"request": "a song"}], f"{name=} {args=} {ran=}")

    check("but not with nothing to pass it",
          _run_text_tool_call("play_media", [tool]) is None, "")
    check("and not when the name is not one of the tools",
          _run_text_tool_call("wibble", [tool], "a song") is None, "")

    class _TwoArgs:
        name = "pair_tool"
        args = {"a": {"type": "string"}, "b": {"type": "string"}}

        def invoke(self, args):  # pragma: no cover - must never be reached
            raise AssertionError("two arguments cannot be guessed")

    check("a tool needing two arguments is left to the model",
          _run_text_tool_call("pair", [tool, _TwoArgs()], "a song") is None, "")

    check("an ordinary sentence is left alone",
          _run_text_tool_call("I opened Spotify for you.", [tool], "play a song")
          is None, "")
    check("broken JSON is left alone",
          _run_text_tool_call("play_media_tool\n{oops}", [tool]) is None, "")
    check("a tool that is not registered is left alone",
          _run_text_tool_call('nope_tool\n{"a": 1}', [tool]) is None, "")
    check("a bare JSON object is not a call",
          _run_text_tool_call('{"request": "x"}', [tool]) is None, "")

    class _Boom:
        name = "calculator"
        args = {"expression": {"type": "string"}}

        def invoke(self, args):
            raise ValueError("bad expression")

    _, _, result = _run_text_tool_call(
        'calculator\n{"expression": "1+"}', [_Boom()])
    check("a tool that raises is reported, not raised",
          result.startswith("Error:"), repr(result))


def test_rag_end_to_end() -> None:
    get_assistant, _, _, _ = _load_stack()
    assistant = get_assistant()

    with tempfile.TemporaryDirectory() as tmp:
        doc = Path(tmp) / "apache_notes.txt"
        doc.write_text(
            "Apache is a wake word based AI assistant. "
            "It uses Ollama for the model and Gradio for the interface. "
            "The code sandbox runs python in isolated mode with a timeout.",
            encoding="utf-8",
        )
        summary = assistant.store.ingest([doc])
        check("store accepts an uploaded document", bool(summary), _short(summary))
        check("store reports a non-empty chunk count",
              assistant.store.chunk_count > 0,
              f"chunks={assistant.store.chunk_count}")

        hits = assistant.store.search("what wake word does it use")
        check("search returns hits", bool(hits), f"hits={len(hits)}")
        if hits:
            check("search ranks the ingested document first",
                  hits[0].source == "apache_notes.txt",
                  f"top={hits[0].source}")
            check("hit carries a numeric score",
                  isinstance(hits[0].score, float), repr(getattr(hits[0], "score", None)))


def test_gradio_ui() -> None:
    _, _, ui, _ = _load_stack()
    get_assistant, _, _, _ = _load_stack()
    assistant = get_assistant()

    demo = ui.build_demo(assistant)
    check("build_demo returns Blocks", type(demo).__name__ == "Blocks",
          type(demo).__name__)
    check("blocks carry the app title", getattr(demo, "title", None) == "Apache",
          repr(getattr(demo, "title", None)))

    blocks = getattr(demo, "blocks", None) or {}
    check("blocks contain components", len(blocks) > 0, f"count={len(blocks)}")

    types = {type(component).__name__ for component in blocks.values()}
    # Match on class names rather than Gradio's lowercase type strings so a
    # rename in either direction is still caught.
    joined = " ".join(types).lower()
    missing = [name for name in EXPECTED_COMPONENTS if name not in joined]
    check("chatbot, timer, upload button and audio are all present", not missing,
          f"missing={missing}")

    handlers = getattr(demo, "fns", None)
    check("event handlers are registered",
          handlers is not None and len(handlers) > 0,
          f"count={len(handlers) if handlers is not None else 'n/a'}")


def test_status_helpers() -> None:
    """Regression guard for the deleted-`supported_tool_names` bug."""
    _, _, ui, _ = _load_stack()
    get_assistant, _, _, _ = _load_stack()
    assistant = get_assistant()

    for name in ("_settings_status_text", "_tools_status_text", "_docs_status_text"):
        helper = getattr(ui, name, None)
        check(f"ui exposes {name}", callable(helper), type(helper).__name__)
        if not callable(helper):
            continue
        try:
            text = helper(assistant)
        except Exception as exc:  # noqa: BLE001
            check(f"{name} runs", False, _short(exc))
            continue
        check(f"{name} runs", isinstance(text, str), type(text).__name__)

    try:
        tools_text = ui._tools_status_text(assistant)
    except Exception as exc:  # noqa: BLE001
        tools_text = f"ERROR {_short(exc)}"

    check("tools status lists the registry by name",
          "unavailable" not in tools_text.lower() and "calculator" in tools_text,
          _short(tools_text, 240))


def test_snapshot_rendering() -> None:
    _, _, ui, _ = _load_stack()
    get_assistant, _, _, _ = _load_stack()
    assistant = get_assistant()

    rendered = ui._render(assistant.snapshot())
    check("_render returns four values",
          isinstance(rendered, tuple) and len(rendered) == 4,
          f"len={len(rendered) if isinstance(rendered, tuple) else type(rendered)}")


def test_agent_turn_degrades_cleanly() -> None:
    """A turn must terminate readably with or without a model available."""
    get_assistant, _, _, _ = _load_stack()
    assistant = get_assistant()

    before = len(assistant.snapshot()[0])
    last = None
    raised = None
    try:
        for snap in assistant.submit("hello"):
            last = snap
    except Exception as exc:  # noqa: BLE001
        raised = exc

    check("agent turn does not raise", raised is None, _short(raised) if raised else "")
    check("agent turn terminates with a snapshot", last is not None)

    if last is not None:
        messages, _activity, _audio, status = last
        check("turn appends messages", len(messages) > before,
              f"{before} -> {len(messages)}")
        check("turn leaves a non-empty reply",
              bool(messages and messages[-1].get("content", "").strip()),
              _short(messages[-1].get("content", "") if messages else ""))
        check("turn settles on a terminal status",
              isinstance(status, str) and bool(status.strip()), repr(status))


def test_stop_path() -> None:
    """A turn in flight must be abandonable, and must release the busy flag.

    The runner is stubbed so this never touches Ollama: the contract under
    test is Apache's own turn machinery, and making it depend on a live model
    would turn a fast deterministic check into a slow flaky one.
    """
    get_assistant, _, _, _ = _load_stack()
    from app.agent import AgentEvent

    assistant = get_assistant()

    check("stop is a no-op when idle", assistant.request_stop() is False)

    class _StubRunner:
        """Yields slowly enough that a stop can land mid-stream."""

        def stream(self, *_args, **_kwargs):
            for index in range(200):
                time.sleep(0.02)
                yield AgentEvent("text", f"chunk {index}")

        def reset_session(self, _session_id):  # pragma: no cover - unused here
            return None

    real_runner = assistant.runner
    assistant.runner = _StubRunner()
    try:
        generator = assistant.submit("please ignore this, it is a test")
        next(generator)  # reaches the first yield: the turn is now live

        check("a live turn claims the busy flag", assistant.busy)
        check("request_stop reports there was something to stop",
              assistant.request_stop() is True)

        last = None
        for snap in generator:
            last = snap

        check("stopped turn yields a final snapshot", last is not None)
        if last is not None:
            _messages, _activity, _audio, status = last
            check("stopped turn reports Stopped.",
                  status == "Stopped.", repr(status))
        check("stopped turn releases the busy flag", not assistant.busy)
        check("stop flag is cleared for the next turn",
              assistant.request_stop() is False)
    finally:
        assistant.runner = real_runner
        assistant.reset()


def test_listener_diagnostics() -> None:
    """Capture health must be visible, not silently swallowed.

    The original listener discarded PortAudio overflow flags, buffer drops and
    ignored utterances without recording them anywhere, which made "the
    microphone is not working" impossible to diagnose from the UI.
    """
    _, _, ui, _ = _load_stack()
    from app.voice.listener import WakeListener

    heard: list[str] = []
    listener = WakeListener(on_command=heard.append)
    info = listener.status()

    required = (
        "state", "message", "transcript", "sample_rate", "device", "muted",
        "wake_word", "level", "threshold", "dropped_blocks", "xruns",
        "ignored", "queued",
    )
    missing = [key for key in required if key not in info]
    check("listener status exposes capture health", not missing,
          f"missing={missing}")
    check("listener reports a threshold for the meter",
          isinstance(info["threshold"], float))
    check("listener starts stopped", info["state"] == "stopped")
    check("listener does not report itself running", listener.running is False)

    text = ui._voice_status_text(listener)
    check("_voice_status_text renders an idle listener",
          isinstance(text, str) and "Microphone is off" in text,
          _short(text, 160))
    check("no device line is shown before the mic is opened",
          "Hz" not in text, _short(text, 160))

    # Mute must not need a stream to be safe -- it runs from the TTS callback.
    listener.mute_for(0)
    listener.mute_for(-5)
    listener.unmute()
    check("mute/unmute are safe without a microphone", True)

    check("stop is safe on a listener that never started",
          listener.stop() == "Microphone is off.",
          repr(_short(listener.stop(), 80)))


def test_capture_feeds_vad() -> None:
    """The capture loop must actually hand audio to the VAD.

    Regression guard. ``_note_level`` was decorated ``@staticmethod`` while its
    body referenced ``self``, so every call raised NameError. The capture loop
    wrapped the batch in ``except Exception: continue``, which swallowed it --
    audio was consumed and discarded, the VAD never saw a frame, and the gate
    sat unmoved at its initial 0.042 forever. Listening simply did nothing,
    with no error surfaced anywhere.

    Both assertions are needed: either one alone would have passed on the
    broken build.
    """
    from app.voice.listener import WakeListener
    from app.voice.vad import EnergyVAD

    listener = WakeListener(on_command=lambda _text: None)

    raised = None
    try:
        listener._note_level([0.5, -0.5, 0.5, -0.5])
    except Exception as exc:  # noqa: BLE001
        raised = exc
    check("_note_level does not raise when called as a method",
          raised is None, repr(raised))
    check("level meter reads the block it was given",
          listener.status()["level"] > 0.0,
          f"level={listener.status()['level']}")

    # Drive the real loop with silence: the noise floor adapts downward, so
    # the gate must fall below its starting value. If no frame reaches the VAD
    # the threshold never moves -- exactly the reported symptom.
    vad = EnergyVAD(sample_rate=16_000, frame_ms=30,
                    min_energy=0.012, multiplier=3.5)
    gate_before = vad.threshold

    with listener._lock:
        listener._running = True
        listener._vad = vad
        listener._pending.clear()
        for _ in range(40):
            listener._pending.append([0.0] * 480)

    worker = threading.Thread(target=listener._capture_loop, daemon=True)
    worker.start()
    deadline = time.time() + 3.0
    while time.time() < deadline and vad.threshold >= gate_before:
        time.sleep(0.05)
    with listener._lock:
        listener._running = False
    worker.join(timeout=2.0)

    check("capture loop delivers audio to the VAD", vad.threshold < gate_before,
          f"gate {gate_before:.5f} -> {vad.threshold:.5f} (never moved)")
    check("capture loop records no block errors",
          listener.status()["capture_errors"] == 0,
          f"errors={listener.status()['capture_errors']}")
    # The meter decays rather than snapping to zero, so loud audio is visible
    # for a moment after it stops. It must fall, not sit pinned at 0.5.
    check("level decays once the loud audio stops",
          listener.status()["level"] < 0.5,
          f"level={listener.status()['level']}")


def test_audio_queue() -> None:
    """Clips queue instead of overwriting each other.

    This is the guard for "Apache is not speaking all responses": replies and
    announcements share one slot, and whichever finished synthesising last
    won it. Three clips completing inside a single 0.6 s tick used to leave
    two of them never played.
    """
    get_assistant, _, _, _ = _load_stack()
    assistant = get_assistant()

    saved_queue = list(assistant._audio_queue)
    saved_free = assistant._audio_free_at
    saved_audio = assistant._audio

    def release() -> None:
        """Pretend whatever was playing has finished."""
        with assistant._lock:
            assistant._audio_free_at = time.monotonic() - 0.01

    try:
        with assistant._lock:
            assistant._audio_queue = [("a.mp3", 3.0), ("b.mp3", 3.0),
                                      ("c.mp3", 3.0)]
            assistant._audio_free_at = 0.0
            assistant._audio = None

        first = assistant.next_audio()
        check("hands out the first clip", first == "a.mp3", repr(first))
        check("one clip leaves the queue", assistant.pending_audio() == 2,
              str(assistant.pending_audio()))
        check("holds the rest while the first plays", assistant.next_audio()
              is None, "released early")
        check("and has not lost them", assistant.pending_audio() == 2,
              str(assistant.pending_audio()))

        release()
        check("releases the next once the first could have ended",
              assistant.next_audio() == "b.mp3", "wrong clip")
        release()
        check("then the last", assistant.next_audio() == "c.mp3", "wrong clip")
        check("nothing left to give", assistant.next_audio() is None,
              "expected None")
        check("queue ended empty", assistant.pending_audio() == 0,
              str(assistant.pending_audio()))

        # The path from a finished reply to the queue, without the network.
        import app.voice.tts as tts_mod

        def fake_synthesize(text, out_path, **kwargs):
            path = Path(out_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"ID3fake")
            return path

        real = tts_mod.synthesize
        tts_mod.synthesize = fake_synthesize
        try:
            with assistant._lock:
                assistant._audio_queue = []
                assistant._audio_free_at = 0.0
                assistant._audio = None
            assistant._synthesize("first reply")
            assistant._synthesize("second reply")
            check("two finished replies both queue",
                  assistant.pending_audio() == 2, str(assistant.pending_audio()))
            check("neither is assigned before it plays",
                  assistant.snapshot()[2] in (None, ""), repr(assistant.snapshot()[2]))
        finally:
            tts_mod.synthesize = real
    finally:
        with assistant._lock:
            assistant._audio_queue = saved_queue
            assistant._audio_free_at = saved_free
            assistant._audio = saved_audio


def test_offline_speech() -> None:
    """A reply must come out of the local engine as a playable file.

    This is the other half of "Apache is not speaking all responses": the
    engine has to write something the browser can decode, under the name that
    says what it actually is.
    """
    get_assistant, _, _, _ = _load_stack()
    get_assistant()

    import wave

    from app.voice import tts

    if not tts.settings.offline:
        print("  skip: APACHE_OFFLINE=0 selects the online engine")
        return
    if not Path(tts.settings.piper_voice).exists():
        # A per-machine download: report it, do not fail the suite on it.
        print(f"  skip: no Piper voice at {tts.settings.piper_voice}")
        return

    out = Path(tempfile.gettempdir()) / "apache-test-speech.wav"
    out.unlink(missing_ok=True)
    try:
        path = tts.synthesize("Testing one two three.", out)
        check("produced a file",
              path.exists() and path.stat().st_size > 0,
              f"{path} ({path.stat().st_size if path.exists() else 0} B)")
        check("named for the format it really is", path.suffix == ".wav",
              path.suffix)
        with wave.open(str(path), "rb") as handle:
            seconds = handle.getnframes() / handle.getframerate()
            rate = handle.getframerate()
        check("holds audible-length audio", 0.2 < seconds < 30.0,
              f"{seconds:.2f}s at {rate} Hz")
    finally:
        out.unlink(missing_ok=True)

    # A voice that is not there must be reported, never raised as a crash --
    # and it must fail before loading, or this would cost another model load.
    saved_path = tts.settings.piper_voice
    saved_cache = dict(tts._PIPER_VOICES)
    try:
        tts.settings.piper_voice = str(
            Path(tempfile.gettempdir()) / "apache-definitely-missing.onnx"
        )
        tts._PIPER_VOICES.clear()
        try:
            tts.synthesize("hi", out)
            outcome = "NO ERROR"
        except tts.TTSUnavailable as exc:
            outcome = str(exc)
        check("a missing voice is reported, not fatal",
              "no offline voice" in outcome, outcome)
    finally:
        tts.settings.piper_voice = saved_path
        tts._PIPER_VOICES.clear()
        tts._PIPER_VOICES.update(saved_cache)
        out.unlink(missing_ok=True)


def test_thinking_animation() -> None:
    """A bubble the model has not answered yet must visibly move."""
    _g, _b, ui, _ = _load_stack()
    get_assistant, _, _, _ = _load_stack()
    assistant = get_assistant()

    pending = [{"role": "assistant", "content": ""}]
    answered = [{"role": "assistant", "content": "Paris."}]

    shown = ui._with_thinking(pending, True)
    check("pending bubble is animated",
          "apache-thinking" in shown[0]["content"], repr(shown[0]["content"]))
    check("the shared history is not mutated",
          pending[0]["content"] == "", repr(pending[0]["content"]))
    check("an answered bubble is left alone",
          ui._with_thinking(answered, True) == answered)
    check("nothing animates while idle",
          ui._with_thinking(pending, False) == pending)
    check("an empty transcript is left alone",
          ui._with_thinking([], True) == [])

    first = ui._with_thinking(pending, True)[0]["content"]
    time.sleep(0.75)  # longer than one frame at 1.7 frames a second
    second = ui._with_thinking(pending, True)[0]["content"]
    check("the frame advances over time", first != second,
          f"{first!r} vs {second!r}")
    check("the stylesheet carries the keyframes", "apache-think" in ui._CSS)

    # Defining the helper is not enough: the ticker has to apply it, or the
    # bubble stays blank no matter what _with_thinking does.
    demo = ui.build_demo(assistant)
    fns = getattr(demo, "fns", None)
    by_name = {}
    for entry in (fns.values() if isinstance(fns, dict) else fns or []):
        fn = getattr(entry, "fn", None)
        if fn is not None:
            by_name[getattr(fn, "__name__", "?")] = fn
    check("ticker handler is registered", "on_tick" in by_name,
          sorted(by_name)[:8])

    saved = list(assistant._history)
    with assistant._lock:
        assistant._history = [{"role": "assistant", "content": ""}]
        assistant._busy = True
    try:
        rendered = by_name["on_tick"]()[0]
        last = rendered[-1] if rendered else {}
        check("on_tick animates the pending bubble",
              "apache-thinking" in str(last.get("content", "")),
              str(last)[:110])
    finally:
        with assistant._lock:
            assistant._busy = False
            assistant._history = saved


def test_jarvis_theme() -> None:
    """The HUD is wired into the app, not merely written down.

    A stylesheet nobody passes to ``launch()`` looks finished and does
    nothing, which is the failure this exists to catch.
    """
    get_assistant, _, _, _ = _load_stack()
    get_assistant()

    import gradio as gr

    from app import ui

    demo = ui.build_demo(get_assistant())
    # str(demo) never renders elem_id; the config Gradio serves is the thing
    # that actually reaches the browser, so that is what to assert against.
    import json

    config = json.dumps(demo.get_config_file(), default=str)
    check("banner is styleable by id", "apache-header" in config,
          "no elem_id on the header")
    check("conversation is styleable by id", "apache-chat" in config,
          "no elem_id on the chatbot")

    check("the stylesheet carries the keyframes", "apache-think" in ui._CSS,
          "thinking animation missing")
    check("banner frame is styled", "#apache-header" in ui._CSS,
          "no banner rules")
    check("console-style bubbles are styled", "#apache-chat .bubble" in ui._CSS,
          "no bubble rules")
    # Gradio 6 ignores Blocks(theme=...); it has to reach launch() to exist.
    check("theme is applied by the launcher",
          "theme=_theme()" in Path(ui.__file__).read_text(encoding="utf-8"),
          "theme not passed to launch()")

    ours = ui._theme()._get_computed_value("primary_500")
    stock = gr.themes.Base()._get_computed_value("primary_500")
    check("our hue differs from Gradio's default", ours != stock,
          f"ours={ours} default={stock}")


def test_audio_delivery() -> None:
    """A reply clip must reach the browser on a machine with no ffmpeg.

    Gradio re-encodes an audio path only when its suffix disagrees with the
    component's ``format`` (audio.py:320), and that re-encode shells out to
    ffmpeg through pydub. This machine has no ffmpeg, so a component still
    claiming ``mp3`` while Piper writes ``.wav`` made every single reply die
    as a ComponentProcessingError inside on_tick -- Apache looked perfectly
    healthy and simply never spoke.
    """
    get_assistant, _, _, _ = _load_stack()
    get_assistant()

    import json
    import wave

    import gradio as gr

    from app import ui
    from app.voice.tts import clip_extension

    demo = ui.build_demo(get_assistant())
    claimed = [
        str((c.get("props") or {}).get("format"))
        for c in demo.get_config_file().get("components", [])
        if str(c.get("type", "")).lower() == "audio"
    ]
    check("the reply player claims the engine's real format",
          claimed and claimed[0] == clip_extension().lstrip("."),
          f"claimed={claimed} engine={clip_extension()}")

    # A minimal but genuinely decodable WAV, so nothing here loads Piper.
    path = Path(tempfile.gettempdir()) / "apache-delivery.wav"
    path.unlink(missing_ok=True)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(16_000)
        handle.writeframes(b"\x00\x00" * 1_600)
    try:
        matched = gr.Audio(format=clip_extension().lstrip("."), type="filepath")
        try:
            served = matched.postprocess(str(path))
            check("a matching clip is served as-is", served is not None,
                  repr(served))
        except Exception as exc:  # noqa: BLE001 - this is the failure
            check("a matching clip is served as-is", False,
                  f"{type(exc).__name__}: {exc}")

        # Characterise the bug so the reason the formats must agree is on the
        # record. Skipped if someone installs ffmpeg, since then it would
        # succeed by conversion rather than by not converting at all.
        import gradio.processing_utils as pu

        if not pu.ffmpeg_installed():
            try:
                gr.Audio(format="mp3", type="filepath").postprocess(str(path))
                outcome = "converted"
            except Exception as exc:  # noqa: BLE001
                outcome = type(exc).__name__
            check("a mismatched format is exactly what needs ffmpeg",
                  outcome != "converted", outcome)
    finally:
        path.unlink(missing_ok=True)


def test_voice_defaults() -> None:
    """Apache should be ready to talk the moment it comes up."""
    get_assistant, _, _, _ = _load_stack()
    assistant = get_assistant()

    check("greets with the line asked for",
          assistant.settings.greeting == "Hello Boss, what will we do today.",
          repr(assistant.settings.greeting))
    check("listens on launch without a click", assistant.settings.mic_autostart,
          repr(assistant.settings.mic_autostart))
    check("answers without the wake word by default",
          not assistant.settings.wake_required,
          repr(assistant.settings.wake_required))
    check("speaks replies by default", assistant.speak_replies,
          repr(assistant.speak_replies))

    from app.voice import tts

    if not tts.settings.offline:
        print("  skip: APACHE_OFFLINE=0 selects the online engine")
        return
    voices = tts.available_voices()
    if not voices:
        print("  skip: no Piper voices downloaded")
        return

    model = tts.resolve_model(assistant.voice)
    check("the configured voice is on disk", model.exists(), str(model))
    check("downloaded voices are offered in the dropdown",
          model.stem in voices, f"{model.stem} not in {voices}")
    check("an offered name resolves to a real file",
          tts.resolve_model(voices[-1]).exists(), voices[-1])
    check("a name typed in capitals still resolves",
          tts.resolve_model(voices[-1].upper()).exists(), voices[-1].upper())
    check("so does the filename form",
          tts.resolve_model(f"{voices[-1]}.onnx").exists(), voices[-1])
    check("an unknown voice falls back instead of failing",
          tts.resolve_model("en_US-not-a-real-voice").exists(),
          "missing model")


def test_background_tab() -> None:
    """A reply must still reach the speaker with another tab in front.

    Gradio's Timer only dispatches when ``document.visibilityState`` reads
    "visible", so going to another tab used to stop the queue being drained
    entirely -- Apache kept thinking and kept answering, and none of it came
    out of the speaker.
    """
    _g, _b, ui, _ = _load_stack()

    js = ui._page_js()
    check("the ticker is told the page is visible",
          "visibilityState" in js and 'return "visible"' in js, js[:120])
    check("the hidden flag is shadowed too", '"hidden"' in js, "missing")
    check("a keep-alive keeps Chrome from throttling it",
          "keepAlive" in js and "data:audio/wav" in js, "missing")
    check("autoplay is unlocked on the first gesture",
          "pointerdown" in js and "unlock" in js, "missing")
    check("a reply the browser rejected is retried",
          "retryBlockedReplies" in js, "missing")
    # The player that speaks is a detached element Gradio's waveform owns, so
    # no scan of the document can ever see it -- the refusal has to be caught
    # on play() itself, which is the one call every player has to make.
    check("the refusal is caught where it happens",
          "HTMLMediaElement.prototype.play" in js, "missing")
    check("the scan that could never see the player is gone",
          "currentSrc" not in js, "still scanning the DOM")
    check("clips are said in order, one at a time",
          "refused.indexOf" in js and "if (!el.paused) return" in js,
          "missing")
    check("the page says why nothing is audible",
          "apache-sound-prompt" in js, "missing")
    check("a Web Audio context is resumed as well",
          "Reflect.construct" in js and ".resume()" in js, "missing")
    # The % formatting that embeds the WAV must have run: a leftover
    # placeholder would inject the literal "%s" into the page.
    check("no unsubstituted placeholder left", "%s" not in js, "found one")
    check("it is an IIFE that cannot leak globals", js.lstrip().startswith("(function"),
          js.lstrip()[:40])

    import base64

    wav = ui._keepalive_wav()
    header = base64.b64decode(wav.split(",", 1)[1])[:4]
    check("the keep-alive decodes to a real WAV", header == b"RIFF", header)


def test_errors_are_spoken() -> None:
    """A failed turn reaches the speaker like any other reply.

    Errors used to be written down and never said out loud, which from the
    user's side of the screen is indistinguishable from Apache deciding not
    to talk -- the one thing they cannot diagnose.
    """
    get_assistant, _, _, _ = _load_stack()
    assistant = get_assistant()

    from app.agent import AgentEvent

    # Self-contained: whatever a previous test left behind must not decide
    # what this one asserts -- and what this one does must not leak either.
    assistant._release()
    with assistant._lock:
        saved_history = list(assistant._history)
        assistant._history.clear()

    spoken: list[str] = []
    original = assistant._synthesize
    assistant._synthesize = lambda text: spoken.append(text)  # type: ignore[method-assign]
    try:
        assistant._begin_turn("why is nothing working")
        assistant._apply(AgentEvent("error", "Ollama is not running."))
        assistant._finish_turn()
        check("an error is spoken", len(spoken) == 1, repr(spoken))
        check("with its message intact",
              bool(spoken) and "Ollama is not running" in spoken[0],
              repr(spoken))
        check("and with the marker already in the history",
              bool(assistant.snapshot()[0])
              and assistant.snapshot()[0][-1]["content"].startswith("⚠"),
              repr(assistant.snapshot()[0][-1]["content"]))

        spoken.clear()
        # A turn is begin/apply/finish; the flag is released between them by
        # submit(). Without it the second turn would be refused and this
        # check would be reading the error it is meant to be away from.
        assistant._release()
        assistant._begin_turn("what is the capital of france")
        assistant._apply(AgentEvent("final", "Paris."))
        assistant._finish_turn()
        check("a normal answer is spoken", len(spoken) == 1, repr(spoken))

        assistant.speak_replies = False
        spoken.clear()
        assistant._release()
        assistant._begin_turn("and again")
        assistant._apply(AgentEvent("final", "Paris."))
        assistant._finish_turn()
        check("the speak switch still silences it", spoken == [], repr(spoken))
    finally:
        assistant._synthesize = original  # type: ignore[method-assign]
        assistant.speak_replies = True
        # _begin_turn claims the busy flag; leaving it set would stop the
        # next test's idle loop from ever firing.
        assistant._release()
        with assistant._lock:
            assistant._history[:] = saved_history


def test_speech_keeps_pace() -> None:
    """Speech starts with the typing, not after it has finished.

    The answer reaches the screen a token at a time, and the speaker used to
    wait for all of it -- so a reply you could read was one you could not yet
    hear. Stopping one halfway has to take the rest of it with it, or the
    words the user silenced are the ones that carry on speaking.
    """
    get_assistant, _, _, _ = _load_stack()
    assistant = get_assistant()

    from app.agent import AgentEvent

    assistant._release()
    with assistant._lock:
        saved_history = list(assistant._history)
        assistant._history.clear()

    spoken: list[str] = []
    original = assistant._synthesize
    assistant._synthesize = lambda text: spoken.append(text)  # type: ignore[method-assign]
    scratch = Path(tempfile.mkdtemp(prefix="apache-stream-"))
    try:
        assistant._begin_turn("tell me something long")
        assistant._apply(AgentEvent("text", "First sentence lands here. Second"))
        check("speech starts before the turn has ended",
              spoken == ["First sentence lands here."], repr(spoken))

        assistant._apply(AgentEvent(
            "text", "First sentence lands here. Second one is still going"))
        check("an unfinished sentence waits its turn",
              len(spoken) == 1, repr(spoken))

        assistant._finish_turn()
        check("the tail is said when the answer ends",
              spoken == ["First sentence lands here.",
                         "Second one is still going"],
              repr(spoken))
        check("and nothing is said twice", len(spoken) == 2, repr(spoken))

        assistant._release()
        spoken.clear()
        assistant._begin_turn("stop me halfway")
        assistant._apply(AgentEvent("text", "One sentence here. Two"))
        mine = scratch / f"reply-{assistant._turn}-deadbeef.wav"
        mine.write_bytes(b"RIFF")
        theirs = scratch / "reply-0-greeting.wav"
        theirs.write_bytes(b"RIFF")
        with assistant._lock:
            assistant._audio_queue.append((str(mine), 1.0))
            assistant._audio_queue.append((str(theirs), 1.0))
        assistant._abort_turn("Stopped.")
        time.sleep(0.3)
        check("stopping drops the reply's unsaid clips", not mine.exists(),
              str(mine))
        check("and leaves a greeting waiting its turn alone",
              theirs.exists(), str(theirs))
        check("the stop itself is still announced",
              "Stopped." in spoken, repr(spoken))
    finally:
        assistant._synthesize = original  # type: ignore[method-assign]
        assistant._release()
        with assistant._lock:
            assistant._history[:] = saved_history
            assistant._audio_queue = [
                (path, dur)
                for path, dur in assistant._audio_queue
                if not path.startswith(str(scratch))
            ]
        for leftover in scratch.glob("*"):
            leftover.unlink(missing_ok=True)
        scratch.rmdir()


def test_presence_prompts() -> None:
    """Greeting at startup, a check-in after silence, and no chatter."""
    _g, _b, ui, _ = _load_stack()
    get_assistant, _, _, _ = _load_stack()
    assistant = get_assistant()

    spoken: list[str] = []
    original_synth = assistant._synthesize
    saved_period = assistant.settings.idle_prompt_s
    saved_speak = assistant.speak_replies
    # Swap synthesis out so neither test opens a network TTS call.
    assistant._synthesize = lambda text: spoken.append(text)
    assistant.speak_replies = True
    assistant.stop_prompts()
    assistant._prompts_started = False

    try:
        assistant.settings.idle_prompt_s = 1  # keep the wait short
        assistant.start_prompts()

        deadline = time.time() + 4.0
        while time.time() < deadline and not spoken:
            time.sleep(0.05)
        check("greets on startup", assistant.settings.greeting in spoken,
              repr(spoken))

        assistant.start_prompts()
        time.sleep(0.4)
        check("the greeting is not repeated",
              spoken.count(assistant.settings.greeting) == 1, repr(spoken))

        deadline = time.time() + 6.0
        idle = assistant.settings.idle_prompt
        while time.time() < deadline and idle not in spoken:
            time.sleep(0.05)
        check("checks in once the idle window passes", idle in spoken,
              repr(spoken))

        before = assistant._last_input
        assistant.note_input()
        check("any interaction restarts the idle clock",
              assistant._last_input >= before,
              f"{assistant._last_input:.3f} vs {before:.3f}")

        # Never talk over a reply in flight.
        with assistant._lock:
            assistant._busy = True
            assistant._last_input = time.monotonic() - 99
        spoken.clear()
        time.sleep(2.5)
        check("stays quiet while a reply is in flight", not spoken,
              repr(spoken))
        with assistant._lock:
            assistant._busy = False

        # Stop the idle thread before the remaining, timing-free checks.
        assistant.stop_prompts()
        time.sleep(1.0)

        spoken.clear()
        assistant.speak_replies = False
        assistant.say("must not be spoken")
        time.sleep(0.4)
        check("honours the speak-replies switch", not spoken, repr(spoken))

        assistant.speak_replies = True
        spoken.clear()
        assistant.say("   ")
        time.sleep(0.4)
        check("blank announcements are ignored", not spoken, repr(spoken))

        check("greeting is configurable", bool(assistant.settings.greeting),
              repr(assistant.settings.greeting))
        check("check-in line is configurable", bool(idle), repr(idle))
    finally:
        assistant._synthesize = original_synth
        assistant.settings.idle_prompt_s = saved_period
        assistant.speak_replies = saved_speak
        assistant.stop_prompts()
        assistant._prompts_started = False
        with assistant._lock:
            assistant._busy = False
            assistant._last_input = time.monotonic()


def main() -> int:
    print("Apache stack tests")
    print("-" * 64)

    _g, _b, _ui, reason = _load_stack()
    if _ui is None:
        print(f"  SKIP the ML/UI stack could not be loaded: {reason}")
        print()
        print("Skipped (native stack unavailable) -- this is expected where")
        print("Smart App Control or a code-integrity policy blocks PyPI wheels.")
        print("Run python run.py --check for the diagnosis.")
        return 0

    for suite in (
        test_assistant,
        test_tool_registry,
        test_text_tool_call,
        test_rag_end_to_end,
        test_gradio_ui,
        test_status_helpers,
        test_snapshot_rendering,
        test_agent_turn_degrades_cleanly,
        test_listener_diagnostics,
        test_capture_feeds_vad,
        test_audio_queue,
        test_offline_speech,
        test_thinking_animation,
        test_jarvis_theme,
        test_audio_delivery,
        test_voice_defaults,
        test_background_tab,
        test_errors_are_spoken,
        test_speech_keeps_pace,
        test_presence_prompts,
        # Kept last: it resets the shared conversation.
        test_stop_path,
    ):
        print(f"-- {suite.__name__}")
        try:
            suite()
        except Exception:  # noqa: BLE001
            traceback.print_exc()
            FAILURES.append(f"{suite.__name__} crashed")
        print()

    if FAILURES:
        print(f"{len(FAILURES)} FAILURE(S):")
        for name in FAILURES:
            print(f"  - {name}")
        return 1
    print("All stack tests passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
