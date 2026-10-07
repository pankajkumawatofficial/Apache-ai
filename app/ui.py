"""Gradio front end for Apache.

Two input paths converge on :class:`app.core.Assistant`:

* **Typed** -- ``on_send`` is a generator that yields a UI snapshot after every
  agent event, so tokens stream into the transcript.
* **Spoken** -- the wake-word listener runs in its own thread and blocks on
  ``Assistant.submit_blocking``. A ``gr.Timer`` polls the shared snapshot and
  pushes whatever the voice path produced into the same components.

Every handler therefore returns views of one shared state; there is no second
copy of the conversation to keep in sync.
"""

from __future__ import annotations

import threading
import time
from typing import Any, Iterator

import gradio as gr

from .config import DEFAULT_WAKE_ALIASES, settings
from .core import Assistant, Snapshot, get_assistant
from .llm import check_ollama
from .voice.listener import WakeListener, list_input_devices
from .voice.tts import clip_extension

__all__ = ["build_demo", "build_and_launch"]

#: A short, high-quality subset of edge-tts voices. Any other name can be
#: typed directly into the field (the dropdown allows custom values).
TTS_VOICES = [
    "en-US-AndrewMultilingualNeural",
    "en-US-AvaMultilingualNeural",
    "en-US-EmmaMultilingualNeural",
    "en-US-BrianMultilingualNeural",
    "en-GB-RyanNeural",
    "en-GB-SoniaNeural",
    "en-IN-PrabhatNeural",
    "en-IN-NeerjaNeural",
]

_HEADER = """\
# APACHE
**Local voice assistant — HUD online**

LangChain agent · Ollama · faster-whisper · piper · fully local
"""


# ---------------------------------------------------------------------------
# Small formatting helpers
# ---------------------------------------------------------------------------
def _level_meter(level: float, threshold: float, width: int = 18) -> str:
    """A one-line bar showing input level against the VAD gate.

    When someone says "it isn't hearing me" the useful question is whether the
    microphone is delivering signal at all and where the gate sits, which is
    otherwise invisible.
    """
    if threshold <= 0:
        return ""
    filled = max(0, min(width, int(round(level / threshold * width))))
    gate = max(0, min(width, int(round(threshold / max(threshold, 1e-9) * width))))
    # 's' marks speech-level, '|' marks the gate. Simpler and more legible in
    # a monospace span than trying to draw the gate inside the bar.
    bar = "█" * filled + "·" * (width - filled)
    verdict = "over gate" if level >= threshold else "below gate"
    return f"`{bar}` {level:.4f} / {threshold:.4f} ({verdict})"


def _voice_choices(assistant: Assistant) -> tuple[list[str], str]:
    """Choices and default for the Voice dropdown, in the engine's own terms.

    Offline the dropdown listed edge-tts names that the engine then ignored,
    so "try a different voice" was a control that did nothing at all. What is
    downloaded now leads, because those are the only voices Piper can speak
    with; the online names follow, since the same field drives
    ``APACHE_OFFLINE=0``.
    """
    from .voice import tts

    local = tts.available_voices()
    if settings.offline and local:
        current = tts.resolve_model(assistant.voice).stem
        return local, current if current in local else local[0]
    return list(TTS_VOICES), assistant.voice


def _voice_status_text(listener: WakeListener) -> str:
    info = listener.status()
    state = info["state"]
    icon = {
        "listening": "🎙️",
        "transcribing": "🎧",
        "command": "⚡",
        "error": "⚠️",
        "stopped": "⏹️",
    }.get(state, "•")

    lines = [f"{icon} **{state.title()}** — {info['message']}"]
    if info["transcript"]:
        lines.append(f"\n_Last heard:_ `{info['transcript']}`")
    lines.append(
        f"\nWake word: **{info['wake_word']}**"
        + (
            " — _optional, any question works_"
            if not info.get("wake_required", True)
            else ""
        )
        + (" · _muted while speaking_" if info["muted"] else "")
    )

    if info.get("sample_rate"):
        lines.append(
            f"\n{_level_meter(info.get('level', 0.0), info.get('threshold', 0.0))}"
        )
        lines.append(
            f"`{info['device']}` · {info['sample_rate']} Hz"
        )

    # Capture health. Every one of these used to be silently swallowed, which
    # made "the microphone is not working" undiagnosable from the UI.
    notes = []
    if info.get("capture_errors"):
        notes.append(f"⚠ {info['capture_errors']} audio block error(s)")
    if info.get("rejected"):
        notes.append(
            f"⚠ {info['rejected']} utterance(s) heard but discarded as too "
            "short — the gate is clipping your voice"
        )
    if info.get("xruns"):
        notes.append(f"⚠ {info['xruns']} audio overrun(s)")
    if info.get("dropped_blocks"):
        notes.append(f"⚠ {info['dropped_blocks']} buffer overrun(s)")
    if info.get("ignored"):
        notes.append(f"· {info['ignored']} utterance(s) ignored while busy")
    if info.get("queued"):
        notes.append(f"· {info['queued']} awaiting transcription")
    # A live meter pinned at zero while supposedly listening means the mic is
    # muted or we opened the wrong device -- worth saying, not leaving implied.
    if state == "listening" and info.get("level", 0.0) <= 0.0:
        notes.append("⚠ no signal reaching the microphone")
    if notes:
        lines.append("\n" + " · ".join(notes))
    return "\n".join(lines)


def _docs_status_text(assistant: Assistant) -> str:
    status = assistant.document_status()
    if not status["chunks"]:
        return "📄 No documents indexed yet."
    sources = ", ".join(status["sources"][:6])
    extra = "" if len(status["sources"]) <= 6 else ", …"
    line = (
        f"📄 **{status['chunks']}** chunks from **{status['documents']}** "
        f"document(s) · retrieval `{status['mode']}`\n\n{sources}{extra}"
    )
    if status["error"]:
        line += f"\n\n⚠️ {status['error']}"
    return line


def _tools_status_text(assistant: Assistant) -> str:
    """The toolset the agent can call, shown so it is visible not implied."""
    try:
        from .tools.registry import build_tools

        tools = build_tools(assistant.store, assistant.workspace, assistant.settings)
        names = [getattr(tool, "name", str(tool)) for tool in tools]
    except Exception as exc:  # noqa: BLE001 - stack may be unavailable
        reason = " ".join(str(exc).split())[:150] or type(exc).__name__
        return f"🧰 Tools unavailable: {reason}"
    if not names:
        return "🧰 No tools registered."
    return "🧰 " + " · ".join(f"`{name}`" for name in names)


def _settings_status_text(assistant: Assistant, ollama: Any = None) -> str:
    ollama = ollama or check_ollama()
    if ollama.ok:
        head = f"✅ {ollama.message}"
    else:
        head = f"❌ {ollama.message}"
    return (
        f"{head}\n\n"
        f"Model: `{assistant.model}` · temperature `{assistant.temperature}` · "
        f"thinking `{'on' if assistant.reasoning else 'off'}` · "
        f"RAG `{assistant.store.status()['mode']}`\n\n"
        f"{_tools_status_text(assistant)}"
    )


def _render(snapshot: Snapshot) -> tuple[Any, Any, Any, Any]:
    messages, activity, _audio, status = snapshot
    # Audio is released only by the ticker, through Assistant.next_audio().
    # Returning it here meant any handler -- uploading a document, applying a
    # setting, resetting -- could push an already-played clip back in front of
    # one that had never been heard.
    return messages, activity, gr.skip(), status


#: Spinner frames for a bubble the model has not answered yet. The frame is
#: derived from wall time, so the ticker needs no state of its own and two
#: viewers of the same page still see the same animation.
_THINKING_FRAMES = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"

#: Applied to the pending bubble. Kept gentle: this is a status cue, not a
#: distraction, and it must stay readable in a dark terminal-style theme.
_CSS = """
/* ==========================================================================
   JARVIS / Iron Man HUD theme.

   A presentational layer and nothing else: every rule either redeclares a
   design token Gradio already resolves its colours from, or styles an element
   this app owns (#apache-header, #apache-chat, .apache-thinking). That is the
   point -- recolouring the tokens retints the entire application without
   depending on Gradio's hashed Svelte class names, which change on release,
   so no behaviour can be disturbed by the theme.
   ========================================================================== */

/* --- 1. Base --------------------------------------------------------------- */
html, body {
    background-color: #04070d !important;
    /* A scanline the eye registers more as texture than as lines. */
    background-image: repeating-linear-gradient(
        180deg,
        rgba(0, 229, 255, .022) 0 1px,
        transparent 1px 4px
    ) !important;
    background-attachment: fixed !important;
}

.gradio-container {
    --body-background-fill: #04070d;
    --background-fill-primary: #070d17;
    --background-fill-secondary: #0a1322;
    --block-background-fill: #0a1322;
    --input-background-fill: #050a12;
    --input-border-color: rgba(0, 229, 255, .26);
    --border-color-primary: rgba(0, 229, 255, .20);
    --body-text-color: #cfe9f5;
    --body-text-color-subdued: #7d97ab;
    --block-label-text-color: #63d3e8;
    --link-text-color: #4fe3ff;
    --link-text-color-hover: #a8f4ff;
    --link-text-color-visited: #4fe3ff;
    --color-accent: #00e5ff;
    --color-accent-soft: 0, 229, 255;
    --primary-500: #00e5ff;
    --primary-600: #00b8d4;
    --button-primary-background-fill: #062831;
    --button-primary-background-fill-hover: #0a4757;
    --button-primary-border-color: #00e5ff;
    --button-primary-border-color-hover: #6ff3ff;
    --button-primary-text-color: #a8f4ff;
    --button-primary-text-color-hover: #ffffff;
    --button-secondary-background-fill: #0a1322;
    --button-secondary-background-fill-hover: #12233a;
    --button-secondary-border-color: rgba(0, 229, 255, .32);
    --button-secondary-text-color: #a8f4ff;
    --button-cancel-background-fill: #2a1015;
    --button-cancel-border-color: rgba(255, 120, 120, .55);
    --button-cancel-text-color: #ff9d9d;
    --font-mono: "JetBrains Mono", "Cascadia Mono", ui-monospace,
                 SFMono-Regular, Menlo, monospace;
    background: #04070d;
    color: #cfe9f5;
}

/* --- 2. The banner --------------------------------------------------------- */
#apache-header {
    position: relative;
    margin: 2px 0 16px;
    padding: 20px 24px 16px;
    background:
        linear-gradient(180deg, rgba(0, 229, 255, .07), rgba(0, 229, 255, 0) 72%),
        #070d17;
    border: 1px solid rgba(0, 229, 255, .30);
    border-radius: 3px;
    box-shadow:
        0 0 30px rgba(0, 229, 255, .10),
        inset 0 0 50px rgba(0, 229, 255, .04);
}
/* Two corner ticks read as a HUD frame without needing four. */
#apache-header::before,
#apache-header::after {
    content: "";
    position: absolute;
    width: 18px;
    height: 18px;
    border: 2px solid #00e5ff;
    opacity: .9;
}
#apache-header::before {
    top: -1px;
    left: -1px;
    border-right: 0;
    border-bottom: 0;
}
#apache-header::after {
    right: -1px;
    bottom: -1px;
    border-left: 0;
    border-top: 0;
}
#apache-header h1 {
    margin: 0;
    color: #eafcff;
    font-size: 2rem;
    font-weight: 700;
    letter-spacing: .34em;
    text-transform: uppercase;
    text-shadow:
        0 0 12px rgba(0, 229, 255, .8),
        0 0 36px rgba(0, 229, 255, .35);
}
#apache-header p {
    margin: 8px 0 0;
    color: #6fd8ea;
    font-family: var(--font-mono);
    font-size: .78rem;
    letter-spacing: .16em;
    text-transform: uppercase;
}
#apache-header p strong { color: #a8f4ff; font-weight: 600; }
#apache-header p + p { margin-top: 4px; color: #4d6b80; }

/* --- 3. Conversation ------------------------------------------------------- */
/* Each entry is a console line: a dark panel behind a cyan rail. */
#apache-chat .bubble {
    background: rgba(10, 19, 34, .9);
    border: 1px solid rgba(0, 229, 255, .16);
    border-left: 2px solid rgba(0, 229, 255, .55);
    border-radius: 2px;
    transition: border-color .18s ease, box-shadow .18s ease;
}
#apache-chat .bubble:hover {
    border-color: rgba(0, 229, 255, .42);
    box-shadow: 0 0 20px rgba(0, 229, 255, .14);
}
#apache-chat .bubble a { color: #4fe3ff; }
#apache-chat .bubble a:hover { color: #a8f4ff; }
#apache-chat .bubble code {
    background: #050a12;
    border: 1px solid rgba(0, 229, 255, .20);
    color: #8ef1ff;
    font-family: var(--font-mono);
}
#apache-chat .bubble pre {
    background: #050a12;
    border: 1px solid rgba(0, 229, 255, .22);
}
#apache-chat .bubble pre code {
    background: transparent;
    border: 0;
    color: #b9e9f5;
}

/* --- 4. Inputs ------------------------------------------------------------- */
.gradio-container textarea,
.gradio-container input:not([type="checkbox"]):not([type="radio"]):not([type="range"]),
.gradio-container select {
    background: #050a12 !important;
    border-color: rgba(0, 229, 255, .26) !important;
    color: #dff6ff;
}
.gradio-container textarea::placeholder,
.gradio-container input::placeholder {
    color: #4d6b80;
}
.gradio-container textarea:focus,
.gradio-container input:focus,
.gradio-container select:focus {
    border-color: #00e5ff !important;
    outline: none;
    box-shadow:
        0 0 0 1px rgba(0, 229, 255, .45),
        0 0 20px rgba(0, 229, 255, .30) !important;
}

/* --- 5. Buttons ------------------------------------------------------------ */
.gradio-container button {
    text-transform: uppercase;
    letter-spacing: .13em;
    font-size: .76rem;
    font-weight: 600;
}
.gradio-container button.primary {
    background: linear-gradient(180deg, #0a4757, #062831) !important;
    border-color: #00e5ff !important;
    color: #a8f4ff !important;
    text-shadow: 0 0 10px rgba(0, 229, 255, .6);
    box-shadow:
        0 0 18px rgba(0, 229, 255, .22),
        inset 0 0 14px rgba(0, 229, 255, .10);
    transition: box-shadow .18s ease, color .18s ease;
}
.gradio-container button.primary:hover:not(:disabled) {
    color: #ffffff !important;
    box-shadow:
        0 0 30px rgba(0, 229, 255, .5),
        inset 0 0 18px rgba(0, 229, 255, .18);
}
.gradio-container button.primary:disabled {
    opacity: .35;
    box-shadow: none;
}
.gradio-container button.secondary {
    border-color: rgba(0, 229, 255, .32) !important;
    color: #a8f4ff !important;
}
.gradio-container button.stop {
    border-color: rgba(255, 120, 120, .55) !important;
    color: #ff9d9d !important;
    text-shadow: none;
}

/* --- 6. Labels and tabs read as instrument text ---------------------------- */
.gradio-container label {
    color: #63d3e8;
    font-family: var(--font-mono);
    font-size: .74rem;
    letter-spacing: .13em;
    text-transform: uppercase;
}
.gradio-container [role="tab"] {
    color: #7d97ab;
    font-family: var(--font-mono);
    font-size: .74rem;
    letter-spacing: .13em;
    text-transform: uppercase;
    transition: color .16s ease;
}
.gradio-container [role="tab"][aria-selected="true"],
.gradio-container [role="tab"].selected {
    color: #00e5ff;
    text-shadow: 0 0 12px rgba(0, 229, 255, .65);
}

/* --- 7. Thinking ----------------------------------------------------------- */
/* The spinner glows rather than merely fading, so "working" reads at a glance. */
.apache-thinking {
    display: inline-block;
    color: #4fe3ff;
    font-family: var(--font-mono);
    letter-spacing: .14em;
    animation: apache-think 1.15s ease-in-out infinite;
}
@keyframes apache-think {
    0%, 100% {
        opacity: .30;
        letter-spacing: .02em;
        text-shadow: 0 0 4px rgba(0, 229, 255, .25);
    }
    50% {
        opacity: 1;
        letter-spacing: .12em;
        text-shadow: 0 0 18px rgba(0, 229, 255, .85);
    }
}

/* --- 8. Scrollbars --------------------------------------------------------- */
.gradio-container ::-webkit-scrollbar { width: 10px; height: 10px; }
.gradio-container ::-webkit-scrollbar-track { background: #050a12; }
.gradio-container ::-webkit-scrollbar-thumb {
    background: rgba(0, 229, 255, .30);
    border: 2px solid #050a12;
    border-radius: 0;
}
.gradio-container ::-webkit-scrollbar-thumb:hover {
    background: rgba(0, 229, 255, .60);
}

/* --- 9. Footer ------------------------------------------------------------- */
.gradio-container footer,
.gradio-container .footer {
    color: #4d6b80 !important;
}

/* --- 10. The one click Chrome will not do without -------------------------- */
/* Chrome refuses to start audio until the page has seen a gesture, and
   Apache's first words arrive before anyone has clicked anything. Saying so
   turns silence with no explanation into one obvious click. */
#apache-sound-prompt {
    position: fixed;
    top: 14px;
    left: 50%;
    transform: translateX(-50%);
    z-index: 999;
    padding: 9px 18px;
    background: rgba(5, 10, 18, .96);
    border: 1px solid rgba(0, 229, 255, .55);
    color: #a8f4ff;
    font-family: var(--font-mono);
    font-size: .74rem;
    letter-spacing: .14em;
    text-transform: uppercase;
    box-shadow:
        0 0 22px rgba(0, 229, 255, .30),
        inset 0 0 14px rgba(0, 229, 255, .08);
    animation: apache-prompt 1.6s ease-in-out infinite;
    cursor: pointer;
}
@keyframes apache-prompt {
    0%, 100% { box-shadow: 0 0 14px rgba(0, 229, 255, .22); }
    50%      { box-shadow: 0 0 34px rgba(0, 229, 255, .55); }
}
"""


def _theme():
    """The Gradio theme the HUD stylesheet sits on top of.

    Gradio derives its stylesheet from this object, which is what makes
    buttons, focus rings, sliders and checked boxes come out cyan rather than
    the default orange -- something raw CSS cannot express reliably, because
    several of those are generated rather than declared. The custom CSS adds
    only what a theme has no notion of: the banner frame, the console-style
    bubbles, and the glow.

    It goes to ``launch()``, not to the ``Blocks`` constructor: Gradio 6 marks
    ``Blocks(theme=...)`` deprecated and drops it, so passing it there would
    silently do nothing.
    """
    return gr.themes.Base(
        primary_hue="cyan",
        secondary_hue="cyan",
        neutral_hue="slate",
        radius_size=gr.themes.utils.sizes.radius_sm,
    )


def _keepalive_wav() -> str:
    """A data URI for 0.2 s of near-silent tone.

    Chrome treats a page that is playing audio as foreground: it stops
    clamping that page's timers, and stops dropping them to one a minute
    after five minutes in the background. Without this, Apache answering into
    a tab you are no longer looking at is delivered up to a minute late.

    The tone is 44 dB below full scale -- non-zero, so it counts as audio,
    and far too quiet to be heard. 0.2 s is exactly twenty cycles at 100 Hz,
    so the loop rejoins in phase and does not click.
    """
    import base64
    import io
    import math
    import struct
    import wave as wave_module

    rate, seconds, hz, amplitude = 8000, 0.2, 100, 20
    frames = int(rate * seconds)
    payload = b"".join(
        struct.pack(
            "<h",
            int(amplitude * math.sin(2 * math.pi * hz * i / rate)),
        )
        for i in range(frames)
    )
    buffer = io.BytesIO()
    with wave_module.open(buffer, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(payload)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:audio/wav;base64,{encoded}"


def _page_js() -> str:
    """Everything about speaking that has to happen in the browser.

    Three independent reasons a reply can synthesise perfectly and still come
    out of the speaker as nothing, none of which are visible from Python:

    1. **Gradio stops ticking when the tab is hidden.** Its Timer reads
       ``document.visibilityState === "visible"`` on every tick and dispatches
       nothing otherwise -- so the moment another tab is in front, the audio
       queue is never drained and the transcript never updates, however
       healthy Apache is. The guard reads the property each time, so
       shadowing it on this document keeps the ticks coming.
    2. **Chrome throttles hidden tabs' timers**, to once a second and then,
       after five minutes, to once a minute -- exempting pages that are
       playing audio. The keep-alive loop buys that exemption.
    3. **Chrome blocks media that starts without a user gesture**, and
       Apache's first words are the greeting, spoken before anyone has
       clicked anything, because the whole point is that you talk to it. The
       refusal has to be caught where it happens -- on the player itself,
       which is detached from the document and invisible to a DOM scan -- and
       retried, in order, once a gesture unlocks the origin. Until then the
       page says so, because silence with no explanation is the one failure
       the user cannot diagnose for themselves.
    """
    wav = _keepalive_wav()
    return """
(function () {
  "use strict";

  var keepAliveWav = "%s";

  /* Every piece of sound state, declared before anything can use it: the
     keep-alive is refused play() during construction of this very block. */
  var keepAlive = null;
  var promptEl = null;
  var unlocked = false;
  var blockedSound = false;
  var refused = [];   // clips the browser turned away, oldest first
  var contexts = [];  // Web Audio contexts, in case a player uses one

  /* --- 1. Keep the ticker running while another tab is in front --------- */
  try {
    Object.defineProperty(document, "visibilityState", {
      configurable: true,
      get: function () { return "visible"; }
    });
    Object.defineProperty(document, "hidden", {
      configurable: true,
      get: function () { return false; }
    });
  } catch (err) {
    /* Non-configurable engine: falls back to ticking only while visible,
       which is exactly the behaviour being fixed, not a new failure. */
  }

  /* --- 2. Stay exempt from Chrome's background-timer throttling --------- */
  function ensureKeepAlive() {
    try {
      if (!keepAlive) {
        keepAlive = document.createElement("audio");
        keepAlive.loop = true;
        keepAlive.preload = "auto";
        keepAlive.setAttribute("aria-hidden", "true");
        keepAlive.src = keepAliveWav;
        document.body.appendChild(keepAlive);
      }
      if (keepAlive.paused) {
        var attempt = keepAlive.play();
        if (attempt && attempt.catch) {
          attempt.catch(function () { /* allowed on a later tick */ });
        }
      }
    } catch (err) {
      keepAlive = null;
    }
  }

  /* --- 3. Catch every play() the browser turns away -------------------- */
  // The element that actually speaks is not in the document. Gradio plays
  // through WaveSurfer, which owns a detached <audio> of its own, so a scan
  // of document.getElementsByTagName("audio") sees only the hidden native
  // element -- which carries no src -- plus the keep-alive above, and never
  // the reply. Wrapping play() is the only thing that can name that element
  // wherever it lives, so the refusal is recorded at the source.
  function noteRefusal(el) {
    if (!el) return;
    blockedSound = true;
    if (!unlocked) showPrompt();
    if (el === keepAlive) return;  // the keep-alive retries itself
    if (refused.indexOf(el) === -1) refused.push(el);
  }

  try {
    var realPlay = HTMLMediaElement.prototype.play;
    HTMLMediaElement.prototype.play = function () {
      var attempt;
      try {
        attempt = realPlay.apply(this, arguments);
      } catch (err) {
        noteRefusal(this);
        throw err;
      }
      if (attempt && attempt.then) {
        var media = this;
        attempt.catch(function () { noteRefusal(media); });
      }
      return attempt;
    };
  } catch (err) { /* an engine that refuses the wrap plays the ordinary way */ }

  // A Web Audio player is silent for a subtler reason: its context is created
  // while the page is still untrusted, stays suspended until a gesture, and
  // only its owner can resume it. Keeping the list is what lets a later
  // gesture do that -- Apache's own context cannot resume someone else's.
  try {
    var RealContext = window.AudioContext || window.webkitAudioContext;
    if (RealContext) {
      var TrackedContext = function () {
        var ctx = Reflect.construct(RealContext, arguments, RealContext);
        contexts.push(ctx);
        return ctx;
      };
      TrackedContext.prototype = RealContext.prototype;
      window.AudioContext = TrackedContext;
      if (window.webkitAudioContext) window.webkitAudioContext = TrackedContext;
    }
  } catch (err) { /* tracking is lost; playback itself is untouched */ }

  /* --- 4. Say the clips in the order they were refused ------------------ */
  function retryBlockedReplies() {
    while (refused.length) {
      var el = refused[0];
      // Finished, sourceless or gone: nothing left to say, move on.
      if (!el || el.ended || !el.src) { refused.shift(); continue; }
      // Speaking: the server has already spaced these by duration, so the
      // next one waits rather than talking over this one.
      if (!el.paused) return;
      try {
        var attempt = el.play();
        if (attempt && attempt.catch) { attempt.catch(function () {}); }
      } catch (err) { /* the next tick tries again */ }
      return; // whether it took is settled on the next pass
    }
  }

  /* --- 5. The one click Chrome insists on -------------------------------- */
  function showPrompt() {
    if (promptEl || unlocked) return;
    try {
      promptEl = document.createElement("div");
      promptEl.id = "apache-sound-prompt";
      promptEl.textContent = "SOUND BLOCKED \\u2014 click anywhere to hear Apache";
      document.body.appendChild(promptEl);
    } catch (err) {
      promptEl = null;  // no body yet; the next refusal tries again
    }
  }

  function hidePrompt() {
    if (!promptEl) return;
    try { promptEl.parentNode.removeChild(promptEl); } catch (err) {}
    promptEl = null;
  }

  function unlock() {
    if (unlocked) return;
    unlocked = true;
    hidePrompt();
    try {
      var Context = window.AudioContext || window.webkitAudioContext;
      var ctx = new Context();
      var buffer = ctx.createBuffer(1, 1, 22050);
      var source = ctx.createBufferSource();
      source.buffer = buffer;
      source.connect(ctx.destination);
      source.start(0);
      if (ctx.state === "suspended" && ctx.resume) ctx.resume();
      window.apacheAudioContext = ctx;
    } catch (err) { /* the next gesture tries again */ }
    // Resume every context this page made while it was still untrusted.
    for (var i = 0; i < contexts.length; i++) {
      try {
        if (contexts[i].state === "suspended" && contexts[i].resume) {
          contexts[i].resume();
        }
      } catch (err) { /* a closed context is not ours to revive */ }
    }
    ensureKeepAlive();
    retryBlockedReplies();
  }
  ["pointerdown", "keydown", "touchstart"].forEach(function (name) {
    document.addEventListener(name, unlock, true);
  });

  ensureKeepAlive();
  setInterval(function () {
    ensureKeepAlive();
    retryBlockedReplies();
    if (blockedSound && !unlocked) showPrompt();
  }, 500);
})();
""" % wav


def _thinking_html() -> str:
    frame = _THINKING_FRAMES[int(time.time() * 1.7) % len(_THINKING_FRAMES)]
    # The span carries the CSS animation. If a stricter sanitisation policy
    # ever strips the tag, the spinner and the word still render, so the
    # bubble is never left blank.
    return f'<span class="apache-thinking">{frame} Thinking…</span>'


def _with_thinking(
    messages: list[dict[str, str]], busy: bool
) -> list[dict[str, str]]:
    """Animate a bubble that exists but has no text yet.

    Only the empty placeholder that :meth:`Assistant._begin_turn` creates
    qualifies -- once the first token arrives the real text replaces it, so
    the animation cannot overwrite an answer in progress.
    """
    if not busy or not messages:
        return messages
    last = messages[-1]
    if last.get("role") != "assistant" or (last.get("content") or "").strip():
        return messages
    out = [dict(m) for m in messages]
    out[-1] = {**last, "content": _thinking_html()}
    return out


def _device_choices() -> tuple[list[str], str]:
    """Microphone picker entries, with the system default listed and selected.

    The default is only a starting point -- Windows may move it while the page
    is open, so the choice is read at click time rather than trusted.
    """
    devices = list_input_devices()
    if not devices:
        return ["(no microphone found)"], ""
    labels = [label for _index, label in devices]
    return labels, labels[0]


def _device_index(label: str | None) -> int | None:
    """Recover the PortAudio index from the label the picker submitted."""
    if not label:
        return None
    head = str(label).split("]", 1)[0].lstrip("[")
    return int(head) if head.isdigit() else None


# ---------------------------------------------------------------------------
# Demo construction
# ---------------------------------------------------------------------------
def build_demo(assistant: Assistant | None = None) -> gr.Blocks:
    """Build the Blocks app. Kept as a factory so tests can inject a stub."""
    assistant = assistant or get_assistant()

    listener = WakeListener(
        on_command=_voice_handler(assistant),
        settings=assistant.settings,
        on_error=lambda message: assistant.log(f"Listener: {message}"),
    )
    # Mute the mic while the reply plays so Apache never transcribes itself.
    assistant.on_reply_audio = lambda _path, seconds: listener.mute_for(
        seconds + assistant.settings.playback_mute_s
    )

    # Gradio 6 takes css on launch(), not on the constructor.
    with gr.Blocks(title="Apache", fill_height=True) as demo:
        # An elem_id so the HUD header can be styled precisely rather than by
        # guessing at Gradio's hashed Svelte class names.
        gr.Markdown(_HEADER, elem_id="apache-header")

        with gr.Row():
            # ---------------- chat column ----------------
            with gr.Column(scale=3, min_width=420):
                chatbot = gr.Chatbot(
                    value=assistant.snapshot()[0],
                    label="Conversation",
                    height=480,
                    layout="bubble",
                    placeholder="Ask something, or say “Apache, …”",
                    elem_id="apache-chat",
                )
                activity = gr.Textbox(
                    value="",
                    label="Agent activity (tool calls)",
                    lines=5,
                    interactive=False,
                    max_lines=12,
                    placeholder=(
                        "Tool calls stream here while the agent works, e.g.\n"
                        "  → calculator(expression=1739 * 42)\n"
                        "    calculator: 73038"
                    ),
                )
                with gr.Row():
                    msg = gr.Textbox(
                        placeholder='Type a message, or start with "Apache, …"',
                        label="Message",
                        scale=4,
                        show_label=False,
                        autofocus=True,
                    )
                    send = gr.Button(
                        "Send", variant="primary", scale=1, min_width=110
                    )
                    stop = gr.Button(
                        "Stop",
                        variant="stop",
                        scale=1,
                        min_width=90,
                        # Enabled only while a turn is actually running.
                        interactive=False,
                    )
                voice_audio = gr.Audio(
                    value=None,
                    label="Voice reply",
                    type="filepath",
                    # Must match what synthesize() actually wrote: claiming mp3
                    # while holding a WAV invites a conversion Gradio can only
                    # do if ffmpeg happens to be installed, and a failed
                    # conversion is silent -- the reply simply never plays.
                    format=clip_extension().lstrip("."),
                    autoplay=True,
                    interactive=False,
                    show_label=True,
                )
                status = gr.Markdown("Ready.")

            # ---------------- side column ----------------
            with gr.Column(scale=1, min_width=320):
                _dev_labels, _dev_default = _device_choices()
                mic_device = gr.Dropdown(
                    choices=_dev_labels,
                    value=_dev_default,
                    label="Microphone",
                    info="Windows may move the default while the page is open; "
                         "pick the one you actually speak into.",
                )
                mic_button = gr.Button("Start listening", variant="primary")
                voice_status = gr.Markdown(_voice_status_text(listener))

                voice_choices, voice_value = _voice_choices(assistant)

                with gr.Tabs():
                    with gr.Tab("Documents"):
                        files = gr.UploadButton(
                            "Upload documents",
                            file_count="multiple",
                            type="filepath",
                        )
                        with gr.Row():
                            clear_docs = gr.Button("Clear index", variant="stop")
                            reset_chat = gr.Button("New conversation")
                        docs_status = gr.Markdown(_docs_status_text(assistant))

                    with gr.Tab("Voice"):
                        speak = gr.Checkbox(
                            value=assistant.speak_replies,
                            label="Speak replies aloud",
                        )
                        voice = gr.Dropdown(
                            choices=voice_choices,
                            value=voice_value,
                            label="Voice",
                            allow_custom_value=True,
                        )
                        wake = gr.Textbox(
                            value=assistant.settings.wake_word,
                            label="Wake word",
                            info="Say this first to ask a question out loud.",
                        )
                        wake_required = gr.Checkbox(
                            value=assistant.settings.wake_required,
                            label="Require the wake word",
                            info=(
                                "Off: answer anything you say, no “Apache” "
                                "needed. On: only run what follows the wake "
                                "word, which keeps a television from talking "
                                "to you."
                            ),
                        )

                    with gr.Tab("Model"):
                        ollama_status = check_ollama()
                        model = gr.Dropdown(
                            choices=ollama_status.choices,
                            value=assistant.model,
                            label="Ollama model",
                            allow_custom_value=True,
                        )
                        temperature = gr.Slider(
                            minimum=0.0,
                            maximum=1.5,
                            value=assistant.temperature,
                            step=0.1,
                            label="Temperature",
                            info="Lower is more deterministic.",
                        )
                        reasoning = gr.Checkbox(
                            value=assistant.reasoning,
                            label="Think before answering",
                            info=(
                                "Chain-of-thought. More careful, but measured "
                                "6.9x slower end-to-end on this machine -- leave "
                                "off for voice."
                            ),
                        )
                        system_prompt = gr.Textbox(
                            value=assistant.system_prompt,
                            label="System prompt",
                            lines=10,
                            max_lines=20,
                        )
                        apply = gr.Button("Apply settings", variant="primary")
                        settings_status = gr.Markdown(
                            _settings_status_text(assistant, ollama_status)
                        )

        # ----------------------------------------------------------
        # Events
        # ----------------------------------------------------------
        def on_send(message: str) -> Iterator[tuple]:
            text = (message or "").strip()
            messages, act, _audio, stat = assistant.snapshot()
            if not text:
                yield messages, act, gr.skip(), stat, ""
                return

            for snapshot in assistant.submit(text):
                messages, act, _audio, stat = snapshot
                # The ticker owns audio. Yielding it here could replay a clip
                # that had already played, or hide one queued behind it.
                yield messages, act, gr.skip(), stat, ""

        def on_tick() -> tuple:
            messages, act, _audio, stat = _render(assistant.snapshot())
            # The timer is the only thing running while the model is silent,
            # so it is what drives the animation in a pending bubble.
            messages = _with_thinking(messages, assistant.busy)
            # One clip per tick at most, and none while the last is still
            # playing -- see Assistant.next_audio.
            clip = assistant.next_audio()
            audio_out = clip if clip else gr.skip()
            busy = assistant.busy
            return (
                messages,
                act,
                audio_out,
                stat,
                _voice_status_text(listener),
                # Stop only makes sense mid-turn; Send is greyed out so a
                # second click cannot race the busy flag.
                gr.update(interactive=busy),
                gr.update(interactive=not busy),
            )

        def on_stop() -> tuple:
            assistant.request_stop()
            messages, _act, _audio, stat = assistant.snapshot()
            return messages, gr.skip(), stat

        def on_mic(device_value: str | None) -> tuple:
            # Reaching for the microphone is the user being present.
            assistant.note_input()
            if listener.running:
                listener.stop()
            else:
                # Read the picker at click time: the system default may have
                # moved since the page loaded, and starting on a silent device
                # looks exactly like a broken one.
                listener.start(device_index=_device_index(device_value))
            label = "Stop listening" if listener.running else "Start listening"
            variant = "stop" if listener.running else "primary"
            return _voice_status_text(listener), gr.update(value=label, variant=variant)

        def on_ingest(paths: list[str] | None) -> tuple:
            snapshot = assistant.ingest(paths or [])
            return (*_render(snapshot), _docs_status_text(assistant))

        def on_clear_docs() -> tuple:
            snapshot = assistant.clear_documents()
            return (*_render(snapshot), _docs_status_text(assistant))

        def on_reset() -> tuple:
            snapshot = assistant.reset()
            return (*_render(snapshot), _voice_status_text(listener))

        def on_apply(
            model_value: str,
            temperature_value: float,
            prompt_value: str,
            speak_value: bool,
            voice_value: str,
            wake_value: str,
            wake_required_value: bool,
            reasoning_value: bool,
        ) -> tuple:
            word = (wake_value or "").strip().lower() or settings.wake_word
            settings.wake_word = word
            settings.wake_aliases = [word] + [
                alias for alias in DEFAULT_WAKE_ALIASES if alias != word
            ]
            # The listener reads this off the shared settings on every
            # utterance, so no restart and no button press is needed to stop
            # asking to be addressed by name.
            settings.wake_required = bool(wake_required_value)
            snapshot = assistant.configure(
                model=(model_value or "").strip() or assistant.model,
                temperature=temperature_value,
                system_prompt=prompt_value,
                speak_replies=speak_value,
                voice=(voice_value or "").strip() or assistant.voice,
                reasoning=bool(reasoning_value),
            )
            return (*_render(snapshot), _settings_status_text(assistant))

        def on_load() -> tuple:
            status = check_ollama()
            # Listening from the moment the page opens. The whole point of the
            # app is saying "Apache", and requiring a first click left it
            # feeling half-started; a machine with no usable input device
            # reports itself through the status line instead of raising.
            if assistant.settings.mic_autostart and not listener.running:
                listener.start()
            running = listener.running
            return (
                _settings_status_text(assistant, status),
                gr.update(choices=status.choices, value=assistant.model),
                _docs_status_text(assistant),
                _voice_status_text(listener),
                # A refresh must not leave the button claiming the mic is off
                # while the listener is still running in this process.
                gr.update(
                    value="Stop listening" if running else "Start listening",
                    variant="stop" if running else "primary",
                ),
            )

        send.click(
            on_send,
            inputs=msg,
            outputs=[chatbot, activity, voice_audio, status, msg],
        )
        msg.submit(
            on_send,
            inputs=msg,
            outputs=[chatbot, activity, voice_audio, status, msg],
        )

        # Polls the shared snapshot so voice-initiated turns appear, and so a
        # long-running typed turn keeps refreshing without a second stream.
        timer = gr.Timer(0.6)
        timer.tick(
            on_tick,
            inputs=[],
            outputs=[
                chatbot,
                activity,
                voice_audio,
                status,
                voice_status,
                stop,
                send,
            ],
        )

        mic_button.click(on_mic, inputs=[mic_device],
                         outputs=[voice_status, mic_button])
        stop.click(on_stop, outputs=[chatbot, voice_audio, status])
        files.upload(on_ingest, inputs=files,
                     outputs=[chatbot, activity, voice_audio, status, docs_status])
        clear_docs.click(on_clear_docs,
                         outputs=[chatbot, activity, voice_audio, status, docs_status])
        reset_chat.click(on_reset,
                         outputs=[chatbot, activity, voice_audio, status, voice_status])
        apply.click(
            on_apply,
            inputs=[
                model, temperature, system_prompt, speak, voice, wake,
                wake_required, reasoning,
            ],
            outputs=[chatbot, activity, voice_audio, status, settings_status],
        )
        demo.load(
            on_load,
            outputs=[settings_status, model, docs_status, voice_status, mic_button],
        )

    return demo


def _voice_handler(assistant: Assistant):
    """Run a wake-word command to completion in the listener's thread."""

    def handle(text: str) -> None:
        assistant.submit_blocking(text)

    return handle


def build_and_launch(
    assistant: Assistant | None = None,
    *,
    server_name: str = "127.0.0.1",
    server_port: int = 7860,
    inbrowser: bool = True,
    share: bool = False,
) -> None:
    demo = build_demo(assistant)
    # Only the real launcher speaks or listens. Tests build demos too, and
    # must not open a TTS call, a microphone, or a thread behind themselves.
    app = assistant or get_assistant()

    # Warm the offline voice and recogniser in the background. Both cost a
    # few seconds the first time (model load plus one-off graph planning), and
    # the greeting queues behind the voice on the same lock -- so the first
    # thing Apache says arrives as a short clip rather than after a long
    # silence, and the first query does not stall on loading Whisper.
    from .voice import stt, tts

    for _warm in (tts.warm_up, stt.warm_up):
        threading.Thread(target=_warm, name=f"apache-{_warm.__module__}-warmup",
                         daemon=True).start()
    app.start_prompts()

    demo.launch(
        server_name=server_name,
        server_port=server_port,
        inbrowser=inbrowser,
        share=share,
        show_error=True,
        theme=_theme(),
        css=_CSS,
        js=_page_js(),
        allowed_paths=[str(assistant.workspace) if assistant else "."],
    )
