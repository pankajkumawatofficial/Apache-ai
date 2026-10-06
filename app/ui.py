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

import time
from typing import Any, Iterator

import gradio as gr

from .config import DEFAULT_WAKE_ALIASES, settings
from .core import Assistant, Snapshot, get_assistant
from .llm import check_ollama
from .voice.listener import WakeListener, list_input_devices

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
# Apache
**Local voice assistant** — LangChain agent · Ollama · Gradio · edge-tts
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
    messages, activity, audio, status = snapshot
    return messages, activity, audio, status


#: Spinner frames for a bubble the model has not answered yet. The frame is
#: derived from wall time, so the ticker needs no state of its own and two
#: viewers of the same page still see the same animation.
_THINKING_FRAMES = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"

#: Applied to the pending bubble. Kept gentle: this is a status cue, not a
#: distraction, and it must stay readable in a dark terminal-style theme.
_CSS = """
.apache-thinking {
    display: inline-block;
    animation: apache-think 1.15s ease-in-out infinite;
}
@keyframes apache-think {
    0%, 100% { opacity: .30; letter-spacing: .02em; }
    50%      { opacity: 1;   letter-spacing: .12em; }
}
"""


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
        gr.Markdown(_HEADER)

        audio_state = gr.State(None)

        with gr.Row():
            # ---------------- chat column ----------------
            with gr.Column(scale=3, min_width=420):
                chatbot = gr.Chatbot(
                    value=assistant.snapshot()[0],
                    label="Conversation",
                    height=480,
                    layout="bubble",
                    placeholder="Ask something, or say “Apache, …”",
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
                    format="mp3",
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
                            choices=TTS_VOICES,
                            value=assistant.voice,
                            label="Voice",
                            allow_custom_value=True,
                        )
                        wake = gr.Textbox(
                            value=assistant.settings.wake_word,
                            label="Wake word",
                            info="Say this first to ask a question out loud.",
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
            messages, act, audio, stat = assistant.snapshot()
            if not text:
                yield messages, act, gr.skip(), stat, ""
                return

            seen = audio
            for snapshot in assistant.submit(text):
                messages, act, audio, stat = snapshot
                audio_out = audio if audio != seen else gr.skip()
                seen = audio
                yield messages, act, audio_out, stat, ""

        def on_tick(last_audio: Any) -> tuple:
            messages, act, audio, stat = _render(assistant.snapshot())
            # The timer is the only thing running while the model is silent,
            # so it is what drives the animation in a pending bubble.
            messages = _with_thinking(messages, assistant.busy)
            audio_out = audio if audio != last_audio else gr.skip()
            busy = assistant.busy
            return (
                messages,
                act,
                audio_out,
                stat,
                _voice_status_text(listener),
                audio,
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
            reasoning_value: bool,
        ) -> tuple:
            word = (wake_value or "").strip().lower() or settings.wake_word
            settings.wake_word = word
            settings.wake_aliases = [word] + [
                alias for alias in DEFAULT_WAKE_ALIASES if alias != word
            ]
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
            inputs=audio_state,
            outputs=[
                chatbot,
                activity,
                voice_audio,
                status,
                voice_status,
                audio_state,
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
            inputs=[model, temperature, system_prompt, speak, voice, wake, reasoning],
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
    # Only the real launcher speaks. Tests build demos too, and must not open
    # a network TTS call or leave a thread running behind them.
    (assistant or get_assistant()).start_prompts()
    demo.launch(
        server_name=server_name,
        server_port=server_port,
        inbrowser=inbrowser,
        share=share,
        show_error=True,
        css=_CSS,
        allowed_paths=[str(assistant.workspace) if assistant else "."],
    )
