"""Paths and tunables for Apache.

Every value can be overridden with an environment variable of the same name
upper-cased and prefixed with ``APACHE_``, e.g. ``APACHE_MODEL=qwen3:8b``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
UPLOADS = DATA / "uploads"      # documents you upload for RAG
WORKSPACE = DATA / "workspace"  # the only directory file tools may touch
AUDIO = DATA / "audio"          # generated TTS clips
MODELS = ROOT / "models"        # speech models, downloaded once, never tracked

for _p in (DATA, UPLOADS, WORKSPACE, AUDIO):
    _p.mkdir(parents=True, exist_ok=True)


def _env(name: str, default: str) -> str:
    return os.environ.get(f"APACHE_{name}", default)


def _env_int(name: str, default: int) -> int:
    try:
        return int(_env(name, str(default)))
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(_env(name, str(default)))
    except ValueError:
        return default


def _env_bool(name: str, default: bool) -> bool:
    raw = _env(name, "1" if default else "0").strip().lower()
    if raw in {"1", "true", "yes", "on"}:
        return True
    if raw in {"0", "false", "no", "off"}:
        return False
    return default


# Google's speech API hears "Apache" as several plausible phrases; accept them
# all so a mis-transcription does not silently drop the command.
DEFAULT_WAKE_ALIASES: list[str] = ["apache", "a patch", "a path", "app patch"]

DEFAULT_SYSTEM_PROMPT = """\
You are Apache, a local voice assistant running on the user's own machine.
You are their friend first and their assistant second: warm, easygoing and
human. Talk the way one person talks to another, never like a help desk.

Communication rules — these matter because your replies are spoken aloud:
- Be conversational. One or two short, natural sentences for a spoken answer
  whenever possible, the way you would actually say them out loud.
- Sound like a person, not a status readout: contractions ("it's", "you're"),
  and a little warmth. Skip the formal openers — no "Certainly!", no "As an
  AI", no restating the question back at them.
- Never emit Markdown, code fences, bullet lists or tables in a spoken reply.
  Use plain prose. If code must be shown, put it on its own line and keep it short.
- Never emit LaTeX or escape sequences: no $...$, no backslashes, no
  \times or \frac. Write arithmetic in plain words instead
  ("1739 times 42"), because the reply is read aloud by a synthesiser.
- If a tool gives you a number or fact, state the result directly, not the reasoning.
- If you do not know something and no tool can help, say so plainly.

Tool guidance:
- Use the calculator for anything arithmetic beyond trivial mental math.
- Use run_python for computation, data processing or plotting tasks.
- Use search_documents whenever the user asks about their uploaded documents.
- Use web_search only for current events or facts you are unsure about.
- File tools are confined to a workspace directory; never claim to read other paths.
"""


# --------------------------------------------------------------------------
# Settings
# --------------------------------------------------------------------------
@dataclass
class Settings:
    # --- model -----------------------------------------------------------
    # qwen3:1.7b rather than 8b: this machine has no GPU runtime installed
    # (Ollama reports 100% CPU) where decode measured ~2.3 tok/s, so an 8b
    # tool turn took 29 s. The 1.7b model is roughly four times quicker and
    # still handles the tool loop; the model dropdown offers 8b back with one
    # click whenever answers matter more than latency.
    model: str = _env("MODEL", "qwen3:1.7b")
    embedding_model: str = _env("EMBEDDING_MODEL", "nomic-embed-text")
    ollama_base_url: str = _env("OLLAMA_BASE_URL", "http://localhost:11434")
    # Ollama unloads a model after keep_alive and reloads it on the next
    # request. Reloading measured 25 s of pure stall. "-1" keeps the weights
    # resident until Ollama itself is restarted.
    ollama_keep_alive: str = _env("OLLAMA_KEEP_ALIVE", "-1")
    temperature: float = _env_float("TEMPERATURE", 0.2)
    request_timeout: int = _env_int("REQUEST_TIMEOUT", 180)
    recursion_limit: int = _env_int("RECURSION_LIMIT", 24)
    # Context window handed to Ollama. RAG and long tool traces need room.
    num_ctx: int = _env_int("NUM_CTX", 8192)
    # Chain-of-thought before the answer. qwen3-family models turn this ON by
    # default when Ollama is not told otherwise, and on CPU-only hardware that
    # measured 16x slower (83.5s vs 5.0s for "say hello in five words").
    # Spoken replies are one or two sentences, so the reasoning buys little.
    reasoning: bool = _env_bool("REASONING", False)
    system_prompt: str = _env("SYSTEM_PROMPT", DEFAULT_SYSTEM_PROMPT)

    # --- voice -----------------------------------------------------------
    wake_word: str = _env("WAKE_WORD", "apache")
    # Extra phrases accepted as the wake word (Google's STT mis-hears often).
    wake_aliases: list[str] = field(
        default_factory=lambda: list(DEFAULT_WAKE_ALIASES)
    )
    sample_rate: int = 16000
    frame_ms: int = 30
    # Silence that ends an utterance. This decides whether a thinking pause
    # inside one question counts as a breath or as the end of the question --
    # too short and the query is chopped, the fragment is sent to the agent,
    # and whatever came after is discarded as stale. 1.5 s is past the point
    # where a person is still gathering their next clause, and costs only the
    # wait before transcription starts.
    silence_end_ms: int = _env_int("SILENCE_END_MS", 1500)
    utterance_min_ms: int = 350
    utterance_max_s: float = 20.0
    # Noise-gate: threshold = max(floor, noise_floor * multiplier).
    #
    # The multiplier is how far above room noise a signal must sit before it
    # counts as speech. Measured live on this machine: room level 0.0073, gate
    # 0.0283 -- 3.9x ambient, more than an ordinary voice sustains for a whole
    # sentence. Speech therefore only partly cleared the gate, the sentence
    # fragmented into pieces, the first reached the agent as "Apache RR" and
    # three later pieces were discarded as too short. 2.0 is the usual
    # noise-gate ratio: speech clears it, and room noise still has to double
    # to trip it.
    vad_min_energy: float = _env_float("VAD_MIN_ENERGY", 0.012)
    vad_multiplier: float = _env_float("VAD_MULTIPLIER", 2.0)
    # Context kept around the voiced span. Google's recogniser segments on
    # silence at both ends; an utterance cut hard at the first and last voiced
    # sample comes back as an empty string even when every word is audible.
    vad_pre_roll_ms: int = _env_int("VAD_PRE_ROLL_MS", 150)
    vad_keep_tail_ms: int = _env_int("VAD_KEEP_TAIL_MS", 300)
    stt_language: str = _env("STT_LANGUAGE", "en-US")
    tts_voice: str = _env("TTS_VOICE", "en-US-AndrewMultilingualNeural")
    # The pace is left at the model's own declared default (its .onnx.json
    # says length_scale: 1) -- a synthesiser is never made more human by
    # being rushed, and Piper already speaks briskly. Set APACHE_TTS_RATE=+10%
    # for quicker, APACHE_TTS_RATE=-10% for a more considered delivery.
    tts_rate: str = _env("TTS_RATE", "+0%")
    # How much prosodic variation the synthesiser is allowed. Higher sounds
    # more alive and less like a machine reading text; too high adds breath
    # noise. Piper's own default is 0.667.
    tts_noise_scale: float = _env_float("TTS_NOISE_SCALE", 0.8)
    # Mute the mic while we are talking so the assistant does not hear itself.
    playback_mute_s: float = _env_float("PLAYBACK_MUTE_S", 2.0)
    # Open the microphone as soon as the app starts, so "Apache" works
    # without a first click on the mic button. 0 starts it stopped.
    mic_autostart: bool = _env_bool("MIC_AUTOSTART", True)
    # Whether Apache insists on being addressed by name. False -- the
    # default -- makes every complete utterance a command and the wake word a
    # convenience rather than a gate. APACHE_WAKE_REQUIRED=1 goes back to
    # waiting for "Apache", which is what a noisy room wants.
    wake_required: bool = _env_bool("WAKE_REQUIRED", False)

    # --- offline speech ---------------------------------------------------
    # Offline is the default. Every utterance over Google's endpoint and every
    # reply through edge-tts costs a network round trip, and recognition is
    # throttled after a burst -- the two things that made voice feel slow and
    # intermittently silent. APACHE_OFFLINE=0 goes back to the online engines.
    offline: bool = _env_bool("OFFLINE", True)
    # Whisper size. "base.en" is ~75 MB and several times real time on this
    # CPU; "small.en" is more accurate and roughly twice as slow.
    whisper_model: str = _env("WHISPER_MODEL", "base.en")
    whisper_dir: Path = Path(_env("WHISPER_DIR", str(MODELS / "whisper")))
    # A Piper .onnx file with its .onnx.json sitting beside it. The default is
    # the medium Ryan; en_US-ryan-high sits beside it as the larger of the two
    # on disk, and the Voice tab switches between them -- or any other
    # downloaded voice -- without a restart.
    piper_voice: str = _env(
        "PIPER_VOICE", str(MODELS / "piper" / "en_US-ryan-medium.onnx")
    )

    # --- presence ---------------------------------------------------------
    # Spoken once when Apache comes up, before anyone has asked anything.
    greeting: str = _env("GREETING", "Hello Boss, what will we do today.")
    # Spoken when this many seconds pass with no input at all. The clock
    # restarts on any interaction and is held back while a reply is in
    # flight, so Apache never talks over its own answer. 0 disables it.
    idle_prompt_s: int = _env_int("IDLE_PROMPT_S", 60)
    idle_prompt: str = _env("IDLE_PROMPT", "Sir! Are you here?")

    # --- agent / tools ---------------------------------------------------
    sandbox_timeout_s: int = _env_int("SANDBOX_TIMEOUT", 20)
    sandbox_max_output: int = _env_int("SANDBOX_MAX_OUTPUT", 8_000)
    web_search_results: int = _env_int("WEB_SEARCH_RESULTS", 5)
    retriever_k: int = _env_int("RETRIEVER_K", 5)
    chunk_size: int = _env_int("CHUNK_SIZE", 900)
    chunk_overlap: int = _env_int("CHUNK_OVERLAP", 150)
    # Trim the visible transcript so the UI never grows without bound.
    max_display_messages: int = 200


settings = Settings()
