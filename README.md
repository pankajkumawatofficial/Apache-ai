# Apache

A local, voice-first AI assistant: a **LangChain** agent running on **Ollama** models, with a **Gradio** web UI, continuous wake-word listening and spoken replies.

Say **"Apache, …"** out loud, or just type. It answers with a voice.

---

## What it does

| Capability | How |
|---|---|
| **Tool-calling agent** | `langchain.agents.create_agent` with a calculator, Python sandbox, file tools, document search, web search and a clock |
| **Conversation memory** | LangGraph `MemorySaver` checkpointer, one thread per session |
| **Chat with your documents (RAG)** | Upload text/PDF → chunked → `OllamaEmbeddings` + cosine retrieval, with an automatic TF-IDF fallback when no embedding model is pulled |
| **Code interpreter sandbox** | `subprocess` running `python -I` with a timeout, no stdin and bounded output |
| **Wake word** | Continuous mic → energy VAD → Google speech-to-text → strip `"Apache"` → run the agent |
| **Voice output** | Microsoft Edge neural voices via `edge-tts`, autoplayed in the browser |
| **Streaming** | Token-by-token replies plus a live tool-activity panel |

---

## Prerequisites

1. **Python 3.10 or newer** (developed and tested on 3.14).
2. **[Ollama](https://ollama.com/download)** installed and running.
3. A **tool-calling** model pulled locally. Small, capable choices:

   ```bash
   ollama pull qwen3:1.7b        # the default: quick enough to feel live
   ollama pull qwen3:8b          # sturdier tool calls, ~4x slower here
   ollama pull llama3.2:3b       # lighter / faster
   ollama pull gpt-oss:20b       # stronger, needs more RAM
   ```

4. An **embedding model** for document Q&A (optional — RAG falls back to TF-IDF without it):

   ```bash
   ollama pull nomic-embed-text
   ```

> **Why tool calling matters:** a model that cannot emit tool calls will never
> use the calculator, sandbox or file tools. If tools never fire, try a
> different model.

---

## Install

Always use `python -m pip`. On machines where the `pip` launcher shim is
blocked by application control policy, `pip.exe` fails while `python -m pip`
works fine.

```bash
python -m pip install -r requirements.txt
```

### Fetch the speech models (once)

Voice runs offline by default, so the two engines need their models on disk
once per machine. Whisper downloads itself into `models/whisper` the first time
it is asked for; the Piper voice is a direct download:

```powershell
New-Item -ItemType Directory -Force models\piper | Out-Null
$base = "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/ryan/medium"
Invoke-WebRequest "$base/en_US-ryan-medium.onnx"      -OutFile models\piper\en_US-ryan-medium.onnx
Invoke-WebRequest "$base/en_US-ryan-medium.onnx.json" -OutFile models\piper\en_US-ryan-medium.onnx.json
```

`models/` is gitignored. If the voice is missing, Apache still runs and still
answers in the chat — `run.py --check` reports it as a line of its own, and the
usual symptom otherwise is an assistant that never speaks. Set
`APACHE_OFFLINE=0` to use Google and edge-tts instead, which need no models.

---

## Verify before you start

```bash
python run.py --check
```

This imports every dependency for real and probes Ollama, the microphone, the
local speech models, and the network endpoints the online fallback depends on.
It distinguishes a **missing**
package from a **blocked** one — if Windows Smart App Control is enforcing,
the unsigned native extensions that PyPI wheels ship (`pydantic_core`,
`numpy`, `_cffi_backend`) are refused at load time, and the report says so
explicitly, naming the policy and listing the options available to you.

Expected output when everything is fine ends with:

```
All checks passed. Start the app with:  python run.py
```

---

## Run

```bash
python run.py                 # starts on http://127.0.0.1:7860 and opens a browser
python run.py --no-browser    # headless
python run.py --port 8080     # different port
```

Then, in the right-hand column:

1. **Documents** tab → upload files → they are indexed for `search_documents`.
2. Click **Start listening** and say *"Apache, what's the square root of 216?"*
3. **Model** tab → pick your Ollama model, adjust temperature, apply.
4. **Voice** tab → choose a voice, change the wake word, or turn spoken replies off.

Typed messages work identically to spoken ones — the wake word is only needed
for microphone input.

---

## Configuration

Every setting can be overridden with an `APACHE_`-prefixed environment variable
(see `app/config.py` for the full list). The common ones:

| Variable | Default | Meaning |
|---|---|---|
| `APACHE_MODEL` | `qwen3:1.7b` | Ollama chat model |
| `APACHE_OFFLINE` | `1` | Local speech engines; `0` switches to Google + edge-tts |
| `APACHE_WHISPER_MODEL` | `base.en` | Local recogniser; `small.en` is more accurate, about twice as slow |
| `APACHE_PIPER_VOICE` | `models/piper/en_US-ryan-medium.onnx` | Local voice file for spoken replies |
| `APACHE_MIC_AUTOSTART` | `1` | Open the microphone as soon as the page opens |
| `APACHE_EMBEDDING_MODEL` | `nomic-embed-text` | RAG embedding model |
| `APACHE_OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama endpoint |
| `APACHE_OLLAMA_KEEP_ALIVE` | `-1` | How long Ollama keeps the model resident; `-1` is forever |
| `APACHE_GREETING` | `Hello boss! What we will do today.` | Spoken once when the app starts |
| `APACHE_IDLE_PROMPT_S` | `60` | Seconds of silence before a check-in; `0` disables it |
| `APACHE_IDLE_PROMPT` | `Sir! Are you here?` | What Apache says when you have been quiet |
| `APACHE_TEMPERATURE` | `0.2` | Sampling temperature |
| `APACHE_NUM_CTX` | `8192` | Context window passed to Ollama |
| `APACHE_WAKE_WORD` | `apache` | Word that activates the microphone |
| `APACHE_TTS_VOICE` | `en-US-AndrewMultilingualNeural` | edge-tts voice |
| `APACHE_STT_LANGUAGE` | `en-US` | Google speech recognition language |
| `APACHE_SILENCE_END_MS` | `1500` | Quiet that ends a spoken query (see below) |
| `APACHE_VAD_MULTIPLIER` | `2.0` | How far above room noise speech must sit (see below) |
| `APACHE_VAD_MIN_ENERGY` | `0.012` | Absolute floor, so a silent room never trips |
| `APACHE_VAD_PRE_ROLL_MS` | `150` | Room tone kept before the first word |
| `APACHE_VAD_KEEP_TAIL_MS` | `300` | Silence kept after the last word |
| `APACHE_SANDBOX_TIMEOUT` | `20` | Seconds a `run_python` call may run |
| `APACHE_SYSTEM_PROMPT` | *(built in)* | Overrides the system prompt |

The wake word also accepts a few aliases (`a patch`, `a path`, `app patch`)
because speech recognition regularly mis-hears it.

---

## Project layout

```
run.py                     entry point  (also: python run.py --check)
app/
  config.py                paths and tunables (APACHE_* env overrides)
  check.py                 environment self-check
  llm.py                   ChatOllama factory + Ollama health probe
  agent.py                 create_agent wiring, streaming, checkpointing
  core.py                  single source of truth shared by text + voice
  rag.py                   ingestion, chunking, embeddings, TF-IDF fallback
  ui.py                    Gradio Blocks front end
  tools/
    registry.py            the @tool wrappers handed to the agent
    calculator.py          AST-based arithmetic evaluator (no eval)
    sandbox.py             python -I subprocess runner
    files.py               workspace-confined read/write/list
    websearch.py           ddgs wrapper
  voice/
    vad.py                 energy voice-activity detection
    wake.py                wake-word matching + speech text cleanup
    stt.py                 Google speech recognition
    tts.py                 edge-tts synthesis
    listener.py            three-stage capture: callback -> VAD -> pipeline
data/
  uploads/  workspace/  audio/     created automatically at first run
tests/
  test_pure.py             calculator, files, VAD, wake word, sandbox
  test_rag.py              chunking, TF-IDF, document store
  test_stack.py            the assembled app: agent, tools, RAG, Gradio UI
```

### Running the tests

```bash
python -m tests.test_pure
python -m tests.test_rag
python -m tests.test_stack
```

The first two exercise only standard-library code plus Apache's own pure
modules, so they pass even where the ML and audio stack cannot be installed.

`tests/test_stack.py` covers the layer those two cannot reach — it builds the
tool registry, ingests and searches a document, constructs the Gradio Blocks
and drives a real agent turn. It **skips** (rather than fails) if the ML stack
will not load, so it stays runnable everywhere.

It exists because of a bug the other two suites were structurally blind to: a
cleanup deleted `agent.supported_tool_names` while `ui._tools_status_text`
still imported it. The broad `except` meant nothing crashed — the UI silently
rendered "Tools unavailable" and both existing suites passed throughout.

---

## Design notes

**One source of truth.** Typed messages and wake-word commands both drive
`Assistant.submit(...)`. The UI never keeps its own copy of the conversation —
it renders `Assistant.snapshot()`. A background `gr.Timer` polls that snapshot
so voice-initiated turns appear without a second streaming path.

**Streaming.** The agent is run with `stream_mode=["updates", "messages"]`:
message chunks from the `model` node stream tokens as they arrive, while
`updates` from the `model` and `tools` nodes supply tool calls and their
results. Node names come from LangChain 1.x's factory (`"model"`, `"tools"`).

**Echo suppression.** The microphone mutes while a reply is playing, for the
estimated duration of the speech plus a small buffer, so Apache does not
transcribe itself. While muted the VAD is skipped entirely rather than merely
having its output discarded: its noise floor adapts from quiet frames, so
feeding it loud TTS would walk the gate up until ordinary speech stopped
crossing it — capture would get worse with every reply.

**Capture never waits on the model.** The listener runs three stages on three
threads: the PortAudio callback only copies samples, a capture worker only runs
the VAD, and a separate pipeline worker does recognition and the agent turn.
The pipeline is the stage that can block for a whole turn, which is exactly why
it is not the one draining the microphone. When all three were one thread, a
single spoken question stalled audio consumption, the block buffer overflowed,
and anything said while Apache answered was silently lost.

**Silent loss is reported.** PortAudio overrun flags, buffer overflows and
utterances discarded because Apache was busy are all counted and shown in the
voice panel, alongside a live input level against the current VAD gate. "It is
not hearing me" becomes a readable diagnosis instead of a guess.

**The gate sits above the room, not above your voice.** The speech gate is the
noise floor times a multiplier, and at 3.5 it measured `0.0283` against a room
level of `0.0073` -- 3.9x ambient, which an ordinary voice does not sustain
for a whole sentence. Speech therefore only partly cleared it: the sentence
fragmented at each dip, the opening clause reached the agent transcribed as
"Apache RR", and three later pieces were discarded as too short. That is what
"it is not taking my full input" looks like from inside the pipeline. The
default is now `APACHE_VAD_MULTIPLIER=2.0`, the usual noise-gate ratio --
speech clears it while room noise still has to double to trip it. The
discarded-utterance counter is what makes this diagnosable at all: before it
existed, a clipped word and a dead microphone produced exactly the same
silence.

**A query ends on silence, not on a breath.** The endpoint decides whether a
pause mid-question is a breath or the end of the question. At 700 ms it fired
mid-thought: the first clause went to the agent as though it were the whole
question, and the remainder was dropped as stale. Worse, speech after the
split was measured against a gate that had not settled from the previous
utterance, so the tail often came in under the minimum-duration floor and
vanished without any error. The default is now 1500 ms, which holds a question
together through a 1.6 s thinking pause and still ends promptly once you have
actually stopped. Each utterance also carries room tone at both ends
(`APACHE_VAD_PRE_ROLL_MS`, `APACHE_VAD_KEEP_TAIL_MS`): the recogniser
segments on silence, and a clip cut hard at the first and last voiced sample
comes back as an empty transcript even when every word is audible.

**Stoppable turns.** Every agent event is checked against a stop flag, and the
agent yields per token, so **Stop** takes effect in roughly one token. A
stopped turn skips speech synthesis, leaves whatever partial text arrived and
releases the busy flag immediately.

**Retrieval degrades rather than breaks.** If `nomic-embed-text` is not
pulled, or Ollama refuses the request, the store falls back to an in-process
TF-IDF index and labels the mode in the UI.

**Chunking** uses LangChain's `RecursiveCharacterTextSplitter` when
`langchain-text-splitters` is installed, and a paragraph/sentence splitter
otherwise.

---

## Example session

Verified end-to-end on this machine — `qwen3:1.7b` on a CPU-only laptop
(i5-1235U, Intel UHD, 16 GB RAM):

```
You   What is 1739 * 42? Please use the calculator tool rather than
       doing it in your head.

       → calculator(expression=1739 * 42)
       calculator: 73038

Apache The result of 1739 multiplied by 42 is 73,038.
```

The activity line is emitted by the `tools` node, not by the model repeating
itself — the arithmetic comes back from the AST evaluator in
`app/tools/calculator.py`, which never runs `eval`.

---

## Honest limitations

### The code sandbox is *not* a security sandbox

`run_python` runs a real interpreter with a real filesystem and real
site-packages, on your machine, as you. It uses `-I` (isolated mode, implying
`-E -P -s`), a wall-clock timeout, no stdin and bounded output — that stops a
runaway or malformed snippet from wedging the UI. It does **not** stop code
from reading or writing files, spawning processes or using the network.

Only run this on a machine you are willing to let the model execute code on.

### File tools are confined; the sandbox is not

`read_file` / `write_file` / `list_files` reject anything that resolves
outside `data/workspace`, including `..`, absolute paths and symlinks. The
Python sandbox deliberately has no such restriction.

### Offline by default

Voice runs locally: **faster-whisper** transcribes the microphone and **piper**
synthesises replies from a model in `models/`. Nothing in the voice path leaves
the machine, there is no round trip per utterance or per reply, and recognition
is not throttled after a burst of queries — all three of which made voice feel
slow or intermittently silent when it depended on the network.

`APACHE_OFFLINE=0` puts Google speech recognition and Edge TTS back in charge.
They are a little more accurate on noisy input, and they need the network; the
engine is chosen once at startup, so switching means a restart.

What still goes out in the default configuration: **DuckDuckGo**, and only when
the model chooses to call `web_search`. Ollama, your documents and your files
stay on your machine.

### No barge-in: one spoken command at a time

While Apache is recognising or answering, the microphone keeps running — but
speech captured in that window is **discarded**, not queued. It would be worse
to execute a command two minutes after it was spoken, so the voice panel counts
the ignored utterances and shows them.

Typed input has the same rule: a second message while a turn is running is
refused rather than queued. Press **Stop** first if you want to change course.

### It is only as fast as your hardware

Apache does no inference of its own — Ollama does, and Ollama reports
`PROCESSOR: 100% CPU` on a machine without a supported GPU. Measured on the
machine this was developed on (i5-1235U), wall clock from submitting a request
to a complete answer:

| Turn | `qwen3:1.7b` (default) | `qwen3:8b` |
|---|---|---|
| Plain question, no tools | 1.4 s | 2.4 s |
| One tool call | 7.5 s | 29.1 s |
| First turn after a pull | +1.4 GB load | +6.7 GB load |

The first turn after `ollama pull` is slower than the rest because the weights
are still on disk. `APACHE_OLLAMA_KEEP_ALIVE=-1` stops Ollama unloading the
model between turns: reloading costs about 25 s, which is far worse than
holding 1.4 GB in memory.

That is the model, not the app: token streaming, the activity panel and the
voice path all work normally, you just wait longer for tokens.

If that is too slow, the lever is the model, not the configuration:

| Model | Size | On this hardware |
|---|---|---|
| `qwen3:1.7b` | 1.4 GB | default; ~7 s / tool turn |
| `qwen3:4b` | 2.6 GB | quicker than 8b, still reliable at tool calls |
| `qwen3:8b` | 5.2 GB | sturdiest tool calls; ~29 s / tool turn |

Set `APACHE_MODEL` or pick a different model in the **Model** tab — no restart
required. A machine with an NVIDIA GPU lands in a different category entirely.

### Windows Smart App Control

If `python run.py --check` reports packages as `BLOCKED`, **Smart App Control**
is enforcing: it allows code that is cloud-rated as safe or signed by a trusted
CA, and blocks everything else — which includes the unsigned extension modules
PyPI wheels ship. Every package is installed and intact; policy is refusing to
load it. This is a machine setting, not a bug in this project, and there is no
per-app exception.

Your options:

1. **Turn it off** — Windows Security → App & browser control → Smart App
   Control → Off. Needs admin rights. Recent Windows updates let you turn it
   back on afterwards, so this is no longer a one-way decision.
2. **Run under WSL2 or a container** — Smart App Control governs Windows
   images, not Linux binaries, so the whole stack runs untouched.
3. **Run it on a machine** where the policy is not enforcing.

Turning it off directly in the registry (`VerifiedAndReputablePolicyState`)
is documented by Microsoft for testing only and can leave the machine blocking
its own system applications — use the Windows Security toggle instead.

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| `Cannot reach Ollama` | Start it: `ollama serve` (or launch the Ollama app) |
| `model '…' not found` | `ollama pull <model>` |
| Tools never fire | Use a tool-calling model; `qwen3:8b` never skips them, the default `qwen3:1.7b` occasionally does |
| Documents answer weakly | `ollama pull nomic-embed-text`, then re-upload |
| No wake word response | Check the **Voice** panel state; the mic may be muted, stopped, or have hit repeated STT failures |
| `Speech recognition failed` repeatedly | The listener stops after 5 consecutive failures to avoid spinning; restart it with the mic button |
| No spoken replies | Untick **Speak replies aloud** only if you meant to. Otherwise run `run.py --check` and read the **Speech models** line — a missing Piper voice fails silently, and Apache looks perfectly healthy while saying nothing |
| Microphone never starts by itself | `APACHE_MIC_AUTOSTART=1` is the default; the **Voice** panel reports a device that refused to open |
| Voice input is slow or stops after a burst | That was Google throttling recognitions. Offline mode (`APACHE_OFFLINE=1`) has no endpoint to throttle |
| `pip.exe` is blocked | Use `python -m pip` instead |
| Native imports fail / "Application Control has blocked this file" | Run `python run.py --check`; likely Smart App Control — see *Windows Smart App Control* above |
