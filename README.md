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
   ollama pull qwen3:8b          # good all-rounder, reliable tool calls
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

---

## Verify before you start

```bash
python run.py --check
```

This imports every dependency for real and probes Ollama, the microphone and
the two network services the voice path needs. It distinguishes a **missing**
package from a **blocked** one — on a machine with Windows Application Control
(WDAC) enforcing code integrity, unsigned native extensions such as
`pydantic_core`, `numpy` and `cffi` are refused at load time and the report
says so explicitly, with the options available to you.

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
| `APACHE_MODEL` | `qwen3:8b` | Ollama chat model |
| `APACHE_EMBEDDING_MODEL` | `nomic-embed-text` | RAG embedding model |
| `APACHE_OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama endpoint |
| `APACHE_TEMPERATURE` | `0.2` | Sampling temperature |
| `APACHE_NUM_CTX` | `8192` | Context window passed to Ollama |
| `APACHE_WAKE_WORD` | `apache` | Word that activates the microphone |
| `APACHE_TTS_VOICE` | `en-US-AndrewMultilingualNeural` | edge-tts voice |
| `APACHE_STT_LANGUAGE` | `en-US` | Google speech recognition language |
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
    listener.py            continuous microphone + wake-word worker
data/
  uploads/  workspace/  audio/     created automatically at first run
tests/
  test_pure.py             calculator, files, VAD, wake word, sandbox
  test_rag.py              chunking, TF-IDF, document store
```

### Running the tests

```bash
python -m tests.test_pure
python -m tests.test_rag
```

Both suites exercise only standard-library code plus Apache's own pure
modules, so they pass even where the ML and audio stack cannot be installed.

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
transcribe itself.

**Retrieval degrades rather than breaks.** If `nomic-embed-text` is not
pulled, or Ollama refuses the request, the store falls back to an in-process
TF-IDF index and labels the mode in the UI.

**Chunking** uses LangChain's `RecursiveCharacterTextSplitter` when
`langchain-text-splitters` is installed, and a paragraph/sentence splitter
otherwise.

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

### Everything local except three network calls

Ollama, your documents and your files stay on your machine. Three things go
out: Google speech recognition (microphone audio), Edge TTS (the text of your
reply), and DuckDuckGo when `web_search` is called. Turn spoken replies off in
the **Voice** tab if you would rather not send reply text for synthesis.

### Windows Application Control (WDAC)

If `python run.py --check` reports packages as `BLOCKED`, Windows is refusing
to load unsigned native extensions from your Python installation. That is a
machine policy, not a bug in this project — no Python code can work around it.
Your options are to have an administrator allow your Python installation, run
the project under WSL2 or a container where the policy does not apply to Linux
binaries, or run it on a machine without the policy.

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| `Cannot reach Ollama` | Start it: `ollama serve` (or launch the Ollama app) |
| `model '…' not found` | `ollama pull <model>` |
| Tools never fire | Use a tool-calling model; try `qwen3:8b` |
| Documents answer weakly | `ollama pull nomic-embed-text`, then re-upload |
| No wake word response | Check the **Voice** panel state; the mic may be muted, stopped, or have hit repeated STT failures |
| `Speech recognition failed` repeatedly | The listener stops after 5 consecutive failures to avoid spinning; restart it with the mic button |
| No spoken replies | Check the network, or untick **Speak replies aloud** |
| `pip.exe` is blocked | Use `python -m pip` instead |
| Native imports fail | Run `python run.py --check` |
