"""Environment self-check:  ``python run.py --check``

Answers "why will this not run?" before anyone has to guess. It imports each
dependency for real, distinguishes a *missing* package from a *blocked* one,
and probes Ollama, the microphone and the two network services the voice path
depends on. Nothing here imports the agent stack, so it works even when the
stack is broken.
"""

from __future__ import annotations

import importlib
import platform
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass, field

BLOCK_SIGNATURE = "Application Control policy has blocked"

#: (import name, what it is for)
DEPENDENCIES: list[tuple[str, str]] = [
    ("gradio", "web UI"),
    ("pydantic", "required by LangChain"),
    ("langchain", "agent loop"),
    ("langchain.agents", "create_agent"),
    ("langchain_ollama", "Ollama model integration"),
    ("langgraph", "agent runtime"),
    ("langgraph.checkpoint.memory", "conversation memory"),
    ("langchain_text_splitters", "document chunking"),
    ("numpy", "required by Gradio and audio handling"),
    ("sounddevice", "microphone capture"),
    ("speech_recognition", "speech to text"),
    ("edge_tts", "text to speech"),
    ("pypdf", "PDF ingestion"),
    ("ddgs", "web search tool"),
]

#: (label, url, what breaks without it)
ENDPOINTS: list[tuple[str, str, str]] = [
    ("Ollama", "http://localhost:11434/api/tags", "the model itself"),
    ("Google STT", "http://www.google.com/speech-api/v2/recognize",
     "wake word and voice input"),
    ("Edge TTS", "https://speech.platform.bing.com/", "spoken replies"),
]


@dataclass
class CheckResult:
    name: str
    ok: bool
    detail: str
    blocked: bool = False
    required: bool = True
    notes: list[str] = field(default_factory=list)


def _check_import(name: str, purpose: str) -> CheckResult:
    try:
        importlib.import_module(name)
    except ImportError as exc:
        message = " ".join(str(exc).split())
        blocked = BLOCK_SIGNATURE in message or "DLL load failed" in message
        if blocked:
            detail = (
                "BLOCKED by Application Control (WDAC) - the package is "
                "installed but its native code is not trusted by policy"
            )
        else:
            detail = f"not importable: {message[:220]}"
        return CheckResult(name, False, detail, blocked=blocked)
    except Exception as exc:  # noqa: BLE001 - any failure is a failure
        return CheckResult(
            name, False, f"raised {type(exc).__name__}: {str(exc)[:220]}"
        )
    return CheckResult(name, True, f"importable ({purpose})")


def _check_endpoint(label: str, url: str, consequence: str,
                    timeout: float = 4.0) -> CheckResult:
    request = urllib.request.Request(  # noqa: S310 - fixed known endpoints
        url, method="GET", headers={"User-Agent": "apache-selfcheck/1.0"}
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout):  # noqa: S310
            pass
    except urllib.error.HTTPError:
        # A 4xx still proves the host is reachable, which is all we need.
        return CheckResult(label, True, "reachable")
    except Exception as exc:  # noqa: BLE001
        return CheckResult(
            label, False, f"unreachable: {str(exc)[:160]}",
            notes=[f"Without it: {consequence}"],
        )
    return CheckResult(label, True, "reachable")


def _check_audio() -> CheckResult:
    try:
        import sounddevice as sd
    except Exception as exc:  # noqa: BLE001 - blocked or missing
        return CheckResult(
            "Microphone", False,
            f"audio input unavailable: {' '.join(str(exc).split())[:200]}",
            required=False,
            notes=["Without it: continuous listening and wake word are unavailable; "
                   "typing still works."],
        )

    try:
        devices = sd.query_devices()
        inputs = [d for d in devices if int(d.get("max_input_channels", 0)) > 0]
    except Exception as exc:  # noqa: BLE001
        return CheckResult("Microphone", False, f"no audio devices: {exc}",
                           required=False)

    if not inputs:
        return CheckResult("Microphone", False, "no input device found",
                           required=False,
                           notes=["Without it: typing still works."])
    default = inputs[0]
    return CheckResult(
        "Microphone", True,
        f"{default['name']} @ {int(default['default_samplerate'])} Hz",
    )


def collect() -> list[CheckResult]:
    results: list[CheckResult] = []

    results.append(
        CheckResult("Python", sys.version_info >= (3, 10),
                    f"{platform.python_version()} at {sys.executable}")
    )
    results.append(
        CheckResult("Platform", True, f"{platform.system()} {platform.release()}")
    )

    for name, purpose in DEPENDENCIES:
        results.append(_check_import(name, purpose))

    for label, url, consequence in ENDPOINTS:
        results.append(_check_endpoint(label, url, consequence))

    results.append(_check_audio())
    return results


def environment_report() -> int:
    """Print a report. Returns 0 when the app should be able to start."""
    results = collect()

    width = max(len(r.name) for r in results)
    print("=" * 72)
    print("Apache environment check")
    print("=" * 72)

    for result in results:
        if result.ok:
            mark = "  OK "
        elif result.blocked:
            mark = "BLOCK"
        else:
            mark = "FAIL "
        print(f"[{mark}] {result.name.ljust(width)}  {result.detail}")
        for note in result.notes:
            print(f"        {'':<{width}}  -> {note}")

    failures = [r for r in results if not r.ok]
    blocked = [r for r in results if r.blocked]

    print("-" * 72)
    if not failures:
        print("All checks passed. Start the app with:  python run.py")
        return 0

    print(f"{len(failures)} problem(s) found.")

    if blocked:
        print()
        print("Root cause: Windows Application Control (WDAC) is enforcing code")
        print("integrity and refuses to load UNSIGNED native extensions.")
        print("Packages affected:", ", ".join(sorted({r.name for r in blocked})))
        print()
        print("This is a machine policy, not a project problem -- no Python code")
        print("can work around it. Options:")
        print("  1. Ask IT/admin to allow your Python installation (or add an")
        print("     allow rule for extension modules from PyPI).")
        print("  2. Run the project in WSL2 or a container, where Windows code")
        print("     integrity policy does not apply to Linux binaries.")
        print("  3. Run it on a machine without this policy.")

    return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(environment_report())
