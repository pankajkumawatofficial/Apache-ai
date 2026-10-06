"""Opening things: web pages, files and folders.

Kept free of LangChain so it can be tested without the ML stack, which is the
split :mod:`app.tools.registry` documents -- substantive logic lives in a
sibling module and the registry only wraps it.

Everything here goes through :func:`_launch`, a single seam that a test can
replace so no test ever actually opens a browser or a window on the developer's
desktop.
"""

from __future__ import annotations

import os
import webbrowser
from pathlib import Path

__all__ = [
    "ControlError",
    "is_launcher_available",
    "open_file",
    "open_folder",
    "open_url",
    "resolve_target",
]

#: Types Apache refuses to open. The tools can open documents; they are not a
#: way to run programs, which is a different and much larger permission.
_EXECUTABLE_SUFFIXES = frozenset(
    {".exe", ".com", ".bat", ".cmd", ".ps1", ".psm1", ".msi", ".scr", ".vbs", ".jar"}
)


class ControlError(RuntimeError):
    """Raised when the system would not or could not perform an action."""


def _launch(target: object) -> None:
    """Hand *target* to the operating system's default handler."""
    if hasattr(os, "startfile"):  # Windows
        os.startfile(target)  # type: ignore[attr-defined]
        return
    webbrowser.open(str(target))  # POSIX / macOS fallback


def is_launcher_available() -> bool:
    """True when this platform has a default-handler launcher we can call."""
    return hasattr(os, "startfile") or bool(webbrowser)


def resolve_target(path: str, workspace: Path) -> Path:
    """Resolve a model-supplied path against the workspace.

    A relative path is looked up inside the workspace, which is the same rule
    the file tools use. An absolute path is accepted as given so the assistant
    can open something outside the workspace when asked directly.
    """
    raw = (path or "").strip().strip('"')
    if not raw:
        raise ControlError("no path given")
    candidate = Path(raw).expanduser()
    if not candidate.is_absolute():
        candidate = Path(workspace) / candidate
    try:
        return candidate.resolve()
    except OSError as exc:  # pragma: no cover - needs an unreadable path
        raise ControlError(f"cannot resolve {raw!r}: {exc}") from exc


def open_url(url: str) -> str:
    """Open a web address in the default browser.

    The scheme is added when the model omits it, and anything carrying a
    different scheme is refused rather than handed to the shell. Checking the
    scheme *before* defaulting matters: blindly prefixing ``https://`` would
    turn ``javascript:...`` into a pass.
    """
    target = (url or "").strip()
    if not target:
        raise ControlError("no URL given")

    # What precedes the first "/" is either "scheme:" or a bare host.
    head = target.split("/", 1)[0]
    if ":" in head:
        scheme = head.split(":", 1)[0].lower()
        if scheme not in ("http", "https"):
            raise ControlError(f"only http and https can be opened, got {scheme!r}")
    else:
        target = "https://" + target

    try:
        _launch(target)
    except ControlError:
        raise
    except Exception as exc:  # noqa: BLE001 - the OS reports these unevenly
        raise ControlError(f"could not open {target}: {exc}") from exc
    return f"Opened {target} in the browser."


def open_file(path: str, workspace: Path) -> str:
    """Open a file with whichever program the system associates with it.

    Directories open in the file manager. Executables are refused: these
    tools are for reaching things, not for running them.
    """
    target = resolve_target(path, workspace)
    if target.is_dir():
        return open_folder(str(target), workspace)
    if not target.exists():
        raise ControlError(f"no such file: {target}")
    if target.suffix.lower() in _EXECUTABLE_SUFFIXES:
        raise ControlError(
            f"refusing to run {target.name}; ask the user to start programs themselves"
        )

    try:
        _launch(str(target))
    except ControlError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise ControlError(f"could not open {target}: {exc}") from exc
    return f"Opened {target}."


def open_folder(path: str, workspace: Path) -> str:
    """Open a folder in the system file manager."""
    target = resolve_target(path, workspace)
    if not target.exists():
        raise ControlError(f"no such folder: {target}")
    if not target.is_dir():
        raise ControlError(f"not a folder: {target}")

    try:
        _launch(str(target))
    except ControlError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise ControlError(f"could not open {target}: {exc}") from exc

    return f"Opened {target}."
