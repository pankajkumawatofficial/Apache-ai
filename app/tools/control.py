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
import shutil
import subprocess
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


#: Where a Chrome that was not installed in the usual place announces itself.
_CHROME_APP_PATH = r"SOFTWARE\Microsoft\Windows\CurrentVersion\App Paths\chrome.exe"

#: The usual places, checked in order when the registry says nothing.
_CHROME_CANDIDATES = (
    r"%ProgramFiles%\Google\Chrome\Application\chrome.exe",
    r"%ProgramFiles(x86)%\Google\Chrome\Application\chrome.exe",
    r"%LocalAppData%\Google\Chrome\Application\chrome.exe",
)


def _chrome_from_registry() -> str | None:
    """Chrome's executable from App Paths, or ``None`` on this platform."""
    if os.name != "nt":  # pragma: no cover - POSIX has no registry
        return None
    try:
        import winreg
    except ImportError:  # pragma: no cover - Windows always ships it
        return None
    for hive in (winreg.HKEY_CURRENT_USER, winreg.HKEY_LOCAL_MACHINE):
        try:
            with winreg.OpenKey(hive, _CHROME_APP_PATH) as key:
                raw, _kind = winreg.QueryValueEx(key, "")
        except OSError:
            continue
        path = Path(raw.strip('"')).expanduser() if raw else None
        if path and path.exists():
            return str(path)
    return None


def browser_for_urls() -> str | None:
    """The browser links should be opened in, or ``None`` for the default.

    Windows gives a URL to the *default* browser, which on this machine is
    Edge -- so "open YouTube" arrived in a window nobody was looking at and
    read as Apache not doing it. Chrome is the browser actually in use here,
    so a link is handed to Chrome whenever Chrome exists and to the default
    handler when it does not. ``APACHE_BROWSER`` names a different browser by
    path for anyone whose preference is neither.
    """
    override = os.environ.get("APACHE_BROWSER", "").strip()
    if override:
        return override
    for source in (_chrome_from_registry, lambda: shutil.which("google-chrome"),
                   lambda: shutil.which("chromium")):
        found = source()
        if found:
            return found
    for candidate in _CHROME_CANDIDATES:
        path = Path(os.path.expandvars(candidate))
        if path.exists():
            return str(path)
    return None


def _spawn(argv: list[str]) -> None:
    """Start *argv* detached so it outlives the thread that asked for it.

    A browser started attached to Apache's console dies with it, which from
    the outside looks like a link that opened and then vanished.
    """
    kwargs: dict = {}
    if os.name == "nt":  # pragma: no cover - POSIX uses process groups
        kwargs["creationflags"] = getattr(subprocess, "DETACHED_PROCESS", 0) | getattr(
            subprocess, "CREATE_NEW_PROCESS_GROUP", 0
        )
    subprocess.Popen(argv, **kwargs)


def _launch_url(target: str) -> None:
    """Open a URL in the preferred browser, or the default when there is none."""
    exe = browser_for_urls()
    if not exe:
        _launch(target)
        return
    _spawn([exe, target])


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
    """Open a web address in a browser.

    The scheme is added when the model omits it, and anything carrying a
    different scheme is refused rather than handed to the shell. Checking the
    scheme *before* defaulting matters: blindly prefixing ``https://`` would
    turn ``javascript:...`` into a pass.

    The browser itself is not the operating system's default: see
    :func:`browser_for_urls`.
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
        _launch_url(target)
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
