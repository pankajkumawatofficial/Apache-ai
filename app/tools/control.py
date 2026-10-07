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
import re
import shutil
import subprocess
import urllib.parse
import webbrowser
from pathlib import Path

from .websearch import find_video

__all__ = [
    "ControlError",
    "is_launcher_available",
    "open_app",
    "open_file",
    "open_folder",
    "open_url",
    "play_media",
    "resolve_target",
]

#: Types Apache refuses to open. The tools can open documents; they are not a
#: way to run programs, which is a different and much larger permission.
_EXECUTABLE_SUFFIXES = frozenset(
    {".exe", ".com", ".bat", ".cmd", ".ps1", ".psm1", ".msi", ".scr", ".vbs", ".jar"}
)

#: Where a program announces itself. One registry read rather than a walk of
#: the disk, and the place classic installers register under.
_APP_PATHS_KEY = r"SOFTWARE\Microsoft\Windows\CurrentVersion\App Paths"

#: Services whose bare name is not a hostname. A model asked for "spotify"
#: has no reason to know it needs a suffix, or which one, and prefixing the
#: word as it stands opens ``https://spotify`` -- a domain that does not
#: exist, which from the speaker's side looks exactly like being ignored.
_KNOWN_SITES = {
    "spotify": "https://open.spotify.com",
    "youtube": "https://www.youtube.com",
    "gmail": "https://mail.google.com",
    "google": "https://www.google.com",
    "netflix": "https://www.netflix.com",
    "github": "https://github.com",
    "reddit": "https://www.reddit.com",
    "facebook": "https://www.facebook.com",
    "instagram": "https://www.instagram.com",
    "amazon": "https://www.amazon.com",
    "twitter": "https://x.com",
    "wikipedia": "https://www.wikipedia.org",
    "maps": "https://maps.google.com",
    "drive": "https://drive.google.com",
}

#: What people ask for when they are not naming a program. "Music" is not
#: installed anywhere and neither is "mail", but there is a sensible place
#: to take both -- and without one the tool answers with an error, which the
#: model then narrates instead of doing anything. Only names with one
#: obvious destination belong here: a confident answer to an ambiguous
#: request is worse than an honest refusal.
_CATEGORY_SITES = {
    "music": "https://open.spotify.com",
    "song": "https://open.spotify.com",
    "songs": "https://open.spotify.com",
    "video": "https://www.youtube.com",
    "videos": "https://www.youtube.com",
    "movie": "https://www.netflix.com",
    "movies": "https://www.netflix.com",
    "tv": "https://www.netflix.com",
    "mail": "https://mail.google.com",
    "email": "https://mail.google.com",
    "chat": "https://chatgpt.com",
    "photos": "https://photos.google.com",
    "news": "https://news.google.com",
}


def _site_for(name: str) -> str | None:
    """The address a bare word stands for, or ``None`` if it names no site."""
    key = (name or "").strip().casefold()
    return _KNOWN_SITES.get(key) or _CATEGORY_SITES.get(key)


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
        # A bare word is a service before it is a host: "spotify" names a
        # player, not a domain, and inventing the suffix for it is not
        # something a small model can be trusted to get right every time.
        target = _site_for(target) or "https://" + target

    try:
        _launch_url(target)
    except ControlError:
        raise
    except Exception as exc:  # noqa: BLE001 - the OS reports these unevenly
        raise ControlError(f"could not open {target}: {exc}") from exc
    return f"Opened {target} in the browser."


def _app_paths_entries() -> list[tuple[str, str]]:
    """``(name, path)`` for every program registered under App Paths."""
    if os.name != "nt":  # pragma: no cover - POSIX has no registry
        return []
    try:
        import winreg
    except ImportError:  # pragma: no cover - Windows always ships it
        return []

    entries: list[tuple[str, str]] = []
    for hive in (winreg.HKEY_CURRENT_USER, winreg.HKEY_LOCAL_MACHINE):
        try:
            root = winreg.OpenKey(hive, _APP_PATHS_KEY)
        except OSError:
            continue
        with root:
            count = winreg.QueryInfoKey(root)[0]
            for index in range(count):
                try:
                    subkey = winreg.EnumKey(root, index)
                    with winreg.OpenKey(root, subkey) as key:
                        raw, _kind = winreg.QueryValueEx(key, "")
                except OSError:  # pragma: no cover - racing uninstall
                    continue
                path = Path(raw.strip('"')).expanduser() if raw else None
                if path and path.exists():
                    entries.append((Path(subkey).stem, str(path)))
    return entries


def _windows_apps_entries() -> list[tuple[str, str]]:
    """``(name, path)`` for Store apps, which publish execution aliases.

    A Store app has no installer directory and no App Paths entry: its only
    launchable handle on disk is a zero-byte reparse point under
    ``WindowsApps``. Spotify is installed that way on this machine, so
    without this source "open Spotify" would have nothing to find.
    """
    base = Path(os.environ.get("LOCALAPPDATA", "")) / "Microsoft" / "WindowsApps"
    if not base.is_dir():
        return []
    try:
        return [(p.stem, str(p)) for p in base.glob("*.exe")]
    except OSError:  # pragma: no cover - unreadable alias directory
        return []


def _start_menu_entries() -> list[tuple[str, str]]:
    """``(name, path)`` for Start Menu shortcuts, walked last of the three.

    It is the only source that reads the disk, and the slowest, so the two
    registry-backed ones get their chance first.
    """
    roots = [
        Path(os.environ.get("APPDATA", ""))
        / "Microsoft" / "Windows" / "Start Menu" / "Programs",
        Path(os.environ.get("PROGRAMDATA", ""))
        / "Microsoft" / "Windows" / "Start Menu" / "Programs",
    ]
    entries: list[tuple[str, str]] = []
    for root in roots:
        if not root.is_dir():
            continue
        try:
            entries.extend((p.stem, str(p)) for p in root.rglob("*.lnk"))
        except OSError:  # pragma: no cover - unreadable Start Menu
            continue
    return entries


def _find_program(wanted: str) -> str | None:
    """The path of an installed program named *wanted*, or ``None``.

    An exact match wins over a loose one **across all three sources**, not
    within each: this machine has ``spotify_cli.exe`` in App Paths and
    ``Spotify.exe`` as a Store alias, and scanning source by source for a
    substring would return the command-line tool when the user meant the
    player. Loose matching stays, because nobody says "Google Chrome" out
    loud when they mean Chrome.
    """
    needle = wanted.casefold()
    loose: str | None = None
    for entries in (_app_paths_entries(), _windows_apps_entries(),
                    _start_menu_entries()):
        for key, path in entries:
            if key.casefold() == needle:
                return path
            if loose is None and needle and needle in key.casefold():
                loose = path
    return loose


def open_app(name: str) -> str:
    """Launch an installed program by name, or reach its website if absent.

    "Open Spotify" is a request for a program, and on a machine where
    Spotify is installed the desktop app is what is meant -- a browser tab
    with the player in it is not the same thing. When nothing of that name
    is installed the request is still answered rather than refused: the
    known website for the name is opened instead, because a reply of "I
    could not find it" is the one outcome that helps nobody.
    """
    wanted = (name or "").strip().strip('"')
    if not wanted:
        raise ControlError("no program name given")

    found = _find_program(wanted)
    if found:
        try:
            _launch(found)
        except Exception as exc:  # noqa: BLE001 - the OS reports these unevenly
            raise ControlError(f"could not start {wanted}: {exc}") from exc
        return f"Opened {Path(found).stem}."

    site = _site_for(wanted)
    if site:
        return f"{open_url(site)} (no {wanted} program is installed here)"
    raise ControlError(f"no program named {wanted!r} is installed here")


#: The words around what is actually meant when a request names Spotify.
_SPOTIFY_FILLER = re.compile(r"\b(on|in|with|from|using|the|app|please|spotify)\b",
                             re.IGNORECASE)


def play_media(request: str) -> str:
    """Open the closest match to *request* so that it starts playing.

    Opening is not playing. Handing back a search page for a song request
    opens something and makes no sound, which reads as Apache failing to do
    the one thing asked of it -- so the lookup here returns a watch page,
    the only kind of page that starts on its own.

    Spotify is the exception worth stating rather than pretending away: the
    desktop app can be pointed at a search but not at *play*, because that
    needs an account key this machine has not got. It is opened on the
    right page and the request is answered honestly instead of quietly
    doing half of it.
    """
    text = (request or "").strip().strip('"')
    if not text:
        raise ControlError("nothing to play was given")

    if "spotify" in text.casefold():
        terms = " ".join(_SPOTIFY_FILLER.sub(" ", text).split())
        if not terms:
            return open_app("spotify")
        uri = "spotify:search:" + urllib.parse.quote(terms, safe="")
        try:
            _launch(uri)
        except Exception as exc:  # noqa: BLE001 - the OS reports these unevenly
            raise ControlError(f"could not open Spotify: {exc}") from exc
        return (f"Opened Spotify on a search for {terms}. Pick the track to "
                "start it -- a song cannot be started in Spotify from here.")

    # `find_video` returning nothing means the search backend was unreachable
    # or found no video at all -- not that the request was wrong, so the
    # results page is the fallback rather than an error.
    found = find_video(text)
    if found:
        url, title = found
        open_url(url)
        return f"Playing {title}."

    page = "https://www.youtube.com/results?search_query=" + urllib.parse.quote(text)
    open_url(page)
    return f"No single match for {text}; opened the results so you can pick one."


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
