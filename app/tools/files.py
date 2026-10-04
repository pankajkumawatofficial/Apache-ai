"""Workspace-confined file access for the agent's file tools.

Everything here is rooted at a single ``data/workspace`` directory.  Resolving
through :meth:`pathlib.Path.resolve` and then checking ancestry means ``..``,
absolute paths and symlinks that point outside the workspace are all rejected
by the same check.
"""

from __future__ import annotations

from pathlib import Path

__all__ = [
    "FileAccessError",
    "resolve_inside",
    "list_files",
    "read_file",
    "write_file",
]

DEFAULT_MAX_READ = 200_000
_MAX_LIST_ENTRIES = 500

# Binary extensions we refuse to open as text rather than flooding the context.
_BINARY_SUFFIXES = {
    ".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp", ".ico",
    ".pdf", ".zip", ".gz", ".tar", ".7z", ".rar",
    ".exe", ".dll", ".so", ".dylib", ".pyd", ".bin",
    ".mp3", ".wav", ".mp4", ".mov", ".avi", ".mkv",
    ".doc", ".docx", ".xls", ".xlsx", ".ppt", ".pptx",
    ".db", ".sqlite", ".sqlite3",
}


class FileAccessError(ValueError):
    """Raised when a path would escape the workspace or the file is unusable."""


def resolve_inside(workspace: Path | str, name: str) -> Path:
    """Resolve *name* against *workspace*, rejecting anything that escapes it."""
    root = Path(workspace).resolve()
    if not name or not str(name).strip():
        raise FileAccessError("a file name is required")

    candidate = Path(str(name).strip())
    if not candidate.is_absolute():
        candidate = root / candidate

    try:
        resolved = candidate.resolve()
    except OSError as exc:  # pragma: no cover - defensive
        raise FileAccessError(f"could not resolve path: {exc}") from exc

    if resolved != root and root not in resolved.parents:
        raise FileAccessError(f"path escapes the workspace: {name}")
    return resolved


def list_files(workspace: Path | str, subdir: str = ".") -> str:
    """List files under *subdir*, recursively, as one path per line."""
    root = Path(workspace).resolve()
    target = resolve_inside(workspace, subdir)
    if not target.exists():
        raise FileAccessError(f"no such directory: {subdir}")
    if not target.is_dir():
        raise FileAccessError(f"not a directory: {subdir}")

    entries: list[str] = []
    for path in sorted(target.rglob("*")):
        if path.is_dir():
            continue
        entries.append(f"{path.relative_to(root).as_posix()}  ({path.stat().st_size} bytes)")
        if len(entries) >= _MAX_LIST_ENTRIES:
            entries.append(f"... [{_MAX_LIST_ENTRIES}+ files, listing truncated]")
            break

    if not entries:
        return f"(empty) {subdir} contains no files."
    return "\n".join(entries)


def read_file(
    workspace: Path | str,
    name: str,
    max_bytes: int = DEFAULT_MAX_READ,
) -> str:
    """Read a UTF-8 text file from the workspace, bounded by *max_bytes*."""
    path = resolve_inside(workspace, name)
    if path.suffix.lower() in _BINARY_SUFFIXES:
        raise FileAccessError(f"{name} looks like a binary file and cannot be read as text")
    if not path.exists():
        raise FileAccessError(f"no such file: {name}")
    if not path.is_file():
        raise FileAccessError(f"not a file: {name}")

    raw = path.read_bytes()[:max_bytes]
    text = raw.decode("utf-8", errors="replace")
    if len(raw) >= max_bytes:
        text += f"\n... [truncated at {max_bytes} bytes]"
    return text


def write_file(workspace: Path | str, name: str, content: str) -> str:
    """Create or overwrite *name* in the workspace with *content*."""
    path = resolve_inside(workspace, name)
    if path.exists() and path.is_dir():
        raise FileAccessError(f"is a directory: {name}")

    path.parent.mkdir(parents=True, exist_ok=True)
    payload = content if isinstance(content, str) else str(content)
    path.write_text(payload, encoding="utf-8")
    relative = path.relative_to(Path(workspace).resolve()).as_posix()
    return f"Wrote {len(payload)} characters to {relative}"
