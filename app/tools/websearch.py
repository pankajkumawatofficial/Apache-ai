"""Web search tool backing.

Kept separate from the ``@tool`` wrapper so the network call and result
shaping can be tested and reused without LangChain. Tolerant of both the
modern ``ddgs`` package and the older ``duckduckgo_search`` name.
"""

from __future__ import annotations

__all__ = ["web_search", "WebSearchUnavailable"]


class WebSearchUnavailable(RuntimeError):
    """Raised when no search backend could be imported."""


def _backend():
    try:
        from ddgs import DDGS

        return DDGS
    except ImportError:
        pass
    try:
        from duckduckgo_search import DDGS  # type: ignore[no-redef]

        return DDGS
    except ImportError as exc:  # neither package present
        raise WebSearchUnavailable(
            "no search backend installed (pip install ddgs)"
        ) from exc


def web_search(query: str, max_results: int = 5, timeout: float = 10.0) -> str:
    """Return the top *max_results* web results for *query* as plain text."""
    query = (query or "").strip()
    if not query:
        return "The search query was empty."

    ddgs_cls = _backend()
    try:
        # `max_results` is the stable kwarg across ddgs 5..9; `timeout` is not
        # universally accepted, so it is only passed when supported.
        try:
            results = ddgs_cls().text(query, max_results=max_results, timeout=timeout)
        except TypeError:
            results = ddgs_cls().text(query, max_results=max_results)
    except Exception as exc:
        return f"Web search failed: {type(exc).__name__}: {exc}"

    if not results:
        return f"No web results found for: {query}"

    lines: list[str] = []
    for number, item in enumerate(results, start=1):
        if not isinstance(item, dict):
            continue
        title = str(item.get("title") or "").strip()
        url = str(item.get("href") or item.get("link") or "").strip()
        body = str(item.get("body") or item.get("snippet") or "").strip()
        if title or body:
            lines.append(f"{number}. {title}\n   {url}\n   {body}")

    if not lines:
        return f"No usable web results found for: {query}"
    return "\n".join(lines)
