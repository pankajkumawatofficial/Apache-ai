"""Web search tool backing.

Kept separate from the ``@tool`` wrapper so the network call and result
shaping can be tested and reused without LangChain. Tolerant of both the
modern ``ddgs`` package and the older ``duckduckgo_search`` name.
"""

from __future__ import annotations

import json
import re
import urllib.parse
import urllib.request

__all__ = ["web_search", "find_video", "WebSearchUnavailable"]


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


def _url_of(item: object) -> str:
    if not isinstance(item, dict):
        return ""
    for key in ("href", "url", "link"):
        value = item.get(key)
        if isinstance(value, str) and value.startswith(("http://", "https://")):
            return value.strip()
    return ""


def _title_of(item: object) -> str:
    if not isinstance(item, dict):
        return ""
    return str(item.get("title") or "").strip()


#: The two shapes a YouTube results page uses for a hit: an eleven-character
#: video id, and a title a short way after it. Both are read straight out of
#: the HTML the page already sends, so no API key and no JavaScript.
_WATCH_ID = re.compile(r'"videoId"\s*:\s*"([A-Za-z0-9_-]{11})"')
_WATCH_TITLE = re.compile(
    r'"title"\s*:\s*\{"runs"\s*:\s*\[\{"text"\s*:\s*"((?:[^"\\]|\\.){1,160})"',
    re.S,
)

_YOUTUBE_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)


def _unquote(raw: str) -> str:
    """Decode the JSON escaping inside a title lifted out of a page."""
    try:
        return str(json.loads(f'"{raw}"'))
    except Exception:  # noqa: BLE001 - a half-escaped title is still a title
        return raw


def _watch_from(html: str) -> tuple[str, str] | None:
    """The first search hit in a YouTube results page: watch URL and title.

    The title is looked for *after* each id rather than matched globally,
    because the page interleaves several lists and pairing the first id with
    the first title would happily name the wrong video. Ids also appear
    outside any result -- in the page's initial payload -- where there is no
    title beside them at all, so an id with nothing near it is remembered as
    a fallback and skipped in favour of the first one that *is* a result.
    """
    if not html:
        return None
    first_id: str | None = None
    for match in _WATCH_ID.finditer(html):
        video_id = match.group(1)
        title = _WATCH_TITLE.search(html, match.end(), match.end() + 6000)
        if title is None:
            if first_id is None:
                first_id = video_id
            continue
        return (
            f"https://www.youtube.com/watch?v={video_id}",
            _unquote(title.group(1)),
        )
    if first_id:
        return f"https://www.youtube.com/watch?v={first_id}", ""
    return None


def _youtube_watch(query: str, timeout: float = 10.0) -> tuple[str, str] | None:
    """The top hit for *query*, read out of YouTube's own results page.

    Preferred over any search backend: it is the page that would have been
    opened anyway, it needs no key, and looking at it cannot be rate-limited
    by a third party. On this machine the results page arrives complete --
    ``ytInitialData`` and all -- without consent pages or rendering.
    """
    url = "https://www.youtube.com/results?search_query=" + urllib.parse.quote(query)
    request = urllib.request.Request(
        url,
        headers={"User-Agent": _YOUTUBE_UA, "Accept-Language": "en-US,en;q=0.9"},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            html = response.read().decode("utf-8", "replace")
    except Exception:  # noqa: BLE001 - offline, blocked, timed out
        return None
    return _watch_from(html)


def find_video(query: str, max_results: int = 8, timeout: float = 10.0) -> tuple[str, str] | None:
    """A watch-page URL for *query* and its title, or ``None``.

    "Play this" is only satisfied by a page that starts playing by itself. A
    search results page opens without a sound, so handing the model one for
    a song request produces exactly the complaint this exists to fix: the
    app opens and nothing is heard. Hence a lookup for the single most
    playable link rather than a page of links.

    YouTube's own results page is asked first, then a search backend -- video
    results first, because they already know which hit is a video, and a
    text search narrowed to YouTube when those come back empty. Only URLs
    that start on their own count from the text search; an article *about* a
    video plays nothing.
    """
    query = (query or "").strip()
    if not query:
        return None

    found = _youtube_watch(query, timeout)
    if found:
        return found[0], found[1] or query

    try:
        ddgs_cls = _backend()
    except WebSearchUnavailable:
        return None

    for kind, term in (("videos", query), ("text", f"site:youtube.com {query}")):
        try:
            client = ddgs_cls()
            call = getattr(client, kind)
            try:
                results = list(call(term, max_results=max_results, timeout=timeout) or [])
            except TypeError:
                results = list(call(term, max_results=max_results) or [])
        except Exception:  # noqa: BLE001 - unavailable, rate limited, ...
            continue

        candidates = [
            candidate
            for candidate in ((_url_of(r), _title_of(r)) for r in results)
            if candidate[0]
        ]
        if not candidates:
            continue

        for url, title in candidates:
            if "youtube.com/watch" in url or "youtu.be/" in url:
                return url, title or query
        if kind == "text":
            # A text search returns articles about videos as readily as it
            # returns videos, and only the videos start.
            continue
        # Every result of a *video* search is one, watch page or not.
        return candidates[0]

    return None
