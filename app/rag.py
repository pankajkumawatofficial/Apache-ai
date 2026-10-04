"""Document ingestion and retrieval for Apache's RAG capability.

Two retrieval back ends, chosen automatically at ingest time:

``embedding``
    Cosine similarity over ``OllamaEmbeddings`` vectors. Best quality, but
    needs an embedding model pulled into Ollama (``nomic-embed-text``).

``tfidf``
    A small in-process TF-IDF index. No network, no extra model -- used
    whenever the embedding model is missing or Ollama refuses, so document
    Q&A degrades instead of breaking.

Everything here is import-light: :mod:`pypdf`, :mod:`langchain_ollama` and the
LangChain text splitter are all pulled in lazily inside the functions that
need them, which keeps the index testable without the ML stack installed.
"""

from __future__ import annotations

import math
import re
import threading
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

from .config import Settings, settings as default_settings

__all__ = ["Hit", "DocumentStore", "split_text", "tokenize", "cosine"]

_TOKEN_RE = re.compile(r"[a-z0-9']{2,}")
_WORD_RE = re.compile(r"[A-Za-z0-9']+")
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")
_PARAGRAPH_RE = re.compile(r"\n\s*\n")

_SUPPORTED_SUFFIXES = {".txt", ".md", ".markdown", ".rst", ".csv", ".json",
                       ".py", ".js", ".ts", ".html", ".css", ".xml", ".yaml",
                       ".yml", ".toml", ".ini", ".log", ".sql", ".sh", ".bat"}
_PDF_SUFFIXES = {".pdf"}


@dataclass(frozen=True)
class Hit:
    """A single retrieved chunk."""

    source: str
    text: str
    score: float
    index: int = 0


# ---------------------------------------------------------------------------
# Text preparation (pure)
# ---------------------------------------------------------------------------
def tokenize(text: str) -> list[str]:
    """Lower-cased word tokens; short and stop-word light by design."""
    return _TOKEN_RE.findall(text.lower())


def split_text(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    """Split *text* into overlapping chunks.

    Uses LangChain's ``RecursiveCharacterTextSplitter`` when installed (the
    idiomatic choice), and falls back to a paragraph/sentence/hard-cut splitter
    so RAG still works if that optional package is absent.
    """
    text = (text or "").strip()
    if not text:
        return []
    if chunk_size <= 0:
        return [text]

    try:
        from langchain_text_splitters import RecursiveCharacterTextSplitter
    except ImportError:
        return _fallback_split(text, chunk_size, chunk_overlap)

    try:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=max(0, min(chunk_overlap, chunk_size // 2)),
            separators=["\n\n", "\n", ". ", " ", ""],
            length_function=len,
        )
        chunks = [c.strip() for c in splitter.split_text(text) if c and c.strip()]
        return chunks or [text]
    except Exception:
        return _fallback_split(text, chunk_size, chunk_overlap)


def _fallback_split(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    overlap = max(0, min(chunk_overlap, chunk_size // 2))
    step = max(1, chunk_size - overlap)

    chunks: list[str] = []
    buffer = ""
    for paragraph in _PARAGRAPH_RE.split(text):
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        if len(buffer) + len(paragraph) + 2 <= chunk_size:
            buffer = f"{buffer}\n\n{paragraph}" if buffer else paragraph
            continue
        if buffer:
            chunks.append(buffer)
            buffer = ""
        if len(paragraph) <= chunk_size:
            buffer = paragraph
        else:
            # Prefer sentence boundaries, then fall back to a hard cut.
            start = 0
            while start < len(paragraph):
                end = min(start + chunk_size, len(paragraph))
                if end < len(paragraph):
                    window = paragraph[start:end]
                    cut = max(
                        window.rfind(". "),
                        window.rfind("! "),
                        window.rfind("? "),
                        window.rfind("\n"),
                    )
                    if cut > chunk_size // 3:
                        end = start + cut + 1
                chunks.append(paragraph[start:end].strip())
                if end >= len(paragraph):
                    break
                start = max(end - overlap, start + step)
            buffer = ""
    if buffer:
        chunks.append(buffer)
    return [c for c in chunks if c]


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    """Cosine similarity, tolerant of differing lengths."""
    if not a or not b:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if not na or not nb:
        return 0.0
    return dot / (na * nb)


# ---------------------------------------------------------------------------
# TF-IDF fallback index (pure)
# ---------------------------------------------------------------------------
class TfidfIndex:
    """A tiny in-memory cosine index over tokenised chunks."""

    def __init__(self, documents: Iterable[str]) -> None:
        self._doc_tfs: list[Counter] = []
        document_frequency: Counter = Counter()

        for document in documents:
            term_freq = Counter(tokenize(document))
            self._doc_tfs.append(term_freq)
            for term in term_freq:
                document_frequency[term] += 1

        count = max(1, len(self._doc_tfs))
        self.idf: dict[str, float] = {
            term: math.log((count + 1) / (freq + 1)) + 1.0
            for term, freq in document_frequency.items()
        }
        self.vectors = [self._vector(tf) for tf in self._doc_tfs]

    def _vector(self, term_freq: Counter) -> dict[str, float]:
        vector = {
            term: (1.0 + math.log(freq)) * self.idf[term]
            for term, freq in term_freq.items()
            if term in self.idf
        }
        norm = math.sqrt(sum(w * w for w in vector.values()))
        if not norm:
            return {}
        return {term: weight / norm for term, weight in vector.items()}

    def scores(self, query: str) -> list[float]:
        query_vector = self._vector(Counter(tokenize(query)))
        results: list[float] = []
        for vector in self.vectors:
            total = 0.0
            for term, weight in query_vector.items():
                other = vector.get(term)
                if other is not None:
                    total += weight * other
            results.append(total)
        return results


# ---------------------------------------------------------------------------
# Reading files
# ---------------------------------------------------------------------------
def _read_text_file(path: Path) -> str:
    raw = path.read_bytes()
    for encoding in ("utf-8", "utf-16", "cp1252"):
        try:
            return raw.decode(encoding)
        except (UnicodeDecodeError, UnicodeError):
            continue
    return raw.decode("utf-8", errors="replace")


def _read_pdf(path: Path) -> str:
    from pypdf import PdfReader  # lazy: optional dependency

    reader = PdfReader(str(path))
    pages = []
    for page in reader.pages:
        try:
            pages.append(page.extract_text() or "")
        except Exception:  # a damaged page should not lose the whole document
            continue
    return "\n\n".join(pages)


def read_document(path: Path) -> str:
    """Extract plain text from a supported file, or raise ``ValueError``."""
    suffix = path.suffix.lower()
    if suffix in _PDF_SUFFIXES:
        return _read_pdf(path)
    if suffix in _SUPPORTED_SUFFIXES or not suffix:
        return _read_text_file(path)
    raise ValueError(f"unsupported file type: {path.name}")


# ---------------------------------------------------------------------------
# The store
# ---------------------------------------------------------------------------
@dataclass
class _Index:
    chunks: list[str] = field(default_factory=list)
    sources: list[str] = field(default_factory=list)
    vectors: list[list[float]] | None = None
    tfidf: TfidfIndex | None = None
    mode: str = "none"


class DocumentStore:
    """Holds the ingested corpus and answers retrieval queries.

    All mutation goes through :class:`DocumentStore`'s lock; the agent's tool
    thread reads while the UI thread ingests.
    """

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or default_settings
        self._lock = threading.RLock()
        self._index = _Index()
        self._revision = 0
        self._last_error: str = ""
        self._embedder: Any = None

    # -- state -----------------------------------------------------------
    @property
    def revision(self) -> int:
        with self._lock:
            return self._revision

    @property
    def chunk_count(self) -> int:
        with self._lock:
            return len(self._index.chunks)

    def sources(self) -> list[str]:
        with self._lock:
            return sorted(set(self._index.sources))

    def status(self) -> dict[str, Any]:
        with self._lock:
            return {
                "chunks": len(self._index.chunks),
                "documents": len(set(self._index.sources)),
                "mode": self._index.mode,
                "sources": sorted(set(self._index.sources)),
                "error": self._last_error,
                "revision": self._revision,
            }

    def clear(self) -> str:
        with self._lock:
            self._index = _Index()
            self._revision += 1
            self._last_error = ""
        return "Document index cleared."

    # -- ingestion -------------------------------------------------------
    def ingest(self, paths: Sequence[str | Path]) -> str:
        """Add files to the index and rebuild it. Returns a summary."""
        texts: list[tuple[str, str]] = []   # (source, chunk)
        errors: list[str] = []

        for raw in paths or []:
            path = Path(raw)
            name = path.name
            try:
                if not path.is_file():
                    errors.append(f"{name}: not a file")
                    continue
                content = read_document(path)
            except Exception as exc:
                errors.append(f"{name}: {exc}")
                continue

            chunks = split_text(
                content, self.settings.chunk_size, self.settings.chunk_overlap
            )
            if not chunks:
                errors.append(f"{name}: no extractable text")
                continue
            for chunk in chunks:
                texts.append((name, chunk))

        if not texts:
            detail = "; ".join(errors) or "nothing was uploaded"
            return f"No documents were indexed ({detail})."

        with self._lock:
            # Re-ingest everything previously held plus the new files, so the
            # index stays a single coherent whole. Files already indexed are
            # replaced rather than duplicated, so re-uploading is idempotent.
            previous = self._index
            replaced = {source for source, _ in texts}
            kept = [
                (source, chunk)
                for source, chunk in zip(previous.sources, previous.chunks)
                if source not in replaced
            ]
            merged_sources = [s for s, _ in kept] + [s for s, _ in texts]
            merged_chunks = [c for _, c in kept] + [c for _, c in texts]
            index = _Index(chunks=merged_chunks, sources=merged_sources)
            index = self._build(index)
            self._index = index
            self._revision += 1
            mode = index.mode

        message = (
            f"Indexed {len(texts)} chunk(s) from {len(paths)} file(s); "
            f"{len(index.chunks)} chunk(s) total; retrieval = {mode}."
        )
        if errors:
            message += " Skipped: " + "; ".join(errors)
        return message

    def _build(self, index: _Index) -> _Index:
        """Attach whichever retrieval back end is usable right now."""
        if not index.chunks:
            return index

        vectors = self._embed(index.chunks)
        if vectors is not None:
            index.vectors = vectors
            index.mode = "embedding"
            return index

        try:
            index.tfidf = TfidfIndex(index.chunks)
            index.mode = "tfidf"
        except Exception:  # pragma: no cover - tokenizer cannot really fail
            index.mode = "none"
        return index

    def _embed(self, texts: list[str]) -> list[list[float]] | None:
        from .llm import build_embeddings

        if self._embedder is None:
            self._embedder = build_embeddings(self.settings.ollama_base_url)
        embeddings = self._embedder
        if embeddings is None:
            self._last_error = "embedding model unavailable"
            return None

        vectors: list[list[float]] = []
        batch_size = 32
        try:
            for start in range(0, len(texts), batch_size):
                batch = texts[start:start + batch_size]
                vectors.extend(embeddings.embed_documents(batch))
        except Exception as exc:
            self._last_error = f"embedding failed: {exc}"
            return None

        if len(vectors) != len(texts) or not vectors:
            self._last_error = "embedding model returned no vectors"
            return None
        self._last_error = ""
        return [[float(x) for x in vector] for vector in vectors]

    # -- retrieval -------------------------------------------------------
    def search(self, query: str, k: int | None = None) -> list[Hit]:
        """Return the *k* chunks most relevant to *query*."""
        top_k = k or self.settings.retriever_k
        query = (query or "").strip()
        if not query:
            return []

        with self._lock:
            index = self._index
            if not index.chunks:
                return []

            if index.mode == "embedding" and index.vectors is not None:
                query_vector = self._embed([query])
                if query_vector:
                    scores = [cosine(query_vector[0], v) for v in index.vectors]
                else:
                    scores = self._fallback_scores(index, query)
            elif index.mode == "tfidf" and index.tfidf is not None:
                scores = index.tfidf.scores(query)
            else:
                scores = self._fallback_scores(index, query)

            order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
            hits = [
                Hit(
                    source=index.sources[i],
                    text=index.chunks[i],
                    score=float(scores[i]),
                    index=i,
                )
                for i in order[: max(1, top_k)]
                if scores[i] > 0
            ]

        if not hits and index.chunks:
            # Nothing scored above zero: hand back the head of the corpus so
            # the model can still answer rather than reporting "not found".
            hits = [
                Hit(source=index.sources[i], text=index.chunks[i], score=0.0, index=i)
                for i in range(min(max(1, top_k), len(index.chunks)))
            ]
        return hits

    @staticmethod
    def _fallback_scores(index: _Index, query: str) -> list[float]:
        try:
            return TfidfIndex(index.chunks).scores(query)
        except Exception:  # pragma: no cover
            return [0.0] * len(index.chunks)


def format_hits(hits: Sequence[Hit]) -> str:
    """Render hits as text the model can quote from."""
    if not hits:
        return "No matching documents were found."
    blocks = []
    for number, hit in enumerate(hits, start=1):
        blocks.append(f"[{number}] source: {hit.source}\n{hit.text.strip()}")
    return "\n\n".join(blocks)
