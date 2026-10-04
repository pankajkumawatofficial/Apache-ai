"""Tests for Apache's retrieval layer.

Run with:  python -m tests.test_rag

These exercise the pure half of :mod:`app.rag` -- splitting, TF-IDF scoring and
the store's fallback behaviour -- without needing Ollama, NumPy or LangChain.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.rag import (  # noqa: E402
    DocumentStore,
    TfidfIndex,
    cosine,
    format_hits,
    read_document,
    split_text,
    tokenize,
)

FAILURES: list[str] = []


def check(label: str, condition: bool, detail: str = "") -> None:
    if condition:
        print(f"  ok   {label}")
    else:
        FAILURES.append(label)
        print(f"  FAIL {label}  {detail}")


DOCUMENTS = {
    "cycling.txt": (
        "A bicycle has two wheels, pedals and a chain. "
        "Cycling is an efficient way to commute across a city. "
        "Wear a helmet when you ride a bicycle on the road."
    ),
    "astronomy.txt": (
        "A telescope collects light from distant stars. "
        "Amateur astronomy is a popular hobby for observing planets. "
        "The Jupiter observation session was cancelled due to cloud cover."
    ),
    "cooking.txt": (
        "Bread dough needs flour, water, yeast and salt. "
        "Prove the dough until it doubles in size before baking. "
        "Sourdough bread uses a natural starter instead of yeast."
    ),
}


def test_tokenize() -> None:
    print("tokenize")
    check("lower-cases", tokenize("Hello World") == ["hello", "world"])
    check("drops one-char tokens", tokenize("a I x") == [],
          repr(tokenize("a I x")))
    check("keeps two-char words", tokenize("of to") == ["of", "to"])
    check("keeps numbers", tokenize("room 101") == ["room", "101"])
    check("strips punctuation", tokenize("don't stop!") == ["don't", "stop"])
    check("empty", tokenize("") == [])


def test_split_text() -> None:
    print("split_text")
    check("empty gives nothing", split_text("   ", 500, 50) == [])

    short = split_text("One short paragraph.", 500, 50)
    check("short text unsplit", len(short) == 1, f"{len(short)}")

    big = " ".join(f"_sentence number {i} ends here." for i in range(400))
    chunks = split_text(big, 400, 80)
    check("long text is chunked", len(chunks) > 1, f"{len(chunks)}")
    check("chunks respect size", all(len(c) <= 400 for c in chunks),
          f"max={max(len(c) for c in chunks)}")
    check("no empty chunks", all(c.strip() for c in chunks))
    # Content must not be lost -- works whether LangChain's splitter or the
    # fallback is in play, since neither invents or drops characters.
    check("content not lost", sum(len(c) for c in chunks) >= 0.9 * len(big),
          f"{sum(len(c) for c in chunks)} vs {len(big)}")

    # Large overlap must not be allowed to swallow the whole chunk.
    weird = split_text("abcdefg " * 400, 200, 5000)
    check("overlap clamped", all(len(c) <= 200 for c in weird),
          f"max={max(len(c) for c in weird)}")

    paragraphs = "\n\n".join(f"Paragraph {i} " + "x" * 300 for i in range(6))
    pc = split_text(paragraphs, 500, 50)
    check("prefers paragraph boundaries", len(pc) >= 3, f"{len(pc)}")

    # CJK / no-space text still gets bounded rather than one giant chunk.
    dense = "字" * 3000
    dc = split_text(dense, 400, 50)
    check("handles text without spaces", all(len(c) <= 400 for c in dc),
          f"max={max(len(c) for c in dc)}")


def test_tfidf() -> None:
    print("tfidf")
    texts = list(DOCUMENTS.values())
    index = TfidfIndex(texts)

    telescope = index.scores("how do I use a telescope to see stars")
    bicycle = index.scores("riding a bicycle safely with a helmet")
    bread = index.scores("baking bread dough with yeast")

    check("telescope ranks astronomy first", telescope[1] == max(telescope),
          f"{telescope}")
    check("bicycle ranks cycling first", bicycle[0] == max(bicycle), f"{bicycle}")
    check("bread ranks cooking first", bread[2] == max(bread), f"{bread}")
    check("scores are non-negative", all(s >= 0 for s in telescope))
    check("irrelevant query scores low",
          max(index.scores("quantum chromodynamics lattice")) <= max(telescope))

    check("cosine identical is 1", abs(cosine([1, 2, 3], [1, 2, 3]) - 1) < 1e-9)
    check("cosine orthogonal is 0", cosine([1, 0], [0, 1]) == 0.0)
    check("cosine empty is 0", cosine([], [1]) == 0.0)
    check("cosine opposite is -1", abs(cosine([1, 0], [-1, 0]) + 1) < 1e-9)


def test_read_document() -> None:
    print("read_document")
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        txt = root / "notes.txt"
        txt.write_text("some notes", encoding="utf-8")
        check("reads txt", "notes" in read_document(txt))

        py = root / "script.py"
        py.write_text("print('hi')", encoding="utf-8")
        check("reads py", "print" in read_document(py))

        bad = root / "image.png"
        bad.write_bytes(b"\x89PNG\r\n")
        try:
            read_document(bad)
            check("rejects unsupported suffix", False)
        except ValueError:
            check("rejects unsupported suffix", True)

        # Undecodable bytes must not crash the ingest.
        weird = root / "data.csv"
        weird.write_bytes(b"\xff\xfe\x00\x01garbage")
        try:
            read_document(weird)
            check("tolerates undecodable bytes", True)
        except Exception as exc:  # noqa: BLE001
            check("tolerates undecodable bytes", False, str(exc))


def test_document_store() -> None:
    print("DocumentStore")
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        paths = []
        for name, body in DOCUMENTS.items():
            path = root / name
            path.write_text(body, encoding="utf-8")
            paths.append(path)

        store = DocumentStore()
        check("starts empty", store.chunk_count == 0)
        check("search empty corpus", store.search("anything") == [])

        summary = store.ingest(paths)
        status = store.status()
        print(f"        ingest -> {summary}")
        check("ingested", status["chunks"] >= 3, str(status))
        check("mode chosen", status["mode"] in {"embedding", "tfidf"},
              status["mode"])
        check("revision bumped", store.revision == 1, str(store.revision))
        check("lists sources", set(status["sources"]) == set(DOCUMENTS))
        check("summary mentions chunks", "chunk" in summary, summary)

        hits = store.search("how do I observe planets with a telescope")
        check("returns hits", len(hits) > 0)
        if hits:
            check("astronomy ranks first", hits[0].source == "astronomy.txt",
                  f"{[h.source for h in hits]}")
            check("hit has score", hits[0].score >= 0)
            check("hit has text", len(hits[0].text) > 0)

        hits = store.search("riding a bicycle with a helmet", k=1)
        check("k is respected", len(hits) <= 1, str(len(hits)))
        if hits:
            check("cycling result for k=1", hits[0].source == "cycling.txt",
                  hits[0].source)

        # Formatting happens while the corpus is still loaded.
        sample = store.search("observing planets with a telescope")
        rendered = format_hits(sample)
        check("format_hits handles hits", "source:" in rendered, rendered[:80])
        check("format_hits numbers hits", "[1]" in rendered, rendered[:80])
        check("format_hits empty", "No matching" in format_hits([]))

        check("blank query returns nothing", store.search("   ") == [])

        # Incremental ingest keeps earlier documents retrievable.
        extra = root / "guitar.txt"
        extra.write_text("Guitar strings vibrate to make music. "
                         "Tune a guitar before you practise.", encoding="utf-8")
        store.ingest([extra])
        after = store.search("how do I tune a guitar")
        check("incremental ingest works", after and after[0].source == "guitar.txt",
              str([h.source for h in after]))
        check("revision tracks ingests", store.revision == 2, str(store.revision))
        still = store.search("observing planets with a telescope")
        check("earlier docs still indexed", still and still[0].source == "astronomy.txt",
              str([h.source for h in still]))

        # A file that cannot be read must not poison the whole batch.
        missing = root / "nope.txt"
        summary = store.ingest([missing])
        check("bad file reported", "No documents" in summary or "Skipped" in summary,
              summary)

        # Re-ingesting an already-indexed file must replace it, not double it.
        before = store.chunk_count
        store.ingest([paths[0]])
        after = store.chunk_count
        check("re-ingest is idempotent", after == before,
              f"{before} -> {after} chunks")
        dupe = store.search("riding a bicycle with a helmet")
        check("re-ingested doc still found once",
              dupe and dupe[0].source == "cycling.txt",
              str([h.source for h in dupe]))

        store.clear()
        check("clear empties index", store.chunk_count == 0)
        check("clear resets search", store.search("telescope") == [])
        check("clear bumps revision", store.revision >= 3, str(store.revision))


def main() -> int:
    for suite in (
        test_tokenize,
        test_split_text,
        test_tfidf,
        test_read_document,
        test_document_store,
    ):
        try:
            suite()
        except Exception as exc:  # noqa: BLE001
            import traceback

            traceback.print_exc()
            FAILURES.append(f"{suite.__name__} crashed: {exc}")
        print()

    if FAILURES:
        print(f"{len(FAILURES)} FAILURE(S):")
        for name in FAILURES:
            print(f"  - {name}")
        return 1
    print("All retrieval tests passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
