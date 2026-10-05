"""Ollama access: health checks and ``ChatOllama`` construction.

The health check deliberately uses only :mod:`urllib` so the UI can report
"Ollama is not running" before any of the LangChain stack is imported.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass

from .config import settings

__all__ = ["OllamaStatus", "check_ollama", "build_chat_model", "build_embeddings"]


@dataclass(frozen=True)
class OllamaStatus:
    ok: bool
    models: list[str]
    message: str

    @property
    def choices(self) -> list[str]:
        """Model names for the UI dropdown; never empty, so it stays usable."""
        return list(self.models) if self.models else ["<no models pulled>"]


def check_ollama(base_url: str | None = None, timeout: float = 3.0) -> OllamaStatus:
    """Probe ``/api/tags`` and report whether Ollama is up and what it has."""
    url = (base_url or settings.ollama_base_url).rstrip("/") + "/api/tags"
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:  # noqa: S310
            payload = json.loads(response.read().decode("utf-8", "replace"))
    except urllib.error.HTTPError as exc:
        return OllamaStatus(False, [], f"Ollama answered with HTTP {exc.code}")
    except Exception as exc:  # connection refused, DNS, timeout
        return OllamaStatus(
            False,
            [],
            f"Cannot reach Ollama at {url} - {exc}. Is `ollama serve` running?",
        )

    models = sorted(
        str(m.get("name", "")).strip()
        for m in payload.get("models", [])
        if m.get("name")
    )
    if not models:
        return OllamaStatus(True, [], "Ollama is running but has no models pulled")
    return OllamaStatus(True, models, f"{len(models)} model(s) available")


def build_chat_model(
    model: str | None = None,
    temperature: float | None = None,
    base_url: str | None = None,
    reasoning: bool | None = None,
):
    """Construct a ``ChatOllama``. Imported lazily to keep startup cheap."""
    from langchain_ollama import ChatOllama

    return ChatOllama(
        model=model or settings.model,
        base_url=(base_url or settings.ollama_base_url).rstrip("/"),
        temperature=settings.temperature if temperature is None else temperature,
        num_ctx=settings.num_ctx,
        # Left as None, Ollama applies the model's own default -- which for
        # qwen3 is thinking ON, ~16x slower on CPU. Apache passes an explicit
        # False unless the operator opted in with APACHE_REASONING=1.
        reasoning=settings.reasoning if reasoning is None else reasoning,
        # Left off: it costs a network round trip on every rebuild, and the
        # first invoke surfaces a clearer error than a constructor would.
        validate_model_on_init=False,
    )


def build_embeddings(base_url: str | None = None):
    """Construct ``OllamaEmbeddings`` for RAG, or ``None`` if unusable."""
    try:
        from langchain_ollama import OllamaEmbeddings

        return OllamaEmbeddings(
            model=settings.embedding_model,
            base_url=(base_url or settings.ollama_base_url).rstrip("/"),
            validate_model_on_init=False,
        )
    except Exception:
        # Missing package, a blocked/broken native dependency, or a bad
        # constructor argument -- all of which mean "fall back to TF-IDF".
        return None
