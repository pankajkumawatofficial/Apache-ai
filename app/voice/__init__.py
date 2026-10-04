"""Voice input/output for Apache.

``wake`` and ``vad`` are pure Python and testable without a microphone;
``stt``, ``tts`` and ``listener`` talk to the network and the sound card.
"""

from __future__ import annotations

__all__ = ["wake", "vad", "stt", "tts", "listener"]
