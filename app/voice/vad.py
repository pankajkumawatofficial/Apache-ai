"""Energy-based voice activity detection.

A dependency-free gate: it measures the RMS of each short frame, tracks a slow
noise floor while nobody is speaking, and treats a jump past
``noise_floor * multiplier`` as speech.  Silence held for ``silence_end_ms``
closes the utterance.

Pure Python over plain sequences so it can be exercised without a microphone
and without NumPy; the listener hands it slices of the ``sounddevice`` stream.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence

__all__ = ["EnergyVAD", "Utterance"]


class Utterance(Sequence[float]):
    """A contiguous block of voiced samples."""

    __slots__ = ("samples", "sample_rate")

    def __init__(self, samples: list[float], sample_rate: int) -> None:
        self.samples = samples
        self.sample_rate = sample_rate

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index):  # pragma: no cover - trivial
        return self.samples[index]

    @property
    def duration_s(self) -> float:
        return len(self.samples) / float(self.sample_rate or 1)


class EnergyVAD:
    """Segment a stream of mono float samples into voiced utterances.

    Parameters
    ----------
    sample_rate:
        Samples per second of the incoming stream.
    frame_ms:
        Analysis window length.
    min_energy:
        Absolute floor for the speech gate, so a truly silent room never
        trips it no matter how the noise floor adapts.
    multiplier:
        ``threshold = max(min_energy, noise_floor * multiplier)``.
    silence_end_ms:
        Consecutive quiet frames required to close an utterance.
    min_utterance_ms:
        Shorter fragments (mouth clicks, the tail of a word) are discarded.
    max_utterance_s:
        Hard cap; a stuck gate cannot buffer the microphone forever.
    """

    def __init__(
        self,
        *,
        sample_rate: int = 16_000,
        frame_ms: int = 30,
        min_energy: float = 0.012,
        multiplier: float = 3.5,
        silence_end_ms: int = 700,
        min_utterance_ms: int = 350,
        max_utterance_s: float = 20.0,
    ) -> None:
        if sample_rate <= 0:
            raise ValueError("sample_rate must be positive")
        if frame_ms <= 0:
            raise ValueError("frame_ms must be positive")

        self.sample_rate = sample_rate
        self.frame_ms = frame_ms
        self.frame_len = max(1, int(round(sample_rate * frame_ms / 1000)))
        self.min_energy = min_energy
        self.multiplier = multiplier
        self.silence_frames_needed = max(1, int(round(silence_end_ms / frame_ms)))
        self.min_utterance_samples = int(sample_rate * min_utterance_ms / 1000)
        self.max_utterance_samples = int(sample_rate * max_utterance_s)

        self._noise_floor = min_energy
        self._buffer: list[float] = []
        self._utterance: list[float] = []
        self._in_speech = False
        self._silent_frames = 0
        self._speech_frames = 0

    # -- introspection ---------------------------------------------------
    @property
    def threshold(self) -> float:
        """The gate currently in force (moves with the noise floor)."""
        return max(self.min_energy, self._noise_floor * self.multiplier)

    @property
    def listening(self) -> bool:
        return self._in_speech

    # -- control ---------------------------------------------------------
    def reset(self) -> None:
        """Drop buffered audio and return to the silent state."""
        self._buffer.clear()
        self._utterance.clear()
        self._in_speech = False
        self._silent_frames = 0
        self._speech_frames = 0
        self._noise_floor = self.min_energy

    def flush(self) -> list[Utterance]:
        """Force-close any in-progress utterance and return it, if usable."""
        self._buffer.clear()
        if not self._in_speech:
            return []
        utterance = self._take_utterance(trim_silence=False)
        return [utterance] if utterance is not None else []

    # -- main loop -------------------------------------------------------
    def feed(self, samples: Iterable[float]) -> list[Utterance]:
        """Consume a block of samples and return any utterances it completed."""
        self._buffer.extend(samples)
        completed: list[Utterance] = []

        while len(self._buffer) >= self.frame_len:
            frame = self._buffer[: self.frame_len]
            del self._buffer[: self.frame_len]
            done = self._process_frame(frame)
            if done is not None:
                completed.append(done)
        return completed

    def _process_frame(self, frame: Sequence[float]) -> Utterance | None:
        energy = _rms(frame)
        voiced = energy >= self.threshold

        if not self._in_speech:
            if voiced:
                # Pass the trigger frame in so its samples are not lost.
                self._begin(frame)
                return None
            # Only ever learn the ambient level from quiet frames. Adapting
            # during speech would walk the threshold up until a long monologue
            # silenced itself.
            self._adapt_noise(energy)
            return None

        # -- inside an utterance ---------------------------------------
        self._utterance.extend(frame)
        self._speech_frames += 1

        if voiced:
            self._silent_frames = 0
        else:
            self._silent_frames += 1

        if self._silent_frames >= self.silence_frames_needed:
            return self._take_utterance(trim_silence=True)

        if len(self._utterance) >= self.max_utterance_samples:
            # Ran into the cap mid-speech: keep everything, it is all voice.
            return self._take_utterance(trim_silence=False)
        return None

    def _begin(self, frame: Sequence[float]) -> None:
        self._in_speech = True
        self._silent_frames = 0
        self._speech_frames = 1
        self._utterance = list(frame)

    def _take_utterance(self, *, trim_silence: bool) -> Utterance | None:
        samples = self._utterance
        self._utterance = []
        self._in_speech = False
        self._silent_frames = 0
        self._speech_frames = 0

        if trim_silence:
            # The frames that triggered the close are trailing silence.
            trim = self.silence_frames_needed * self.frame_len
            if trim and len(samples) > trim:
                samples = samples[: len(samples) - trim]

        if len(samples) < self.min_utterance_samples:
            return None
        return Utterance(samples, self.sample_rate)

    def _adapt_noise(self, energy: float, rate: float = 0.05) -> None:
        """Follow the ambient level using quiet frames only."""
        self._noise_floor += (energy - self._noise_floor) * max(rate, 0.001)
        # Keep the floor from collapsing to zero in a truly silent room.
        self._noise_floor = max(self._noise_floor, 1e-5)


def _rms(frame: Sequence[float]) -> float:
    if not frame:
        return 0.0
    total = 0.0
    for sample in frame:
        total += float(sample) * float(sample)
    return math.sqrt(total / len(frame))
