from pathlib import Path

import pytest

from cassandra_yt_mcp.services.fallback_transcriber import (
    FallbackTranscriber,
    LocalTranscriptionUnavailableError,
)
from cassandra_yt_mcp.services.transcriber import UnsupportedLanguageError
from cassandra_yt_mcp.types import TranscriptResult


class DummyLocalTranscriber:
    def __init__(self, behavior: str) -> None:
        self.behavior = behavior

    def transcribe(self, audio_path: Path) -> TranscriptResult:
        if self.behavior == "ok":
            return TranscriptResult(text="local", segments=[], language="en")
        if self.behavior == "unsupported":
            raise UnsupportedLanguageError("ja")
        if self.behavior == "missing":
            raise ImportError("torch not installed")
        raise AssertionError(f"Unknown behavior: {self.behavior}")


class DummyFallbackTranscriber:
    def transcribe(self, audio_path: Path) -> TranscriptResult:
        return TranscriptResult(text="fallback", segments=[], language="en")


def test_uses_fallback_when_local_disabled(tmp_path: Path) -> None:
    transcriber = FallbackTranscriber(
        local=DummyLocalTranscriber("ok"),
        fallback=DummyFallbackTranscriber(),
        enable_local=False,
    )

    result = transcriber.transcribe(tmp_path / "audio.mp3")

    assert result.text == "fallback"


def test_uses_fallback_when_local_runtime_missing(tmp_path: Path) -> None:
    transcriber = FallbackTranscriber(
        local=DummyLocalTranscriber("missing"),
        fallback=DummyFallbackTranscriber(),
        enable_local=True,
    )

    result = transcriber.transcribe(tmp_path / "audio.mp3")

    assert result.text == "fallback"


def test_raises_when_local_disabled_and_no_fallback(tmp_path: Path) -> None:
    transcriber = FallbackTranscriber(local=None, fallback=None, enable_local=False)

    with pytest.raises(LocalTranscriptionUnavailableError):
        transcriber.transcribe(tmp_path / "audio.mp3")


def test_uses_fallback_for_unsupported_language(tmp_path: Path) -> None:
    transcriber = FallbackTranscriber(
        local=DummyLocalTranscriber("unsupported"),
        fallback=DummyFallbackTranscriber(),
        enable_local=True,
    )

    result = transcriber.transcribe(tmp_path / "audio.mp3")

    assert result.text == "fallback"
