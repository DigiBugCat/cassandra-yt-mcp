from __future__ import annotations

import logging
from pathlib import Path

from cassandra_yt_mcp.services.transcriber import AssemblyAITranscriber, UnsupportedLanguageError
from cassandra_yt_mcp.types import TranscriptResult

logger = logging.getLogger(__name__)


class LocalTranscriptionUnavailableError(RuntimeError):
    """Raised when local GPU transcription is disabled or dependencies are missing."""


class FallbackTranscriber:
    def __init__(
        self,
        *,
        local: object | None,
        fallback: AssemblyAITranscriber | None = None,
        enable_local: bool = True,
    ) -> None:
        self.local = local if enable_local else None
        self.fallback = fallback

    def transcribe(self, audio_path: Path) -> TranscriptResult:
        if self.local is not None:
            try:
                return self.local.transcribe(audio_path)
            except UnsupportedLanguageError as exc:
                if self.fallback is None:
                    raise RuntimeError(
                        f"Detected unsupported language '{exc.language}' "
                        "but no ASSEMBLYAI_API_KEY configured for fallback"
                    ) from exc
                logger.info("Falling back to AssemblyAI for language '%s'", exc.language)
                return self.fallback.transcribe(audio_path)
            except (ImportError, OSError) as exc:
                logger.warning(
                    "Local GPU transcription failed (missing deps or CUDA): %s. "
                    "Attempting AssemblyAI fallback.",
                    exc,
                )
                if self.fallback is None:
                    raise LocalTranscriptionUnavailableError(
                        f"Local GPU transcription unavailable ({exc}) "
                        "and no ASSEMBLYAI_API_KEY configured for fallback"
                    ) from exc
                return self.fallback.transcribe(audio_path)

        if self.fallback is None:
            raise LocalTranscriptionUnavailableError(
                "Local transcription is disabled (ENABLE_LOCAL_TRANSCRIPTION=false) "
                "and no ASSEMBLYAI_API_KEY configured for fallback"
            )
        logger.info("Local transcription disabled; using AssemblyAI")
        return self.fallback.transcribe(audio_path)
