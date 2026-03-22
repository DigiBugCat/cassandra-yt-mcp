"""FluidAudio transcriber — Apple Silicon CoreML/ANE via cassandra-transcriber server."""

from __future__ import annotations

import logging
from pathlib import Path

import httpx

from cassandra_yt_mcp.types import TranscriptResult, TranscriptSegment

logger = logging.getLogger(__name__)

_TRANSCRIBE_TIMEOUT = 600.0  # 10 min max


class FluidAudioTranscriber:
    def __init__(self, base_url: str, diarize: bool = True) -> None:
        self.base_url = base_url.rstrip("/")
        self.diarize = diarize
        self.last_transcriber_used: str = "fluidaudio"

    def transcribe(self, audio_path: Path) -> TranscriptResult:
        content_type = _content_type(audio_path)
        with httpx.Client(timeout=_TRANSCRIBE_TIMEOUT) as client:
            with audio_path.open("rb") as f:
                resp = client.post(
                    f"{self.base_url}/transcribe",
                    params={"diarize": str(self.diarize).lower()},
                    files={"file": (audio_path.name, f, content_type)},
                )

        if resp.status_code >= 400:
            raise RuntimeError(f"FluidAudio failed ({resp.status_code}): {resp.text[:400]}")

        data = resp.json()
        text = data.get("text", "")
        segments = _build_segments(text, data.get("diarization"))

        return TranscriptResult(
            text=text,
            segments=segments,
            language="en",
        )


def _build_segments(
    text: str, diarization: dict | None
) -> list[TranscriptSegment]:
    """Build segments from diarization data, or a single segment from raw text."""
    if diarization and diarization.get("segments"):
        return [
            TranscriptSegment(
                start=seg["startTime"],
                end=seg["endTime"],
                text=seg.get("text", ""),
                speaker=f"SPEAKER_{seg['speakerId']}",
            )
            for seg in diarization["segments"]
        ]

    # No diarization — return full text as a single segment
    if text:
        return [TranscriptSegment(start=0.0, end=0.0, text=text, speaker=None)]
    return []


def _content_type(path: Path) -> str:
    ext = path.suffix.lower()
    return {
        ".wav": "audio/wav",
        ".mp3": "audio/mpeg",
        ".m4a": "audio/mp4",
        ".ogg": "audio/ogg",
        ".opus": "audio/ogg",
        ".flac": "audio/flac",
        ".webm": "audio/webm",
        ".mp4": "video/mp4",
    }.get(ext, "application/octet-stream")
