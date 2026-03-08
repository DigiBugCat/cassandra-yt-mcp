from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import httpx

from cassandra_yt_mcp.types import TranscriptResult, TranscriptSegment


@runtime_checkable
class Transcriber(Protocol):
    def transcribe(self, audio_path: Path) -> TranscriptResult: ...


class UnsupportedLanguageError(Exception):
    def __init__(self, language: str) -> None:
        self.language = language
        super().__init__(f"Unsupported language: {language}")


class AssemblyAITranscriber:
    def __init__(
        self,
        api_key: str,
        poll_interval_seconds: float = 3.0,
        timeout_seconds: float = 600.0,
        max_wait_seconds: float = 3600.0,
    ) -> None:
        self.api_key = api_key
        self.poll_interval_seconds = poll_interval_seconds
        self.timeout_seconds = timeout_seconds
        self.max_wait_seconds = max_wait_seconds
        self.base_url = "https://api.assemblyai.com/v2"

    def transcribe(self, audio_path: Path) -> TranscriptResult:
        headers = {"authorization": self.api_key}
        with httpx.Client(timeout=self.timeout_seconds) as client:
            audio_url = self._upload_audio(client, headers, audio_path)
            transcript_id = self._start_transcript(client, headers, audio_url)
            payload = self._poll_transcript(client, headers, transcript_id)
            sentences = self._fetch_sentences(client, headers, transcript_id)

        text = str(payload.get("text") or "").strip()
        segments = self._extract_segments(sentences) or self._extract_segments(payload.get("utterances"))
        if not text and segments:
            text = "\n".join(segment.text for segment in segments).strip()
        language = payload.get("language_code") or payload.get("language")
        return TranscriptResult(text=text, segments=segments, language=str(language) if language else None)

    def _upload_audio(self, client: httpx.Client, headers: dict[str, str], audio_path: Path) -> str:
        with audio_path.open("rb") as audio_stream:
            response = client.post(f"{self.base_url}/upload", headers=headers, content=audio_stream)
        if response.status_code >= 400:
            raise RuntimeError(f"AssemblyAI upload failed ({response.status_code}): {response.text[:400]}")
        payload = response.json()
        upload_url = payload.get("upload_url")
        if not upload_url:
            raise RuntimeError("AssemblyAI upload response missing upload_url")
        return str(upload_url)

    def _start_transcript(self, client: httpx.Client, headers: dict[str, str], audio_url: str) -> str:
        response = client.post(
            f"{self.base_url}/transcript",
            headers=headers,
            json={"audio_url": audio_url, "speaker_labels": True, "punctuate": True, "format_text": True},
        )
        if response.status_code >= 400:
            raise RuntimeError(
                f"AssemblyAI transcript create failed ({response.status_code}): {response.text[:400]}"
            )
        payload = response.json()
        transcript_id = payload.get("id")
        if not transcript_id:
            raise RuntimeError("AssemblyAI transcript response missing id")
        return str(transcript_id)

    def _poll_transcript(self, client: httpx.Client, headers: dict[str, str], transcript_id: str) -> dict[str, Any]:
        started = time.monotonic()
        while True:
            response = client.get(f"{self.base_url}/transcript/{transcript_id}", headers=headers)
            if response.status_code >= 400:
                raise RuntimeError(
                    f"AssemblyAI transcript poll failed ({response.status_code}): {response.text[:400]}"
                )
            payload = response.json()
            status = str(payload.get("status") or "").lower()
            if status == "completed":
                return dict(payload)
            if status == "error":
                raise RuntimeError(str(payload.get("error") or "AssemblyAI reported error status"))
            if time.monotonic() - started >= self.max_wait_seconds:
                raise RuntimeError("AssemblyAI transcription polling timed out")
            time.sleep(self.poll_interval_seconds)

    def _fetch_sentences(
        self,
        client: httpx.Client,
        headers: dict[str, str],
        transcript_id: str,
    ) -> list[dict[str, Any]]:
        response = client.get(f"{self.base_url}/transcript/{transcript_id}/sentences", headers=headers)
        if response.status_code >= 400:
            return []
        payload = response.json()
        sentences = payload.get("sentences")
        return sentences if isinstance(sentences, list) else []

    def _extract_segments(self, utterances: object) -> list[TranscriptSegment]:
        if not isinstance(utterances, list):
            return []
        segments: list[TranscriptSegment] = []
        for item in utterances:
            if not isinstance(item, dict):
                continue
            text = str(item.get("text") or "").strip()
            if not text:
                continue
            segments.append(
                TranscriptSegment(
                    start=self._ms_to_seconds(item.get("start")),
                    end=self._ms_to_seconds(item.get("end")),
                    speaker=str(item.get("speaker")) if item.get("speaker") is not None else None,
                    text=text,
                )
            )
        return segments

    @staticmethod
    def _ms_to_seconds(value: object) -> float:
        try:
            return float(str(value)) / 1000.0 if value is not None else 0.0
        except (TypeError, ValueError):
            return 0.0
