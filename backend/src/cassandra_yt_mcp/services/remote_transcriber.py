from __future__ import annotations

import logging
import subprocess
import time
from pathlib import Path

import httpx

from cassandra_yt_mcp.types import TranscriptResult, TranscriptSegment

logger = logging.getLogger(__name__)

_HEALTHZ_TIMEOUT = 5.0
_TRANSCRIBE_TIMEOUT = 600.0  # 10 min max per request
_HEALTH_CHECK_INTERVAL = 30.0


class NoHealthyWorkerError(RuntimeError):
    """All GPU workers are unhealthy or unreachable."""


class RemoteTranscriber:
    """Dispatches audio files to remote GPU workers for transcription."""

    def __init__(self, worker_urls: list[str]) -> None:
        if not worker_urls:
            raise ValueError("GPU_WORKERS must not be empty in coordinator mode")
        self.worker_urls = worker_urls
        self._next_index = 0
        self._health_cache: dict[str, tuple[bool, float]] = {}
        self.last_transcriber_used: str = "remote"

    def transcribe(self, audio_path: Path) -> TranscriptResult:
        healthy = self._get_healthy_workers()
        if not healthy:
            raise NoHealthyWorkerError(
                f"No healthy GPU workers among: {self.worker_urls}"
            )

        errors: list[str] = []
        for url in healthy:
            try:
                return self._send_to_worker(url, audio_path)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Worker %s failed: %s", url, exc)
                self._health_cache[url] = (False, time.monotonic())
                errors.append(f"{url}: {exc}")

        raise NoHealthyWorkerError(
            f"All workers failed. Errors: {'; '.join(errors)}"
        )

    def _get_healthy_workers(self) -> list[str]:
        """Return workers in round-robin order, skipping recently unhealthy ones."""
        now = time.monotonic()
        healthy: list[str] = []
        for url in self.worker_urls:
            cached = self._health_cache.get(url)
            if cached is not None:
                is_healthy, checked_at = cached
                if not is_healthy and (now - checked_at) < _HEALTH_CHECK_INTERVAL:
                    continue
            if self._check_health(url):
                healthy.append(url)

        if not healthy:
            return []

        # Round-robin: rotate list so next worker is first
        idx = self._next_index % len(healthy)
        self._next_index = idx + 1
        return healthy[idx:] + healthy[:idx]

    def _check_health(self, url: str) -> bool:
        try:
            resp = httpx.get(f"{url}/worker/healthz", timeout=_HEALTHZ_TIMEOUT)
            ok = resp.status_code == 200 and resp.json().get("ok", False)
            self._health_cache[url] = (ok, time.monotonic())
            return ok
        except Exception:  # noqa: BLE001
            self._health_cache[url] = (False, time.monotonic())
            return False

    def _send_to_worker(self, url: str, audio_path: Path) -> TranscriptResult:
        logger.info("Dispatching %s to worker %s", audio_path.name, url)
        send_path = self._ensure_wav(audio_path)
        try:
            with send_path.open("rb") as f:
                resp = httpx.post(
                    f"{url}/worker/transcribe",
                    files={"audio": (send_path.name, f, "application/octet-stream")},
                    timeout=_TRANSCRIBE_TIMEOUT,
                )
            resp.raise_for_status()
        finally:
            if send_path != audio_path and send_path.exists():
                send_path.unlink()
        data = resp.json()

        segments = [
            TranscriptSegment(
                start=s["start"],
                end=s["end"],
                text=s["text"],
                speaker=s.get("speaker"),
            )
            for s in data.get("segments", [])
        ]
        return TranscriptResult(
            text=data["text"],
            segments=segments,
            language=data.get("language"),
        )

    @staticmethod
    def _ensure_wav(audio_path: Path) -> Path:
        """Convert audio to 16kHz mono WAV on the coordinator so the GPU worker skips ffmpeg."""
        if audio_path.suffix == ".wav":
            return audio_path
        wav_path = audio_path.with_suffix(".16k.wav")
        t0 = time.monotonic()
        result = subprocess.run(
            ["ffmpeg", "-y", "-i", str(audio_path),
             "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le",
             str(wav_path)],
            capture_output=True, timeout=300,
        )
        if result.returncode != 0:
            logger.warning("ffmpeg conversion failed, sending original: %s", result.stderr[-200:])
            return audio_path
        logger.info("Converted %s → WAV in %.1fs", audio_path.name, time.monotonic() - t0)
        return wav_path
