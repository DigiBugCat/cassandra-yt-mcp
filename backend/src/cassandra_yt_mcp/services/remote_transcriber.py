from __future__ import annotations

import logging
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
        with audio_path.open("rb") as f:
            resp = httpx.post(
                f"{url}/worker/transcribe",
                files={"audio": (audio_path.name, f, "application/octet-stream")},
                timeout=_TRANSCRIBE_TIMEOUT,
            )
        resp.raise_for_status()
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
