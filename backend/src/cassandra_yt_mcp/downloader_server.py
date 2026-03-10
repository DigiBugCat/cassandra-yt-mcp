"""Minimal FastAPI app for downloader mode.

Runs the download loop — claims queued jobs, downloads audio to shared PVC.
No transcription, no storage — purely download + status updates.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from cassandra_yt_mcp.config import Settings, load_settings
from cassandra_yt_mcp.runtime import DownloaderRuntime

logger = logging.getLogger(__name__)


def create_downloader_app(settings: Settings | None = None) -> FastAPI:
    app_settings = settings or load_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        runtime = DownloaderRuntime(app_settings)
        runtime.start()
        app.state.runtime = runtime
        logger.info(
            "Downloader ready — concurrency=%d, poll=%ds",
            app_settings.download_concurrency,
            app_settings.poll_interval_seconds,
        )
        try:
            yield
        finally:
            runtime.close()

    app = FastAPI(title="cassandra-yt-mcp-downloader", version="0.1.0", lifespan=lifespan)

    @app.get("/healthz")
    def healthz() -> dict[str, object]:
        runtime: DownloaderRuntime = app.state.runtime
        return {
            "ok": True,
            "role": "downloader",
            "worker_running": runtime.worker.is_running,
        }

    return app
