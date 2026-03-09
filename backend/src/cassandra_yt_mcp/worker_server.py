"""Minimal FastAPI app for GPU worker mode.

Receives audio files, runs ASR + diarization, returns TranscriptResult JSON.
No database, no storage, no job queue — purely stateless.
"""

from __future__ import annotations

import logging
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, UploadFile, status

from cassandra_yt_mcp.config import Settings, load_settings
from cassandra_yt_mcp.services.local_transcriber import LocalTranscriber

logger = logging.getLogger(__name__)


def create_worker_app(settings: Settings | None = None) -> FastAPI:
    app_settings = settings or load_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        transcriber = LocalTranscriber(
            huggingface_token=app_settings.huggingface_token,
        )
        # Warm models into VRAM on startup
        transcriber._load_models()
        app.state.transcriber = transcriber
        logger.info("Worker ready — models loaded")
        try:
            yield
        finally:
            pass

    app = FastAPI(title="cassandra-yt-mcp-worker", version="0.1.0", lifespan=lifespan)

    @app.get("/worker/healthz")
    def healthz() -> dict[str, object]:
        transcriber: LocalTranscriber = app.state.transcriber
        gpu_info: dict[str, object] = {"available": False}
        try:
            import torch  # noqa: PLC0415

            gpu_info["available"] = torch.cuda.is_available()
            if torch.cuda.is_available():
                gpu_info["device"] = torch.cuda.get_device_name(0)
                vram = torch.cuda.get_device_properties(0).total_mem
                gpu_info["vram_gb"] = round(vram / (1024**3), 1)
        except Exception:  # noqa: BLE001
            pass

        return {
            "ok": True,
            "gpu": gpu_info,
            "model_loaded": transcriber._asr_model is not None,
        }

    @app.post("/worker/transcribe")
    async def transcribe(audio: UploadFile) -> dict[str, object]:
        if audio.filename is None:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No audio file provided",
            )

        transcriber: LocalTranscriber = app.state.transcriber

        # Write uploaded audio to temp file
        suffix = Path(audio.filename).suffix or ".wav"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            content = await audio.read()
            tmp.write(content)
            tmp_path = Path(tmp.name)

        try:
            result = transcriber.transcribe(tmp_path)
            return {
                "text": result.text,
                "language": result.language,
                "segments": [
                    {
                        "start": s.start,
                        "end": s.end,
                        "text": s.text,
                        "speaker": s.speaker,
                    }
                    for s in result.segments
                ],
            }
        finally:
            tmp_path.unlink(missing_ok=True)

    return app
