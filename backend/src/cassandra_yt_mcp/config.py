from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


@dataclass(slots=True)
class Settings:
    host: str
    port: int
    poll_interval_seconds: int
    data_dir: Path
    database_path: Path
    assemblyai_api_key: str | None
    huggingface_token: str | None
    max_workers: int
    backend_api_token: str | None
    enable_local_transcription: bool


def _as_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    return int(raw)


def load_settings() -> Settings:
    load_dotenv()
    data_dir = Path(os.getenv("DATA_DIR", "/data")).resolve()
    database_path = Path(
        os.getenv("DATABASE_PATH", str(data_dir / "cassandra_yt_mcp.sqlite3"))
    ).resolve()

    return Settings(
        host=os.getenv("HOST", "0.0.0.0"),
        port=_as_int("PORT", 3000),
        poll_interval_seconds=_as_int("POLL_INTERVAL_SECONDS", 5),
        data_dir=data_dir,
        database_path=database_path,
        assemblyai_api_key=os.getenv("ASSEMBLYAI_API_KEY", "").strip() or None,
        huggingface_token=os.getenv("HUGGINGFACE_TOKEN", "").strip() or None,
        max_workers=_as_int("MAX_WORKERS", 3),
        backend_api_token=os.getenv("BACKEND_API_TOKEN", "").strip() or None,
        enable_local_transcription=os.getenv("ENABLE_LOCAL_TRANSCRIPTION", "true").lower()
        in ("true", "1", "yes"),
    )
