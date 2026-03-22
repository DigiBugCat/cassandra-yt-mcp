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
    fluidaudio_url: str  # e.g. "http://172.20.1.179:8420"
    max_workers: int
    backend_api_token: str | None
    role: str  # "standalone" | "mcp"
    download_concurrency: int
    # MCP server settings (role=mcp)
    mcp_port: int = 3003
    auth_url: str = ""  # Auth service URL for key validation
    auth_secret: str = ""  # Shared secret for auth service
    auth_yaml_path: str = "/app/acl.yaml"  # Path to bundled acl.yaml for local enforcement


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

    role = os.getenv("ROLE", "standalone").lower()
    if role not in ("standalone", "mcp"):
        raise ValueError(f"ROLE must be standalone|mcp, got '{role}'")

    fluidaudio_url = os.getenv("FLUIDAUDIO_URL", "").strip()
    if not fluidaudio_url and role != "mcp":
        raise ValueError("FLUIDAUDIO_URL is required")

    return Settings(
        host=os.getenv("HOST", "0.0.0.0"),
        port=_as_int("PORT", 3000),
        poll_interval_seconds=_as_int("POLL_INTERVAL_SECONDS", 5),
        data_dir=data_dir,
        database_path=database_path,
        fluidaudio_url=fluidaudio_url,
        max_workers=_as_int("MAX_WORKERS", 3),
        backend_api_token=os.getenv("BACKEND_API_TOKEN", "").strip() or None,
        role=role,
        download_concurrency=_as_int("DOWNLOAD_CONCURRENCY", 2),
        mcp_port=_as_int("MCP_PORT", 3003),
        auth_url=os.getenv("AUTH_URL", "").strip(),
        auth_secret=os.getenv("AUTH_SECRET", "").strip(),
        auth_yaml_path=os.getenv("AUTH_YAML_PATH", "/app/acl.yaml"),
    )
