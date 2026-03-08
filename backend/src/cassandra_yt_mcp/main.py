from __future__ import annotations

import logging

import uvicorn

from cassandra_yt_mcp.api.app import create_app
from cassandra_yt_mcp.config import load_settings

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s - %(message)s")


def app():
    return create_app(load_settings())


def cli() -> None:
    settings = load_settings()
    uvicorn.run(
        "cassandra_yt_mcp.main:app",
        host=settings.host,
        port=settings.port,
        factory=True,
        reload=False,
    )


if __name__ == "__main__":
    cli()
