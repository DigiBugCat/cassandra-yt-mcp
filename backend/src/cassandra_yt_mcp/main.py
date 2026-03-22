from __future__ import annotations

import logging

import uvicorn

from cassandra_yt_mcp.config import load_settings

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s - %(message)s")

logger = logging.getLogger(__name__)


def app():
    from cassandra_yt_mcp.api.app import create_app  # noqa: PLC0415

    return create_app(load_settings())


def mcp_app():
    from cassandra_yt_mcp.mcp_server import create_mcp_server  # noqa: PLC0415

    return create_mcp_server(load_settings())


def cli() -> None:
    settings = load_settings()

    if settings.role == "mcp":
        logger.info("Starting in MCP mode on port %d", settings.mcp_port)
        mcp_server = mcp_app()
        mcp_server.run(
            transport="streamable-http",
            host=settings.host,
            port=settings.mcp_port,
        )
    else:
        logger.info("Starting in %s mode on port %d", settings.role.upper(), settings.port)
        uvicorn.run(
            "cassandra_yt_mcp.main:app",
            host=settings.host,
            port=settings.port,
            factory=True,
            reload=False,
            h11_max_incomplete_event_size=256 * 1024,
        )


if __name__ == "__main__":
    cli()
