# Cassandra YT MCP Backend

Private HTTP API for `cassandra-yt-mcp`.

This service owns:

- job queue and retry state
- yt-dlp download flow
- transcript storage and search
- local-first transcription with optional AssemblyAI fallback
- backend-only routes used by the public Cloudflare Worker MCP edge

## Routes

- `POST /api/jobs/transcribe`
- `GET /api/jobs/{job_id}`
- `GET /api/transcripts`
- `GET /api/transcripts/search`
- `GET /api/transcripts/{video_id}`
- `GET /api/youtube/search`
- `GET /api/youtube/metadata`
- `GET /api/youtube/comments`
- `GET /healthz`

## Authentication

If `BACKEND_API_TOKEN` is set, all `/api/*` routes require:

```text
Authorization: Bearer <token>
```

This is intended to be layered on top of Cloudflare Access in production.

## Local Development

```bash
pip install -e ".[dev]"
uvicorn cassandra_yt_mcp.main:app --reload --factory
pytest
```
