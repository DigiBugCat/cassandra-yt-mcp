from __future__ import annotations

from prometheus_client import Counter, Gauge, Histogram

# --- Job lifecycle ---

jobs_total = Counter(
    "yt_mcp_jobs_total",
    "Total transcription jobs by final status and transcriber",
    ["status", "transcriber"],
)

jobs_in_progress = Gauge(
    "yt_mcp_jobs_in_progress",
    "Jobs currently being processed",
    ["phase"],
)

jobs_queued = Gauge(
    "yt_mcp_jobs_queued",
    "Number of jobs waiting in the queue",
)

retries_total = Counter(
    "yt_mcp_retries_total",
    "Total job retry attempts",
)

# --- Timing ---

download_duration_seconds = Histogram(
    "yt_mcp_download_duration_seconds",
    "Time spent downloading audio via yt-dlp",
    buckets=[5, 10, 20, 30, 60, 120, 300, 600],
)

transcription_duration_seconds = Histogram(
    "yt_mcp_transcription_duration_seconds",
    "Time spent transcribing audio",
    buckets=[10, 30, 60, 120, 300, 600, 1200, 1800, 3600],
)

# --- Content characteristics ---

audio_duration_seconds = Histogram(
    "yt_mcp_audio_duration_seconds",
    "Duration of the source audio/video in seconds",
    buckets=[60, 120, 300, 600, 1200, 1800, 3600, 7200],
)

speed_ratio = Histogram(
    "yt_mcp_speed_ratio",
    "Realtime speed ratio (audio_duration / transcription_time)",
    buckets=[0.5, 1, 2, 3, 5, 8, 10, 15, 20, 30],
)

word_count = Histogram(
    "yt_mcp_word_count",
    "Word count of transcribed text",
    buckets=[100, 500, 1000, 2000, 5000, 10000, 20000, 50000],
)

speaker_count = Histogram(
    "yt_mcp_speaker_count",
    "Number of speakers detected",
    buckets=[1, 2, 3, 4, 5, 8, 10],
)

# --- Fallback ---

fallback_total = Counter(
    "yt_mcp_fallback_total",
    "Times AssemblyAI fallback was triggered",
    ["reason"],
)

# --- Stored transcripts ---

transcripts_stored = Gauge(
    "yt_mcp_transcripts_stored",
    "Total number of stored transcripts",
)

# --- API layer ---

api_requests_total = Counter(
    "yt_mcp_api_requests_total",
    "HTTP requests to the backend API",
    ["method", "endpoint", "status"],
)

api_request_duration_seconds = Histogram(
    "yt_mcp_api_request_duration_seconds",
    "HTTP request duration",
    ["method", "endpoint"],
    buckets=[0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5],
)
