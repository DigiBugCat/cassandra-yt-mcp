import pytest

from cassandra_yt_mcp.utils.url import extract_youtube_video_id, normalize_url


@pytest.mark.parametrize(
    "url",
    [
        "https://www.youtube.com/watch?v=abc123",
        "https://youtube.com/watch?v=abc123",
        "https://m.youtube.com/watch?v=abc123",
        "https://youtu.be/abc123",
        "https://youtube.com/watch?v=abc123&t=30",
        "https://youtube.com/shorts/abc123",
        "https://youtube.com/embed/abc123",
        "https://youtube.com/v/abc123",
        "https://www.youtube.com/live/abc123",
    ],
)
def test_youtube_canonical(url: str) -> None:
    assert normalize_url(url) == "https://youtube.com/watch?v=abc123"


def test_generic_strips_www() -> None:
    assert normalize_url("https://www.example.com/path?b=2&a=1") == "https://example.com/path?a=1&b=2"


def test_extract_video_id() -> None:
    assert extract_youtube_video_id("https://youtube.com/watch?v=abc123") == "abc123"
