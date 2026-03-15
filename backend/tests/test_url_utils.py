import pytest

from cassandra_yt_mcp.utils.url import (
    extract_video_id,
    extract_youtube_video_id,
    is_playlist_url,
    is_youtube_url,
    normalize_url,
    url_based_video_id,
)


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


@pytest.mark.parametrize(
    "url",
    [
        "https://www.youtube.com/playlist?list=PLrAXtmErZgOeiKm4sgNOknGvNjby9efdf",
        "https://youtube.com/playlist?list=PLrAXtmErZgOeiKm4sgNOknGvNjby9efdf",
        "https://youtube.com/watch?v=abc123&list=PLrAXtmErZgOeiKm4sgNOknGvNjby9efdf",
        "https://www.youtube.com/watch?list=PLrAXtmErZgOeiKm4sgNOknGvNjby9efdf",
    ],
)
def test_is_playlist_url_true(url: str) -> None:
    assert is_playlist_url(url) is True


@pytest.mark.parametrize(
    "url",
    [
        "https://youtube.com/watch?v=abc123",
        "https://youtu.be/abc123",
        "https://youtube.com/shorts/abc123",
        "https://example.com/playlist?list=xyz",
    ],
)
def test_is_playlist_url_false(url: str) -> None:
    assert is_playlist_url(url) is False


def test_extract_video_id_youtube() -> None:
    assert extract_video_id("https://youtube.com/watch?v=abc123") == "abc123"


def test_extract_video_id_non_youtube() -> None:
    assert extract_video_id("https://www.twitch.tv/videos/12345") is None


def test_is_youtube_url() -> None:
    assert is_youtube_url("https://youtube.com/watch?v=abc123") is True
    assert is_youtube_url("https://www.twitch.tv/videos/12345") is False
    assert is_youtube_url("https://twitter.com/user/status/123") is False


def test_url_based_video_id_stable() -> None:
    url = "https://www.twitch.tv/videos/12345"
    id1 = url_based_video_id(url)
    id2 = url_based_video_id(url)
    assert id1 == id2
    assert len(id1) == 16


def test_normalize_non_youtube() -> None:
    assert normalize_url("https://www.twitch.tv/videos/12345") == "https://twitch.tv/videos/12345"
