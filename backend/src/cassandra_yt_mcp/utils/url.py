from __future__ import annotations

from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

_YT_HOSTNAMES = frozenset({"youtube.com", "www.youtube.com", "m.youtube.com", "youtu.be"})


def _extract_youtube_video_id(parsed) -> str | None:
    netloc = parsed.netloc.lower()
    path = parsed.path
    if netloc == "youtu.be":
        return path.strip("/") or None
    clean_path = path.rstrip("/")
    if clean_path == "/watch":
        params = dict(parse_qsl(parsed.query, keep_blank_values=False))
        return params.get("v") or None
    for prefix in ("/shorts/", "/embed/", "/v/", "/live/"):
        if clean_path.startswith(prefix):
            return clean_path[len(prefix) :].strip("/") or None
    return None


def extract_youtube_video_id(url: str) -> str | None:
    parsed = urlparse(url.strip())
    if not parsed.scheme:
        parsed = urlparse(f"https://{url.strip()}")
    netloc = parsed.netloc.lower()
    if netloc.startswith("www."):
        netloc = netloc[4:]
    if netloc not in _YT_HOSTNAMES:
        return None
    return _extract_youtube_video_id(parsed)


def is_playlist_url(url: str) -> bool:
    """Check if a URL points to a YouTube playlist."""
    parsed = urlparse(url.strip())
    if not parsed.scheme:
        parsed = urlparse(f"https://{url.strip()}")
    netloc = parsed.netloc.lower()
    if netloc.startswith("www."):
        netloc = netloc[4:]
    if netloc not in _YT_HOSTNAMES:
        return False
    params = dict(parse_qsl(parsed.query, keep_blank_values=False))
    clean_path = parsed.path.rstrip("/")
    # Explicit playlist page
    if clean_path == "/playlist" and "list" in params:
        return True
    # Watch URL with list= param but no specific video
    if "list" in params and "v" not in params:
        return True
    # Watch URL with both v= and list= — treat as playlist
    if "list" in params and "v" in params:
        return True
    return False


def normalize_url(url: str) -> str:
    parsed = urlparse(url.strip())
    if not parsed.scheme:
        parsed = urlparse(f"https://{url.strip()}")
    netloc = parsed.netloc.lower()
    if netloc.startswith("www."):
        netloc = netloc[4:]
    if netloc in _YT_HOSTNAMES:
        video_id = _extract_youtube_video_id(parsed)
        if video_id:
            return f"https://youtube.com/watch?v={video_id}"
    query = urlencode(sorted(parse_qsl(parsed.query, keep_blank_values=False)))
    return urlunparse((parsed.scheme.lower(), netloc, parsed.path.rstrip("/"), "", query, ""))
