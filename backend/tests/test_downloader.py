import json
from pathlib import Path
from unittest.mock import patch

import pytest

from cassandra_yt_mcp.services.downloader import Downloader


def test_expand_playlist_parses_entries(tmp_path: Path) -> None:
    downloader = Downloader(tmp_path)

    fake_stdout = "\n".join([
        json.dumps({"id": "vid1", "title": "Video 1", "url": "https://www.youtube.com/watch?v=vid1"}),
        json.dumps({"id": "vid2", "title": "Video 2", "url": "https://www.youtube.com/watch?v=vid2"}),
        json.dumps({"id": "vid3", "title": "Video 3"}),
    ])

    with patch("cassandra_yt_mcp.services.downloader.subprocess.run") as mock_run:
        mock_run.return_value.returncode = 0
        mock_run.return_value.stdout = fake_stdout
        mock_run.return_value.stderr = ""

        entries = downloader.expand_playlist("https://youtube.com/playlist?list=PLtest")

    assert len(entries) == 3
    assert entries[0]["id"] == "vid1"
    assert entries[0]["url"] == "https://www.youtube.com/watch?v=vid1"
    assert entries[2]["url"] == ""  # no URL or webpage_url in metadata


def test_expand_playlist_raises_on_empty(tmp_path: Path) -> None:
    downloader = Downloader(tmp_path)

    with patch("cassandra_yt_mcp.services.downloader.subprocess.run") as mock_run:
        mock_run.return_value.returncode = 0
        mock_run.return_value.stdout = ""
        mock_run.return_value.stderr = ""

        with pytest.raises(RuntimeError, match="No videos found"):
            downloader.expand_playlist("https://youtube.com/playlist?list=PLtest")


def test_expand_playlist_raises_on_failure(tmp_path: Path) -> None:
    downloader = Downloader(tmp_path)

    with patch("cassandra_yt_mcp.services.downloader.subprocess.run") as mock_run:
        mock_run.return_value.returncode = 1
        mock_run.return_value.stdout = ""
        mock_run.return_value.stderr = "ERROR: playlist not found"

        with pytest.raises(RuntimeError, match="playlist not found"):
            downloader.expand_playlist("https://youtube.com/playlist?list=PLtest")
