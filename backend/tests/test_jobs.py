from pathlib import Path

from cassandra_yt_mcp.db.database import Database
from cassandra_yt_mcp.db.jobs import JobsRepository


def _make_repo(tmp_path: Path) -> JobsRepository:
    db = Database(tmp_path / "test.sqlite3")
    return JobsRepository(db)


def test_enqueue_and_claim(tmp_path: Path) -> None:
    repo = _make_repo(tmp_path)
    job = repo.enqueue(url="https://youtube.com/watch?v=abc", normalized_url="https://youtube.com/watch?v=abc")
    assert job["status"] == "queued"

    claimed = repo.claim_next()
    assert claimed is not None
    assert claimed["id"] == job["id"]
    assert claimed["status"] == "downloading"


def test_mark_downloaded_and_claim(tmp_path: Path) -> None:
    repo = _make_repo(tmp_path)
    job = repo.enqueue(url="https://youtube.com/watch?v=abc", normalized_url="https://youtube.com/watch?v=abc")

    # Claim as downloading
    claimed = repo.claim_next()
    assert claimed is not None

    # Mark downloaded with audio path
    repo.mark_downloaded(claimed["id"], "/data/_work/job1/abc.opus")

    refreshed = repo.get(claimed["id"])
    assert refreshed is not None
    assert refreshed["status"] == "downloaded"
    assert refreshed["audio_path"] == "/data/_work/job1/abc.opus"


def test_claim_next_downloaded(tmp_path: Path) -> None:
    repo = _make_repo(tmp_path)
    job = repo.enqueue(url="https://youtube.com/watch?v=abc", normalized_url="https://youtube.com/watch?v=abc")

    # Nothing downloaded yet
    assert repo.claim_next_downloaded() is None

    # Simulate download flow
    claimed = repo.claim_next()
    repo.mark_downloaded(claimed["id"], "/data/_work/job1/abc.opus")

    # Now claim_next_downloaded should find it
    transcribe_job = repo.claim_next_downloaded()
    assert transcribe_job is not None
    assert transcribe_job["id"] == job["id"]
    assert transcribe_job["status"] == "transcribing"
    assert transcribe_job["audio_path"] == "/data/_work/job1/abc.opus"


def test_claim_next_downloaded_empty_when_none(tmp_path: Path) -> None:
    repo = _make_repo(tmp_path)
    assert repo.claim_next_downloaded() is None


def test_downloaded_in_active_statuses(tmp_path: Path) -> None:
    repo = _make_repo(tmp_path)
    job = repo.enqueue(url="https://youtube.com/watch?v=abc", normalized_url="https://youtube.com/watch?v=abc")

    # Claim and mark downloaded
    claimed = repo.claim_next()
    repo.mark_downloaded(claimed["id"], "/data/_work/job1/abc.opus")

    # Should be found as active
    active = repo.find_active_by_normalized_url("https://youtube.com/watch?v=abc")
    assert active is not None
    assert active["status"] == "downloaded"
