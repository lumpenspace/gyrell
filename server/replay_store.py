"""Replay persistence: mirror the local replay dir to a GCS bucket.

Cloud Run's filesystem is ephemeral — replays and traces written under
REPLAY_DIR vanish on every restart, taking the leaderboard, the lineup
rotation state, and the fallback diagnostics with them. When
GCS_REPLAYS_BUCKET is set, pull the bucket's contents down at startup and
push each new archive up as it lands. Any GCS failure degrades to
local-only operation: the show must go on.
"""

from __future__ import annotations

import os
import threading
from pathlib import Path

BUCKET = os.environ.get("GCS_REPLAYS_BUCKET")


def _bucket():
    # Imported lazily: the google-cloud-storage dependency only exists in
    # the server image, and local dev runs fine without it.
    from google.cloud import storage

    return storage.Client().bucket(BUCKET)


def sync_down(replay_dir: Path) -> int:
    """Download every bucket object missing locally. Returns the count."""
    if not BUCKET:
        return 0
    fetched = 0
    try:
        for blob in _bucket().list_blobs():
            target = replay_dir / blob.name
            if target.exists():
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            blob.download_to_filename(str(target))
            fetched += 1
        print(f"replay_store: synced {fetched} object(s) down from gs://{BUCKET}")
    except Exception as error:
        print(f"replay_store: sync from gs://{BUCKET} failed, serving local only: {error!r}")
    return fetched


def _upload(replay_dir: Path, paths: tuple[Path, ...]) -> None:
    try:
        bucket = _bucket()
        for path in paths:
            bucket.blob(str(path.relative_to(replay_dir))).upload_from_filename(str(path))
    except Exception as error:
        print(f"replay_store: upload to gs://{BUCKET} failed: {error!r}")


def upload_async(replay_dir: Path, *paths: Path) -> None:
    """Fire-and-forget upload so archiving never stalls the broadcast loop."""
    if not BUCKET or not paths:
        return
    threading.Thread(target=_upload, args=(replay_dir, tuple(paths)), daemon=True).start()
