"""GCS helpers for baseline sync and upload archival."""

import logging
import os
from pathlib import Path

from google.cloud import storage

log = logging.getLogger(__name__)


def _bucket_name() -> str | None:
    return os.getenv("GCS_BUCKET_NAME")


def gcs_enabled() -> bool:
    return bool(_bucket_name())


def _prefix(name: str, default: str) -> str:
    value = os.getenv(name, default).strip().strip("/")
    return value


def _client() -> storage.Client:
    return storage.Client(project=os.getenv("GOOGLE_PROJECT_ID"))


def sync_baseline_from_gcs(local_data_dir: Path) -> bool:
    """Download baseline data snapshot from GCS into local data/."""
    bucket_name = _bucket_name()
    if not bucket_name:
        log.info("GCS_BUCKET_NAME not set; skipping baseline sync")
        return False

    baseline_prefix = _prefix("GCS_BASELINE_PREFIX", "baseline")
    client = _client()
    bucket = client.bucket(bucket_name)
    blobs = list(client.list_blobs(bucket, prefix=f"{baseline_prefix}/"))
    if not blobs:
        log.warning("No baseline objects found at gs://%s/%s/", bucket_name, baseline_prefix)
        return False

    # Reset local data dir so startup is always baseline-only.
    if local_data_dir.exists():
        for path in sorted(local_data_dir.rglob("*"), reverse=True):
            if path.is_file():
                path.unlink()
            elif path.is_dir():
                path.rmdir()
    local_data_dir.mkdir(parents=True, exist_ok=True)

    downloaded = 0
    for blob in blobs:
        if blob.name.endswith("/"):
            continue
        rel = blob.name[len(f"{baseline_prefix}/") :]
        local_path = local_data_dir / rel
        local_path.parent.mkdir(parents=True, exist_ok=True)
        blob.download_to_filename(local_path)
        downloaded += 1

    log.info(
        "Baseline sync complete from gs://%s/%s/ (%d objects)",
        bucket_name,
        baseline_prefix,
        downloaded,
    )
    return True


def archive_uploaded_artifacts(pdf_path: Path, chunks_path: Path | None = None) -> None:
    """Persist uploaded doc artifacts to GCS archive path."""
    bucket_name = _bucket_name()
    if not bucket_name:
        return

    uploads_prefix = _prefix("GCS_UPLOADS_PREFIX", "uploads")
    client = _client()
    bucket = client.bucket(bucket_name)

    targets: list[tuple[Path, str]] = [
        (pdf_path, f"{uploads_prefix}/1_raw/{pdf_path.name}"),
    ]
    if chunks_path and chunks_path.exists():
        targets.append((chunks_path, f"{uploads_prefix}/2_processed/{chunks_path.name}"))

    for local_path, blob_name in targets:
        if not local_path.exists():
            continue
        blob = bucket.blob(blob_name)
        blob.upload_from_filename(local_path)
        log.info("Archived upload to gs://%s/%s", bucket_name, blob_name)
