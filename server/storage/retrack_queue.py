"""
Filesystem-backed retrack queue for managing mask retracking jobs.

Implements FIFO queue with support for parallel workers. Queue state is stored
as JSON files for durability and observability.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Optional

ISO_FORMAT = "%Y-%m-%dT%H:%M:%SZ"


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def isoformat(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).strftime(ISO_FORMAT)


@dataclass
class RetrackJob:
    """A single retrack job in the queue."""

    study_uid: str
    series_uid: str
    editor: str
    previous_version_id: Optional[str]
    new_version_id: str
    uploaded_masks_path: Path  # Temporary directory containing extracted masks
    queued_at: str
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    failed_at: Optional[str] = None
    error_message: Optional[str] = None
    status: str = "pending"  # pending | processing | completed | failed

    def to_dict(self) -> dict:
        """Serialize to dict, converting Path to string."""
        data = asdict(self)
        data["uploaded_masks_path"] = str(self.uploaded_masks_path)
        return data

    @classmethod
    def from_dict(cls, data: dict) -> RetrackJob:
        """Deserialize from dict, converting string to Path."""
        data = data.copy()
        data["uploaded_masks_path"] = Path(data["uploaded_masks_path"])
        return cls(**data)


class RetrackQueue:
    """FIFO queue for retrack jobs with filesystem persistence."""

    def __init__(self, queue_file: Path):
        self.queue_file = queue_file
        self.queue_file.parent.mkdir(parents=True, exist_ok=True)
        self._lock = Lock()

    def _load_queue(self) -> list[RetrackJob]:
        """Load queue from filesystem."""
        if not self.queue_file.exists():
            return []

        with self.queue_file.open() as f:
            data = json.load(f)

        return [RetrackJob.from_dict(item) for item in data]

    def _save_queue(self, jobs: list[RetrackJob]) -> None:
        """Save queue to filesystem."""
        data = [job.to_dict() for job in jobs]
        with self.queue_file.open("w") as f:
            json.dump(data, f, indent=2)

    def enqueue(
        self,
        study_uid: str,
        series_uid: str,
        editor: str,
        previous_version_id: Optional[str],
        uploaded_masks_path: Path,
    ) -> RetrackJob:
        """
        Add a new retrack job to the queue.

        Returns the created job with generated version_id.
        """
        with self._lock:
            jobs = self._load_queue()

            # Generate new version ID: hash(timestamp + editor email)
            timestamp = utc_now()
            version_string = f"{isoformat(timestamp)}_{editor}"
            new_version_id = hashlib.sha256(
                version_string.encode()
            ).hexdigest()[:16]

            job = RetrackJob(
                study_uid=study_uid,
                series_uid=series_uid,
                editor=editor,
                previous_version_id=previous_version_id,
                new_version_id=new_version_id,
                uploaded_masks_path=uploaded_masks_path,
                queued_at=isoformat(timestamp),
            )

            jobs.append(job)
            self._save_queue(jobs)

            return job

    def dequeue(self) -> Optional[RetrackJob]:
        """
        Get next pending job and mark it as processing.

        Returns None if no pending jobs available.
        """
        with self._lock:
            jobs = self._load_queue()

            # Find first pending job
            for job in jobs:
                if job.status == "pending":
                    job.status = "processing"
                    job.started_at = isoformat(utc_now())
                    self._save_queue(jobs)
                    return job

            return None

    def mark_completed(
        self, study_uid: str, series_uid: str, new_version_id: str
    ) -> None:
        """Mark a job as completed."""
        with self._lock:
            jobs = self._load_queue()

            for job in jobs:
                if (
                    job.study_uid == study_uid
                    and job.series_uid == series_uid
                    and job.new_version_id == new_version_id
                    and job.status == "processing"
                ):
                    job.status = "completed"
                    job.completed_at = isoformat(utc_now())
                    self._save_queue(jobs)
                    return

    def mark_failed(
        self,
        study_uid: str,
        series_uid: str,
        new_version_id: str,
        error_message: str,
    ) -> None:
        """Mark a job as failed."""
        with self._lock:
            jobs = self._load_queue()

            for job in jobs:
                if (
                    job.study_uid == study_uid
                    and job.series_uid == series_uid
                    and job.new_version_id == new_version_id
                    and job.status == "processing"
                ):
                    job.status = "failed"
                    job.failed_at = isoformat(utc_now())
                    job.error_message = error_message
                    self._save_queue(jobs)
                    return

    def get_job_status(
        self, study_uid: str, series_uid: str, version_id: Optional[str] = None
    ) -> Optional[RetrackJob]:
        """
        Get status of a specific job.

        If version_id is provided, matches by version_id. Otherwise returns
        the most recent job for the series.
        """
        jobs = self._load_queue()

        if version_id:
            for job in reversed(jobs):  # Most recent first
                if job.new_version_id == version_id:
                    return job
        else:
            # Return most recent job for this series
            for job in reversed(jobs):
                if job.study_uid == study_uid and job.series_uid == series_uid:
                    return job

        return None

    def has_active_job(self, study_uid: str, series_uid: str) -> bool:
        """Check if there's an active (pending or processing) job for this series."""
        jobs = self._load_queue()

        for job in reversed(jobs):  # Most recent first
            if job.study_uid == study_uid and job.series_uid == series_uid:
                if job.status in {"pending", "processing"}:
                    return True
                # If we found a completed/failed job, no active job exists
                return False

        return False

    def get_queue_position(
        self, study_uid: str, series_uid: str, version_id: str
    ) -> int:
        """Get position in queue for a specific job (1-indexed, 0 if not found)."""
        jobs = self._load_queue()
        pending_jobs = [j for j in jobs if j.status == "pending"]

        for idx, job in enumerate(pending_jobs, start=1):
            if (
                job.study_uid == study_uid
                and job.series_uid == series_uid
                and job.new_version_id == version_id
            ):
                return idx

        return 0
