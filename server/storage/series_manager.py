"""
Filesystem-backed state store for series metadata, activity tracking, and
selection logic. This module is shared by the Flask API and background workers.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from pathlib import Path
from threading import Lock
from typing import Optional

import mdai
import pandas as pd

from lib.config import ServerConfig
from lib.mask_archive import (
    get_mask_count,
    get_version_id,
    mask_series_dir,
)
from track import find_annotations_file, find_images_dir

ISO_FORMAT = "%Y-%m-%dT%H:%M:%SZ"


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def isoformat(dt: Optional[datetime]) -> Optional[str]:
    if not dt:
        return None
    return dt.astimezone(timezone.utc).strftime(ISO_FORMAT)


def parse_time(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    return datetime.strptime(value, ISO_FORMAT).replace(tzinfo=timezone.utc)


@dataclass
class SeriesMetadata:
    """Minimal series metadata. Most fields derived from filesystem or annotations."""

    study_uid: str
    series_uid: str
    exam_number: Optional[int]
    series_number: Optional[int]
    dataset_name: str
    video_path: str
    status: str = "pending"  # pending | completed (user completion status only)


class SeriesManager:
    def __init__(self, config: ServerConfig):
        self.config = config
        self.mask_root = Path(config.mask_storage_path)
        self.flow_method = config.flow_method
        self._lock = Lock()
        # In-memory index: populated from MD.ai annotations on startup
        self._index: list[SeriesMetadata] = []
        self._build_index_from_annotations()

    # ------------------------------------------------------------------ Helpers
    def _series_output_dir(self, study_uid: str, series_uid: str) -> Path:
        """Get the series output directory (parent of retrack/)."""
        return mask_series_dir(
            self.mask_root, self.flow_method, study_uid, series_uid
        )

    def _status_path(self, study_uid: str, series_uid: str) -> Path:
        """Get status.json path in series output directory (user completion status and activity)."""
        return self._series_output_dir(study_uid, series_uid) / "status.json"

    def _frametype_path(self, study_uid: str, series_uid: str) -> Path:
        """Get frametype.json path (prefer retrack, fallback to initial)."""
        output_dir = self._series_output_dir(study_uid, series_uid)
        retrack_frametype = output_dir / "retrack" / "frametype.json"
        if retrack_frametype.exists():
            return retrack_frametype
        return output_dir / "frametype.json"

    # --------------------------------------------------------------- Filesystem helpers
    def _get_version_id(self, study_uid: str, series_uid: str) -> Optional[str]:
        """Get version_id for a series (frametype first, then masks.tar metadata)."""
        frametype_path = self._frametype_path(study_uid, series_uid)
        if frametype_path.exists():
            try:
                with frametype_path.open() as f:
                    data = json.load(f)
                    if "_version_id" in data:
                        return data.get("_version_id") or None
            except (json.JSONDecodeError, KeyError, TypeError):
                pass

        # Fallback: read from masks.tar metadata (if built)
        archive_path = self._get_archive_path(study_uid, series_uid)
        return get_version_id(archive_path)

    def _get_archive_path(self, study_uid: str, series_uid: str) -> Path:
        """Get masks.tar path (prefer retrack, fallback to initial)."""
        output_dir = self._series_output_dir(study_uid, series_uid)
        retrack_archive = output_dir / "retrack" / "masks.tar"
        if retrack_archive.exists():
            return retrack_archive
        return output_dir / "masks.tar"

    def _get_mask_count(self, study_uid: str, series_uid: str) -> int:
        """Get mask_count from masks.tar metadata.json."""
        archive_path = self._get_archive_path(study_uid, series_uid)
        return get_mask_count(archive_path)

    def _get_tracking_status(self, study_uid: str, series_uid: str) -> str:
        """Compute tracking_status from filesystem (no persistence)."""
        output_dir = self._series_output_dir(study_uid, series_uid)

        # Check for retracking in progress
        retrack_temp_dir = output_dir / "retrack" / "masks_temp"
        if retrack_temp_dir.exists() and list(retrack_temp_dir.glob("*.webp")):
            return "retracking"

        # Check for retracked archive
        retrack_archive = output_dir / "retrack" / "masks.tar"
        if retrack_archive.exists():
            return "completed"

        # Check for initial tracking archive (server creates masks.tar, track.py creates masks.tar.gz)
        initial_archive_tar = output_dir / "masks.tar"
        initial_archive_targz = output_dir / "masks.tar.gz"
        if initial_archive_tar.exists() or initial_archive_targz.exists():
            return "completed"

        return "never_run"

    def _read_status(self, study_uid: str, series_uid: str) -> dict:
        """Read status.json: contains status and activity tracking."""
        status_path = self._status_path(study_uid, series_uid)
        if status_path.exists():
            try:
                with status_path.open() as f:
                    return json.load(f)
            except (json.JSONDecodeError, KeyError):
                pass
        return {"status": "pending", "activity": {}}

    def _write_status(
        self,
        study_uid: str,
        series_uid: str,
        status: Optional[str] = None,
        user_email: Optional[str] = None,
    ) -> None:
        """Update status.json with status and/or activity."""
        status_path = self._status_path(study_uid, series_uid)
        data = self._read_status(study_uid, series_uid)

        if status is not None:
            data["status"] = status
        if user_email is not None:
            if "activity" not in data:
                data["activity"] = {}
            data["activity"][user_email] = isoformat(utc_now())

        status_path.parent.mkdir(parents=True, exist_ok=True)
        with status_path.open("w") as f:
            json.dump(data, f, indent=2)

    def _get_status(self, study_uid: str, series_uid: str) -> str:
        """Get user completion status from status.json."""
        data = self._read_status(study_uid, series_uid)
        return data.get("status", "pending")

    # --------------------------------------------------------------- Bootstrapping
    def _build_index_from_annotations(self) -> None:
        try:
            annotations_path = find_annotations_file(
                str(self.config.data_dir),
                self.config.project_id,
                self.config.dataset_id,
            )
            images_dir = find_images_dir(
                str(self.config.data_dir),
                self.config.project_id,
                self.config.dataset_id,
            )
        except FileNotFoundError as exc:
            raise RuntimeError(
                "Unable to locate MD.ai dataset. Run `uv run python3 track.py` once "
                "to download the project locally before starting the server."
            ) from exc

        annotations_blob = mdai.common_utils.json_to_dataframe(annotations_path)
        annotations_df = pd.DataFrame(annotations_blob["annotations"])
        studies_df = pd.DataFrame(annotations_blob["studies"])
        studies_lookup = {
            row["StudyInstanceUID"]: row for _, row in studies_df.iterrows()
        }

        records: list[dict] = []
        for (study_uid, series_uid), video_df in annotations_df.groupby([
            "StudyInstanceUID",
            "SeriesInstanceUID",
        ]):
            exam_number = None
            series_number = None
            dataset_name = video_df.iloc[0].get("dataset", "Unknown")

            study_info = studies_lookup.get(study_uid)
            if study_info is not None:
                exam_number = study_info.get("number")

            # Series number may live under different keys depending on export
            raw_series_number = video_df.iloc[0].get(
                "SeriesNumber"
            ) or video_df.iloc[0].get("seriesNumber")
            if raw_series_number is not None:
                try:
                    series_number = int(raw_series_number)
                except (ValueError, TypeError):
                    series_number = None

            video_path = Path(images_dir) / study_uid / f"{series_uid}.mp4"
            metadata = SeriesMetadata(
                study_uid=study_uid,
                series_uid=series_uid,
                exam_number=int(exam_number)
                if exam_number not in (None, "")
                else None,
                series_number=series_number,
                dataset_name=str(dataset_name),
                video_path=str(video_path),
            )
            records.append(metadata)

        # Store in-memory index
        self._index = records

    # ------------------------------------------------------------- CRUD operations
    def _read_metadata_from_index(
        self, study_uid: str, series_uid: str
    ) -> Optional[SeriesMetadata]:
        """Read series metadata from in-memory index."""
        for metadata in self._index:
            if (
                metadata.study_uid == study_uid
                and metadata.series_uid == series_uid
            ):
                return metadata
        return None

    def list_series(self) -> list[SeriesMetadata]:
        """List all series from in-memory index, with status from filesystem."""
        items = []
        for metadata in self._index:
            # Get status from filesystem (reads per-series status.json on demand)
            metadata.status = self._get_status(
                metadata.study_uid, metadata.series_uid
            )
            items.append(metadata)
        return items

    def get_series(self, study_uid: str, series_uid: str) -> SeriesMetadata:
        """Get series metadata with status from filesystem."""
        metadata = self._read_metadata_from_index(study_uid, series_uid)
        if not metadata:
            raise FileNotFoundError(f"No metadata for {study_uid}/{series_uid}")
        metadata.status = self._get_status(study_uid, series_uid)
        return metadata

    def mark_complete(
        self, study_uid: str, series_uid: str, user_email: Optional[str] = None
    ) -> SeriesMetadata:
        """Mark series as completed by user."""
        with self._lock:
            self._write_status(
                study_uid, series_uid, status="completed", user_email=user_email
            )
            return self.get_series(study_uid, series_uid)

    def reopen(self, study_uid: str, series_uid: str) -> SeriesMetadata:
        """Reopen series (mark as pending)."""
        with self._lock:
            self._write_status(study_uid, series_uid, status="pending")
            return self.get_series(study_uid, series_uid)

    def clear_activity(
        self, study_uid: str, series_uid: str
    ) -> SeriesMetadata:
        """Clear all activity tracking for a series."""
        with self._lock:
            status_path = self._status_path(study_uid, series_uid)
            data = self._read_status(study_uid, series_uid)
            # Clear activity but preserve status
            data["activity"] = {}
            status_path.parent.mkdir(parents=True, exist_ok=True)
            with status_path.open("w") as f:
                json.dump(data, f, indent=2)
            return self.get_series(study_uid, series_uid)

    def mark_activity(
        self, study_uid: str, series_uid: str, user_email: Optional[str]
    ) -> SeriesMetadata:
        """Record user activity on series."""
        with self._lock:
            if user_email:
                self._write_status(study_uid, series_uid, user_email=user_email)
            return self.get_series(study_uid, series_uid)

    def record_view(
        self, study_uid: str, series_uid: str, user_email: Optional[str]
    ) -> SeriesMetadata:
        """Record that a user viewed/selected this series (same as mark_activity)."""
        return self.mark_activity(study_uid, series_uid, user_email)

    # ----------------------------------------------------------- Activity helpers
    def activity_history(
        self, study_uid: str, series_uid: str
    ) -> dict[str, str]:
        """Get activity history: dict of user_email -> last_activity_at."""
        data = self._read_status(study_uid, series_uid)
        return data.get("activity", {})

    # ----------------------------------------------------------- Public helpers
    def get_version_id(self, study_uid: str, series_uid: str) -> Optional[str]:
        """Get version_id from frametype.json (tied to tracking revision)."""
        return self._get_version_id(study_uid, series_uid)

    def get_tracking_status(self, study_uid: str, series_uid: str) -> str:
        """Get tracking_status from filesystem."""
        return self._get_tracking_status(study_uid, series_uid)

    def get_mask_count(self, study_uid: str, series_uid: str) -> int:
        """Get mask_count from masks.tar metadata.json."""
        return self._get_mask_count(study_uid, series_uid)

    # -------------------------------------------------------------- Selection logic
    def select_next_series(
        self, user_email: Optional[str] = None
    ) -> Optional[SeriesMetadata]:
        """
        Get the series for a user to work on.

        Logic: Prefer series where most recent activity is from another user (not current user).
        This ensures users work on different series and don't conflict.

        Note: In-progress series (active by another user) are still available, just sorted lower.
        """
        # Debug: check index size
        import logging
        import sys

        logger = logging.getLogger(__name__)
        logger.debug(f"select_next_series: Index has {len(self._index)} series")
        print(
            f"DEBUG: select_next_series: Index has {len(self._index)} series",
            file=sys.stderr,
            flush=True,
        )

        all_series = self.list_series()
        logger.debug(
            f"select_next_series: list_series() returned {len(all_series)} series"
        )
        print(
            f"DEBUG: select_next_series: list_series() returned {len(all_series)} series",
            file=sys.stderr,
            flush=True,
        )

        candidates: list[SeriesMetadata] = []
        status_counts = {"completed": 0, "pending": 0, "other": 0}
        for metadata in all_series:
            status = metadata.status
            if status == "completed":
                status_counts["completed"] += 1
                continue
            if status == "pending":
                status_counts["pending"] += 1
            else:
                status_counts["other"] += 1
            candidates.append(metadata)

        if not candidates:
            # Debug: log why no candidates
            import logging
            import sys

            logger = logging.getLogger(__name__)
            warning_msg = (
                f"select_next_series: No candidates. Total series: {len(all_series)}, "
                f"Status breakdown: {status_counts}"
            )
            logger.warning(warning_msg)
            print(warning_msg, file=sys.stderr, flush=True)
            return None

        # Filter out series with recent activity from other users (within 24 hours)
        # This prevents conflicts by not assigning series that others are actively working on
        RECENT_VIEW_THRESHOLD = timedelta(hours=24)
        now = utc_now()
        filtered_candidates = []
        
        for item in candidates:
            data = self._read_status(item.study_uid, item.series_uid)
            activity = data.get("activity", {})
            
            # Check if another user has been active recently
            has_recent_other_activity = False
            for other_user, timestamp_str in activity.items():
                if other_user == user_email:
                    continue  # Skip current user
                timestamp = parse_time(timestamp_str)
                if timestamp and (now - timestamp) < RECENT_VIEW_THRESHOLD:
                    has_recent_other_activity = True
                    break
            
            # Skip series with recent activity from other users (unless current user was also recently active)
            if has_recent_other_activity:
                # Only skip if current user hasn't been active recently
                current_user_time = None
                if user_email and user_email in activity:
                    current_user_time = parse_time(activity[user_email])
                
                if not current_user_time or (now - current_user_time) >= RECENT_VIEW_THRESHOLD:
                    continue  # Skip this series - another user is actively working on it
            
            filtered_candidates.append(item)
        
        # If all candidates were filtered out, fall back to all candidates (sorted by priority)
        if not filtered_candidates:
            filtered_candidates = candidates

        def sort_key(item: SeriesMetadata):
            data = self._read_status(item.study_uid, item.series_uid)
            activity = data.get("activity", {})

            # Find most recent activity from current user (higher priority = lower sort value)
            current_user_time = None
            if user_email and user_email in activity:
                timestamp = parse_time(activity[user_email])
                if timestamp:
                    current_user_time = timestamp

            # Find most recent activity from other users (lower priority = higher sort value)
            most_recent_other_time = None
            for other_user, timestamp_str in activity.items():
                if other_user == user_email:
                    continue  # Skip current user
                timestamp = parse_time(timestamp_str)
                if timestamp and (
                    most_recent_other_time is None
                    or timestamp > most_recent_other_time
                ):
                    most_recent_other_time = timestamp

            # Sorting logic:
            # 1. Series with recent activity by current user (highest priority) -> negative timestamp (sorted first)
            # 2. Series with no activity (medium priority) -> 0.0
            # 3. Series with recent activity by others (lowest priority) -> positive timestamp (sorted last)
            if current_user_time:
                # Recent activity by current user -> highest priority (negative = sorted first)
                return -current_user_time.timestamp()
            elif most_recent_other_time:
                # Recent activity by others -> lowest priority (positive = sorted last)
                return most_recent_other_time.timestamp()
            else:
                # No activity -> medium priority
                return 0.0

        selected = sorted(filtered_candidates, key=sort_key)[0]
        return self.record_view(
            selected.study_uid, selected.series_uid, user_email
        )
