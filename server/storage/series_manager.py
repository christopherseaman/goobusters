"""
Filesystem-backed state store for series metadata, activity tracking, and
selection logic. This module is shared by the Flask API and background workers.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from threading import Lock
from typing import Optional

import mdai
import pandas as pd

from lib.config import ServerConfig
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
    study_uid: str
    series_uid: str
    exam_number: Optional[int]
    series_number: Optional[int]
    dataset_name: str
    video_path: str
    status: str = "pending"  # pending | in_progress | completed
    tracking_status: str = "never_run"  # never_run | pending | completed | failed
    current_version_id: Optional[str] = None
    last_editor: Optional[str] = None
    last_viewed_at: Optional[str] = None
    last_activity_at: Optional[str] = None
    last_activity_by: Optional[str] = None
    mask_count: int = 0

    def touch_view(self, user_email: Optional[str]) -> None:
        now = isoformat(utc_now())
        self.last_viewed_at = now
        if user_email:
            self.last_activity_by = user_email

    def touch_activity(self, user_email: Optional[str]) -> None:
        now = isoformat(utc_now())
        self.last_activity_at = now
        if user_email:
            self.last_activity_by = user_email


class SeriesManager:
    def __init__(self, config: ServerConfig):
        self.config = config
        self.state_root = config.server_state_path
        self.series_root = self.state_root / "series"
        self.activity_log = self.state_root / "activity_log.json"
        self.index_file = self.state_root / "series_index.json"
        self.series_root.mkdir(parents=True, exist_ok=True)
        self._lock = Lock()
        self._ensure_index()

    # ------------------------------------------------------------------ Helpers
    def _series_key(self, study_uid: str, series_uid: str) -> str:
        return f"{study_uid}__{series_uid}"

    def _series_dir(self, study_uid: str, series_uid: str) -> Path:
        return self.series_root / self._series_key(study_uid, series_uid)

    def _metadata_path(self, study_uid: str, series_uid: str) -> Path:
        return self._series_dir(study_uid, series_uid) / "metadata.json"

    def _activity_path(self, study_uid: str, series_uid: str) -> Path:
        return self._series_dir(study_uid, series_uid) / "activity.json"

    # --------------------------------------------------------------- Bootstrapping
    def _ensure_index(self) -> None:
        if self.index_file.exists():
            return
        self._build_index_from_annotations()

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
        for (study_uid, series_uid), video_df in annotations_df.groupby(
            ["StudyInstanceUID", "SeriesInstanceUID"]
        ):
            exam_number = None
            series_number = None
            dataset_name = video_df.iloc[0].get("dataset", "Unknown")

            study_info = studies_lookup.get(study_uid)
            if study_info is not None:
                exam_number = study_info.get("number")

            # Series number may live under different keys depending on export
            raw_series_number = (
                video_df.iloc[0].get("SeriesNumber")
                or video_df.iloc[0].get("seriesNumber")
            )
            if raw_series_number is not None:
                try:
                    series_number = int(raw_series_number)
                except (ValueError, TypeError):
                    series_number = None

            video_path = Path(images_dir) / study_uid / f"{series_uid}.mp4"
            metadata = SeriesMetadata(
                study_uid=study_uid,
                series_uid=series_uid,
                exam_number=int(exam_number) if exam_number not in (None, "") else None,
                series_number=series_number,
                dataset_name=str(dataset_name),
                video_path=str(video_path),
            )
            self._write_metadata(metadata)
            records.append(asdict(metadata))

        with self.index_file.open("w") as f:
            json.dump(records, f, indent=2)

    # ------------------------------------------------------------- CRUD operations
    def _write_metadata(self, metadata: SeriesMetadata) -> None:
        series_dir = self._series_dir(metadata.study_uid, metadata.series_uid)
        series_dir.mkdir(parents=True, exist_ok=True)
        path = series_dir / "metadata.json"
        with path.open("w") as f:
            json.dump(asdict(metadata), f, indent=2)

    def _read_metadata(self, study_uid: str, series_uid: str) -> SeriesMetadata:
        path = self._metadata_path(study_uid, series_uid)
        if not path.exists():
            raise FileNotFoundError(f"No metadata for {study_uid}/{series_uid}")
        with path.open() as f:
            payload = json.load(f)
        return SeriesMetadata(**payload)

    def list_series(self) -> list[SeriesMetadata]:
        items: list[SeriesMetadata] = []
        for metadata_path in self.series_root.glob("*/metadata.json"):
            with metadata_path.open() as f:
                payload = json.load(f)
            items.append(SeriesMetadata(**payload))
        return items

    def get_series(self, study_uid: str, series_uid: str) -> SeriesMetadata:
        return self._read_metadata(study_uid, series_uid)

    def mark_complete(self, study_uid: str, series_uid: str) -> SeriesMetadata:
        with self._lock:
            metadata = self._read_metadata(study_uid, series_uid)
            metadata.status = "completed"
            self._write_metadata(metadata)
            return metadata

    def reopen(self, study_uid: str, series_uid: str) -> SeriesMetadata:
        with self._lock:
            metadata = self._read_metadata(study_uid, series_uid)
            metadata.status = "pending"
            self._write_metadata(metadata)
            return metadata

    def mark_activity(self, study_uid: str, series_uid: str, user_email: Optional[str]) -> SeriesMetadata:
        with self._lock:
            metadata = self._read_metadata(study_uid, series_uid)
            metadata.touch_activity(user_email)
            self._append_activity_event(study_uid, series_uid, "activity", user_email)
            self._write_metadata(metadata)
            return metadata

    def record_view(self, study_uid: str, series_uid: str, user_email: Optional[str]) -> SeriesMetadata:
        with self._lock:
            metadata = self._read_metadata(study_uid, series_uid)
            metadata.touch_view(user_email)
            self._append_activity_event(study_uid, series_uid, "view", user_email)
            self._write_metadata(metadata)
            return metadata

    def update_version(self, study_uid: str, series_uid: str, version_id: str, editor: str) -> SeriesMetadata:
        with self._lock:
            metadata = self._read_metadata(study_uid, series_uid)
            metadata.current_version_id = version_id
            metadata.last_editor = editor
            metadata.touch_activity(editor)
            self._write_metadata(metadata)
            return metadata

    def update_tracking_status(self, study_uid: str, series_uid: str, status: str, mask_count: int = 0) -> SeriesMetadata:
        with self._lock:
            metadata = self._read_metadata(study_uid, series_uid)
            metadata.tracking_status = status
            metadata.mask_count = mask_count or metadata.mask_count
            self._write_metadata(metadata)
            return metadata

    # ----------------------------------------------------------- Activity helpers
    def _append_activity_event(self, study_uid: str, series_uid: str, event_type: str, user_email: Optional[str]) -> None:
        path = self._activity_path(study_uid, series_uid)
        events = []
        if path.exists():
            with path.open() as f:
                events = json.load(f)
        events.append(
            {
                "event": event_type,
                "at": isoformat(utc_now()),
                "user": user_email,
            }
        )
        with path.open("w") as f:
            json.dump(events[-50:], f, indent=2)  # keep last 50 events

    def activity_history(self, study_uid: str, series_uid: str) -> list[dict]:
        path = self._activity_path(study_uid, series_uid)
        if not path.exists():
            return []
        with path.open() as f:
            return json.load(f)

    # -------------------------------------------------------------- Selection logic
    def select_next_series(self, user_email: Optional[str] = None) -> Optional[SeriesMetadata]:
        threshold = utc_now() - timedelta(minutes=self.config.recent_view_threshold_minutes)
        candidates: list[SeriesMetadata] = []
        for metadata in self.list_series():
            if metadata.status == "completed":
                continue
            last_activity = parse_time(metadata.last_activity_at)
            if last_activity and last_activity > threshold:
                continue
            candidates.append(metadata)

        if not candidates:
            return None

        def sort_key(item: SeriesMetadata):
            last_view = parse_time(item.last_viewed_at) or datetime.fromtimestamp(0, tz=timezone.utc)
            return last_view

        selected = sorted(candidates, key=sort_key)[0]
        return self.record_view(selected.study_uid, selected.series_uid, user_email)
