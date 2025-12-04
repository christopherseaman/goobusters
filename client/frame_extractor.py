"""
Frame extraction utilities for the client.

Converts downloaded MP4 videos into per-frame PNGs stored under the configured
frames cache directory. These frames are served directly to the WebView so no
PHI ever leaves the iPad.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import cv2


class FrameExtractionError(RuntimeError):
    """Raised when frames cannot be extracted from a video."""


class FrameExtractor:
    def __init__(self, frames_root: Path) -> None:
        self.frames_root = Path(frames_root)
        self.frames_root.mkdir(parents=True, exist_ok=True)

    def frame_dir(self, study_uid: str, series_uid: str) -> Path:
        return self.frames_root / f"{study_uid}_{series_uid}"

    def manifest_path(self, study_uid: str, series_uid: str) -> Path:
        return self.frame_dir(study_uid, series_uid) / "manifest.json"

    def ensure_frames(self, video_path: Path, study_uid: str, series_uid: str) -> Path:
        """
        Extract frames for the requested video if missing or stale.
        Returns the directory containing PNGs.
        """
        target_dir = self.frame_dir(study_uid, series_uid)
        target_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = self.manifest_path(study_uid, series_uid)

        source_mtime = os.path.getmtime(video_path)
        if manifest_path.exists():
            with manifest_path.open() as f:
                manifest = json.load(f)
            if manifest.get("source_mtime") == source_mtime:
                return target_dir

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise FrameExtractionError(f"Unable to open video: {video_path}")

        frame_index = 0
        success = True
        while success:
            success, frame = cap.read()
            if not success:
                break
            filename = target_dir / f"frame_{frame_index:06d}.png"
            if not cv2.imwrite(str(filename), frame):
                cap.release()
                raise FrameExtractionError(f"Failed to write frame {filename}")
            frame_index += 1

        cap.release()

        manifest = {
            "study_uid": study_uid,
            "series_uid": series_uid,
            "frame_count": frame_index,
            "source_video": str(video_path),
            "source_mtime": source_mtime,
        }
        with manifest_path.open("w") as f:
            json.dump(manifest, f, indent=2)
        return target_dir

    def list_frames(self, study_uid: str, series_uid: str) -> list[str]:
        """
        Return sorted list of frame filenames for the given series.
        """
        target_dir = self.frame_dir(study_uid, series_uid)
        if not target_dir.exists():
            return []
        return sorted(
            [path.name for path in target_dir.glob("frame_*.png") if path.is_file()]
        )
