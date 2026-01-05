"""
Frame extraction utilities for the client.

Converts downloaded MP4 videos into per-frame WebP images stored under the configured
frames cache directory. These frames are served directly to the WebView so no
PHI ever leaves the iPad. WebP format matches the server's frame format.
"""

from __future__ import annotations

import json
import os
import tarfile
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

    def frames_tar_path(self, study_uid: str, series_uid: str) -> Path:
        """Path to cached .tar archive of frames (no gzip, WebP already compressed)."""
        return self.frame_dir(study_uid, series_uid) / "frames.tar"

    def frames_exist(self, video_path: Path, study_uid: str, series_uid: str) -> bool:
        """
        Fast check if frames exist and are up-to-date.
        Does NOT extract frames - just checks existence.
        Returns True if frames.tar and manifest.json exist and are current.
        """
        manifest_path = self.manifest_path(study_uid, series_uid)
        tar_path = self.frames_tar_path(study_uid, series_uid)
        
        if not manifest_path.exists() or not tar_path.exists():
            return False
        
        try:
            source_mtime = os.path.getmtime(video_path)
            with manifest_path.open() as f:
                manifest = json.load(f)
            return manifest.get("source_mtime") == source_mtime
        except (OSError, json.JSONDecodeError, KeyError):
            return False

    def ensure_frames(
        self, video_path: Path, study_uid: str, series_uid: str
    ) -> Path:
        """
        Extract frames for the requested video if missing or stale.
        Returns the directory containing WebP frames.
        """
        target_dir = self.frame_dir(study_uid, series_uid)
        target_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = self.manifest_path(study_uid, series_uid)

        source_mtime = os.path.getmtime(video_path)
        tar_path = self.frames_tar_path(study_uid, series_uid)

        # Check if frames and tar are up-to-date
        if manifest_path.exists() and tar_path.exists():
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
            filename = target_dir / f"frame_{frame_index:06d}.webp"
            # Use WebP format with quality 95 (matches server format)
            if not cv2.imwrite(
                str(filename), frame, [cv2.IMWRITE_WEBP_QUALITY, 95]
            ):
                cap.release()
                raise FrameExtractionError(f"Failed to write frame {filename}")
            frame_index += 1

        cap.release()

        # Build cached tar archive (no gzip - WebP already compressed)
        tar_path = self.frames_tar_path(study_uid, series_uid)
        with tarfile.open(tar_path, mode="w") as tar:
            for frame_path in sorted(target_dir.glob("frame_*.webp")):
                tar.add(frame_path, arcname=f"frames/{frame_path.name}")

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
        return sorted([
            path.name
            for path in target_dir.glob("frame_*.webp")
            if path.is_file()
        ])
