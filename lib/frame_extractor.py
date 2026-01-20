"""
Shared frame extraction utilities using imageio.

Uses imageio with ffmpeg backend for cross-platform video→frame extraction.
Works on server (with system ffmpeg) and iOS (with mobile-ffmpeg).
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Iterator, Tuple

import imageio.v3 as iio


class FrameExtractionError(RuntimeError):
    """Raised when frames cannot be extracted from a video."""


def get_video_properties(video_path: str | Path) -> Tuple[int, int, int, float]:
    """
    Get video properties without extracting frames.

    Args:
        video_path: Path to the video file

    Returns:
        Tuple of (total_frames, width, height, fps)
    """
    video_path = Path(video_path)
    if not video_path.exists():
        raise FrameExtractionError(f"Video not found: {video_path}")

    props = iio.improps(video_path, plugin="pyav")
    meta = iio.immeta(video_path, plugin="pyav")

    # props.shape is (n_frames, height, width, channels)
    n_frames = props.shape[0] if props.shape else 0
    height = props.shape[1] if len(props.shape) > 1 else 0
    width = props.shape[2] if len(props.shape) > 2 else 0
    fps = meta.get("fps", 30.0)

    return n_frames, width, height, fps


def iter_frames(video_path: str | Path) -> Iterator[Tuple[int, any]]:
    """
    Iterate over frames in a video.

    Args:
        video_path: Path to the video file

    Yields:
        Tuple of (frame_index, frame_array) where frame_array is numpy RGB
    """
    video_path = Path(video_path)
    if not video_path.exists():
        raise FrameExtractionError(f"Video not found: {video_path}")

    for idx, frame in enumerate(iio.imiter(video_path, plugin="pyav")):
        yield idx, frame


def extract_frame(video_path: str | Path, frame_index: int) -> any:
    """
    Extract a single frame from a video.

    Args:
        video_path: Path to the video file
        frame_index: 0-based frame index

    Returns:
        Frame as numpy array (RGB)
    """
    video_path = Path(video_path)
    if not video_path.exists():
        raise FrameExtractionError(f"Video not found: {video_path}")

    return iio.imread(video_path, index=frame_index, plugin="pyav")


class FrameExtractor:
    """
    Extracts frames from videos and caches them as WebP images.

    This class handles the video→frames extraction pipeline, storing
    frames as WebP images with a manifest for cache validation.
    """

    def __init__(self, frames_root: Path) -> None:
        self.frames_root = Path(frames_root)
        self.frames_root.mkdir(parents=True, exist_ok=True)

    def frame_dir(self, study_uid: str, series_uid: str) -> Path:
        return self.frames_root / f"{study_uid}_{series_uid}"

    def manifest_path(self, study_uid: str, series_uid: str) -> Path:
        return self.frame_dir(study_uid, series_uid) / "manifest.json"

    def frames_exist(self, video_path: Path, study_uid: str, series_uid: str) -> bool:
        """
        Check if frames exist and are up-to-date.

        Returns True if manifest exists and source video hasn't changed.
        """
        manifest_path = self.manifest_path(study_uid, series_uid)

        if not manifest_path.exists():
            return False

        try:
            source_mtime = os.path.getmtime(video_path)
            with manifest_path.open() as f:
                manifest = json.load(f)
            return manifest.get("source_mtime") == source_mtime
        except (OSError, json.JSONDecodeError, KeyError):
            return False

    def ensure_frames(
        self, video_path: Path, study_uid: str, series_uid: str, quality: int = 95
    ) -> Path:
        """
        Extract frames for the requested video if missing or stale.

        Args:
            video_path: Path to source video
            study_uid: Study UID for organizing output
            series_uid: Series UID for organizing output
            quality: WebP quality (0-100)

        Returns:
            Path to directory containing WebP frames
        """
        video_path = Path(video_path)
        target_dir = self.frame_dir(study_uid, series_uid)
        target_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = self.manifest_path(study_uid, series_uid)

        source_mtime = os.path.getmtime(video_path)

        # Check if frames are up-to-date
        if manifest_path.exists():
            try:
                with manifest_path.open() as f:
                    manifest = json.load(f)
                if manifest.get("source_mtime") == source_mtime:
                    return target_dir
            except (json.JSONDecodeError, KeyError):
                pass

        # Extract frames
        frame_count = 0
        for frame_index, frame in iter_frames(video_path):
            filename = target_dir / f"frame_{frame_index:06d}.webp"
            iio.imwrite(filename, frame, quality=quality)
            frame_count += 1

        # Write manifest
        manifest = {
            "study_uid": study_uid,
            "series_uid": series_uid,
            "frame_count": frame_count,
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
            [path.name for path in target_dir.glob("frame_*.webp") if path.is_file()]
        )
