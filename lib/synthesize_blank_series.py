"""
Synthesize openable artifacts for a series that has a video on disk but ZERO
annotations of any labelId.

The normal tracking pipeline requires at least one annotation to run, so a
zero-annotation series can never produce the frames.tar / masks.tar /
frametype.json the viewer needs. This module materializes those artifacts
directly from the video: every frame extracted as WebP, an all-empty
frametype.json, and a blank masks.tar (zero mask files, every frame
has_mask=false). After synthesis the series serves through the exact same
endpoints as a tracked series.
"""

from __future__ import annotations

import logging
import tarfile
from pathlib import Path

import cv2

from lib.config import ServerConfig
from lib.mask_archive import build_mask_archive, build_mask_metadata

logger = logging.getLogger(__name__)

# Runaway guard: no real series video exceeds this many frames.
_MAX_FRAMES = 100_000


def synthesize_blank_series(
    study_uid: str,
    series_uid: str,
    video_path: Path | str,
    output_dir: Path | str,
    config: ServerConfig,
) -> Path:
    """
    Materialize frames.tar, masks.tar and frametype.json for a zero-annotation
    series, all frames marked empty (no mask). Returns output_dir.

    Idempotent: if masks.tar already exists, returns immediately. masks.tar is
    written last, so a crash mid-synthesis leaves no completion marker and the
    next call re-runs.
    """
    video_path = Path(video_path)
    output_dir = Path(output_dir)

    masks_tar = output_dir / "masks.tar"
    if masks_tar.exists():
        return output_dir

    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    frames_dir = output_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    masks_dir = output_dir / "masks"
    masks_dir.mkdir(parents=True, exist_ok=True)

    # Extract every decodable frame to WebP. Read until cap.read() returns
    # False rather than trusting CAP_PROP_FRAME_COUNT, which some codecs
    # over- or under-report.
    frame_count = 0
    cap = cv2.VideoCapture(str(video_path))
    try:
        while frame_count < _MAX_FRAMES:
            ret, frame = cap.read()
            if not ret:
                break
            frame_file = frames_dir / f"frame_{frame_count:06d}.webp"
            # cv2.imwrite returns False (does not raise) on disk-full / encode
            # failure. Fail loudly so the job is marked failed and retried,
            # rather than shipping a series with silently-missing frames.
            if not cv2.imwrite(
                str(frame_file), frame, [cv2.IMWRITE_WEBP_QUALITY, 85]
            ):
                raise IOError(
                    f"Failed to write frame {frame_count} (disk full?): {frame_file}"
                )
            frame_count += 1
    finally:
        cap.release()

    if frame_count == 0:
        raise ValueError(f"No decodable frames in video: {video_path}")
    if frame_count >= _MAX_FRAMES:
        # Hit the runaway guard: a corrupt/looping stream. Fail rather than
        # ship a truncated series the idempotency marker would lock in forever.
        raise ValueError(
            f"Video exceeded {_MAX_FRAMES} frame cap (corrupt/looping?): {video_path}"
        )

    # All-empty frametype.json: every frame present, no mask, empty label.
    empty_id = config.empty_id
    frametype = {
        str(i): {
            "type": "empty",
            "is_annotation": False,
            "has_mask": False,
            "track_id": f"track_{i}",
            "label_id": empty_id,
        }
        for i in range(frame_count)
    }
    import json

    (output_dir / "frametype.json").write_text(json.dumps(frametype, indent=2))

    # Blank masks.tar: build_mask_metadata reads the frametype.json above
    # (from masks_dir.parent) and emits frame_count entries, all has_mask=false.
    series = {"study_uid": study_uid, "series_uid": series_uid}
    metadata = build_mask_metadata(series, masks_dir, config.flow_method, config)
    archive_bytes = build_mask_archive(masks_dir, metadata)

    # frames.tar (arcname "frames" -> members "frames/frame_NNNNNN.webp").
    # Write to a temp path then atomically rename, so a concurrent reader (or a
    # crash) never sees a half-written archive — same pattern as the queue and
    # status writers elsewhere in this codebase.
    frames_tar = output_dir / "frames.tar"
    frames_tmp = output_dir / "frames.tar.tmp"
    with tarfile.open(frames_tmp, "w") as tar:
        tar.add(frames_dir, arcname="frames")
    frames_tmp.replace(frames_tar)

    # masks.tar written last as the completion marker — also atomic, so a crash
    # mid-write can never leave a corrupt marker the idempotency check trusts.
    masks_tmp = output_dir / "masks.tar.tmp"
    masks_tmp.write_bytes(archive_bytes)
    masks_tmp.replace(masks_tar)

    logger.info(
        f"Synthesized blank series {study_uid}/{series_uid}: "
        f"{frame_count} frames, 0 masks"
    )
    return output_dir
