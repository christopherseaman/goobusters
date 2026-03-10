"""
Shared tracking pipeline and archive utilities.

Contains run_tracking_pipeline() (the shared core for optical flow tracking)
and build_mask_archive_from_directory() (for building server-format archives).
Called by retrack_worker.py for both initial and retrack jobs.
"""

from __future__ import annotations

import logging
import os
import shutil
import sys
from pathlib import Path
from typing import Optional

# Add paths: project root (for lib imports) and lib/server (for server package imports)
import os

_project_root = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
_lib_server = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

if _project_root not in sys.path:
    sys.path.insert(0, _project_root)
if _lib_server not in sys.path:
    sys.path.insert(0, _lib_server)

import pandas as pd  # noqa: E402

from lib.config import ServerConfig  # noqa: E402
from lib.mask_archive import (  # noqa: E402
    build_mask_archive,
    build_mask_metadata,
    mask_series_dir,
)
from lib.multi_frame_tracker import (  # noqa: E402
    process_video_with_multi_frame_tracking,
    set_label_ids,
)
from lib.optical import create_identity_file, copy_annotations_to_output  # noqa: E402
from lib.opticalflowprocessor import OpticalFlowProcessor  # noqa: E402
from server.storage.series_manager import SeriesManager  # noqa: E402
from track import find_images_dir  # noqa: E402

logger = logging.getLogger(__name__)


def build_mask_archive_from_directory(
    study_uid: str,
    series_uid: str,
    masks_dir: Path,
    output_dir: Path,
    config: ServerConfig,
    series_manager: SeriesManager,
) -> None:
    """
    Build masks.tar and frames.tar archives from existing directories.

    Used when masks already exist (e.g., from track.py) and we just need to
    build the server format archives with metadata.json.
    """
    import tarfile

    series = series_manager.get_series(study_uid, series_uid)
    metadata = build_mask_metadata(
        series, masks_dir, config.flow_method, config
    )
    archive_path = output_dir / "masks.tar"
    archive_bytes = build_mask_archive(masks_dir, metadata)
    with archive_path.open("wb") as f:
        f.write(archive_bytes)
    logger.info(
        f"Built masks.tar from existing masks for {study_uid}/{series_uid}"
    )

    # Also create frames.tar if frames/ exists but frames.tar doesn't
    frames_dir = output_dir / "frames"
    frames_tar_path = output_dir / "frames.tar"
    if frames_dir.exists() and not frames_tar_path.exists():
        frame_files = list(frames_dir.glob("*.webp"))
        if frame_files:
            with tarfile.open(frames_tar_path, mode="w") as tar:
                for frame_path in sorted(frame_files):
                    tar.add(frame_path, arcname=f"frames/{frame_path.name}")
            logger.info(
                f"Built frames.tar from existing frames for {study_uid}/{series_uid}"
            )


def run_tracking_pipeline(
    study_uid: str,
    series_uid: str,
    annotations_df: pd.DataFrame,
    studies_df: pd.DataFrame,
    annotations_blob: dict,
    config: ServerConfig,
    is_retrack: bool = False,
    version_id: Optional[str] = None,
) -> Path:
    """
    Shared tracking pipeline used by both initial tracking and retrack.
    Returns the output directory path containing masks and metadata.

    Args:
        is_retrack: If True, write to retrack subdirectory to avoid overwriting original masks
        version_id: Version ID to write to frametype.json (for retrack).
                   For initial tracking, this is None (no version yet).
                   For retrack, this is the version_id for the retracked masks.
    """
    # Determine images dir and video path (matches track.py behavior)
    images_dir = find_images_dir(
        str(config.data_dir),
        config.project_id,
        config.dataset_id,
    )

    def construct_video_path(base_dir, study_uid, series_uid):
        return os.path.join(base_dir, study_uid, f"{series_uid}.mp4")

    annotations_df = annotations_df.copy()
    # Handle input that may not have StudyInstanceUID/SeriesInstanceUID columns (retrack uploads)
    if "StudyInstanceUID" not in annotations_df.columns:
        annotations_df["StudyInstanceUID"] = study_uid
    if "SeriesInstanceUID" not in annotations_df.columns:
        annotations_df["SeriesInstanceUID"] = series_uid

    # labelName should already be present (added by caller for retrack, or from original annotations for initial tracking)
    if "labelName" not in annotations_df.columns:
        raise ValueError(
            f"annotations_df must have 'labelName' column. Columns: {list(annotations_df.columns)}"
        )

    annotations_df["video_path"] = annotations_df.apply(
        lambda row: construct_video_path(
            images_dir,
            row["StudyInstanceUID"],
            row["SeriesInstanceUID"],
        ),
        axis=1,
    )

    video_path = annotations_df.iloc[0]["video_path"]
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")

    # Build output dir
    # For retrack, write to retrack subdirectory to avoid overwriting original masks
    base_output_dir = mask_series_dir(
        Path(config.mask_storage_path),
        config.flow_method,
        study_uid,
        series_uid,
    )
    if is_retrack:
        series_output_dir = base_output_dir / "retrack"
        # Ensure retrack directory is clean (remove if it exists from previous failed retrack)
        if series_output_dir.exists():
            shutil.rmtree(series_output_dir)
        series_output_dir.mkdir(parents=True, exist_ok=True)
    else:
        series_output_dir = base_output_dir
        series_output_dir.mkdir(parents=True, exist_ok=True)

    # Verify labelName exists and is valid before calling create_identity_file
    if "labelName" not in annotations_df.columns:
        raise ValueError(
            f"labelName missing before create_identity_file. Columns: {list(annotations_df.columns)}"
        )

    # Ensure labelName column has no NaN values (should have been populated by caller)
    if annotations_df["labelName"].isna().any():
        missing_frames = (
            annotations_df[annotations_df["labelName"].isna()][
                "frameNumber"
            ].tolist()
            if "frameNumber" in annotations_df.columns
            else "unknown"
        )
        raise ValueError(
            f"labelName has NaN values. Missing labelName for frames: {missing_frames}. "
            f"Columns: {list(annotations_df.columns)}"
        )

    # Create identity file (same as track.py)
    create_identity_file(
        str(series_output_dir),
        study_uid,
        series_uid,
        annotations_df,
        studies_df,
    )

    # Copy annotations to output (same as track.py)
    copy_annotations_to_output(
        str(series_output_dir), annotations_df, annotations_blob
    )

    # Use version_id (renamed from new_version_id for consistency)
    resolved_version_id = version_id

    import time as _time
    _tp0 = _time.perf_counter()

    # Initialize optical flow processor
    set_label_ids(config.label_id, config.empty_id)
    logger.info(f"Optical flow: method={config.flow_method}, preset={config.flow_preset}")
    flow_processor = OpticalFlowProcessor(config.flow_method, preset=config.flow_preset)
    _tp1 = _time.perf_counter()
    logger.info(f"  flow processor init: {_tp1 - _tp0:.2f}s")

    # Process the video with multi-frame tracking (same as track.py)
    # For retrack, don't pass label_id_machine (TRACK_ID) - only use label_id and empty_id
    label_id_machine = "" if is_retrack else os.getenv("TRACK_ID", "")
    set_label_ids(config.label_id, config.empty_id)
    process_video_with_multi_frame_tracking(
        video_path=video_path,
        annotations_df=annotations_df,
        study_uid=study_uid,
        series_uid=series_uid,
        flow_processor=flow_processor,
        output_dir=str(series_output_dir),
        mdai_client=None,
        label_id_fluid=config.label_id,
        label_id_machine=label_id_machine,
        upload_to_mdai=False,
        project_id=config.project_id,
        dataset_id=config.dataset_id,
        version_id=resolved_version_id,
        masks_only=is_retrack,
    )

    _tp2 = _time.perf_counter()
    logger.info(f"  process_video_with_multi_frame_tracking: {_tp2 - _tp1:.2f}s")

    # Clean up GPU memory after processing
    flow_processor.cleanup_memory()

    return series_output_dir


