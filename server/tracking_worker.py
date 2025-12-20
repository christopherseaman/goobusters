"""
Background worker for lazy tracking of series on first request.

Runs optical flow tracking for a single series when masks are first requested,
using the same logic as track.py but for individual series.
"""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

import cv2
import mdai
import pandas as pd

# Add project root to Python path for imports
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from lib.config import ServerConfig
from lib.mask_archive import (
    build_mask_archive,
    build_mask_metadata,
    MaskArchiveError,
)
from lib.multi_frame_tracker import process_video_with_multi_frame_tracking
from lib.optical import create_identity_file, copy_annotations_to_output
from lib.opticalflowprocessor import OpticalFlowProcessor
from server.storage.series_manager import SeriesManager
from track import find_annotations_file, find_images_dir


def run_tracking_pipeline(
    study_uid: str,
    series_uid: str,
    annotations_df: pd.DataFrame,
    studies_df: pd.DataFrame,
    annotations_blob: dict,
    config: ServerConfig,
    is_retrack: bool = False,
) -> Path:
    """
    Shared tracking pipeline used by both initial tracking and retrack.
    Returns the output directory path containing masks and metadata.

    Args:
        is_retrack: If True, write to retrack subdirectory to avoid overwriting original masks
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
    base_output_dir = (
        Path(config.mask_storage_path)
        / config.flow_method
        / f"{study_uid}_{series_uid}"
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

    # Initialize optical flow processor
    flow_processor = OpticalFlowProcessor(config.flow_method)

    # Process the video with multi-frame tracking (same as track.py)
    # For retrack, don't pass label_id_machine (TRACK_ID) - only use label_id and empty_id
    label_id_machine = "" if is_retrack else os.getenv("TRACK_ID", "")
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
    )

    # Clean up GPU memory after processing
    flow_processor.cleanup_memory()

    return series_output_dir


def run_tracking_for_series(
    study_uid: str,
    series_uid: str,
    config: ServerConfig,
    series_manager: SeriesManager,
) -> None:
    """
    Run optical flow tracking for a single series.

    This is the lazy tracking trigger - called when masks are first requested.
    Uses the same logic as track.py but for a single series.

    Args:
        study_uid: Study Instance UID
        series_uid: Series Instance UID
        config: Server configuration
        series_manager: SeriesManager instance
    """
    try:
        # Update status to pending
        series_manager.update_tracking_status(study_uid, series_uid, "pending")

        # Find annotations file and images directory
        annotations_path = find_annotations_file(
            str(config.data_dir),
            config.project_id,
            config.dataset_id,
        )
        images_dir = find_images_dir(
            str(config.data_dir),
            config.project_id,
            config.dataset_id,
        )

        # Load annotations - need both DataFrame format and original JSON structure
        # Use MD.ai utility to properly parse the JSON structure
        annotations_blob = mdai.common_utils.json_to_dataframe(annotations_path)
        annotations_df = pd.DataFrame(annotations_blob["annotations"])
        studies_df = pd.DataFrame(annotations_blob["studies"])

        # Filter for this specific series
        series_annotations = annotations_df[
            (annotations_df["StudyInstanceUID"] == study_uid)
            & (annotations_df["SeriesInstanceUID"] == series_uid)
        ].copy()

        if series_annotations.empty:
            raise ValueError(
                f"No annotations found for series {study_uid}/{series_uid}"
            )

        # Check if series is trackable using shared logic
        from lib.trackable_series import is_series_trackable

        if not is_series_trackable(study_uid, series_uid, config):
            raise ValueError(
                f"Series {study_uid}/{series_uid} is not trackable (no free fluid annotations or video missing)"
            )

        # Filter for LABEL_ID and EMPTY_ID only (ignore TRACK_ID)
        label_id = config.label_id
        empty_id = config.empty_id

        free_fluid_annotations = series_annotations[
            (series_annotations["labelId"] == label_id)
            | (series_annotations["labelId"] == empty_id)
        ].copy()

        if free_fluid_annotations.empty:
            raise ValueError(
                f"No free fluid annotations (LABEL_ID or EMPTY_ID) found for series {study_uid}/{series_uid}"
            )

        # Ensure labelName is present (may be missing in some annotation exports)
        if "labelName" not in free_fluid_annotations.columns:
            if (
                "labelId" in annotations_df.columns
                and "labelName" in annotations_df.columns
            ):
                label_map = dict(
                    annotations_df[["labelId", "labelName"]]
                    .drop_duplicates()
                    .values
                )
                free_fluid_annotations["labelName"] = free_fluid_annotations[
                    "labelId"
                ].map(label_map)
            else:
                raise ValueError(
                    "Original annotations must have both 'labelId' and 'labelName' columns"
                )

        output_dir = run_tracking_pipeline(
            study_uid,
            series_uid,
            free_fluid_annotations,
            studies_df,
            annotations_blob,
            config,
        )

        # Use masks/ written by tracking pipeline (frame_XXXXXX_mask.webp)
        masks_dir = output_dir / "masks"
        if not masks_dir.exists():
            raise FileNotFoundError(
                f"Expected masks directory not found: {masks_dir}"
            )

        # Count masks for metadata
        mask_count = len(list(masks_dir.glob("*.webp")))

        # Build mask archive and metadata (completion marker)
        series = series_manager.get_series(study_uid, series_uid)
        try:
            metadata = build_mask_metadata(
                series, masks_dir, config.flow_method
            )
            archive_path = output_dir / "masks.tar"
            archive_bytes = build_mask_archive(masks_dir, metadata)
            with archive_path.open("wb") as f:
                f.write(archive_bytes)
        except (MaskArchiveError, Exception) as exc:
            # Log error but don't fail tracking - masks are still valid
            print(
                f"Warning: Failed to build mask archive for {study_uid}/{series_uid}: {exc}"
            )

        # Update series metadata
        series_manager.update_tracking_status(
            study_uid, series_uid, "completed", mask_count
        )

    except Exception as exc:
        # Update status to failed
        import traceback

        error_msg = f"{type(exc).__name__}: {str(exc)}"
        print(
            f"ERROR in tracking worker for {study_uid}/{series_uid}: {error_msg}"
        )
        traceback.print_exc()
        series_manager.update_tracking_status(study_uid, series_uid, "failed")
        raise


def trigger_lazy_tracking(
    study_uid: str,
    series_uid: str,
    config: ServerConfig,
    series_manager: SeriesManager,
) -> None:
    """
    Trigger lazy tracking in a background thread.

    This is called from the API endpoint when masks are requested but don't exist yet.
    """
    import threading

    def _track_in_background():
        try:
            run_tracking_for_series(
                study_uid, series_uid, config, series_manager
            )
        except Exception as exc:
            # Error already logged in run_tracking_for_series
            print(f"Error in lazy tracking for {study_uid}/{series_uid}: {exc}")

    thread = threading.Thread(target=_track_in_background, daemon=True)
    thread.start()
