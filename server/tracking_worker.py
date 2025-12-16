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
from lib.mask_archive import build_mask_archive, build_mask_metadata, MaskArchiveError
from lib.multi_frame_tracker import process_video_with_multi_frame_tracking
from lib.optical import create_identity_file, copy_annotations_to_output
from lib.opticalflowprocessor import OpticalFlowProcessor
from server.storage.series_manager import SeriesManager
from track import find_annotations_file, find_images_dir


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

        # Get video path
        video_path = Path(images_dir) / study_uid / f"{series_uid}.mp4"
        if not video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")

        # Create output directory
        mask_root = Path(config.mask_storage_path)
        output_dir = (
            mask_root / config.flow_method / f"{study_uid}_{series_uid}"
        )
        output_dir.mkdir(parents=True, exist_ok=True)

        # Generate identity file
        create_identity_file(
            str(output_dir),
            study_uid,
            series_uid,
            free_fluid_annotations,
            studies_df,
        )

        # Copy original annotations
        copy_annotations_to_output(
            str(output_dir), free_fluid_annotations, annotations_blob
        )

        # Initialize optical flow processor
        flow_processor = OpticalFlowProcessor(config.flow_method)

        # Run tracking - this writes frametype.json, masks.json, masks/, frames/
        result = process_video_with_multi_frame_tracking(
            video_path=str(video_path),
            annotations_df=free_fluid_annotations,
            study_uid=study_uid,
            series_uid=series_uid,
            flow_processor=flow_processor,
            output_dir=str(output_dir),
            mdai_client=None,  # Not needed for tracking
            label_id_fluid=label_id,
            label_id_machine=os.getenv("TRACK_ID", ""),
            upload_to_mdai=False,
            project_id=config.project_id,
            dataset_id=config.dataset_id,
        )

        # Clean up flow processor
        flow_processor.cleanup_memory()

        # Use masks/ written by tracking pipeline (frame_XXXXXX_mask.webp)
        masks_dir = output_dir / "masks"
        if not masks_dir.exists():
            raise FileNotFoundError(f"Expected masks directory not found: {masks_dir}")

        # Count masks for metadata
        mask_count = len(list(masks_dir.glob("*.webp")))

        # Build mask archive and metadata (completion marker)
        # This is done once on completion, not on every API request
        series = series_manager.get_series(study_uid, series_uid)
        try:
            metadata = build_mask_metadata(series, masks_dir, config.flow_method)
            archive_path = output_dir / "masks.tar"
            archive_bytes = build_mask_archive(masks_dir, metadata)
            with archive_path.open("wb") as f:
                f.write(archive_bytes)
        except (MaskArchiveError, Exception) as exc:
            # Log error but don't fail tracking - masks are still valid
            print(f"Warning: Failed to build mask archive for {study_uid}/{series_uid}: {exc}")

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
