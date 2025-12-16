"""
Background worker for processing retrack jobs from the queue.

Runs optical flow tracking on uploaded mask edits and updates the server
storage with new tracked masks.
"""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

import cv2

# Add project root to Python path for imports
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from lib.config import ServerConfig, load_config
from lib.mask_archive import build_mask_archive, build_mask_metadata, MaskArchiveError
from lib.multi_frame_tracker import process_video_with_multi_frame_tracking
from lib.opticalflowprocessor import OpticalFlowProcessor
from lib.uploaded_masks import convert_uploaded_masks_to_annotations_df
from server.storage.retrack_queue import RetrackQueue
from server.storage.series_manager import SeriesManager


def process_retrack_job(
    job, config: ServerConfig, series_manager: SeriesManager
) -> None:
    """
    Process a single retrack job.

    Args:
        job: RetrackJob from queue
        config: Server configuration
        series_manager: SeriesManager instance
    """
    try:
        # Get video path from series metadata
        series = series_manager.get_series(job.study_uid, job.series_uid)
        video_path = Path(series.video_path)

        if not video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")

        # Load uploaded annotations (archive already extracted)
        annotations_df, metadata = convert_uploaded_masks_to_annotations_df(
            None,
            job.uploaded_masks_path,  # None indicates already extracted
        )

        if annotations_df.empty:
            raise ValueError("No valid annotations found in uploaded masks")

        # Get label IDs from config
        label_id_fluid = config.label_id
        label_id_empty = config.empty_id
        label_id_track = os.getenv("TRACK_ID", "")

        # Filter annotations to match tracking behavior (LABEL_ID and EMPTY_ID only)
        # This ensures retracking uses the same annotation filtering as tracking
        filtered_annotations = annotations_df[
            (annotations_df["labelId"] == label_id_fluid)
            | (annotations_df["labelId"] == label_id_empty)
        ].copy()

        if filtered_annotations.empty:
            raise ValueError(
                f"No annotations with label_id {label_id_fluid} or {label_id_empty} found in uploaded masks"
            )

        # Create optical flow processor
        flow_processor = OpticalFlowProcessor(config.flow_method)

        # Determine output directory structure
        mask_root = Path(config.mask_storage_path)
        output_dir = (
            mask_root / config.flow_method / f"{job.study_uid}_{job.series_uid}"
        )
        retrack_dir = output_dir / "retrack"
        masks_dir = retrack_dir / "masks"
        masks_temp_dir = retrack_dir / "masks_temp"

        # Create identity file and copy annotations (same as tracking)
        # This ensures retracking produces the same output structure as tracking
        from lib.optical import create_identity_file, copy_annotations_to_output
        import pandas as pd
        import json

        # Create studies_df from metadata if available, otherwise empty
        studies_df = (
            pd.DataFrame(metadata.get("studies", []))
            if metadata.get("studies")
            else pd.DataFrame()
        )

        # Create identity file (same as tracking)
        create_identity_file(
            str(retrack_dir),
            job.study_uid,
            job.series_uid,
            filtered_annotations,
            studies_df,
        )

        # Copy annotations (same as tracking)
        # Create annotations blob structure for copy_annotations_to_output
        annotations_blob = {
            "annotations": filtered_annotations.to_dict("records"),
            "studies": studies_df.to_dict("records")
            if not studies_df.empty
            else [],
        }
        copy_annotations_to_output(
            str(retrack_dir), filtered_annotations, annotations_blob
        )

        # Use temp directory for retrack output
        import tempfile

        with tempfile.TemporaryDirectory() as temp_dir:
            # Run tracking (same as tracking worker, with same parameters).
            # This writes frametype.json, masks.json, masks/, frames/ into temp_dir.
            result = process_video_with_multi_frame_tracking(
                video_path=str(video_path),
                annotations_df=filtered_annotations,  # Use filtered annotations
                study_uid=job.study_uid,
                series_uid=job.series_uid,
                flow_processor=flow_processor,
                output_dir=temp_dir,
                mdai_client=None,
                label_id_fluid=label_id_fluid,
                label_id_machine=label_id_track,
                upload_to_mdai=False,
                project_id=config.project_id,  # Same as tracking
                dataset_id=config.dataset_id,  # Same as tracking
            )

            temp_output_dir = Path(temp_dir)
            temp_masks_dir = temp_output_dir / "masks"
            if not temp_masks_dir.exists():
                raise FileNotFoundError(
                    f"Expected masks directory not found in retrack output: {temp_masks_dir}"
                )

            # Clear and create temp masks directory
            if masks_temp_dir.exists():
                shutil.rmtree(masks_temp_dir)
            shutil.copytree(temp_masks_dir, masks_temp_dir)

            # Promote temp to production: move masks_temp to masks
            if masks_dir.exists():
                shutil.rmtree(masks_dir)
            masks_temp_dir.rename(masks_dir)

            # Count masks for metadata
            mask_count = len(list(masks_dir.glob("*.webp")))

            # Copy frametype.json / masks.json into retrack_dir for canonical metadata
            for json_name in ("frametype.json", "masks.json"):
                src_json = temp_output_dir / json_name
                if src_json.exists():
                    shutil.copy2(src_json, retrack_dir / json_name)

            # Build mask archive and metadata (completion marker)
            # This is done once on completion, not on every API request
            series = series_manager.get_series(job.study_uid, job.series_uid)
            try:
                metadata = build_mask_metadata(series, masks_dir, config.flow_method)
                archive_path = retrack_dir / "masks.tar"
                archive_bytes = build_mask_archive(masks_dir, metadata)
                with archive_path.open("wb") as f:
                    f.write(archive_bytes)
            except (MaskArchiveError, Exception) as exc:
                # Log error but don't fail retracking - masks are still valid
                print(f"Warning: Failed to build mask archive for {job.study_uid}/{job.series_uid}: {exc}")

            # Clean up temp version file if it exists
            series_key = f"{job.study_uid}__{job.series_uid}"
            version_temp_file = (
                series_manager.series_root / series_key / "version_temp.json"
            )
            if version_temp_file.exists():
                version_temp_file.unlink()

            # Update series metadata
            series_manager.update_version(
                job.study_uid, job.series_uid, job.new_version_id, job.editor
            )
            series_manager.update_tracking_status(
                job.study_uid, job.series_uid, "completed", mask_count
            )

            # Mark job as completed
            queue_file = config.server_state_path / "retrack_queue.json"
            retrack_queue = RetrackQueue(queue_file)
            retrack_queue.mark_completed(
                job.study_uid, job.series_uid, job.new_version_id
            )

            # Clean up uploaded masks directory
            if job.uploaded_masks_path.exists():
                shutil.rmtree(job.uploaded_masks_path)

        # Clean up flow processor
        flow_processor.cleanup_memory()

    except Exception as exc:
        # Mark job as failed
        queue_file = config.server_state_path / "retrack_queue.json"
        retrack_queue = RetrackQueue(queue_file)
        retrack_queue.mark_failed(
            job.study_uid, job.series_uid, job.new_version_id, str(exc)
        )

        # Update series tracking status
        series_manager.update_tracking_status(
            job.study_uid, job.series_uid, "failed"
        )

        # Clean up temp directory on failure
        mask_root = Path(config.mask_storage_path)
        output_dir = (
            mask_root / config.flow_method / f"{job.study_uid}_{job.series_uid}"
        )
        retrack_dir = output_dir / "retrack"
        masks_temp_dir = retrack_dir / "masks_temp"
        if masks_temp_dir.exists():
            shutil.rmtree(masks_temp_dir)

        # Clean up version_temp.json
        series_key = f"{job.study_uid}__{job.series_uid}"
        version_temp_file = (
            series_manager.series_root / series_key / "version_temp.json"
        )
        if version_temp_file.exists():
            version_temp_file.unlink()

        raise


def worker_loop(config: ServerConfig, series_manager: SeriesManager) -> None:
    """
    Main worker loop that processes retrack jobs from the queue.

    Runs continuously, checking for pending jobs and processing them.
    """
    import time

    queue_file = config.server_state_path / "retrack_queue.json"
    retrack_queue = RetrackQueue(queue_file)

    while True:
        job = retrack_queue.dequeue()
        if job:
            try:
                process_retrack_job(job, config, series_manager)
            except Exception as exc:
                # Error already handled in process_retrack_job
                print(f"Error processing retrack job: {exc}")
        else:
            # No jobs available, sleep briefly
            time.sleep(1)


def main() -> None:
    """Entry point for retrack worker process."""
    import logging

    logging.basicConfig(level=logging.INFO)
    config = load_config("server")
    series_manager = SeriesManager(config)

    logging.info("Starting retrack worker...")
    worker_loop(config, series_manager)


if __name__ == "__main__":
    main()
