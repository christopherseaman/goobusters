"""
Background worker for processing retrack jobs from the queue.

Uses the same tracking pipeline as track.py and tracking_worker.py,
just with different input (uploaded masks) and output location (retrack/).
"""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

# Add project root to Python path for imports
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import mdai
import pandas as pd

from lib.config import ServerConfig, load_config
from lib.mask_archive import (
    build_mask_archive,
    build_mask_metadata,
    MaskArchiveError,
)
from lib.uploaded_masks import convert_uploaded_masks_to_annotations_df
from server.storage.retrack_queue import RetrackQueue
from server.storage.series_manager import SeriesManager
from server.tracking_worker import run_tracking_pipeline
from track import find_annotations_file, find_images_dir


def process_retrack_job(
    job, config: ServerConfig, series_manager: SeriesManager
) -> None:
    """
    Process a single retrack job using the same pipeline as track.py.
    
    The only differences from initial tracking:
    1. Input: uploaded masks instead of original annotations
    2. Output: retrack/ subdirectory instead of main output directory
    3. label_id_machine: "" (no TRACK_ID) instead of TRACK_ID
    """
    print(f"[RETRACK] Starting retrack job: {job.study_uid}/{job.series_uid} (version: {job.new_version_id})", flush=True)
    
    try:
        # Load uploaded masks and convert to annotations_df
        uploaded_masks_path = Path(job.uploaded_masks_path)
        if not uploaded_masks_path.exists():
            raise FileNotFoundError(
                f"Uploaded masks directory not found: {uploaded_masks_path}"
            )
        
        print(f"[RETRACK] Loading annotations from {uploaded_masks_path}", flush=True)
        annotations_df, _ = convert_uploaded_masks_to_annotations_df(
            None,  # Archive already extracted
            uploaded_masks_path,
        )
        
        if annotations_df.empty:
            raise ValueError("No valid annotations found in uploaded masks")
        
        print(f"[RETRACK] Loaded {len(annotations_df)} annotations", flush=True)
        
        # Filter annotations for LABEL_ID and EMPTY_ID only (same as track.py)
        label_id_fluid = config.label_id
        label_id_empty = config.empty_id
        
        filtered_annotations = annotations_df[
            (annotations_df["labelId"] == label_id_fluid)
            | (annotations_df["labelId"] == label_id_empty)
        ].copy()
        
        if filtered_annotations.empty:
            raise ValueError(
                f"No annotations with label_id {label_id_fluid} or {label_id_empty} found"
            )
        
        print(f"[RETRACK] Filtered to {len(filtered_annotations)} annotations", flush=True)
        
        # Ensure required identifiers are present
        if "StudyInstanceUID" not in filtered_annotations.columns:
            filtered_annotations["StudyInstanceUID"] = job.study_uid
        if "SeriesInstanceUID" not in filtered_annotations.columns:
            filtered_annotations["SeriesInstanceUID"] = job.series_uid
        
        # Load original annotations to get labelName and studies (same as track.py)
        annotations_path = find_annotations_file(
            str(config.data_dir),
            config.project_id,
            config.dataset_id,
        )
        original_annotations_data = mdai.common_utils.json_to_dataframe(annotations_path)
        original_annotations_df = pd.DataFrame(original_annotations_data["annotations"])
        original_studies_df = pd.DataFrame(original_annotations_data.get("studies", []))
        
        # Add labelName from original annotations (required by create_identity_file)
        if "labelName" not in filtered_annotations.columns:
            if "labelId" in original_annotations_df.columns and "labelName" in original_annotations_df.columns:
                label_map = dict(original_annotations_df[["labelId", "labelName"]].drop_duplicates().values)
                filtered_annotations["labelName"] = filtered_annotations["labelId"].map(label_map)
                if filtered_annotations["labelName"].isna().any():
                    missing_label_ids = filtered_annotations[filtered_annotations["labelName"].isna()]["labelId"].unique().tolist()
                    raise ValueError(f"Could not find labelName for labelIds: {missing_label_ids}")
            else:
                raise ValueError("Original annotations must have both 'labelId' and 'labelName' columns")
        
        # Use original studies_df (same as track.py)
        studies_df = original_studies_df
        
        # Create annotations_blob structure (same as track.py)
        annotations_blob = {
            "annotations": original_annotations_data["annotations"],
            "studies": original_annotations_data.get("studies", []),
            "labels": original_annotations_data.get("labels", []),
        }
        
        # Run tracking pipeline (same as track.py, but is_retrack=True)
        print(f"[RETRACK] Running tracking pipeline...", flush=True)
        output_dir = run_tracking_pipeline(
            job.study_uid,
            job.series_uid,
            filtered_annotations,
            studies_df,
            annotations_blob,
            config,
            is_retrack=True,  # This makes it write to retrack/ subdirectory
        )
        
        print(f"[RETRACK] Tracking pipeline completed. Output dir: {output_dir}", flush=True)
        
        # Count masks for metadata
        masks_dir = output_dir / "masks"
        mask_count = len(list(masks_dir.glob("*.webp")))
        
        # Build mask archive and metadata (same as tracking_worker.py)
        # Write tarball to main output directory (not retrack/)
        main_output_dir = output_dir.parent
        series = series_manager.get_series(job.study_uid, job.series_uid)
        
        print(f"[RETRACK] Building mask archive and metadata", flush=True)
        try:
            metadata = build_mask_metadata(
                series, masks_dir, config.flow_method
            )
            archive_path = main_output_dir / "masks.tar"
            archive_bytes = build_mask_archive(masks_dir, metadata)
            with archive_path.open("wb") as f:
                f.write(archive_bytes)
            print(f"[RETRACK] Created tarball at {archive_path}", flush=True)
        except (MaskArchiveError, Exception) as exc:
            print(
                f"[RETRACK] Warning: Failed to build mask archive: {exc}",
                flush=True
            )
        
        # Update series metadata (atomic operation - only on successful retrack completion)
        print(f"[RETRACK] Updating version and tracking status", flush=True)
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
        print(f"[RETRACK] Marked job as completed", flush=True)
        
        # Clean up uploaded masks directory (atomic operation - only persist on success)
        if uploaded_masks_path.exists():
            shutil.rmtree(uploaded_masks_path)
            print(f"[RETRACK] Cleaned up uploaded masks directory", flush=True)
        
        print(f"[RETRACK] Retrack job completed successfully", flush=True)
        
    except Exception as exc:
        # Mark job as failed
        queue_file = config.server_state_path / "retrack_queue.json"
        retrack_queue = RetrackQueue(queue_file)
        retrack_queue.mark_failed(
            job.study_uid, job.series_uid, job.new_version_id, str(exc)
        )
        
        # Revert version ID to previous version on failure
        if job.previous_version_id is not None:
            series_manager.update_version(
                job.study_uid,
                job.series_uid,
                job.previous_version_id,
                job.editor,
            )
        else:
            # No previous version - clear it
            import json
            series_key = f"{job.study_uid}__{job.series_uid}"
            metadata_file = (
                series_manager.series_root / series_key / "metadata.json"
            )
            if metadata_file.exists():
                with metadata_file.open("r+") as f:
                    metadata_data = json.load(f)
                    metadata_data["current_version_id"] = None
                    f.seek(0)
                    f.truncate()
                    json.dump(metadata_data, f, indent=2)
        
        # Update series tracking status
        series_manager.update_tracking_status(
            job.study_uid, job.series_uid, "failed"
        )
        
        # Clean up retrack directory on failure
        mask_root = Path(config.mask_storage_path)
        output_dir = (
            mask_root / config.flow_method / f"{job.study_uid}_{job.series_uid}"
        )
        retrack_dir = output_dir / "retrack"
        if retrack_dir.exists():
            shutil.rmtree(retrack_dir)
            print(f"[RETRACK] Cleaned up retrack directory on failure: {retrack_dir}", flush=True)
        
        # Clean up uploaded masks directory on failure
        uploaded_masks_path = Path(job.uploaded_masks_path)
        if uploaded_masks_path.exists():
            shutil.rmtree(uploaded_masks_path)
            print(f"[RETRACK] Cleaned up uploaded masks on failure: {uploaded_masks_path}", flush=True)
        
        # Don't re-raise - let the worker loop continue processing other jobs
        print(f"[RETRACK] Job marked as failed. Exception: {exc}", flush=True)
        import traceback
        traceback.print_exc()


def worker_loop(config: ServerConfig, series_manager: SeriesManager) -> None:
    """
    Main worker loop that processes retrack jobs from the queue.
    """
    import time
    import logging

    logger = logging.getLogger(__name__)
    queue_file = config.server_state_path / "retrack_queue.json"
    retrack_queue = RetrackQueue(queue_file)

    logger.info("Retrack worker loop started, waiting for jobs...")
    
    # Reset any stale processing jobs (from previous worker crash)
    reset_count = retrack_queue.reset_stale_processing_jobs()
    if reset_count > 0:
        print(f"[RETRACK WORKER] Reset {reset_count} stale processing job(s) on startup", flush=True)
        logger.info(f"Reset {reset_count} stale processing job(s) on startup")

    while True:
        try:
            job = retrack_queue.dequeue()
            if job:
                print(
                    f"[RETRACK WORKER] Processing retrack job: {job.study_uid}/{job.series_uid} (version: {job.new_version_id})",
                    flush=True,
                )
                logger.info(
                    f"Processing retrack job: {job.study_uid}/{job.series_uid} "
                    f"(version: {job.new_version_id})"
                )
                try:
                    # Process the job - it handles its own exceptions and marks job as failed
                    process_retrack_job(job, config, series_manager)
                    print(
                        f"[RETRACK WORKER] Completed retrack job: {job.study_uid}/{job.series_uid} (version: {job.new_version_id})",
                        flush=True,
                    )
                    logger.info(
                        f"Completed retrack job: {job.study_uid}/{job.series_uid} "
                        f"(version: {job.new_version_id})"
                    )
                except Exception as exc:
                    # This should not happen - process_retrack_job handles all exceptions
                    # But catch it anyway to prevent worker loop from crashing
                    print(
                        f"[RETRACK WORKER] Unexpected error (job should have been marked failed): {exc}",
                        flush=True,
                    )
                    import traceback
                    traceback.print_exc()
                    logger.error(
                        f"Unexpected error processing retrack job {job.study_uid}/{job.series_uid}: {exc}",
                        exc_info=True,
                    )
                    # Mark job as failed as a safety measure
                    retrack_queue.mark_failed(
                        job.study_uid, job.series_uid, job.new_version_id, str(exc)
                    )
            else:
                # No jobs available, sleep briefly
                time.sleep(1)
        except Exception as exc:
            print(f"[RETRACK WORKER] Error in worker loop: {exc}", flush=True)
            import traceback
            traceback.print_exc()
            logger.error(f"Error in worker loop: {exc}", exc_info=True)
            time.sleep(5)  # Wait longer on error before retrying


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
