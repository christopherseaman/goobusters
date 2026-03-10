"""
Background worker for processing retrack jobs from the queue.

Uses the same tracking pipeline as track.py, with uploaded masks as input
and output to retrack/ subdirectory.
"""

from __future__ import annotations

import logging
import shutil
import sys
from pathlib import Path

# Add paths: project root (for lib imports) and lib/server (for server package imports)
import os
_project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_lib_server = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

if _project_root not in sys.path:
    sys.path.insert(0, _project_root)
if _lib_server not in sys.path:
    sys.path.insert(0, _lib_server)

import mdai
import pandas as pd

from lib.config import ServerConfig, load_config
from lib.mask_archive import (
    build_mask_archive,
    build_mask_metadata,
)
from lib.uploaded_masks import convert_uploaded_masks_to_annotations_df
from server.storage.retrack_queue import RetrackQueue
from server.storage.series_manager import SeriesManager
from server.tracking_worker import run_tracking_pipeline
from track import find_annotations_file, find_images_dir

logger = logging.getLogger(__name__)


def _merge_annotations(uploaded_df, study_uid, series_uid):
    """Process uploaded masks (client sends ALL annotation frames)."""
    # Client now sends ALL annotation frames (label_id and empty_id), so uploaded_df
    # contains the complete set of annotations. No merge with original needed.
    
    # Ensure compatible schemas
    if "mask" not in uploaded_df.columns:
        uploaded_df["mask"] = None
    if "data" not in uploaded_df.columns:
        uploaded_df["data"] = None

    # Set required identifiers
    uploaded_df["StudyInstanceUID"] = uploaded_df.get(
        "StudyInstanceUID", study_uid
    ).fillna(study_uid)
    uploaded_df["SeriesInstanceUID"] = uploaded_df.get(
        "SeriesInstanceUID", series_uid
    ).fillna(series_uid)

    # Sort by frame number
    merged = uploaded_df.sort_values("frameNumber").reset_index(drop=True)

    return merged


def _add_label_names(annotations_df, config):
    """Ensure labelName is populated for all rows using config mappings."""
    # Normalize labelId types to match config (handle string/int mismatches)
    # Convert both labelId column and config values to strings for consistent comparison
    annotations_df["labelId"] = annotations_df["labelId"].astype(str)
    label_id_str = str(config.label_id)
    empty_id_str = str(config.empty_id)

    # Build label map from config (source of truth)
    label_map = {
        label_id_str: "Fluid",
        empty_id_str: "Empty",
    }

    # Always set labelName from labelId mapping (overwrite any existing values)
    # This ensures labelName is always correct based on labelId, regardless of what was in the uploaded data
    annotations_df["labelName"] = annotations_df["labelId"].map(label_map)

    if annotations_df["labelName"].isna().any():
        missing_frames = (
            annotations_df[annotations_df["labelName"].isna()][
                "frameNumber"
            ].tolist()
            if "frameNumber" in annotations_df.columns
            else "unknown"
        )
        missing_label_ids = (
            annotations_df[annotations_df["labelName"].isna()]["labelId"]
            .unique()
            .tolist()
        )
        raise ValueError(
            f"Could not find labelName for labelIds: {missing_label_ids}; "
            f"frames: {missing_frames}. Only {config.label_id} (Fluid) and {config.empty_id} (Empty) are supported."
        )


def _add_video_paths(annotations_df, images_dir):
    """Add video_path column to annotations DataFrame."""

    def construct_video_path(base_dir, study_uid, series_uid):
        import os

        return os.path.join(base_dir, study_uid, f"{series_uid}.mp4")

    annotations_df["video_path"] = annotations_df.apply(
        lambda row: construct_video_path(
            images_dir,
            row["StudyInstanceUID"],
            row["SeriesInstanceUID"],
        ),
        axis=1,
    )


def _check_masks_already_exist(output_dir: Path) -> bool:
    """Check if masks already exist for all frames (skip-if-exists optimization)."""
    import json

    masks_dir = output_dir / "masks"
    frametype_path = output_dir / "frametype.json"

    if not masks_dir.exists() or not frametype_path.exists():
        return False

    try:
        with frametype_path.open() as f:
            frametype_data = json.load(f)

        expected_frames = set()
        for key, info in frametype_data.items():
            if key == "_version_id":
                continue
            try:
                frame_num = int(key)
                if isinstance(info, dict) and info.get("has_mask", False):
                    expected_frames.add(frame_num)
            except (ValueError, TypeError):
                continue

        if not expected_frames:
            return False

        existing_masks = {
            int(f.stem.split("_")[1])
            for f in masks_dir.glob("frame_*_mask.webp")
            if len(f.stem.split("_")) >= 2
            and f.stem.split("_")[1].isdigit()
        }

        return expected_frames.issubset(existing_masks)
    except (OSError, json.JSONDecodeError, KeyError, ValueError):
        return False


def process_initial_job(
    job, config: ServerConfig, series_manager: SeriesManager
) -> None:
    """Process an initial tracking job using MD.ai annotations as source.

    Mirrors process_retrack_job() but loads annotations from MD.ai dataset
    instead of uploaded masks.
    """
    from lib.trackable_series import is_series_trackable
    from server.tracking_worker import build_mask_archive_from_directory

    try:
        import time as _time
        _t0 = _time.perf_counter()

        queue_file = config.server_state_path / "retrack_queue.json"
        retrack_queue = RetrackQueue(queue_file)

        if not is_series_trackable(job.study_uid, job.series_uid, config):
            logger.info(f"Series {job.study_uid}/{job.series_uid} is not trackable, skipping")
            retrack_queue.mark_completed(job.study_uid, job.series_uid, job.new_version_id)
            return

        # Check if masks already exist (e.g. from track.py or previous run)
        from lib.mask_archive import mask_series_dir
        output_dir = mask_series_dir(
            Path(config.mask_storage_path), config.flow_method,
            job.study_uid, job.series_uid,
        )
        if _check_masks_already_exist(output_dir):
            logger.info(
                f"Masks already exist for {job.study_uid}/{job.series_uid}. "
                f"Building archive only."
            )
            build_mask_archive_from_directory(
                job.study_uid, job.series_uid,
                output_dir / "masks", output_dir,
                config, series_manager,
            )
            retrack_queue.mark_completed(job.study_uid, job.series_uid, job.new_version_id)
            return

        # Load MD.ai annotations
        annotations_path = find_annotations_file(
            str(config.data_dir), config.project_id, config.dataset_id
        )
        original_data = mdai.common_utils.json_to_dataframe(annotations_path)
        annotations_df = pd.DataFrame(original_data["annotations"])
        studies_df = pd.DataFrame(original_data.get("studies", []))

        # Filter for this series
        series_annotations = annotations_df[
            (annotations_df["StudyInstanceUID"] == job.study_uid)
            & (annotations_df["SeriesInstanceUID"] == job.series_uid)
        ].copy()

        if series_annotations.empty:
            raise ValueError(f"No annotations found for series {job.study_uid}/{job.series_uid}")

        # Filter for label_id + empty_id only
        filtered = series_annotations[
            (series_annotations["labelId"] == config.label_id)
            | (series_annotations["labelId"] == config.empty_id)
        ].copy()

        if filtered.empty:
            raise ValueError(
                f"No free fluid annotations for series {job.study_uid}/{job.series_uid}"
            )

        # Ensure labelName is populated
        _add_label_names(filtered, config)

        _t_prep = _time.perf_counter()
        logger.info(f"Initial tracking prep (annotations load): {_t_prep - _t0:.2f}s")

        # Run tracking pipeline (is_retrack=False writes to base dir)
        annotations_blob = {
            "annotations": original_data["annotations"],
            "studies": original_data.get("studies", []),
            "labels": original_data.get("labels", []),
        }

        output_dir = run_tracking_pipeline(
            job.study_uid, job.series_uid,
            filtered, studies_df, annotations_blob,
            config, is_retrack=False,
        )

        _t_track = _time.perf_counter()
        logger.info(f"Initial tracking pipeline: {_t_track - _t_prep:.2f}s")

        # Build mask archive
        masks_dir = output_dir / "masks"
        mask_count = len(list(masks_dir.glob("*.webp")))
        series = series_manager.get_series(job.study_uid, job.series_uid)
        logger.info(f"Initial tracking produced {mask_count} masks in {masks_dir}")

        metadata = build_mask_metadata(series, masks_dir, config.flow_method, config)
        archive_path = output_dir / "masks.tar"
        archive_bytes = build_mask_archive(masks_dir, metadata)
        with archive_path.open("wb") as f:
            f.write(archive_bytes)

        _t_archive = _time.perf_counter()
        logger.info(
            f"Wrote masks.tar ({len(archive_bytes)} bytes) to {archive_path} "
            f"(archive: {_t_archive - _t_track:.2f}s)"
        )

        retrack_queue.mark_completed(job.study_uid, job.series_uid, job.new_version_id)

    except Exception as exc:
        queue_file = config.server_state_path / "retrack_queue.json"
        retrack_queue = RetrackQueue(queue_file)
        retrack_queue.mark_failed(
            job.study_uid, job.series_uid, job.new_version_id, str(exc)
        )
        logger.error(f"Initial tracking failed for {job.study_uid}/{job.series_uid}: {exc}")
        import traceback
        traceback.print_exc()


def process_retrack_job(
    job, config: ServerConfig, series_manager: SeriesManager
) -> None:
    """Process a single retrack job using the same pipeline as track.py."""
    try:
        import time as _time
        _t0 = _time.perf_counter()

        # Load original annotations
        annotations_path = find_annotations_file(
            str(config.data_dir), config.project_id, config.dataset_id
        )
        original_data = mdai.common_utils.json_to_dataframe(annotations_path)
        original_annotations_df = pd.DataFrame(original_data["annotations"])
        original_studies_df = pd.DataFrame(original_data.get("studies", []))

        # Load uploaded masks (client now sends ALL annotation frames)
        uploaded_masks_path = Path(job.uploaded_masks_path)
        if not uploaded_masks_path.exists():
            raise FileNotFoundError(
                f"Uploaded masks directory not found: {uploaded_masks_path}"
            )

        uploaded_df, _ = convert_uploaded_masks_to_annotations_df(
            None, uploaded_masks_path
        )

        if uploaded_df.empty:
            raise ValueError("No valid annotations found in uploaded masks")

        # Filter uploaded annotations for LABEL_ID and EMPTY_ID only
        uploaded_filtered = uploaded_df[
            (uploaded_df["labelId"] == config.label_id)
            | (uploaded_df["labelId"] == config.empty_id)
        ].copy()

        if uploaded_filtered.empty:
            raise ValueError(
                f"No uploaded annotations with label_id {config.label_id} or {config.empty_id} found"
            )

        # Set required identifiers
        uploaded_filtered["StudyInstanceUID"] = job.study_uid
        uploaded_filtered["SeriesInstanceUID"] = job.series_uid

        # Client sends ALL annotation frames, so use uploaded directly (no merge needed)
        filtered_annotations = _merge_annotations(
            uploaded_filtered, job.study_uid, job.series_uid
        )

        # Add labelName and video_path (fills all labelName values or raises with details)
        # _add_label_names normalizes types and handles the mapping
        _add_label_names(filtered_annotations, config)

        images_dir = find_images_dir(
            str(config.data_dir), config.project_id, config.dataset_id
        )
        _add_video_paths(filtered_annotations, images_dir)

        # Run tracking pipeline
        annotations_blob = {
            "annotations": original_data["annotations"],
            "studies": original_data.get("studies", []),
            "labels": original_data.get("labels", []),
        }

        _t_prep = _time.perf_counter()
        logger.info(f"Retrack prep (annotations+masks load): {_t_prep - _t0:.2f}s")

        # Pass version_id to tracking pipeline so it's written to frametype.json
        # This is the newly generated version_id (different from job.previous_version_id)
        output_dir = run_tracking_pipeline(
            job.study_uid,
            job.series_uid,
            filtered_annotations,
            original_studies_df,
            annotations_blob,
            config,
            is_retrack=True,
            version_id=job.new_version_id,
        )

        _t_track = _time.perf_counter()
        logger.info(f"Retrack tracking pipeline: {_t_track - _t_prep:.2f}s")

        # Build mask archive (version_id is also written to masks.tar metadata.json for client)
        masks_dir = output_dir / "masks"
        mask_count = len(list(masks_dir.glob("*.webp")))
        series = series_manager.get_series(job.study_uid, job.series_uid)
        logger.info(
            f"Retrack produced {mask_count} masks in {masks_dir}"
        )

        # build_mask_metadata reads version_id from frametype.json
        metadata = build_mask_metadata(
            series, masks_dir, config.flow_method, config
        )
        archive_path = output_dir.parent / "masks.tar"
        archive_bytes = build_mask_archive(masks_dir, metadata)
        with archive_path.open("wb") as f:
            f.write(archive_bytes)
        _t_archive = _time.perf_counter()
        logger.info(
            f"Wrote masks.tar ({len(archive_bytes)} bytes) to {archive_path} (archive: {_t_archive - _t_track:.2f}s)"
        )

        # Mark job as completed
        queue_file = config.server_state_path / "retrack_queue.json"
        retrack_queue = RetrackQueue(queue_file)
        retrack_queue.mark_completed(
            job.study_uid, job.series_uid, job.new_version_id
        )

        # Clean up uploaded masks
        if uploaded_masks_path.exists():
            shutil.rmtree(uploaded_masks_path)

    except Exception as exc:
        # Mark job as failed
        queue_file = config.server_state_path / "retrack_queue.json"
        retrack_queue = RetrackQueue(queue_file)
        retrack_queue.mark_failed(
            job.study_uid, job.series_uid, job.new_version_id, str(exc)
        )

        # Version ID is stored in masks.tar metadata.json - no need to revert on failure
        # Tracking status is computed from filesystem

        # Clean up on failure
        mask_root = Path(config.mask_storage_path)
        retrack_dir = (
            mask_root
            / config.flow_method
            / f"{job.study_uid}_{job.series_uid}"
            / "retrack"
        )
        if retrack_dir.exists():
            shutil.rmtree(retrack_dir)

        uploaded_masks_path = Path(job.uploaded_masks_path)
        if uploaded_masks_path.exists():
            shutil.rmtree(uploaded_masks_path)

        print(f"[RETRACK] Failed: {exc}", flush=True)
        import traceback

        traceback.print_exc()


def worker_loop(config: ServerConfig, series_manager: SeriesManager) -> None:
    """Main worker loop that processes retrack jobs from the queue."""
    import time

    # Track parent PID so we can exit if parent server dies (prevents orphaned workers)
    parent_pid = os.getppid()

    queue_file = config.server_state_path / "retrack_queue.json"
    retrack_queue = RetrackQueue(queue_file)

    logger.info("Tracking worker started, waiting for jobs... (parent PID: %d)", parent_pid)

    try:
        import objc
        _has_objc = True
    except ImportError:
        _has_objc = False

    while True:
        # Exit if parent server process died (prevents orphaned workers racing on queue)
        if os.getppid() != parent_pid:
            logger.info("Parent process changed (server restarted?), exiting worker.")
            return
        try:
            job = retrack_queue.dequeue()
            if job:
                job_label = f"{job.job_type}:{job.study_uid}/{job.series_uid}"
                print(f"[TRACKING WORKER] Processing {job_label}", flush=True)
                logger.info(f"Processing {job_label}")
                try:
                    if job.job_type == "initial":
                        _process_job = lambda: process_initial_job(job, config, series_manager)
                    else:
                        _process_job = lambda: process_retrack_job(job, config, series_manager)

                    # Wrap in autorelease pool to drain Vision framework GPU memory
                    if _has_objc:
                        with objc.autorelease_pool():
                            _process_job()
                    else:
                        _process_job()
                    print(f"[TRACKING WORKER] Completed {job_label}", flush=True)
                    logger.info(f"Completed {job_label}")
                except Exception as exc:
                    print(f"[TRACKING WORKER] Failed {job_label}: {exc}", flush=True)
                    import traceback

                    traceback.print_exc()
                    logger.error(f"Failed {job_label}: {exc}", exc_info=True)
                    retrack_queue.mark_failed(
                        job.study_uid,
                        job.series_uid,
                        job.new_version_id,
                        str(exc),
                    )
            else:
                time.sleep(1)
        except Exception as exc:
            print(f"[TRACKING WORKER] Error in worker loop: {exc}", flush=True)
            import traceback

            traceback.print_exc()
            logger.error(f"Error in worker loop: {exc}", exc_info=True)
            time.sleep(5)


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
