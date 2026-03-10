"""
Server startup initialization: MD.ai dataset download and initial mask generation.

This module implements the core server operation as specified in DISTRIBUTED_ARCHITECTURE.md:
1. Download MD.ai dataset on startup
2. Generate masks for all series on startup (before serving clients)
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional

import mdai

# Add paths: project root (for lib imports) and lib/server (for server package imports)
import os
_project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_lib_server = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

if _project_root not in sys.path:
    sys.path.insert(0, _project_root)
if _lib_server not in sys.path:
    sys.path.insert(0, _lib_server)

from lib.config import ServerConfig
from lib.mask_archive import mask_series_dir
from server.storage.series_manager import SeriesManager
from track import find_annotations_file, find_images_dir

logger = logging.getLogger(__name__)


def download_mdai_dataset(config: ServerConfig) -> Path:
    """
    Download MD.ai dataset using token + project_id + dataset_id.

    Returns the images directory path.

    Raises:
        RuntimeError: If download fails
    """
    logger.info(f"  Project ID: {config.project_id}")
    logger.info(f"  Dataset ID: {config.dataset_id}")
    logger.info(f"  Data directory: {config.data_dir}")

    try:
        client = mdai.Client(
            domain=config.domain, access_token=config.mdai_token
        )
        project = client.project(
            project_id=config.project_id,
            dataset_id=config.dataset_id,
            path=str(config.data_dir),
        )
        images_dir = Path(project.images_dir)
        logger.info(f"Images directory: {images_dir}")
        return images_dir
    except Exception as exc:
        raise RuntimeError(f"Failed to download MD.ai dataset: {exc}") from exc


def generate_masks_for_all_series(
    config: ServerConfig,
    series_manager: SeriesManager,
    max_workers: Optional[int] = None,
) -> None:
    """
    Generate masks for all series on startup.

    This processes all series in the dataset, generating mask images for each.
    Uses the same tracking logic as track.py but for individual series.

    Args:
        config: Server configuration
        series_manager: SeriesManager instance
        max_workers: Maximum number of parallel workers (None = sequential)
    """
    logger.info("Generating masks for all series on startup...")

    # Ensure series index is built
    # Index is built in SeriesManager.__init__, no need to call _ensure_index()

    # Get all series
    all_series = series_manager.list_series()
    total = len(all_series)

    if total == 0:
        logger.warning("No series found in dataset. Skipping mask generation.")
        return

    logger.info(f"Found {total} series. Generating masks...")

    # Use shared logic to get trackable series (same as track.py)
    from lib.trackable_series import get_trackable_series

    trackable_series_set = get_trackable_series(config)
    
    logger.info(
        f"Found {len(trackable_series_set)} series with free fluid annotations and existing videos"
    )
    
    mask_root = Path(config.mask_storage_path)
    flow_method = config.flow_method
    
    untracked = []
    for series in all_series:
        # Only attempt series that track.py would process (have annotations + video exists)
        if (series.study_uid, series.series_uid) not in trackable_series_set:
            # Skip series without annotations - don't mark as failed, just skip
            continue
        
        # Check if tracking already completed by looking for mask archive
        # track.py creates masks.tar.gz, server creates masks.tar
        # If either exists, tracking is complete
        output_dir = mask_series_dir(mask_root, flow_method, series.study_uid, series.series_uid)
        archive_tar = output_dir / "masks.tar"
        archive_targz = output_dir / "masks.tar.gz"
        
        # Skip series that have already failed (don't retry them)
        tracking_status = series_manager.get_tracking_status(series.study_uid, series.series_uid)
        if tracking_status == "failed":
            logger.debug(
                f"Skipping failed series: {series.study_uid}/{series.series_uid}"
        )
            continue
        
        # Series needs tracking if no archive exists (neither track.py nor server has completed)
        if not archive_tar.exists() and not archive_targz.exists():
            untracked.append(series)

    if not untracked:
        logger.info("All series already have masks generated. Skipping.")
        return

    logger.info(f"Generating masks for {len(untracked)} untracked series...")

    # Wrap each series in an ObjC autorelease pool to drain Vision framework GPU memory
    # (VNGenerateOpticalFlowRequest creates CVPixelBuffers that accumulate without this)
    try:
        import objc
        _has_objc = True
    except ImportError:
        _has_objc = False

    # Process series sequentially
    for idx, series in enumerate(untracked, 1):
        logger.info(
            f"[{idx}/{len(untracked)}] Starting tracking for {series.study_uid}/{series.series_uid}"
        )

        try:
            if _has_objc:
                with objc.autorelease_pool():
                    run_tracking_for_series(
                        series.study_uid,
                        series.series_uid,
                        config,
                        series_manager,
                    )
            else:
                run_tracking_for_series(
                    series.study_uid,
                    series.series_uid,
                    config,
                    series_manager,
                )
            logger.info(
                f"[{idx}/{len(untracked)}] ✓ Completed tracking for {series.study_uid}/{series.series_uid}"
            )
        except Exception as exc:
            logger.error(
                f"[{idx}/{len(untracked)}] ✗ Failed tracking for {series.study_uid}/{series.series_uid}: {exc}",
                exc_info=True,
            )
            # Continue with other series even if one fails

    # Verify completion
    completed = sum(
        1
        for s in series_manager.list_series()
        if series_manager.get_tracking_status(s.study_uid, s.series_uid) == "completed"
    )
    logger.info(
        f"Mask generation complete. {completed}/{total} series have masks."
    )


def initialize_server(
    config: ServerConfig,
    series_manager: SeriesManager,
) -> None:
    """
    Initialize server: verify dataset, start workers, generate masks.

    Dataset must already be downloaded by the caller. This function:
    1. Verifies dataset exists on disk
    2. Initializes status.json files for all series
    3. Starts tracking workers
    4. Enqueues initial tracking jobs for untracked series
    5. Blocks until all initial jobs complete

    Args:
        config: Server configuration
        series_manager: SeriesManager instance
    """
    import time

    from server.storage.retrack_queue import RetrackQueue

    # Verify dataset exists
    try:
        find_annotations_file(
            str(config.data_dir),
            config.project_id,
            config.dataset_id,
        )
        find_images_dir(
            str(config.data_dir),
            config.project_id,
            config.dataset_id,
        )
    except FileNotFoundError as exc:
        raise RuntimeError(
            f"Dataset not found on disk. Download must complete before initialization: {exc}"
        ) from exc

    # Verify series index
    series_count = len(series_manager.list_series())
    logger.info(f"Series index ready. Found {series_count} series.")

    # Initialize status.json files for all series
    initialized_count = 0
    error_count = 0
    for series in series_manager.list_series():
        try:
            status_path = series_manager._status_path(series.study_uid, series.series_uid)
            if not status_path.exists():
                series_manager._write_status(series.study_uid, series.series_uid)
                initialized_count += 1
        except Exception as exc:
            error_count += 1
            logger.warning(f"Failed to initialize status.json for {series.study_uid}/{series.series_uid}: {exc}")
    if initialized_count > 0:
        logger.info(f"Initialized {initialized_count} status.json files.")
    if error_count > 0:
        logger.warning(f"Failed to initialize {error_count} status.json files.")

    # Find untracked series that need initial mask generation
    from lib.trackable_series import get_trackable_series

    all_series = series_manager.list_series()
    trackable_set = get_trackable_series(config)
    mask_root = Path(config.mask_storage_path)
    flow_method = config.flow_method

    untracked = []
    for series in all_series:
        if (series.study_uid, series.series_uid) not in trackable_set:
            continue
        tracking_status = series_manager.get_tracking_status(series.study_uid, series.series_uid)
        if tracking_status == "failed":
            continue
        output_dir = mask_series_dir(mask_root, flow_method, series.study_uid, series.series_uid)
        if not (output_dir / "masks.tar").exists() and not (output_dir / "masks.tar.gz").exists():
            untracked.append(series)

    if not untracked:
        logger.info("All series already have masks. Starting workers for retrack jobs.")
        from server.start import start_tracking_workers
        start_tracking_workers(config)
        logger.info("Server initialization complete.")
        return

    logger.info(f"Found {len(untracked)} untracked series. Enqueueing initial tracking jobs...")

    # Start workers before enqueueing so they can begin processing immediately
    from server.start import start_tracking_workers
    start_tracking_workers(config)

    # Enqueue initial tracking jobs
    queue_file = config.server_state_path / "retrack_queue.json"
    retrack_queue = RetrackQueue(queue_file)
    for series in untracked:
        retrack_queue.enqueue_initial(series.study_uid, series.series_uid)

    logger.info(f"Enqueued {len(untracked)} initial tracking jobs. Waiting for completion...")

    # Block until all jobs complete (workers process in parallel)
    while True:
        active = retrack_queue.count_active()
        if active == 0:
            break
        logger.info(f"Waiting for {active} tracking job(s) to complete...")
        time.sleep(5)

    logger.info("Server initialization complete. Ready to serve clients.")
