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
