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

# Add project root to Python path BEFORE any imports
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from lib.config import ServerConfig
from server.storage.series_manager import SeriesManager
from server.tracking_worker import run_tracking_for_series
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
    series_manager._ensure_index()

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
        f"Found {len(trackable_series_set)} series with free fluid annotations and existing videos (same logic as track.py)"
    )

    mask_root = Path(config.mask_storage_path)
    flow_method = config.flow_method

    untracked = []
    for series in all_series:
        # Only attempt series that track.py would process (have annotations + video exists)
        if (series.study_uid, series.series_uid) not in trackable_series_set:
            # Skip series without annotations - don't mark as failed, just skip
            continue

        # Check if tracking completed by looking for the server's archive file
        # Archive is built as the LAST step of tracking, so it's the completion marker
        # Only check for masks.tar (server format), not masks.tar.gz (track.py format)
        # The server must generate its own masks, not rely on track.py output
        output_dir = (
            mask_root / flow_method / f"{series.study_uid}_{series.series_uid}"
        )
        archive_tgz = output_dir / "masks.tar"

        # Skip series that have already failed (don't retry them)
        if series.tracking_status == "failed":
            logger.debug(
                f"Skipping failed series: {series.study_uid}/{series.series_uid}"
            )
            continue

        # Series needs tracking if server's archive doesn't exist (server never completed tracking)
        if not archive_tgz.exists():
            untracked.append(series)

    if not untracked:
        logger.info("All series already have masks generated. Skipping.")
        return

    logger.info(f"Generating masks for {len(untracked)} untracked series...")

    # Process series sequentially (can be parallelized later if needed)
    # For now, sequential is safer and easier to debug
    for idx, series in enumerate(untracked, 1):
        logger.info(
            f"[{idx}/{len(untracked)}] {series.study_uid}/{series.series_uid}"
        )

        try:
            run_tracking_for_series(
                series.study_uid,
                series.series_uid,
                config,
                series_manager,
            )
        except Exception as exc:
            logger.error(
                f"  ✗ Failed: {series.study_uid}/{series.series_uid} - {exc}",
                exc_info=True,
            )
            # Continue with other series even if one fails

    # Verify completion
    completed = sum(
        1
        for s in series_manager.list_series()
        if s.tracking_status == "completed"
    )
    logger.info(
        f"Mask generation complete. {completed}/{total} series have masks."
    )


def initialize_server(
    config: ServerConfig,
    series_manager: SeriesManager,
    skip_download: bool = False,
    skip_mask_generation: bool = False,
) -> None:
    """
    Initialize server: download dataset and generate masks for all series.

    This implements the core server operation as specified in DISTRIBUTED_ARCHITECTURE.md.
    The server must complete this initialization before serving clients.

    Args:
        config: Server configuration
        series_manager: SeriesManager instance
        skip_download: If True, skip MD.ai dataset download (use existing data)
        skip_mask_generation: If True, skip mask generation (use existing masks)
    """
    # Step 1: Download MD.ai dataset
    if not skip_download:
        try:
            download_mdai_dataset(config)
        except Exception as exc:
            logger.error(f"Dataset download failed: {exc}", exc_info=True)
            raise
    else:
        logger.info("Skipping MD.ai dataset download (using existing data)")
        # Verify data exists
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
            logger.info("Existing dataset verified")
        except FileNotFoundError as exc:
            raise RuntimeError(
                "Dataset download skipped but data not found. "
                "Run with skip_download=False or ensure data exists."
            ) from exc

    # Step 2: Build series index (if not already built)
    logger.info("Building series index...")
    series_manager._ensure_index()
    series_count = len(series_manager.list_series())
    logger.info(f"Series index built. Found {series_count} series.")

    # Step 3: Generate masks for all series
    if not skip_mask_generation:
        try:
            generate_masks_for_all_series(config, series_manager)
        except Exception as exc:
            logger.error(f"Mask generation failed: {exc}", exc_info=True)
            raise
    else:
        logger.info("Skipping mask generation (using existing masks)")

    logger.info("Server initialization complete. Ready to serve clients.")
