#!/usr/bin/env python
"""
One-off utility to rebuild mask archives (.tar + metadata.json) for all series.

This is intended to fix older tracking runs where archives were built with
incorrect metadata. It DOES NOT rerun optical flow; it just:

- reads existing masks under output/{flow_method}/{study}_{series}/masks/
- (and retrack/masks/ if present)
- uses the current lib.mask_archive.build_mask_metadata (frametype.json-aware)
- writes fresh masks.tar (and retrack/masks.tar) in-place
"""

from __future__ import annotations

from pathlib import Path
import sys

# Ensure project root is on sys.path so we can import lib and server.*
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lib.config import load_config, ServerConfig  # type: ignore
from lib.mask_archive import (  # type: ignore
    build_mask_archive,
    build_mask_metadata,
    MaskArchiveError,
)
from server.storage.series_manager import SeriesManager  # type: ignore


def rebuild_for_series(
    series, config: ServerConfig, series_manager: SeriesManager
) -> None:
    mask_root = Path(config.mask_storage_path)
    flow_method = config.flow_method
    base_dir = mask_root / flow_method / f"{series.study_uid}_{series.series_uid}"

    def _rebuild(masks_dir: Path, archive_path: Path) -> None:
        if not masks_dir.exists() or not list(masks_dir.glob("*.webp")):
            return

        print(f"  rebuilding {archive_path} from {masks_dir}")
        metadata = build_mask_metadata(series, masks_dir, flow_method, config)
        archive_bytes = build_mask_archive(masks_dir, metadata)
        archive_path.parent.mkdir(parents=True, exist_ok=True)
        with archive_path.open("wb") as f:
            f.write(archive_bytes)

    # Initial tracking masks
    masks_dir = base_dir / "masks"
    archive_path = base_dir / "masks.tar"
    try:
        _rebuild(masks_dir, archive_path)
    except MaskArchiveError as exc:
        print(f"  ! failed to rebuild {archive_path}: {exc}")

    # Retrack masks (if any)
    retrack_masks_dir = base_dir / "retrack" / "masks"
    retrack_archive_path = base_dir / "retrack" / "masks.tar"
    try:
        _rebuild(retrack_masks_dir, retrack_archive_path)
    except MaskArchiveError as exc:
        print(f"  ! failed to rebuild {retrack_archive_path}: {exc}")


def main() -> None:
    config = load_config("server")
    series_manager = SeriesManager(config)

    series_list = series_manager.list_series()
    print(f"Found {len(series_list)} series. Rebuilding mask archives where masks exist...")

    for series in series_list:
        print(f"- {series.study_uid}/{series.series_uid}")
        rebuild_for_series(series, config, series_manager)

    print("Done.")


if __name__ == "__main__":
    main()

