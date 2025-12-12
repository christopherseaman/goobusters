#!/usr/bin/env python3
"""
Verify that server startup completed successfully:
1. MD.ai dataset exists (or was downloaded)
2. Masks were generated for all series

This test runs after the server starts to verify the core server operation.
NO MOCKS, NO FALLBACKS, NO FAKE DATA - uses real MD.ai dataset and real tracking results.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from lib.config import load_config
from server.storage.series_manager import SeriesManager


def verify_dataset_exists() -> bool:
    """Verify MD.ai dataset exists (was downloaded on startup)."""
    try:
        from track import find_annotations_file, find_images_dir

        config = load_config("server")
        annotations_path = Path(
            find_annotations_file(
                str(config.data_dir),
                config.project_id,
                config.dataset_id,
            )
        )
        images_dir = Path(
            find_images_dir(
                str(config.data_dir),
                config.project_id,
                config.dataset_id,
            )
        )

        if not annotations_path.exists():
            print(f"✗ Annotations file not found: {annotations_path}")
            return False

        video_count = sum(1 for _ in images_dir.rglob("*.mp4"))
        if video_count == 0:
            print(f"✗ No video files found in {images_dir}")
            return False

        print(
            f"✓ Dataset verified: {video_count} videos, annotations at {annotations_path.name}"
        )
        return True
    except Exception as exc:
        print(f"✗ Dataset verification failed: {exc}")
        return False


def verify_masks_generated() -> bool:
    """Verify masks were generated for all series on startup."""
    config = load_config("server")
    series_manager = SeriesManager(config)

    all_series = series_manager.list_series()
    if not all_series:
        print("✗ No series found in dataset")
        return False

    completed_count = 0
    failed_count = 0
    never_run_count = 0
    missing_masks_count = 0

    mask_root = Path(config.mask_storage_path)
    flow_method = config.flow_method

    for series in all_series:
        if series.tracking_status == "completed":
            completed_count += 1

            # Verify masks actually exist on disk
            mask_dir = (
                mask_root
                / flow_method
                / f"{series.study_uid}_{series.series_uid}"
                / "masks"
            )

            if not mask_dir.exists():
                print(
                    f"✗ Series {series.study_uid[:20]}.../{series.series_uid[:20]}...: "
                    f"status=completed but masks directory missing"
                )
                missing_masks_count += 1
                continue

            mask_files = list(mask_dir.glob("*.webp"))
            if not mask_files:
                print(
                    f"✗ Series {series.study_uid[:20]}.../{series.series_uid[:20]}...: "
                    f"status=completed but no mask files found"
                )
                missing_masks_count += 1
                continue

        elif series.tracking_status == "failed":
            failed_count += 1
        elif series.tracking_status == "never_run":
            never_run_count += 1

    print(f"\nMask generation status:")
    print(f"  Total series: {len(all_series)}")
    print(f"  Completed: {completed_count}")
    print(f"  Failed: {failed_count}")
    print(f"  Never run: {never_run_count}")
    print(f"  Missing masks (despite completed): {missing_masks_count}")

    # Success: all series completed with masks
    if never_run_count > 0 or failed_count > 0 or missing_masks_count > 0:
        print(f"\n✗ Not all series have masks generated")
        if never_run_count > 0:
            print(
                f"  - {never_run_count} series never run (startup may still be in progress)"
            )
        if failed_count > 0:
            print(f"  - {failed_count} series failed (check server logs)")
        if missing_masks_count > 0:
            print(f"  - {missing_masks_count} series missing mask files")
        return False

    if completed_count == len(all_series):
        print(f"\n✓ All {completed_count} series have masks generated")
        return True
    else:
        print(f"\n✗ Only {completed_count}/{len(all_series)} series completed")
        return False


def main() -> None:
    """Run startup verification tests."""
    print("=" * 60)
    print("Server Startup Verification")
    print("=" * 60)
    print("\nVerifying core server operation:")
    print("  1. MD.ai dataset exists (downloaded on startup)")
    print("  2. Masks generated for all series (on startup)")
    print("\nNO MOCKS, NO FALLBACKS, NO FAKE DATA")
    print("=" * 60)

    # Test 1: Dataset exists
    print("\n[1/2] Verifying dataset...")
    dataset_ok = verify_dataset_exists()

    # Test 2: Masks generated
    print("\n[2/2] Verifying mask generation...")
    masks_ok = verify_masks_generated()

    # Summary
    print("\n" + "=" * 60)
    print("Verification Summary")
    print("=" * 60)

    if dataset_ok and masks_ok:
        print("✓ All verifications passed")
        print("  Server startup completed successfully")
        sys.exit(0)
    else:
        print("✗ Some verifications failed")
        if not dataset_ok:
            print("  - Dataset verification failed")
        if not masks_ok:
            print("  - Mask generation verification failed")
        print(
            "\nNote: If server just started, startup may still be in progress."
        )
        print("      Wait a few minutes and re-run this test.")
        sys.exit(1)


if __name__ == "__main__":
    main()


