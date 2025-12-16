#!/usr/bin/env python3
"""
Test script for lazy tracking functionality (INITIAL tracking, not retracking).

Tests that INITIAL tracking is triggered automatically when masks are first requested:
1. Delete existing masks (simulates "never tracked" scenario)
2. Request masks for a series
3. Verify tracking is triggered (status becomes "pending")
4. Wait for tracking to complete (15-25 seconds)
5. Verify masks are available

Note: This tests INITIAL tracking, not retracking. Retracking is tested in test_server_api.py.

Uses real data and real tracking pipeline.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import httpx

from lib.config import load_config


def test_lazy_tracking(base_url: str, study_uid: str, series_uid: str) -> bool:
    """
    Test that lazy tracking is triggered on first mask request for an untracked series.

    This tests the actual lazy tracking scenario: a series that has NEVER been tracked
    gets tracked automatically when masks are first requested.
    """
    print("\n[1] Testing lazy tracking for untracked series...")
    print(f"  Series: {study_uid[:20]}.../{series_uid[:20]}...")

    # Get config to find mask storage path
    from lib.config import load_config

    config = load_config("server")
    mask_root = Path(config.mask_storage_path)
    flow_method = config.flow_method
    masks_dir = mask_root / flow_method / f"{study_uid}_{series_uid}" / "masks"

    # For a never_run series, masks should not exist, but clean up if they do (data inconsistency)
    if masks_dir.exists():
        print(
            "  ⚠ Found existing masks for untracked series (data inconsistency)"
        )
        print("     Deleting to ensure clean test state...")
        import shutil

        shutil.rmtree(masks_dir)
        print("  ✓ Masks deleted")
    else:
        print("  ✓ Verified: No masks exist (expected for never_run series)")

    try:
        # Request masks - this should trigger lazy tracking for a never_run series
        print("\n[2] Requesting masks (should trigger lazy tracking)...")
        response = httpx.get(
            f"{base_url}/api/masks/{study_uid}/{series_uid}",
            timeout=10,
        )

        if response.status_code == 200:
            # Masks returned immediately - lazy tracking was not triggered
            print(
                "  ✗ Masks returned immediately - lazy tracking was not triggered"
            )
            print(
                "     For a never_run series, this should have triggered tracking"
            )
            return False

        if response.status_code == 202:
            data = response.json()
            error_code = data.get("error_code")
            print(f"  ✓ Tracking triggered: {error_code}")

            # Wait for tracking to complete - poll until status changes to completed or failed
            print("  Waiting for tracking to complete...")
            print(
                "  (Tracking typically takes 15-25 seconds, but allowing up to 30 minutes)"
            )
            start_time = time.time()
            max_wait = 1800  # 30 minute absolute timeout (much longer than typical 15-25s)
            last_status = None
            last_print_time = 0

            while time.time() - start_time < max_wait:
                elapsed = int(time.time() - start_time)
                response = httpx.get(
                    f"{base_url}/api/masks/{study_uid}/{series_uid}",
                    timeout=10,
                )

                if response.status_code == 200:
                    # Tracking completed - verify masks are actually there
                    elapsed = int(time.time() - start_time)
                    archive_bytes = response.content

                    if len(archive_bytes) == 0:
                        print(
                            f"  ✗ Tracking completed but masks archive is empty"
                        )
                        return False

                    # Verify archive can be extracted and has masks
                    from lib.mask_archive import (
                        extract_mask_archive,
                        MASK_METADATA_FILENAME,
                    )
                    import tempfile

                    with tempfile.TemporaryDirectory() as temp_dir:
                        extract_path = Path(temp_dir)
                        extract_mask_archive(archive_bytes, extract_path)

                        # Check metadata exists
                        metadata_path = extract_path / MASK_METADATA_FILENAME
                        if not metadata_path.exists():
                            print(
                                f"  ✗ Tracking completed but metadata missing"
                            )
                            return False

                        # Check for mask files
                        mask_files = list(extract_path.glob("*.webp"))
                        if not mask_files:
                            print(
                                f"  ✗ Tracking completed but no mask files found"
                            )
                            return False

                        print(
                            f"  ✓ Tracking completed successfully in {elapsed}s"
                        )
                        print(
                            f"  ✓ Verified: {len(mask_files)} mask files in archive ({len(archive_bytes)} bytes)"
                        )
                        print(
                            f"\n✓ Lazy tracking test PASSED - series now has completed tracking"
                        )
                        print(
                            f"   This series can be used for retracking tests."
                        )
                        return True

                elif response.status_code == 202:
                    data = response.json()
                    error_code = data.get("error_code")
                    # Only print status changes or every 10 seconds
                    if (
                        error_code != last_status
                        or (elapsed - last_print_time) >= 10
                    ):
                        if error_code == "TRACK_PROCESSING":
                            print(f"  ⏳ Tracking in progress... ({elapsed}s)")
                        elif error_code == "TRACK_PENDING":
                            print(f"  ⏳ Tracking pending... ({elapsed}s)")
                        last_status = error_code
                        last_print_time = elapsed
                elif response.status_code == 500:
                    data = response.json()
                    error_code = data.get("error_code")
                    error_message = data.get("error_message", "")
                    if error_code == "TRACK_FAILED":
                        elapsed = int(time.time() - start_time)
                        print(f"  ✗ Tracking failed after {elapsed}s")
                        if error_message:
                            print(f"     Error: {error_message}")
                        print(
                            f"\n✗ Lazy tracking test FAILED - tracking did not complete"
                        )
                        print(
                            f"   Without successful tracking, retracking tests cannot run."
                        )
                        return False

                time.sleep(2)  # Check every 2 seconds

            elapsed = int(time.time() - start_time)
            print(f"  ✗ Timeout waiting for tracking completion ({elapsed}s)")
            return False

    except Exception as exc:
        print(f"  ✗ Failed: {exc}")
        return False


def find_test_series(base_url: str) -> dict | None:
    """
    Find a series that has NEVER been tracked (tracking_status == "never_run").

    This is the proper test for lazy tracking - testing that a series
    that has never been tracked gets tracked when masks are first requested.
    """
    print("\n[0] Finding untracked series (never_run status)...")

    try:
        # Get all series
        response = httpx.get(f"{base_url}/api/series", timeout=10)
        response.raise_for_status()
        series_list = response.json()

        # Look for series that has NEVER been tracked
        # Skip failed series - they can't be tracked (e.g., no annotations)
        for series in series_list:
            status = series.get("tracking_status")
            if status == "never_run":
                study_uid = series.get("study_uid")
                series_uid = series.get("series_uid")
                print(
                    f"  ✓ Found untracked series: {study_uid[:20]}.../{series_uid[:20]}..."
                )
                print(f"  ✓ Exam number: {series.get('exam_number')}")
                return series
            elif status == "failed":
                # Skip failed series - they can't be tracked
                study_uid = series.get("study_uid")
                series_uid = series.get("series_uid")
                print(
                    f"  ⚠ Skipping failed series: {study_uid[:20]}.../{series_uid[:20]}... (no annotations or other error)"
                )
                continue

        # No never_run series found - reset a completed one for testing
        # But only if we haven't found any failed series (failed series can't be tracked)
        print(
            "  ⚠ No untracked series found - resetting a completed series for testing..."
        )
        for series in series_list:
            status = series.get("tracking_status")
            if status == "completed":
                study_uid = series.get("study_uid")
                series_uid = series.get("series_uid")
                print(
                    f"  ✓ Found completed series to reset: {study_uid[:20]}.../{series_uid[:20]}..."
                )
                print(f"  ✓ Exam number: {series.get('exam_number')}")

                # Reset tracking status to never_run
                from lib.config import load_config
                from server.storage.series_manager import SeriesManager

                config = load_config("server")
                series_manager = SeriesManager(config)
                try:
                    series_manager.update_tracking_status(
                        study_uid, series_uid, "never_run"
                    )
                    print("  ✓ Reset tracking status to 'never_run'")

                    # Also delete masks if they exist
                    mask_root = Path(config.mask_storage_path)
                    flow_method = config.flow_method
                    masks_dir = (
                        mask_root
                        / flow_method
                        / f"{study_uid}_{series_uid}"
                        / "masks"
                    )
                    if masks_dir.exists():
                        import shutil

                        shutil.rmtree(masks_dir)
                        print("  ✓ Deleted existing masks")
                    
                    # Also delete archive if it exists
                    archive_path = (
                        mask_root
                        / flow_method
                        / f"{study_uid}_{series_uid}"
                        / "masks.tgz"
                    )
                    if archive_path.exists():
                        archive_path.unlink()
                        print("  ✓ Deleted existing archive")

                    return series
                except Exception as exc:
                    print(f"  ⚠ Could not reset series: {exc}")
                    continue

        print("  ✗ No suitable series found to reset")
        print("     Note: Failed series cannot be used for testing (they have no annotations)")
        return None

    except Exception as exc:
        print(f"  ✗ Failed: {exc}")
        return None


def main() -> None:
    """Run lazy tracking tests."""
    print("=" * 60)
    print("Lazy Tracking Test Suite")
    print("=" * 60)
    print("\n⚠️  This test uses REAL data and requires:")
    print("   1. Server running (uv run python server/server.py)")
    print(
        "   2. At least one series with tracking_status == 'never_run' (never been tracked)"
    )
    print("   3. Video files available in data/ directory")
    print("   4. Tracking takes 15-25 seconds - test will wait for completion")
    print()
    print("This test verifies LAZY TRACKING:")
    print("  - Finds an untracked series (never_run)")
    print("  - Requests masks for it")
    print("  - Verifies tracking is automatically triggered")
    print("  - Waits for tracking to complete")
    print("  - Verifies masks are created")
    print()

    config = load_config("client")
    base_url = config.server_url

    print(f"Server URL: {base_url}")
    print()

    # Find test series (prefer one with existing masks)
    series = find_test_series(base_url)
    if not series:
        print("\n✗ No untracked series available for testing")
        print(
            "   Need a series with tracking_status == 'never_run' to test lazy tracking"
        )
        print(
            "   This test verifies that untracked series get tracked when masks are first requested."
        )
        sys.exit(
            1
        )  # Fail - can't test lazy tracking without an untracked series

    study_uid = series["study_uid"]
    series_uid = series["series_uid"]

    # Test lazy tracking
    success = test_lazy_tracking(base_url, study_uid, series_uid)

    print("\n" + "=" * 60)
    if success:
        print("✓ Lazy tracking test completed successfully")
    else:
        print("✗ Lazy tracking test failed")
    print("=" * 60)


if __name__ == "__main__":
    main()
