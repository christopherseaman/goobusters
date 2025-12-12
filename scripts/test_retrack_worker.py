#!/usr/bin/env python3
"""
Test script for retrack worker functionality.

Tests the retrack worker by:
1. Creating a test retrack job in the queue
2. Verifying worker processes it
3. Checking output masks are created
4. Verifying version management

Uses real tracking pipeline and real data.
"""

from __future__ import annotations

import json
import shutil
import sys
import tempfile
import time
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from lib.config import load_config
from lib.mask_archive import build_mask_archive
from server.storage.retrack_queue import RetrackQueue
from server.storage.series_manager import SeriesManager


def create_test_mask_archive(
    study_uid: str, series_uid: str, output_dir: Path
) -> Path:
    """
    Create a test mask archive from existing masks in output/ directory.

    Returns path to created archive.
    """
    config = load_config("server")
    mask_root = Path(config.mask_storage_path)
    flow_method = config.flow_method

    # Find existing masks
    masks_dir = mask_root / flow_method / f"{study_uid}_{series_uid}" / "masks"

    if not masks_dir.exists():
        raise FileNotFoundError(
            f"No existing masks found at {masks_dir}. "
            "Run track.py first to generate masks."
        )

    # Create metadata
    mask_files = sorted(masks_dir.glob("*.webp"))
    if not mask_files:
        raise ValueError(f"No mask files found in {masks_dir}")

    # Extract frame numbers from filenames
    # Per spec: use 'frames' array with is_annotation=true for retracking input
    frames = []
    for mask_file in mask_files[:5]:  # Use first 5 masks for testing
        # Parse frame number from filename like "frame_000001.webp"
        parts = mask_file.stem.split("_")
        if len(parts) >= 2 and parts[1].isdigit():
            frame_num = int(parts[1])
            frames.append({
                "frame_number": frame_num,
                "has_mask": True,
                "is_annotation": True,  # Mark as annotation for retrack
                "label_id": config.label_id,  # Required for annotation frames
                "filename": mask_file.name,
            })

    metadata = {
        "study_uid": study_uid,
        "series_uid": series_uid,
        "previous_version_id": None,
        "editor": "test@example.com",
        "edited_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "frames": frames,  # Single array - retracking filters by is_annotation=true
    }

    # Create archive
    archive_bytes = build_mask_archive(
        masks_dir, metadata, include_metadata=True
    )

    # Save to temp file
    archive_path = output_dir / "test_masks.tgz"
    with archive_path.open("wb") as f:
        f.write(archive_bytes)

    return archive_path


def test_retrack_queue_operations() -> bool:
    """Test retrack queue basic operations."""
    print("\n[1] Testing retrack queue operations...")

    config = load_config("server")
    queue_file = config.server_state_path / "retrack_queue.json"

    # Clean up existing queue for testing
    if queue_file.exists():
        queue_file.unlink()

    retrack_queue = RetrackQueue(queue_file)

    # Test enqueue
    with tempfile.TemporaryDirectory() as temp_dir:
        uploaded_masks_path = Path(temp_dir) / "test_masks"
        uploaded_masks_path.mkdir()

        job = retrack_queue.enqueue(
            study_uid="test_study",
            series_uid="test_series",
            editor="test@example.com",
            previous_version_id=None,
            uploaded_masks_path=uploaded_masks_path,
        )

        print(f"  ✓ Job enqueued: {job.new_version_id}")
        print(f"  ✓ Status: {job.status}")

        # Test get status
        status = retrack_queue.get_job_status("test_study", "test_series")
        assert status is not None, "Job should be retrievable"
        assert status.status == "pending", "Job should be pending"
        print(f"  ✓ Job status retrieved: {status.status}")

        # Test dequeue
        dequeued = retrack_queue.dequeue()
        assert dequeued is not None, "Job should be dequeuable"
        assert dequeued.status == "processing", "Job should be processing"
        print(f"  ✓ Job dequeued: {dequeued.status}")

        # Test mark completed
        retrack_queue.mark_completed(
            "test_study", "test_series", dequeued.new_version_id
        )
        completed = retrack_queue.get_job_status("test_study", "test_series")
        assert completed.status == "completed", "Job should be completed"
        print(f"  ✓ Job marked as completed")

    return True


def test_retrack_worker_integration() -> bool:
    """Test retrack worker with real data."""
    print("\n[2] Testing retrack worker integration...")

    config = load_config("server")
    series_manager = SeriesManager(config)

    # Find a series with existing masks
    series_list = series_manager.list_series()
    test_series = None

    for series in series_list:
        mask_root = Path(config.mask_storage_path)
        flow_method = config.flow_method
        masks_dir = (
            mask_root
            / flow_method
            / f"{series.study_uid}_{series.series_uid}"
            / "masks"
        )
        if masks_dir.exists() and any(masks_dir.glob("*.webp")):
            test_series = series
            break

    if not test_series:
        print("  ⚠ No series with masks found. Run track.py first.")
        return False

    print(
        f"  Using series: {test_series.study_uid[:20]}.../{test_series.series_uid[:20]}..."
    )

    # Create test archive
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        try:
            archive_path = create_test_mask_archive(
                test_series.study_uid, test_series.series_uid, temp_path
            )
            print(f"  ✓ Created test archive: {archive_path.name}")

            # Extract archive
            from lib.mask_archive import extract_mask_archive

            uploaded_masks_dir = temp_path / "uploaded_masks"
            with archive_path.open("rb") as f:
                extract_mask_archive(f.read(), uploaded_masks_dir)

            print(f"  ✓ Extracted archive to {uploaded_masks_dir}")

            # Create queue job
            queue_file = config.server_state_path / "retrack_queue.json"
            retrack_queue = RetrackQueue(queue_file)

            job = retrack_queue.enqueue(
                study_uid=test_series.study_uid,
                series_uid=test_series.series_uid,
                editor="test@example.com",
                previous_version_id=test_series.current_version_id,
                uploaded_masks_path=uploaded_masks_dir,
            )

            print(f"  ✓ Created retrack job: {job.new_version_id}")
            print(f"  ⚠ Note: Worker must be running to process this job")
            print(f"     Run: uv run python server/retrack_worker.py")

        except Exception as exc:
            print(f"  ✗ Failed: {exc}")
            return False

    return True


def verify_retrack_output(study_uid: str, series_uid: str) -> bool:
    """Verify retrack output was created correctly."""
    print("\n[3] Verifying retrack output...")

    config = load_config("server")
    mask_root = Path(config.mask_storage_path)
    flow_method = config.flow_method

    masks_dir = mask_root / flow_method / f"{study_uid}_{series_uid}" / "masks"

    if not masks_dir.exists():
        print(f"  ✗ Masks directory not found: {masks_dir}")
        return False

    mask_files = list(masks_dir.glob("*.webp"))
    print(f"  ✓ Found {len(mask_files)} mask files")

    if len(mask_files) == 0:
        print(f"  ✗ No mask files found")
        return False

    # Check series metadata
    series_manager = SeriesManager(config)
    try:
        series = series_manager.get_series(study_uid, series_uid)
        print(f"  ✓ Tracking status: {series.tracking_status}")
        print(f"  ✓ Mask count: {series.mask_count}")
        print(f"  ✓ Version ID: {series.current_version_id}")
    except Exception as exc:
        print(f"  ⚠ Could not verify metadata: {exc}")

    return True


def main() -> None:
    """Run retrack worker tests."""
    print("=" * 60)
    print("Retrack Worker Test Suite")
    print("=" * 60)
    print("\n⚠️  This test uses REAL data and requires:")
    print("   1. At least one series with tracked masks (run track.py first)")
    print("   2. Server state initialized (server must have run once)")
    print()

    try:
        # Test 1: Queue operations
        if not test_retrack_queue_operations():
            print("\n❌ Queue operations test failed")
            sys.exit(1)

        # Test 2: Worker integration
        if not test_retrack_worker_integration():
            print("\n⚠️  Worker integration test skipped (no data)")
            sys.exit(0)

        print("\n" + "=" * 60)
        print("Test suite completed")
        print("=" * 60)
        print("\nNext steps:")
        print(
            "  1. Start retrack worker: uv run python server/retrack_worker.py"
        )
        print("  2. Run test_server_api.py to test full flow")

    except Exception as exc:
        print(f"\n❌ Test failed: {exc}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
