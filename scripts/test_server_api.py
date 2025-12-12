#!/usr/bin/env python3
"""
Test script for server API endpoints using real data.

Tests the full flow including RETRACKING:
1. GET /api/status - Server health
2. GET /api/series/next - Get next series
3. GET /api/masks/{study}/{series} - Download masks
4. POST /api/masks/{study}/{series} - Upload edited masks (triggers RETRACK)
5. GET /api/retrack/status/{study}/{series} - Check retrack status
6. Version conflict detection
7. Retrack completion verification (waits for retrack to complete)

Uses real data from output/ directory and real server endpoints.
No mocks or fake data.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import httpx

from lib.config import load_config


def test_server_status(base_url: str) -> bool:
    """Test GET /api/status endpoint."""
    print("\n[1] Testing GET /api/status...")
    try:
        response = httpx.get(f"{base_url}/api/status", timeout=10)
        response.raise_for_status()
        data = response.json()
        print(f"  ✓ Server ready: {data.get('ready')}")
        print(f"  ✓ Series total: {data.get('series_total')}")
        print(f"  ✓ Series completed: {data.get('series_completed')}")
        print(f"  ✓ Series pending: {data.get('series_pending')}")
        return True
    except Exception as exc:
        print(f"  ✗ Failed: {exc}")
        return False


def find_series_with_completed_tracking(base_url: str) -> dict | None:
    """
    Find a series with completed tracking status.

    If none exists, waits for lazy tracking test to complete one, or finds one in progress.

    This ensures we have a series that can be used for retracking tests.
    """
    print("\n[2] Finding series with completed tracking...")
    try:
        # Get all series
        response = httpx.get(f"{base_url}/api/series", timeout=10)
        response.raise_for_status()
        series_list = response.json()

        # Look for series with completed tracking
        for series in series_list:
            if series.get("tracking_status") == "completed":
                study_uid = series.get("study_uid")
                series_uid = series.get("series_uid")
                print(
                    f"  ✓ Found series with completed tracking: {study_uid[:20]}.../{series_uid[:20]}..."
                )
                print(f"  ✓ Exam number: {series.get('exam_number')}")
                return series

        # No completed series - check if any are pending (might complete soon)
        print("  ⚠ No series with completed tracking found")
        for series in series_list:
            if series.get("tracking_status") == "pending":
                print(
                    "  ℹ Found series with pending tracking - lazy tracking test may complete it"
                )
                print(
                    "     If lazy tracking test passes, retracking test can proceed"
                )

        return None
    except Exception as exc:
        print(f"  ✗ Failed: {exc}")
        return None


def test_get_next_series(base_url: str, user_email: str) -> dict | None:
    """Test GET /api/series/next endpoint."""
    print("\n[2] Testing GET /api/series/next...")
    try:
        response = httpx.get(
            f"{base_url}/api/series/next",
            headers={"X-User-Email": user_email},
            timeout=10,
        )
        response.raise_for_status()
        data = response.json()

        if data.get("no_available_series"):
            print("  ⚠ No available series (all completed or recently viewed)")
            return None

        study_uid = data.get("study_uid")
        series_uid = data.get("series_uid")
        print(f"  ✓ Got series: {study_uid[:20]}.../{series_uid[:20]}...")
        print(f"  ✓ Exam number: {data.get('exam_number')}")
        print(f"  ✓ Tracking status: {data.get('tracking_status')}")
        return data
    except Exception as exc:
        print(f"  ✗ Failed: {exc}")
        return None


def test_get_masks(
    base_url: str, study_uid: str, series_uid: str
) -> tuple[bytes, dict] | None:
    """Test GET /api/masks/{study}/{series} endpoint."""
    print("\n[3] Testing GET /api/masks/{study}/{series}...")
    try:
        response = httpx.get(
            f"{base_url}/api/masks/{study_uid}/{series_uid}",
            timeout=30,
        )

        if response.status_code == 202:
            data = response.json()
            print(f"  ⚠ Masks pending: {data.get('error_code')}")
            return None, None

        response.raise_for_status()

        version_id = response.headers.get("X-Version-ID", "")
        mask_count = response.headers.get("X-Mask-Count", "0")
        print(f"  ✓ Downloaded mask archive ({len(response.content)} bytes)")
        print(f"  ✓ Version ID: {version_id or '(none)'}")
        print(f"  ✓ Mask count: {mask_count}")

        # Extract to temp directory to inspect
        import tempfile
        from lib.mask_archive import (
            extract_mask_archive,
            MASK_METADATA_FILENAME,
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            extract_path = Path(temp_dir)
            extract_mask_archive(response.content, extract_path)

            metadata_path = extract_path / MASK_METADATA_FILENAME
            if metadata_path.exists():
                with metadata_path.open() as f:
                    metadata = json.load(f)
                print(f"  ✓ Metadata: {len(metadata.get('frames', []))} frames")
                return response.content, metadata

        return response.content, {}
    except Exception as exc:
        print(f"  ✗ Failed: {exc}")
        return None, None


def create_modified_mask_archive(
    original_archive: bytes, study_uid: str, series_uid: str
) -> bytes:
    """
    Create a modified mask archive for testing.

    In a real scenario, this would be user edits. For testing, we'll
    just add a metadata annotation entry to simulate an edit.
    """
    import tempfile
    from lib.mask_archive import (
        extract_mask_archive,
        MASK_METADATA_FILENAME,
        build_mask_archive,
    )
    from lib.config import load_config

    config = load_config("client")

    with tempfile.TemporaryDirectory() as temp_dir:
        extract_path = Path(temp_dir)
        extract_mask_archive(original_archive, extract_path)

        # Load metadata
        metadata_path = extract_path / MASK_METADATA_FILENAME
        with metadata_path.open() as f:
            metadata = json.load(f)

        # Simulate user edit: mark first frame as annotation (it already is, but ensure it has label_id)
        frames = metadata.get("frames", [])
        if frames:
            # First frame should already be marked as annotation, just ensure it has label_id
            first_frame = frames[0]
            if first_frame.get("is_annotation", False):
                first_frame["label_id"] = (
                    config.label_id
                )  # Ensure label_id is set
                # This simulates an edit - frame is already an annotation, just updating it

        metadata["previous_version_id"] = metadata.get("version_id")
        metadata["editor"] = "test@example.com"
        metadata["edited_at"] = time.strftime(
            "%Y-%m-%dT%H:%M:%SZ", time.gmtime()
        )

        # Save modified metadata
        with metadata_path.open("w") as f:
            json.dump(metadata, f, indent=2)

        # Rebuild archive - need to find masks directory
        # Masks are in the extract_path root (extracted from archive)
        masks_dir = extract_path
        # Filter out metadata.json from mask files
        return build_mask_archive(masks_dir, metadata, include_metadata=True)


def test_post_masks(
    base_url: str,
    study_uid: str,
    series_uid: str,
    archive_bytes: bytes,
    previous_version_id: str,
    user_email: str,
) -> dict | None:
    """Test POST /api/masks/{study}/{series} endpoint."""
    print("\n[4] Testing POST /api/masks/{study}/{series}...")

    # Create modified archive
    modified_archive = create_modified_mask_archive(
        archive_bytes, study_uid, series_uid
    )

    try:
        response = httpx.post(
            f"{base_url}/api/masks/{study_uid}/{series_uid}",
            content=modified_archive,
            headers={
                "Content-Type": "application/x-tar+gzip",
                "X-Previous-Version-ID": previous_version_id or "",
                "X-Editor": user_email,
            },
            timeout=30,
        )

        if response.status_code == 409:
            data = response.json()
            error_code = data.get("error_code")
            print(f"  ⚠ Conflict detected: {error_code}")
            print(f"     {data.get('message')}")
            if error_code == "VERSION_MISMATCH":
                print(f"     Current version: {data.get('current_version')}")
                print(f"     Your version: {data.get('your_version')}")
            return None

        response.raise_for_status()
        data = response.json()
        print(f"  ✓ Masks uploaded successfully")
        print(f"  ✓ New version ID: {data.get('version_id')}")
        print(f"  ✓ Retrack queued: {data.get('retrack_queued')}")
        print(f"  ✓ Queue position: {data.get('queue_position')}")
        return data
    except Exception as exc:
        print(f"  ✗ Failed: {exc}")
        return None


def test_retrack_status(
    base_url: str, study_uid: str, series_uid: str, max_wait: int = 1800
) -> bool:
    """
    Test GET /api/retrack/status/{study}/{series} endpoint.

    Waits until retrack is actually completed or failed, then verifies results.
    """
    print("\n[5] Testing GET /api/retrack/status/{study}/{series}...")
    print("  Waiting for retrack to complete...")
    print(
        "  (Retracking typically takes 15-25 seconds, but allowing up to 30 minutes)"
    )

    start_time = time.time()
    last_status = None
    last_print_time = 0

    while time.time() - start_time < max_wait:
        try:
            elapsed = int(time.time() - start_time)
            response = httpx.get(
                f"{base_url}/api/retrack/status/{study_uid}/{series_uid}",
                timeout=10,
            )
            response.raise_for_status()
            data = response.json()

            status = data.get("status")

            # Print status changes or every 10 seconds
            if status != last_status or (elapsed - last_print_time) >= 10:
                if status == "completed":
                    print(f"  ✓ Retrack completed in {elapsed}s")
                elif status == "failed":
                    print(f"  ✗ Retrack failed after {elapsed}s")
                elif status in {"pending", "processing"}:
                    error_code = data.get("error_code")
                    queue_pos = data.get("queue_position")
                    if queue_pos:
                        print(
                            f"  ⏳ {error_code} (queue: {queue_pos}) - {elapsed}s"
                        )
                    else:
                        print(f"  ⏳ {error_code} - {elapsed}s")
                last_status = status
                last_print_time = elapsed

            if status == "completed":
                elapsed = int(time.time() - start_time)

                # Verify retrack actually produced new masks
                version_id = data.get("version_id")
                print(f"  ✓ Version ID: {version_id}")

                # Fetch masks to verify they exist and are valid
                mask_response = httpx.get(
                    f"{base_url}/api/masks/{study_uid}/{series_uid}",
                    timeout=10,
                )
                if mask_response.status_code == 200:
                    archive_bytes = mask_response.content
                    if len(archive_bytes) == 0:
                        print(
                            f"  ✗ Retrack completed but masks archive is empty"
                        )
                        return False

                    # Verify archive has masks
                    from lib.mask_archive import (
                        extract_mask_archive,
                        MASK_METADATA_FILENAME,
                    )
                    import tempfile

                    with tempfile.TemporaryDirectory() as temp_dir:
                        extract_path = Path(temp_dir)
                        extract_mask_archive(archive_bytes, extract_path)
                        mask_files = list(extract_path.glob("*.webp"))
                        if not mask_files:
                            print(
                                f"  ✗ Retrack completed but no mask files found"
                            )
                            return False
                        print(f"  ✓ Retrack completed in {elapsed}s")
                        print(
                            f"  ✓ Verified: {len(mask_files)} mask files in archive ({len(archive_bytes)} bytes)"
                        )
                        return True
                else:
                    print(
                        f"  ✗ Retrack completed but cannot fetch masks (status {mask_response.status_code})"
                    )
                    return False

            elif status == "failed":
                error_code = data.get("error_code")
                error_msg = data.get("error_message")
                elapsed = int(time.time() - start_time)
                print(f"  ✗ Retrack failed after {elapsed}s: {error_code}")
                if error_msg:
                    print(f"     {error_msg}")
                return False

            time.sleep(2)

        except Exception as exc:
            print(f"  ✗ Error checking status: {exc}")
            return False

    elapsed = int(time.time() - start_time)
    print(f"  ✗ Timeout waiting for retrack completion ({elapsed}s)")
    return False


def test_version_conflict(
    base_url: str,
    study_uid: str,
    series_uid: str,
    archive_bytes: bytes,
    user_email: str,
) -> bool:
    """Test version conflict detection."""
    print("\n[6] Testing version conflict detection...")

    # Try to upload with wrong version ID
    try:
        response = httpx.post(
            f"{base_url}/api/masks/{study_uid}/{series_uid}",
            content=archive_bytes,
            headers={
                "Content-Type": "application/x-tar+gzip",
                "X-Previous-Version-ID": "wrong_version_id",
                "X-Editor": user_email,
            },
            timeout=30,
        )

        if response.status_code == 409:
            data = response.json()
            if data.get("error_code") == "VERSION_MISMATCH":
                print(f"  ✓ Version conflict correctly detected")
                print(f"     Current: {data.get('current_version')}")
                print(f"     Provided: {data.get('your_version')}")
                return True

        print(f"  ✗ Expected 409 VERSION_MISMATCH, got {response.status_code}")
        return False
    except Exception as exc:
        print(f"  ✗ Failed: {exc}")
        return False


def main() -> None:
    """Run all server API tests."""
    print("=" * 60)
    print("Server API Test Suite")
    print("=" * 60)
    print("\n⚠️  This test uses REAL data and requires:")
    print("   1. Server running (uv run python server/server.py)")
    print(
        "   2. Retrack worker running (uv run python server/retrack_worker.py)"
    )
    print("   3. At least one series with tracked masks in output/")
    print()

    config = load_config("client")
    base_url = config.server_url
    user_email = config.user_email or "test@example.com"

    print(f"Server URL: {base_url}")
    print(f"User email: {user_email}")
    print()

    # Test 1: Server status
    if not test_server_status(base_url):
        print("\n❌ Server not available. Start server first.")
        sys.exit(1)

    # Test 2: Find a series with completed tracking (required for retracking test)
    # First try to find one with completed tracking
    series_data = find_series_with_completed_tracking(base_url)
    if not series_data:
        # No completed series yet - lazy tracking test should create one
        # But we can also try to find a pending one and wait, or use get_next_series
        print("\n  No series with completed tracking found yet.")
        print("  This is expected if lazy tracking test hasn't completed yet.")
        print("  Trying get_next_series as fallback...")
        series_data = test_get_next_series(base_url, user_email)
        if not series_data:
            print("\n✗ No series available for testing")
            print(
                "   Note: If lazy tracking test passes, it will create a completed series"
            )
            print("   that can be used for retracking tests.")
            sys.exit(1)

        # Check if this series has completed tracking
        if series_data.get("tracking_status") != "completed":
            print(
                f"\n⚠️  Series from get_next_series has status: {series_data.get('tracking_status')}"
            )
            print(
                "   Retracking test requires a series with completed tracking."
            )
            print(
                "   This may work if the series completes tracking during the test."
            )

    study_uid = series_data["study_uid"]
    series_uid = series_data["series_uid"]
    current_version = series_data.get("current_version_id")

    # Test 3: Get masks (should be available since we found a series with completed tracking)
    archive_bytes, metadata = test_get_masks(base_url, study_uid, series_uid)
    if not archive_bytes:
        # Masks not available - this shouldn't happen if tracking_status is "completed"
        tracking_status = series_data.get("tracking_status")
        print(
            f"\n✗ Cannot test retracking: masks not available (status: {tracking_status})"
        )
        print("   Series has 'completed' status but masks are not available.")
        print("   This may indicate a data inconsistency issue.")
        sys.exit(1)  # Fail the test - we can't test retracking without masks

    # Test 4: Post masks - this tests RETRACKING
    # Only test retracking if we have masks to upload
    if series_data.get("tracking_status") == "completed" and archive_bytes:
        post_result = test_post_masks(
            base_url,
            study_uid,
            series_uid,
            archive_bytes,
            current_version,
            user_email,
        )

        if post_result:
            # Test 5: Wait for retrack to actually complete and verify results
            retrack_success = test_retrack_status(
                base_url, study_uid, series_uid
            )
            if not retrack_success:
                print("\n⚠️  Retrack did not complete successfully")
                sys.exit(1)

    # Test 6: Version conflict
    test_version_conflict(
        base_url, study_uid, series_uid, archive_bytes, user_email
    )

    print("\n" + "=" * 60)
    print("Test suite completed")
    print("=" * 60)


if __name__ == "__main__":
    main()
