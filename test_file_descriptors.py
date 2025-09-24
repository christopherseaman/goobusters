#!/usr/bin/env python3
"""
Test script to verify file descriptor management fixes.

This script checks that video capture resources are properly released
and file descriptors don't leak.
"""

import os
import resource
import cv2
import sys
import tempfile
import numpy as np

# Add the lib directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'lib'))

from video_capture_manager import video_capture, get_video_properties


def get_open_file_count():
    """Get the current number of open file descriptors."""
    import psutil
    process = psutil.Process()
    return len(process.open_files())


def test_video_capture_cleanup():
    """Test that video captures are properly cleaned up."""
    print("Testing video capture cleanup...")

    # Create a temporary test video
    temp_video = tempfile.mktemp(suffix='.mp4')

    # Create a simple test video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(temp_video, fourcc, 30.0, (640, 480))
    for _ in range(30):
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        out.write(frame)
    out.release()

    # Get initial file descriptor count
    try:
        import psutil
        initial_count = get_open_file_count()
        print(f"Initial open file count: {initial_count}")
    except ImportError:
        print("psutil not available, using resource limits instead")
        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        print(f"File descriptor limits - Soft: {soft}, Hard: {hard}")
        initial_count = None

    # Test multiple video capture operations
    print("\nTesting 100 video capture operations...")
    for i in range(100):
        try:
            # Test with context manager
            with video_capture(temp_video) as cap:
                ret, frame = cap.read()
                if not ret:
                    print(f"  Failed to read frame at iteration {i}")

            # Test get_video_properties
            props = get_video_properties(temp_video)

            if i % 20 == 0:
                print(f"  Completed {i} iterations...")
        except Exception as e:
            print(f"  Error at iteration {i}: {str(e)}")
            break

    # Check final file descriptor count
    if initial_count is not None:
        final_count = get_open_file_count()
        print(f"\nFinal open file count: {final_count}")

        if final_count > initial_count + 5:
            print("⚠️  WARNING: Possible file descriptor leak detected!")
            print(f"   File descriptors increased by {final_count - initial_count}")
        else:
            print("✅ No file descriptor leak detected")

    # Clean up
    try:
        os.remove(temp_video)
    except:
        pass

    print("\nTest completed successfully!")


def test_error_handling():
    """Test error handling with invalid video paths."""
    print("\nTesting error handling with invalid paths...")

    try:
        with video_capture("/nonexistent/video.mp4") as cap:
            print("This should not print")
    except RuntimeError as e:
        print(f"✅ Correctly caught error: {str(e)}")
    except Exception as e:
        print(f"⚠️  Unexpected error: {str(e)}")

    print("Error handling test completed")


if __name__ == "__main__":
    print("=" * 60)
    print("File Descriptor Management Test")
    print("=" * 60)

    test_video_capture_cleanup()
    test_error_handling()

    print("\n" + "=" * 60)
    print("All tests completed!")
    print("=" * 60)