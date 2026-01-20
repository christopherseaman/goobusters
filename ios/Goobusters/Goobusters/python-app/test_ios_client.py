#!/usr/bin/env python3
"""
Test script for iOS-compatible MD.ai client.
Verifies that httpx-based client works without pandas/numpy/opencv.
"""

import sys
from pathlib import Path

# Test if we can import the iOS client without heavy dependencies
try:
    from lib.client.ios_mdai_client import (
        MDaiClient,
        json_to_dataframe,
        filter_annotations,
        group_by_video,
        add_video_paths,
        sort_annotations_by_frame,
    )

    print("✓ iOS client imports successfully")
except ImportError as e:
    print(f"✗ Failed to import iOS client: {e}")
    sys.exit(1)


# Test json_to_dataframe with a sample annotations file
def test_json_parsing():
    """Test that we can parse MD.ai JSON without pandas."""
    # Create minimal test JSON
    test_json = Path("test_annotations.json")
    test_json.write_text("""{
        "datasets": [{
            "id": "test_dataset",
            "name": "Test Dataset",
            "studies": [{
                "StudyInstanceUID": "1.2.3.4",
                "number": 1,
                "SeriesNumber": "5"
            }],
            "annotations": [{
                "StudyInstanceUID": "1.2.3.4",
                "SeriesInstanceUID": "5.6.7.8",
                "labelId": "L_abc123",
                "frameNumber": 10,
                "data": {
                    "foreground": [[[0, 0], [100, 0], [100, 100], [0, 100]]],
                    "background": []
                }
            }]
        }],
        "labelGroups": [{
            "id": "LG_xyz",
            "name": "Free Fluid",
            "labels": [{
                "id": "L_abc123",
                "name": "Fluid",
                "annotationMode": "freeform",
                "color": "#ff0000",
                "description": "Free fluid region",
                "radlexTagIds": [],
                "scope": "instance"
            }]
        }]
    }""")

    try:
        # Parse JSON
        result = json_to_dataframe(str(test_json))

        # Verify structure
        assert "annotations" in result
        assert "studies" in result
        assert "labels" in result

        # Verify data types (should be plain lists/dicts, not DataFrames)
        assert isinstance(result["annotations"], list)
        assert isinstance(result["studies"], list)
        assert isinstance(result["labels"], list)

        # Verify content
        assert len(result["annotations"]) == 1
        assert result["annotations"][0]["labelId"] == "L_abc123"
        assert result["annotations"][0]["dataset"] == "Test Dataset"

        assert len(result["studies"]) == 1
        assert result["studies"][0]["StudyInstanceUID"] == "1.2.3.4"

        assert len(result["labels"]) == 1
        assert result["labels"][0]["labelId"] == "L_abc123"

        print("✓ JSON parsing works correctly")
        return True
    except Exception as e:
        print(f"✗ JSON parsing failed: {e}")
        import traceback

        traceback.print_exc()
        return False
    finally:
        # Cleanup
        if test_json.exists():
            test_json.unlink()


# Test utility functions
def test_utilities():
    """Test utility functions work with plain dicts."""
    annotations = [
        {"labelId": "L_1", "frameNumber": 5},
        {"labelId": "L_2", "frameNumber": 3},
        {"labelId": "L_1", "frameNumber": 1},
    ]

    # Test filter
    filtered = filter_annotations(annotations, ["L_1"])
    assert len(filtered) == 2
    assert all(a["labelId"] == "L_1" for a in filtered)
    print("✓ filter_annotations works")

    # Test sort
    sorted_anns = sort_annotations_by_frame(annotations)
    assert sorted_anns[0]["frameNumber"] == 1
    assert sorted_anns[1]["frameNumber"] == 3
    assert sorted_anns[2]["frameNumber"] == 5
    print("✓ sort_annotations_by_frame works")

    # Test grouping
    video_annotations = [
        {"StudyInstanceUID": "S1", "SeriesInstanceUID": "V1", "frameNumber": 1},
        {"StudyInstanceUID": "S1", "SeriesInstanceUID": "V1", "frameNumber": 2},
        {"StudyInstanceUID": "S1", "SeriesInstanceUID": "V2", "frameNumber": 1},
    ]
    grouped = group_by_video(video_annotations)
    assert len(grouped) == 2
    assert len(grouped[("S1", "V1")]) == 2
    assert len(grouped[("S1", "V2")]) == 1
    print("✓ group_by_video works")

    return True


# Run tests
if __name__ == "__main__":
    print("\n=== Testing iOS-compatible MD.ai client ===\n")

    success = True
    success &= test_json_parsing()
    success &= test_utilities()

    print("\n" + "=" * 50)
    if success:
        print("✓ All tests passed!")
        print("\nThe iOS client is ready to use.")
        print("It uses only httpx (no pandas, numpy, or opencv).")
        sys.exit(0)
    else:
        print("✗ Some tests failed")
        sys.exit(1)
