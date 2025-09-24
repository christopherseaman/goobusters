#!/usr/bin/env python3
"""Check if labels are correctly assigned in the tracking output."""
import os
import sys
import json
from pathlib import Path
from dotenv import load_dotenv

load_dotenv('dot.env')
sys.path.insert(0, 'lib')

# Run the optimized tracker with labels
from opticalflowprocessor import OpticalFlowProcessor
from multi_frame_tracker_optimized import process_annotations_optimized

TEST_STUDY_UID = os.getenv('TEST_STUDY_UID')
TEST_SERIES_UID = os.getenv('TEST_SERIES_UID')
DATA_DIR = os.getenv('DATA_DIR')
LABEL_ID = os.getenv('LABEL_ID')
TRACK_ID = os.getenv('TRACK_ID')

print(f"Label IDs configured:")
print(f"  Human-annotated (LABEL_ID): {LABEL_ID}")
print(f"  Machine-tracked (TRACK_ID): {TRACK_ID}")

# Create simple test annotations
import numpy as np
test_annotations = [
    {'frame': 0, 'mask': np.ones((100, 100), dtype=np.uint8) * 255, 'type': 'fluid'},
    {'frame': 5, 'mask': np.ones((100, 100), dtype=np.uint8) * 255, 'type': 'fluid'},
    {'frame': 10, 'mask': np.ones((100, 100), dtype=np.uint8) * 255, 'type': 'fluid'},
]

# Mock video path (we'll create a dummy one)
video_path = "test_video.mp4"

# Create a dummy video file to avoid file not found errors
import cv2
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(video_path, fourcc, 1.0, (100, 100))
for i in range(15):
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    out.write(frame)
out.release()

try:
    # Create flow processor
    flow_processor = OpticalFlowProcessor(method='farneback')

    # Run tracking
    result = process_annotations_optimized(
        annotations=test_annotations,
        video_path=video_path,
        flow_processor=flow_processor,
        output_dir='output/label_test'
    )

    # Check results
    print("\n\nFrame Analysis:")
    print("-" * 60)

    original_frames = []
    tracked_frames = []

    for frame_num in sorted(result.keys()):
        frame_data = result[frame_num]
        label = frame_data.get('label_id', 'MISSING')
        is_ann = frame_data.get('is_annotation', False)
        frame_type = frame_data.get('type', 'unknown')

        if is_ann:
            original_frames.append(frame_num)
            status = "✓" if label == LABEL_ID else f"✗ (got {label})"
            print(f"Frame {frame_num:3d}: Original annotation - Label: {label} {status}")
        else:
            tracked_frames.append(frame_num)
            status = "✓" if label == TRACK_ID else f"✗ (got {label})"
            print(f"Frame {frame_num:3d}: Tracked ({frame_type:8s}) - Label: {label} {status}")

    print("\nSummary:")
    print(f"  Original annotation frames ({len(original_frames)}): {original_frames}")
    print(f"  Tracked frames ({len(tracked_frames)}): {tracked_frames}")

    # Verify all tracked frames have correct label
    all_correct = all(
        result[f].get('label_id') == TRACK_ID
        for f in tracked_frames
    ) and all(
        result[f].get('label_id') == LABEL_ID
        for f in original_frames
    )

    if all_correct:
        print("\n✅ All frames have correct labels!")
    else:
        print("\n❌ Some frames have incorrect labels")

finally:
    # Clean up test video
    if os.path.exists(video_path):
        os.remove(video_path)