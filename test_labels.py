#!/usr/bin/env python3
"""Test that tracked frames are labeled correctly."""
import sys
import os
from dotenv import load_dotenv

load_dotenv('dot.env')
sys.path.insert(0, 'lib')

from multi_frame_tracker_optimized import process_annotations_optimized
from opticalflowprocessor import OpticalFlowProcessor

# Test with a simple annotation set
test_annotations = [
    {'frame': 10, 'mask': None, 'type': 'fluid'},
    {'frame': 20, 'mask': None, 'type': 'fluid'},
]

# Mock masks (normally would be numpy arrays)
import numpy as np
for ann in test_annotations:
    ann['mask'] = np.zeros((480, 640), dtype=np.uint8)

# Create flow processor
flow_processor = OpticalFlowProcessor(method='farneback')

# Run processing
result = process_annotations_optimized(
    annotations=test_annotations,
    video_path='dummy.mp4',  # Will fail but we'll check labels first
    flow_processor=flow_processor,
    output_dir='output/test_labels'
)

# Check label assignments
LABEL_ID = os.getenv('LABEL_ID')
TRACK_ID = os.getenv('TRACK_ID')

print(f"Human-annotated label: {LABEL_ID}")
print(f"Machine-annotated label: {TRACK_ID}")
print("\nChecking frame labels:")

for frame_num in sorted(result.keys()):
    frame_data = result[frame_num]
    label = frame_data.get('label_id', 'MISSING')
    is_ann = frame_data.get('is_annotation', False)

    if is_ann:
        expected = LABEL_ID
        status = "✓" if label == expected else "✗"
        print(f"Frame {frame_num}: {label} (original annotation) {status}")
    else:
        expected = TRACK_ID
        status = "✓" if label == expected else "✗"
        print(f"Frame {frame_num}: {label} (tracked) {status}")