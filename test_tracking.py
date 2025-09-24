#!/usr/bin/env python3
"""Quick test to verify multi-frame tracking is working."""

import sys
import os
sys.path.insert(0, 'lib')

# Set environment
os.environ['LABEL_ID_NO_FLUID'] = 'L_75K42J'
os.environ['MDAI_TOKEN'] = 'd2b086facd41171613d918a9abefe499'

# Fix import issue
sys.modules['lib.opticalflowprocessor'] = type('MockModule', (), {
    'OpticalFlowProcessor': type('MockProcessor', (), {
        '__init__': lambda self, method: None,
        'calculate_flow': lambda self, prev, curr: np.zeros((540, 720, 2), dtype=np.float32)
    })
})()

from multi_frame_tracker import MultiFrameTracker, SharedParams
import cv2
import numpy as np

# Test with a simple video
video_path = "data/mdai_ucsf_project_x9N2LJBZ_images_2025-09-18-050340/1.2.826.0.1.3680043.8.498.72482083786325365098389308815031363477/1.2.826.0.1.3680043.8.498.21145625036577365260308414819346452349.mp4"

# Check video exists
if not os.path.exists(video_path):
    print(f"Video not found: {video_path}")
    exit(1)

print("Testing multi-frame tracking...")
print("-" * 50)

# Initialize tracker
shared_params = SharedParams()
from opticalflowprocessor import OpticalFlowProcessor
flow_processor = OpticalFlowProcessor(method='farneback')
tracker = MultiFrameTracker(
    flow_processor=flow_processor,
    output_dir='output/test',
    debug_mode=True
)
tracker.shared_params = shared_params  # Add shared params

# Create a test annotation at frame 75 (last frame)
cap = cv2.VideoCapture(video_path)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
cap.release()

print(f"Video info: {total_frames} frames, {width}x{height}")

# Create a dummy mask (circular region)
mask = np.zeros((height, width), dtype=np.uint8)
cv2.circle(mask, (width//2, height//2), 50, 255, -1)

# Create annotation at last frame
annotations = [{
    'frame': total_frames - 1,  # Last frame
    'type': 'fluid',
    'mask': mask
}]

print(f"Created annotation at frame {total_frames - 1}")
print("Processing annotations...")

# Open video for tracker
tracker.cap = cv2.VideoCapture(video_path)
tracker.total_frames = total_frames

# Call internal tracking method directly
print(f"Tracking backward from frame {total_frames - 1} to 0...")
result = tracker._track_backward(
    start_frame=total_frames - 1,
    end_frame=0,
    initial_mask=mask
)

print("-" * 50)
print(f"RESULT: Processed {len(result)} frames")

# Count annotation vs predicted frames
annotated = sum(1 for r in result.values() if r.get('is_annotation', False))
predicted = len(result) - annotated

print(f"  - Annotated frames: {annotated}")
print(f"  - Predicted frames: {predicted}")

if predicted > 0:
    print(f"✅ Multi-frame tracking IS WORKING! Propagated to {predicted} frames")
else:
    print(f"❌ Multi-frame tracking NOT working - no frames were propagated")