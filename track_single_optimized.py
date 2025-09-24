#!/usr/bin/env python3
"""
Run optimized multi-frame tracking on the 34-annotation test video.
"""
import os
import sys
import json
from pathlib import Path
from dotenv import load_dotenv
import cv2
import numpy as np
from datetime import datetime

# Load environment
load_dotenv('dot.env')

# Add lib to path
sys.path.insert(0, 'lib')

from opticalflowprocessor import OpticalFlowProcessor
from multi_frame_tracker_optimized import process_annotations_optimized

# Get test study/series from environment
TEST_STUDY_UID = os.getenv('TEST_STUDY_UID')
TEST_SERIES_UID = os.getenv('TEST_SERIES_UID')
DATA_DIR = os.getenv('DATA_DIR')
ANNOTATIONS_FILE = os.path.join(DATA_DIR, os.getenv('ANNOTATIONS'))
LABEL_ID = os.getenv('LABEL_ID')
FLOW_METHOD = os.getenv('FLOW_METHOD', 'farneback').split(',')[0]  # Use first method

print(f"Running optimized tracking on 34-annotation video")
print(f"Study UID: {TEST_STUDY_UID}")
print(f"Series UID: {TEST_SERIES_UID}")
print(f"Using flow method: {FLOW_METHOD}")

# Load annotations
with open(ANNOTATIONS_FILE) as f:
    annotations_data = json.load(f)

# Find annotations for this study/series
study_annotations = []
for dataset in annotations_data['datasets']:
    for annotation in dataset.get('annotations', []):
        if (annotation.get('StudyInstanceUID') == TEST_STUDY_UID and
            annotation.get('SeriesInstanceUID') == TEST_SERIES_UID and
            annotation.get('labelId') == LABEL_ID):
            study_annotations.append(annotation)

print(f"Found {len(study_annotations)} annotations")

# Build video path
video_path = Path(DATA_DIR) / f"mdai_ucsf_project_x9N2LJBZ_images_2025-09-18-050340" / TEST_STUDY_UID / f"{TEST_SERIES_UID}.mp4"

if not video_path.exists():
    print(f"ERROR: Video not found at {video_path}")
    sys.exit(1)

# Process annotations to extract masks
def decode_mask(annotation_data):
    """Decode mask from annotation (polygon format)."""
    if 'data' not in annotation_data:
        return None

    data = annotation_data['data']

    # Handle different data formats
    if isinstance(data, str):
        # Try to parse as JSON
        try:
            import json
            data = json.loads(data)
        except:
            return None

    if isinstance(data, dict) and 'foreground' in data:
        # Polygon data
        polygons = data['foreground']
        height = annotation_data.get('height', 480)
        width = annotation_data.get('width', 640)

        # Create mask from polygons
        mask = np.zeros((height, width), dtype=np.uint8)

        # Handle nested list structure - foreground contains list of polygons
        if isinstance(polygons, list):
            for polygon in polygons:
                if isinstance(polygon, list) and len(polygon) >= 6:  # At least 3 points
                    # Convert flat list of coordinates to points
                    points = np.array(polygon).reshape(-1, 2).astype(np.int32)
                    cv2.fillPoly(mask, [points], 255)

        return mask

    return None

# Convert annotations to our format
annotations = []
for ann in study_annotations:
    mask = decode_mask(ann)
    if mask is not None:
        annotations.append({
            'frame': ann.get('frameNumber', 0),
            'mask': mask,
            'type': 'fluid'
        })

print(f"Processed {len(annotations)} valid annotations with masks")

# Create flow processor
flow_processor = OpticalFlowProcessor(method=FLOW_METHOD)

# Run optimized tracking
output_dir = f"output/{FLOW_METHOD}_optimized/{TEST_STUDY_UID}_{TEST_SERIES_UID}"

print(f"\nStarting optimized tracking...")
start_time = datetime.now()

result = process_annotations_optimized(
    annotations=annotations,
    video_path=str(video_path),
    flow_processor=flow_processor,
    output_dir=output_dir
)

end_time = datetime.now()
duration = (end_time - start_time).total_seconds()

print(f"\n{'='*60}")
print(f"Tracking completed in {duration:.1f} seconds")
print(f"Results saved to: {output_dir}")
print(f"{'='*60}")