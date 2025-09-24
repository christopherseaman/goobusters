#!/usr/bin/env python3
"""
Run multi-frame tracking on a single study/series.
"""
import os
import sys
from dotenv import load_dotenv

# Load environment variables
load_dotenv('dot.env')

# Get the test study/series from environment
TEST_STUDY_UID = os.getenv('TEST_STUDY_UID', '1.2.826.0.1.3680043.8.498.44600297086425666247367139389036414147')
TEST_SERIES_UID = os.getenv('TEST_SERIES_UID', '1.2.826.0.1.3680043.8.498.40380241154250908958461848377748917341')

print(f"Running tracking on single study/series:")
print(f"  Study UID: {TEST_STUDY_UID}")
print(f"  Series UID: {TEST_SERIES_UID}")

# Import and run the tracking
sys.path.insert(0, 'lib')
from dotenv import load_dotenv
import mdai
import pandas as pd
import json
import cv2
import numpy as np
from pathlib import Path

load_dotenv('dot.env')

ACCESS_TOKEN = os.getenv('MDAI_TOKEN')
DATA_DIR = os.getenv('DATA_DIR')
ANNOTATIONS = os.path.join(DATA_DIR, os.getenv('ANNOTATIONS'))
LABEL_ID = os.getenv('LABEL_ID')
FLOW_METHOD = os.getenv('FLOW_METHOD', 'farneback').split(',')

print(f"Using flow methods: {FLOW_METHOD}")
print(f"Loading annotations from: {ANNOTATIONS}")

# Load annotations
with open(ANNOTATIONS) as f:
    annotations_data = json.load(f)

# Find annotations for this study/series
study_annotations = []
for dataset in annotations_data['datasets']:
    for annotation in dataset.get('annotations', []):
        if (annotation.get('StudyInstanceUID') == TEST_STUDY_UID and
            annotation.get('SeriesInstanceUID') == TEST_SERIES_UID and
            annotation.get('labelId') == LABEL_ID):
            study_annotations.append(annotation)

print(f"Found {len(study_annotations)} annotations for this video")

# Build video path
video_path = Path(DATA_DIR) / f"mdai_ucsf_project_x9N2LJBZ_images_2025-09-18-050340" / TEST_STUDY_UID / f"{TEST_SERIES_UID}.mp4"

if not video_path.exists():
    print(f"ERROR: Video not found at {video_path}")
    sys.exit(1)

print(f"Video path: {video_path}")

# Get video info
cap = cv2.VideoCapture(str(video_path))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = cap.get(cv2.CAP_PROP_FPS)
cap.release()
print(f"Video info: {total_frames} frames at {fps:.1f} fps")

# Run tracking for each flow method
from multi_frame_tracker import process_video_with_multi_frame_tracking
from opticalflowprocessor import OpticalFlowProcessor

for method in FLOW_METHOD:
    print(f"\n{'='*60}")
    print(f"Running {method} on video with {len(study_annotations)} annotations")
    print(f"{'='*60}")

    try:
        # Create flow processor for this method
        flow_processor = OpticalFlowProcessor(method=method)

        # Create DataFrame with annotations
        annotations_df = pd.DataFrame(study_annotations)

        result = process_video_with_multi_frame_tracking(
            video_path=str(video_path),
            annotations_df=annotations_df,
            study_uid=TEST_STUDY_UID,
            series_uid=TEST_SERIES_UID,
            flow_processor=flow_processor,
            output_dir=f"output/{method}/{TEST_STUDY_UID}_{TEST_SERIES_UID}",
            upload_to_mdai=False,
            mdai_client=None,
            label_id_fluid=LABEL_ID,
            project_id=None,
            dataset_id=None
        )

        # Count results
        if result:
            annotated = sum(1 for r in result.values() if r.get('is_annotation', False))
            predicted = len(result) - annotated
            print(f"\n✅ {method} completed:")
            print(f"   - Total frames processed: {len(result)}")
            print(f"   - Annotated frames: {annotated}")
            print(f"   - Predicted frames: {predicted}")
            print(f"   - Coverage: {len(result)/total_frames*100:.1f}% of video")
        else:
            print(f"❌ {method} failed - no results returned")

    except Exception as e:
        print(f"❌ {method} failed with error: {e}")
        import traceback
        traceback.print_exc()

print("\n" + "="*60)
print("Single study tracking complete!")