# test_multi_frame_tracker.py

import os
import cv2
import pandas as pd
import numpy as np
from datetime import datetime

from opticalflowprocessor import OpticalFlowProcessor
from multi_frame_tracker import MultiFrameTracker
import json

# Set custom label IDs for testing
TEST_LABEL_ID_FLUID = "L_7BGQQl"  #temp testing label 
TEST_LABEL_ID_MACHINE_GROUP = "G_RJY6Qn"  # Use existing or another test ID

# Setup test directories
TEST_DIR = os.path.join(os.getcwd(), "test_multi_frame")
os.makedirs(TEST_DIR, exist_ok=True)

print(f"Created test directory: {TEST_DIR}")
print(f"Using test label ID: {TEST_LABEL_ID_FLUID}")

# Hard-coded path to a known working video
VIDEO_PATH = "/Users/Shreya1/Documents/GitHub/goobusters/data/mdai_ucsf_project_x9N2LJBZ_images_2025-03-31-175635/1.2.826.0.1.3680043.8.498.12762211632497404572246503032980657292/1.2.826.0.1.3680043.8.498.90262783102403545676047413537747709850.mp4"
STUDY_UID = "1.2.826.0.1.3680043.8.498.12762211632497404572246503032980657292"
SERIES_UID = "1.2.826.0.1.3680043.8.498.90262783102403545676047413537747709850"
FRAME_NUMBER = 53

# Load the annotations file to get the corresponding free_fluid_foreground data
ANNOTATIONS_PATH = "/Users/Shreya1/Documents/GitHub/goobusters/data/mdai_ucsf_project_x9N2LJBZ_annotations_2025-04-01-035131.json"

print(f"Using video file: {VIDEO_PATH}")
print(f"Study UID: {STUDY_UID}")
print(f"Series UID: {SERIES_UID}")
print(f"Frame number: {FRAME_NUMBER}")

# Check if video file exists
if not os.path.exists(VIDEO_PATH):
    print(f"ERROR: Video file not found: {VIDEO_PATH}")
    exit(1)

# Load annotations to get the polygon data for this frame
try:
    with open(ANNOTATIONS_PATH, 'r') as f:
        annotations_json = json.load(f)

    # Extract polygon data
    from pandas import json_normalize
    all_annotations = []
    for dataset in annotations_json.get('datasets', []):
        if 'annotations' in dataset:
            all_annotations.extend(dataset['annotations'])

    annotations_df = json_normalize(all_annotations, sep='_')

    # Filter to find this exact video and frame
    ACTUAL_LABEL_ID_FREE_FLUID = os.getenv('LABEL_ID_FREE_FLUID', 'L_13yPq1')
    target_annotation = annotations_df[
        (annotations_df['labelId'] == ACTUAL_LABEL_ID_FREE_FLUID) &
        (annotations_df['StudyInstanceUID'] == STUDY_UID) &
        (annotations_df['SeriesInstanceUID'] == SERIES_UID) &
        (annotations_df['frameNumber'] == FRAME_NUMBER)
    ]

    if len(target_annotation) == 0:
        print(f"ERROR: No annotation found for frame {FRAME_NUMBER} in video {SERIES_UID}")
        exit(1)

    sample_annotation = target_annotation.iloc[0].copy()
    sample_annotation['free_fluid_foreground'] = sample_annotation['data_foreground']
    print(f"Found annotation at index {target_annotation.index[0]}")
    
except Exception as e:
    print(f"Error loading annotations: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

cap = cv2.VideoCapture(VIDEO_PATH)
ret, frame = cap.read()
if not ret:
    print(f"ERROR: Could not read frame from video: {VIDEO_PATH}")
    exit(1)

frame_height, frame_width = frame.shape[:2]
print(f"Video dimensions: {frame_width}x{frame_height}")
cap.release()

# Initialise optical flow processor
FLOW_METHOD = 'dis'  
print(f"Initializing OpticalFlowProcessor with method: {FLOW_METHOD}")
flow_processor = OpticalFlowProcessor(FLOW_METHOD)

# Initialise MultiFrameTracker
print("Initializing MultiFrameTracker...")
tracker = MultiFrameTracker(flow_processor, TEST_DIR, debug_mode=True)

# Create sample df for the tracker
# Make sure to include the frame number and polygon data in the correct format
sample_data = {
    'frameNumber': FRAME_NUMBER,
    'free_fluid_foreground': sample_annotation['free_fluid_foreground'],
    'id': sample_annotation.get('id', 'test_id')
}
sample_df = pd.DataFrame([sample_data])

print("Created sample DataFrame with actual annotation data")

# Try processing the annotation
print("\nTesting MultiFrameTracker.process_annotations...")
try:
    result = tracker.process_annotations(
        annotations_df=sample_df,
        video_path=VIDEO_PATH,
        study_uid=STUDY_UID,
        series_uid=SERIES_UID
    )
    
    print("\nSUCCESS: MultiFrameTracker processed the annotation")
    print(f"Number of frames processed: {len(result)}")
    print(f"Frame numbers: {sorted(list(result.keys()))[:10]}... (showing first 10)")
    
    # Check output files
    output_video = os.path.join(TEST_DIR, "multi_frame_tracking.mp4")
    if os.path.exists(output_video):
        print(f"Output video created: {output_video}")
        # Get video info
        cap = cv2.VideoCapture(output_video)
        output_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        output_fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        print(f"Output video stats: {output_frames} frames, {output_fps} fps")
    else:
        print(f"WARNING: Output video not found: {output_video}")

    # Now test the full process_video_with_multi_frame_tracking function
    print("\nTesting process_video_with_multi_frame_tracking...")
    from multi_frame_tracker import process_video_with_multi_frame_tracking
    
    full_result = process_video_with_multi_frame_tracking(
        video_path=VIDEO_PATH,
        annotations_df=sample_df,
        study_uid=STUDY_UID,
        series_uid=SERIES_UID,
        flow_processor=flow_processor,
        output_dir=os.path.join(TEST_DIR, "full_test"),
        mdai_client=None,  # Skip uploading for test
        label_id_fluid=TEST_LABEL_ID_FLUID,  # Use the test label ID here
        label_id_machine=TEST_LABEL_ID_MACHINE_GROUP,
        upload_to_mdai=False  # Make sure to set this to False for testing
    )
    
    print("\nSUCCESS: Full process_video_with_multi_frame_tracking completed")
    print(f"Result summary: {full_result}")
    
    print(f"\nTest completed successfully using test label ID: {TEST_LABEL_ID_FLUID}")
    print("You can now integrate MultiFrameTracker into your main script.")
    
except Exception as e:
    print(f"\nERROR during processing: {str(e)}")
    import traceback
    traceback.print_exc()
    print("\nMultiFrameTracker is not compatible with your current setup.")