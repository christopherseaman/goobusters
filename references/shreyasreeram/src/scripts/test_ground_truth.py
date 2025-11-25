#!/usr/bin/env python3
import os
import json
import numpy as np
import pandas as pd
from src.utils import get_annotations_for_study_series, create_ground_truth_dataset

def test_ground_truth_utils():
    """Test the ground truth utilities with mock data"""
    print("Testing ground truth utilities...")
    
    # Create test directory
    test_output_dir = "src/output/test_ground_truth"
    os.makedirs(test_output_dir, exist_ok=True)
    
    # Create mock data
    mock_study_uid = "1.2.3.4.5"
    mock_series_uid = "1.2.3.4.5.1"
    
    # Create mock annotations DataFrame
    matched_annotations = pd.DataFrame({
        'StudyInstanceUID': [mock_study_uid],
        'SeriesInstanceUID': [mock_series_uid],
        'frameNumber': [1],
        'data': [{'foreground': [[[100, 100], [200, 100], [200, 200], [100, 200]]]}]
    })
    
    free_fluid_annotations = pd.DataFrame({
        'StudyInstanceUID': [mock_study_uid],
        'SeriesInstanceUID': [mock_series_uid],
        'frameNumber': [1],
        'data': [{'foreground': [[[100, 100], [200, 100], [200, 200], [100, 200]]]}]
    })
    
    # Create a mock video file
    video_path = os.path.join(test_output_dir, "test_video.mp4")
    if not os.path.exists(video_path):
        import cv2
        out = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v'), 30, (640, 480))
        for _ in range(10):  # 10 frames
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            out.write(frame)
        out.release()
    
    # Test create_ground_truth_dataset
    try:
        result = create_ground_truth_dataset(
            video_paths=[video_path],
            study_series_pairs=[(mock_study_uid, mock_series_uid)],
            flow_processor=None,  # Mock flow processor
            output_dir=test_output_dir,
            mdai_client=None,  # Mock client
            project_id="test_project",
            dataset_id="test_dataset",
            ground_truth_label_id="test_label",
            matched_annotations=matched_annotations,
            free_fluid_annotations=free_fluid_annotations,
            upload=False
        )
        print("\nGround truth dataset creation test completed")
        
    except Exception as e:
        print(f"Error during ground truth dataset creation: {str(e)}")
        raise

if __name__ == "__main__":
    test_ground_truth_utils() 