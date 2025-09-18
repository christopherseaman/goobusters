#!/usr/bin/env python3
import numpy as np
import os
from src.utils import visualize_comparison

def test_visualization_utils():
    """Test the visualization utilities with simple test data"""
    print("Testing visualization utilities...")
    
    # Create test directory
    test_output_dir = "src/output/test_visualization"
    os.makedirs(test_output_dir, exist_ok=True)
    
    # Create a simple test video file
    video_path = os.path.join(test_output_dir, "test_video.mp4")
    if not os.path.exists(video_path):
        # Create a blank video file for testing
        import cv2
        out = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v'), 30, (640, 480))
        for _ in range(10):  # 10 frames
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            out.write(frame)
        out.release()
    
    # Create test masks
    algorithm_masks = {}
    ground_truth_masks = {}
    
    for i in range(10):  # 10 frames
        # Create random masks
        algo_mask = np.zeros((480, 640), dtype=np.uint8)
        gt_mask = np.zeros((480, 640), dtype=np.uint8)
        
        # Add some shapes to the masks
        cv2.circle(algo_mask, (320, 240), 50, 1, -1)  # Circle in the middle
        cv2.rectangle(gt_mask, (300, 220), (340, 260), 1, -1)  # Rectangle in the middle
        
        algorithm_masks[i] = algo_mask
        ground_truth_masks[i] = gt_mask
    
    # Test visualization
    output_path = os.path.join(test_output_dir, "test_comparison.mp4")
    try:
        result_path = visualize_comparison(
            video_path=video_path,
            algorithm_masks=algorithm_masks,
            ground_truth_masks=ground_truth_masks,
            output_path=output_path
        )
        
        assert os.path.exists(output_path), "Output video file should exist"
        print(f"\nVisualization test passed! Output saved to: {output_path}")
        
    except Exception as e:
        print(f"Error during visualization: {str(e)}")
        raise

if __name__ == "__main__":
    test_visualization_utils() 