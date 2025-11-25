#!/usr/bin/env python3
import os
import numpy as np
import cv2
from src.utils import process_video_with_multi_frame_tracking_enhanced, filter_mask, calculate_mask_stats, create_overlay

class MockFlowProcessor:
    """Mock flow processor for testing"""
    def track_mask(self, mask, frame_step=1):
        # Just return the mask slightly modified for testing
        kernel = np.ones((3,3), np.uint8)
        return cv2.dilate(mask, kernel, iterations=1)

def test_shared_utils():
    """Test the shared utilities with mock data"""
    print("Testing shared utilities...")
    
    # Create test directory
    test_output_dir = "src/output/test_shared_utils"
    os.makedirs(test_output_dir, exist_ok=True)
    
    # Create a test video file
    video_path = os.path.join(test_output_dir, "test_video.mp4")
    if not os.path.exists(video_path):
        # Create a blank video file for testing
        out = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v'), 30, (640, 480))
        for _ in range(10):  # 10 frames
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            out.write(frame)
        out.release()
    
    # Create a test mask that meets filtering criteria
    initial_mask = np.zeros((480, 640), dtype=np.uint8)
    # Draw a solid rectangle that meets size and shape requirements
    cv2.rectangle(initial_mask, (100, 100), (200, 200), 1, -1)  # 100x100 square
    
    # Test filter_mask
    print("\nTesting filter_mask...")
    filtered_mask = filter_mask(
        initial_mask,
        threshold=0.5,
        min_area=1000,  # The square is 10000 pixels
        max_area=20000,
        min_height=50,   # The square is 100 pixels high
        min_width=50,    # The square is 100 pixels wide
        max_height=150,
        max_width=150,
        min_solidity=0.5,  # Square has perfect solidity
        min_extent=0.3,    # Square has perfect extent
        min_eccentricity=0.1,  # Square has balanced eccentricity
        max_eccentricity=1.0   # Allow perfect squares (eccentricity = 1.0)
    )
    print(f"Original mask sum: {np.sum(initial_mask)}")
    print(f"Filtered mask sum: {np.sum(filtered_mask)}")
    assert np.sum(filtered_mask) > 0, "Filtered mask should not be empty"
    
    # Test calculate_mask_stats
    print("\nTesting calculate_mask_stats...")
    stats = calculate_mask_stats(filtered_mask)
    print("Mask statistics:", stats)
    assert 'num_regions' in stats, "Stats should include num_regions"
    assert 'total_area' in stats, "Stats should include total_area"
    assert stats['num_regions'] > 0, "Should have at least one region"
    assert stats['total_area'] > 0, "Should have non-zero area"
    
    # Test create_overlay
    print("\nTesting create_overlay...")
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    overlay = create_overlay(frame, filtered_mask)
    assert overlay.shape == (480, 640, 3), "Overlay should have same dimensions as frame"
    
    # Test process_video_with_multi_frame_tracking_enhanced
    print("\nTesting video processing...")
    try:
        frames, masks, overlays, debug_info = process_video_with_multi_frame_tracking_enhanced(
            video_path=video_path,
            initial_mask=initial_mask,
            initial_frame_number=0,
            flow_processor=MockFlowProcessor(),
            output_dir=test_output_dir,
            save_frames=True,
            save_masks=True,
            save_overlays=True,
            save_debug_info=True
        )
        
        print("Video processing results:")
        print(f"Number of frames processed: {len(frames)}")
        print(f"Number of masks generated: {len(masks)}")
        print(f"Number of overlays created: {len(overlays)}")
        print(f"Debug info keys: {list(debug_info.keys())}")
        
        assert len(frames) > 0, "Should have processed frames"
        assert len(masks) > 0, "Should have generated masks"
        assert len(overlays) > 0, "Should have created overlays"
        
        print("\nAll shared utilities tests passed!")
        
    except Exception as e:
        print(f"Error during video processing: {str(e)}")
        raise

if __name__ == "__main__":
    test_shared_utils() 