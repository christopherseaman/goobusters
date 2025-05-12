import sys
import os
import logging
import json
import shutil
import traceback
from pathlib import Path
from datetime import datetime
import time
from dotenv import load_dotenv
import mdai
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from pandas import json_normalize
import argparse

from multi_frame_tracking.opticalflowprocessor import OpticalFlowProcessor
from multi_frame_tracking.multi_frame_tracker import MultiFrameTracker
from multi_frame_tracking.utils import track_frames, polygons_to_mask, visualize_flow, debug_visualize, verify_directory, find_exam_number, convert_numpy_to_python
from evaluation_utils import evaluate_with_iou, calculate_iou, calculate_dice
from visualisation_utils import visualize_comparison
from ground_truth_utils import get_annotations_for_study_series, create_ground_truth_dataset,create_feedback_loop_report

# Initialize module-level variables
MULTI_FRAME_AVAILABLE = True  
TRACKING_MODE = 'multi'  
DEBUG_MODE = False 

#.env path: 
dotenv_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '.env')
load_dotenv(dotenv_path)

# Create a dedicated debug log file
debug_log_path = f"debug_no_fluid_{int(time.time())}.log"
debug_log = open(debug_log_path, "w")

def debug_print(message):
    """Write debug messages to a separate file"""
    debug_log.write(f"{message}\n")
    debug_log.flush()  # Ensure it's written immediately
    
debug_print(f"=== DEBUG LOG STARTED AT {time.ctime()} ===")


# Enable debug mode
DEBUG_MODE = True
DEBUG_SAMPLE_SIZE = 2
DEBUG_ISSUE_TYPES = ["multiple_distinct"]

# Set debugging options for tracking
TARGET_FRAMES = []  # Empty list by default
VERBOSE_DEBUGGING = False

# Set tracking mode: 'single', 'multi', or 'both'
TRACKING_MODE = 'multi'  # this can be changed to switch between tracking modes

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(PROJECT_DIR, "output")
MULTI_FRAME_DIR = os.path.join(OUTPUT_DIR, "multi_frame_output")  # Subfolder for multi-frame outputs

FLOW_METHOD = ['dis']
MASK_MIN_SIZE = 100   
INTENSITY_THRESHOLD = 30

#logging to capture all console output
def setup_logging(output_dir):
    """Setup logging to capture all console output to a file"""
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(output_dir, f'tracking_log_{timestamp}.log')
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)

    # Clear existing handlers to prevent duplicate logs
    if root_logger.hasHandlers():
        root_logger.handlers.clear()
    
    # File handler
    file_handler = logging.FileHandler(log_file, mode='w')
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_formatter)
    root_logger.addHandler(file_handler)
    
    # Console handler (to keep console output)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.DEBUG)
    console_formatter = logging.Formatter('%(message)s')
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)
    
    class PrintToLogger:
        def write(self, message):
            if message.strip():  # Only log non-empty messages
                root_logger.info(message.strip())
        def flush(self):
            pass
    
    logging.info(f"Logging started, output to: {log_file}")
    return log_file


def polygons_to_mask(polygons, frame_height, frame_width):
    """Convert polygon points to a binary mask"""
    mask = np.zeros((frame_height, frame_width), dtype=np.uint8)
    for polygon in polygons:
        clipped_polygon = [
            [max(0, min(point[0], frame_width - 1)), max(0, min(point[1], frame_height - 1))]
            for point in polygon
        ]
        points = np.array(clipped_polygon, dtype=np.int32)
        cv2.fillPoly(mask, [points], 1)
    return mask

def print_mask_stats(mask, frame_num):
    """Print coverage statistics for a mask"""
    coverage = np.mean(mask) * 100
    print(f"Frame {frame_num} - Mask coverage: {coverage:.2f}%")

def visualize_flow(frame, flow, skip=8):
    """
    Visualize optical flow for debugging
    
    Args:
        frame: Input frame
        flow: Flow field (x and y displacements)
        skip: Spacing between displayed vectors
        
    Returns:
        Visualization image
    """
    h, w = frame.shape[:2]
    
    # Create empty visualization image
    vis = np.zeros((h * 2, w, 3), dtype=np.uint8)
    
    # Copy original frame to top half
    if len(frame.shape) == 2:  # Convert grayscale to color if needed
        vis[:h, :] = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    else:
        vis[:h, :] = frame.copy()
    
    # Create flow visualization
    # Calculate flow magnitude and angle
    flow_x = flow[..., 0]
    flow_y = flow[..., 1]
    magnitude, angle = cv2.cartToPolar(flow_x, flow_y)
    
    # Create HSV image for flow visualization
    hsv = np.zeros((h, w, 3), dtype=np.uint8)
    hsv[..., 0] = angle * 180 / np.pi / 2  # Hue: direction
    hsv[..., 1] = 255                       # Saturation: max
    hsv[..., 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)  # Value: magnitude
    
    # Convert HSV to BGR
    flow_vis = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    
    # Copy flow visualization to bottom half
    vis[h:, :] = flow_vis
    
    # Draw flow vectors on top half
    for y in range(0, h, skip):
        for x in range(0, w, skip):
            fx, fy = flow[y, x]
            
            # Only draw significant flow
            if np.sqrt(fx*fx + fy*fy) > 1:
                cv2.arrowedLine(vis[:h], (x, y), (int(x+fx), int(y+fy)), (0, 255, 0), 1, tipLength=0.3)
    
    # Add labels
    cv2.putText(vis, "Original frame", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
    cv2.putText(vis, "Flow visualization", (10, h+20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
    
    return vis

def debug_visualize(frame, initial_mask, flow_mask, adjusted_mask, final_mask, frame_number, flow=None):
    """
    Enhanced debug visualisation with improved visualisation to match numerical data.
    """
    h, w = frame.shape[:2]
    
    # Create a larger grid for visualisation including flow
    if flow is not None:
        grid = np.zeros((h * 3, w * 3, 3), dtype=np.uint8)  # 3x3 grid
    else:
        grid = np.zeros((h * 2, w * 3, 3), dtype=np.uint8)  # 2x3 grid (original size)
    
    # Convert masks to binary for contour detection and area calculation
    binary_initial = (initial_mask > 0.5).astype(np.uint8) if initial_mask is not None else None
    binary_flow = (flow_mask > 0.5).astype(np.uint8) if flow_mask is not None else None
    binary_adjusted = (adjusted_mask > 0.5).astype(np.uint8) if adjusted_mask is not None else None
    binary_final = (final_mask > 0.5).astype(np.uint8) if final_mask is not None else None
    
    # Original frame (Top Left)
    grid[:h, :w] = frame
    cv2.putText(grid, "Original Frame", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    # Initial Mask (Top Middle)
    if initial_mask is not None:
        # Create visualization using thresholding
        initial_viz = frame.copy()
        
        # Only color pixels above threshold
        initial_viz[initial_mask > 0.5] = initial_viz[initial_mask > 0.5] * 0.7 + np.array([0, 0, 255], dtype=np.uint8) * 0.3
        
        # Add contours to initial mask
        if binary_initial is not None:
            contours, _ = cv2.findContours(binary_initial, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(initial_viz, contours, -1, (0, 255, 255), 1)  # Yellow contour
            
            # Add area metrics
            initial_area = np.sum(binary_initial)
            cv2.putText(initial_viz, f"Area: {initial_area}", (10, h-20), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
            cv2.putText(initial_viz, f"Sum: {np.sum(initial_mask):.1f}", (10, h-50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
            
        grid[:h, w:w*2] = initial_viz
    cv2.putText(grid, "Initial Mask", (w + 10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    # Flow Mask (Top Right)
    if flow_mask is not None:
        flow_viz = frame.copy()
        
        # Only color pixels above threshold
        flow_viz[flow_mask > 0.5] = flow_viz[flow_mask > 0.5] * 0.7 + np.array([255, 0, 0], dtype=np.uint8) * 0.3
        
        # Add contours to flow mask
        if binary_flow is not None:
            contours, _ = cv2.findContours(binary_flow, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(flow_viz, contours, -1, (0, 255, 255), 1)  # Yellow contour
            
            # Add area metrics
            flow_area = np.sum(binary_flow)
            cv2.putText(flow_viz, f"Area: {flow_area}", (10, h-20), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
            cv2.putText(flow_viz, f"Sum: {np.sum(flow_mask):.1f}", (10, h-50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
            
            # Add IoU if both masks exist
            if binary_initial is not None:
                intersection = np.sum(np.logical_and(binary_initial, binary_flow))
                union = np.sum(np.logical_or(binary_initial, binary_flow))
                iou = intersection / union if union > 0 else 0
                cv2.putText(flow_viz, f"IoU: {iou:.4f}", (10, h-80), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
                
                # Add area ratio
                area_ratio = flow_area / initial_area if initial_area > 0 else 0
                cv2.putText(flow_viz, f"Ratio: {area_ratio:.4f}", (10, h-110), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
            
        grid[:h, w*2:w*3] = flow_viz
    cv2.putText(grid, "Flow Mask", (w*2 + 10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    # Create Difference Visualisation (Middle Left)
    
    if initial_mask is not None and flow_mask is not None:
        diff_viz = frame.copy()
        
        # Create a mask that shows where the masks differ
        diff_mask = np.zeros_like(binary_initial)
        diff_mask[(binary_initial > 0) & (binary_flow == 0)] = 1  # Initial only - show in red
        diff_mask[(binary_initial == 0) & (binary_flow > 0)] = 2  # Flow only - show in blue
        
        # Apply the difference visualization
        diff_viz[diff_mask == 1] = diff_viz[diff_mask == 1] * 0.7 + np.array([0, 0, 255], dtype=np.uint8) * 0.3  # Red
        diff_viz[diff_mask == 2] = diff_viz[diff_mask == 2] * 0.7 + np.array([255, 0, 0], dtype=np.uint8) * 0.3  # Blue
        
        # Add metrics
        diff_count = np.sum(diff_mask > 0)
        cv2.putText(diff_viz, f"Diff pixels: {diff_count}", (10, h-20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
        cv2.putText(diff_viz, f"Red: Initial only", (10, h-50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 1)
        cv2.putText(diff_viz, f"Blue: Flow only", (10, h-80), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 1)
        
        grid[h:h*2, :w] = diff_viz
    cv2.putText(grid, "Mask Difference", (10, h + 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    # Adjusted Mask (Middle Middle)
    if adjusted_mask is not None:
        adjusted_viz = frame.copy()
        
        # Only colour pixels above threshold (0.5)
        adjusted_viz[adjusted_mask > 0.5] = adjusted_viz[adjusted_mask > 0.5] * 0.7 + np.array([0, 0, 255], dtype=np.uint8) * 0.3
        
        # Add contours
        if binary_adjusted is not None:
            contours, _ = cv2.findContours(binary_adjusted, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(adjusted_viz, contours, -1, (0, 255, 255), 1)  # Yellow contour
        
        grid[h:h*2, w:w*2] = adjusted_viz
    cv2.putText(grid, "Adjusted Mask", (w + 10, h + 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    # Final Mask (Middle Right)
    if final_mask is not None:
        final_viz = frame.copy()
        
        # Only colour pixels above threshold
        final_viz[final_mask > 0.5] = final_viz[final_mask > 0.5] * 0.7 + np.array([0, 255, 0], dtype=np.uint8) * 0.3
        
        # Add contours to final mask
        if binary_final is not None:
            contours, _ = cv2.findContours(binary_final, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(final_viz, contours, -1, (0, 255, 255), 1)  # Yellow contour
            
            # Add area metrics
            final_area = np.sum(binary_final)
            cv2.putText(final_viz, f"Area: {final_area}", (10, h-20), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
            cv2.putText(final_viz, f"Sum: {np.sum(final_mask):.1f}", (10, h-50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
            
            # Add comparison to initial mask if it exists
            if binary_initial is not None:
                initial_area = np.sum(binary_initial)
                area_ratio = final_area / initial_area if initial_area > 0 else 0
                cv2.putText(final_viz, f"Area ratio: {area_ratio:.4f}", (10, h-80), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
            
        grid[h:h*2, w*2:w*3] = final_viz
    cv2.putText(grid, "Final Mask", (w*2 + 10, h + 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    # Add flow visualization (Bottom row) if flow is provided
    if flow is not None:
        try:
            # Flow Vectors (Bottom Left)
            flow_vis = visualize_flow(frame, flow, skip=8)
            
            # If flow_vis has expected two-part format (vectors+heatmap)
            if flow_vis.shape[0] > h:
                # Top part: Flow Vectors
                grid[h*2:h*3, :w] = flow_vis[:h]
                cv2.putText(grid, "Flow Vectors", (10, h*2 + 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                
                # Bottom part: Flow Heatmap (will be cropped if too large)
                heatmap_h = min(h, flow_vis.shape[0] - h)
                grid[h*2:h*2+heatmap_h, w:w*2] = flow_vis[h:h+heatmap_h]
                cv2.putText(grid, "Flow Heatmap", (w + 10, h*2 + 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            else:
                # If visualization format is different, just use what we have
                grid[h*2:h*3, :w] = flow_vis
                cv2.putText(grid, "Flow Visualization", (10, h*2 + 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            # Mask with Flow Vectors (Bottom Right)
            mask_with_vectors = frame.copy()
            # Add mask overlay
            mask_with_vectors[final_mask > 0.5] = mask_with_vectors[final_mask > 0.5] * 0.7 + np.array([0, 255, 0], dtype=np.uint8) * 0.3
            
            # Draw vectors only in mask area
            if flow is not None and final_mask is not None:
                # Calculate magnitude for thresholding
                mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
                mag_norm = cv2.normalize(mag, None, 0, 1, cv2.NORM_MINMAX)
                
                # Draw arrows in mask region
                for y in range(0, h, 8):  # Smaller skip for more arrows
                    for x in range(0, w, 8):
                        if final_mask[y, x] > 0.5 and mag_norm[y, x] > 0.05:  # Only in mask and with sufficient magnitude
                            fx = flow[y, x, 0]
                            fy = flow[y, x, 1]
                            # Use yellow for better visibility on green mask
                            cv2.arrowedLine(mask_with_vectors, (x, y), 
                                         (int(x + fx), int(y + fy)),
                                         (0, 255, 255), 2, tipLength=0.3)
            
            grid[h*2:h*3, w*2:w*3] = mask_with_vectors
            cv2.putText(grid, "Mask + Flow Vectors", (w*2 + 10, h*2 + 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                        
        except Exception as e:
            print(f"Error creating flow visualization: {str(e)}")
            
    # Add frame number and timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cv2.putText(grid, f"Frame: {frame_number} | {timestamp}", 
                (10, grid.shape[0] - 10), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    
    return grid


def create_difference_map(frame1, frame2, flow=None):
    """
    Creates a difference map between two frames and overlays flow vectors
    
    Args:
        frame1: First frame (numpy array)
        frame2: Second frame (numpy array)
        flow: Optional flow field between the frames (numpy array)
        
    Returns:
        Visualization showing frame differences and enhanced flow vectors
    """
    # Convert frames to grayscale if they're not already
    if len(frame1.shape) == 3:
        gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    else:
        gray1 = frame1
        gray2 = frame2
    
    # Calculate absolute difference between frames
    diff = cv2.absdiff(gray1, gray2)
    
    # Enhance difference for better visibility
    diff_enhanced = cv2.normalize(diff, None, 0, 255, cv2.NORM_MINMAX)
    
    # Apply color map for better visualization
    diff_color = cv2.applyColorMap(diff_enhanced, cv2.COLORMAP_JET)
    
    # If flow is provided, overlay flow vectors on areas with significant differences
    if flow is not None:
        # Create a mask of areas with significant differences
        threshold = 8  # Lower threshold to show more arrows
        diff_mask = diff > threshold
        
        # Calculate flow magnitude
        flow_mag = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
        flow_mag_norm = cv2.normalize(flow_mag, None, 0, 1, cv2.NORM_MINMAX)
        
        # Create a slightly darkened version of the diff map to make arrows stand out more
        diff_color_darkened = diff_color.copy()
        diff_color_darkened = cv2.addWeighted(diff_color_darkened, 0.7, np.zeros_like(diff_color_darkened), 0.3, 0)
        
        # Draw flow vectors on areas with significant differences
        h, w = diff.shape
        skip = 12  # Use smaller skip to show more arrows
        
        for y in range(0, h, skip):
            for x in range(0, w, skip):
                if diff_mask[y, x] and flow_mag_norm[y, x] > 0.05:  # Lower threshold to show more arrows
                    # Amplify the vectors significantly
                    fx = flow[y, x, 0] * 3.0  # Much larger scale factor
                    fy = flow[y, x, 1] * 3.0
                    
                    # First draw a black outline/shadow
                    cv2.arrowedLine(
                        diff_color_darkened, 
                        (x, y), 
                        (int(x + fx), int(y + fy)),
                        (0, 0, 0),      # Black outline
                        5,              # Thicker line for outline
                        tipLength=0.5   # Larger arrow tip
                    )
                    
                    # Then draw a bright arrow on top
                    cv2.arrowedLine(
                        diff_color_darkened, 
                        (x, y), 
                        (int(x + fx), int(y + fy)),
                        (0, 255, 255),  # Bright yellow
                        3,              # Thicker line
                        tipLength=0.5   # Larger arrow tip
                    )
        
        
        diff_color = diff_color_darkened
    
    # Create a combined visualization
    result = np.zeros((frame1.shape[0] * 2, frame1.shape[1], 3), dtype=np.uint8)
    
    # Add original frames
    if len(frame1.shape) == 3:
        result[:frame1.shape[0], :] = frame1
    else:
        result[:frame1.shape[0], :] = cv2.cvtColor(frame1, cv2.COLOR_GRAY2BGR)
    
    # Add difference map
    result[frame1.shape[0]:, :] = diff_color
    
    # Add labels
    cv2.putText(result, "Original Frame", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(result, "Difference Map with Flow", (10, frame1.shape[0] + 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    return result

def calculate_tracking_metrics(single_predictions, multi_predictions, ground_truth=None):
    """
    Calculate various metrics to compare tracking approaches.
    
    Args:
        single_predictions: Dictionary of single-frame tracking predictions {frame_idx: mask}
        multi_predictions: Dictionary of multi-frame tracking predictions {frame_idx: mask}
        ground_truth: Optional dictionary of ground truth annotations {frame_idx: mask}
        
    Returns:
        Dictionary of metrics
    """
    metrics = {
        'single': {},
        'multi': {}
    }
    
    # Get all frames that have predictions from either method
    all_frames = sorted(set(list(single_predictions.keys()) + list(multi_predictions.keys())))
    
    # 1. Temporal consistency (frame-to-frame IoU)
    single_ious = []
    multi_ious = []
    
    for i in range(len(all_frames)-1):
        curr_frame = all_frames[i]
        next_frame = all_frames[i+1]
        
        # Single-frame IoU
        if curr_frame in single_predictions and next_frame in single_predictions:
            curr_mask = single_predictions[curr_frame]
            next_mask = single_predictions[next_frame]
            
            if isinstance(curr_mask, dict) and 'mask' in curr_mask:
                curr_mask = curr_mask['mask']
            if isinstance(next_mask, dict) and 'mask' in next_mask:
                next_mask = next_mask['mask']
                
            binary_curr = (curr_mask > 0.5).astype(np.uint8)
            binary_next = (next_mask > 0.5).astype(np.uint8)
            
            intersection = np.sum(np.logical_and(binary_curr, binary_next))
            union = np.sum(np.logical_or(binary_curr, binary_next))
            iou = intersection / union if union > 0 else 0
            single_ious.append(iou)
        
        # Multi-frame IoU
        if curr_frame in multi_predictions and next_frame in multi_predictions:
            curr_mask = multi_predictions[curr_frame]
            next_mask = multi_predictions[next_frame]
            
            if isinstance(curr_mask, dict) and 'mask' in curr_mask:
                curr_mask = curr_mask['mask']
            if isinstance(next_mask, dict) and 'mask' in next_mask:
                next_mask = next_mask['mask']
                
            binary_curr = (curr_mask > 0.5).astype(np.uint8)
            binary_next = (next_mask > 0.5).astype(np.uint8)
            
            intersection = np.sum(np.logical_and(binary_curr, binary_next))
            union = np.sum(np.logical_or(binary_curr, binary_next))
            iou = intersection / union if union > 0 else 0
            multi_ious.append(iou)
    
    metrics['single']['temporal_consistency'] = np.mean(single_ious) if single_ious else 0
    metrics['multi']['temporal_consistency'] = np.mean(multi_ious) if multi_ious else 0
    
    # 2. Number of prediction failures (frames where method produced no prediction)
    single_fail_count = sum(1 for frame in all_frames if frame not in single_predictions)
    multi_fail_count = sum(1 for frame in all_frames if frame not in multi_predictions)
    
    metrics['single']['failure_count'] = single_fail_count
    metrics['multi']['failure_count'] = multi_fail_count
    
    # 3. Ground truth IoU 
    if ground_truth:
        single_gt_ious = []
        multi_gt_ious = []
        
        for frame in all_frames:
            if frame in ground_truth:
                gt_mask = ground_truth[frame]
                if isinstance(gt_mask, dict) and 'mask' in gt_mask:
                    gt_mask = gt_mask['mask']
                binary_gt = (gt_mask > 0.5).astype(np.uint8)
                
                # Single-frame IoU with ground truth
                if frame in single_predictions:
                    mask = single_predictions[frame]
                    if isinstance(mask, dict) and 'mask' in mask:
                        mask = mask['mask']
                    binary_mask = (mask > 0.5).astype(np.uint8)
                    
                    intersection = np.sum(np.logical_and(binary_gt, binary_mask))
                    union = np.sum(np.logical_or(binary_gt, binary_mask))
                    iou = intersection / union if union > 0 else 0
                    single_gt_ious.append(iou)
                
                # Multi-frame IoU with ground truth
                if frame in multi_predictions:
                    mask = multi_predictions[frame]
                    if isinstance(mask, dict) and 'mask' in mask:
                        mask = mask['mask']
                    binary_mask = (mask > 0.5).astype(np.uint8)
                    
                    intersection = np.sum(np.logical_and(binary_gt, binary_mask))
                    union = np.sum(np.logical_or(binary_gt, binary_mask))
                    iou = intersection / union if union > 0 else 0
                    multi_gt_ious.append(iou)
        
        metrics['single']['gt_iou'] = np.mean(single_gt_ious) if single_gt_ious else 0
        metrics['multi']['gt_iou'] = np.mean(multi_gt_ious) if multi_gt_ious else 0
    
    # 4. Mask area stability (standard deviation of mask areas)
    single_areas = []
    multi_areas = []
    
    for frame in all_frames:
        # Single-frame mask area
        if frame in single_predictions:
            mask = single_predictions[frame]
            if isinstance(mask, dict) and 'mask' in mask:
                mask = mask['mask']
            binary_mask = (mask > 0.5).astype(np.uint8)
            single_areas.append(np.sum(binary_mask))
        
        # Multi-frame mask area
        if frame in multi_predictions:
            mask = multi_predictions[frame]
            if isinstance(mask, dict) and 'mask' in mask:
                mask = mask['mask']
            binary_mask = (mask > 0.5).astype(np.uint8)
            multi_areas.append(np.sum(binary_mask))
    
    # Calculate mean and std of areas
    single_mean_area = np.mean(single_areas) if single_areas else 0
    multi_mean_area = np.mean(multi_areas) if multi_areas else 0
    
    # Calculate normalised std dev (std / mean) to compare stability regardless of mask size
    metrics['single']['area_stability'] = (np.std(single_areas) / single_mean_area) if single_mean_area > 0 else 0
    metrics['multi']['area_stability'] = (np.std(multi_areas) / multi_mean_area) if multi_mean_area > 0 else 0
    
    # Lower value is better (less fluctuation)
    metrics['single']['area_stability'] = 1 - metrics['single']['area_stability']
    metrics['multi']['area_stability'] = 1 - metrics['multi']['area_stability']
    
    return metrics

def create_comparison_video(video_path, single_predictions, multi_predictions, ground_truth, output_path):
    """
    Create a side-by-side comparison video between single-frame and multi-frame tracking.
    
    Args:
        video_path: Path to the original video
        single_predictions: Dictionary of single-frame tracking predictions {frame_idx: mask}
        multi_predictions: Dictionary of multi-frame tracking predictions {frame_idx: mask}
        ground_truth: Dictionary of ground truth annotations {frame_idx: mask} (optional)
        output_path: Path to save the comparison video
    """
    # Open the original video
    cap = cv2.VideoCapture(video_path)
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Create a video writer for the comparison video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    
    # Width will be 3x the original width (or 2x if no ground truth)
    if ground_truth:
        comparison_width = width * 3
        out = cv2.VideoWriter(output_path, fourcc, fps, (comparison_width, height))
    else:
        comparison_width = width * 2
        out = cv2.VideoWriter(output_path, fourcc, fps, (comparison_width, height))
    
    # colour maps for different prediction types
    colors = {
        'single': (0, 0, 255),    # Red for single-frame
        'multi': (0, 255, 0),     # Green for multi-frame
        'ground_truth': (255, 0, 0)  # Blue for ground truth
    }
    
    # Process each frame
    for frame_idx in range(frame_count):
        ret, frame = cap.read()
        if not ret:
            break
        
        # Create the comparison frame
        if ground_truth:
            comparison = np.zeros((height, comparison_width, 3), dtype=np.uint8)
            
            # Original frame with single-frame overlay
            single_view = frame.copy()
            if frame_idx in single_predictions:
                mask = single_predictions[frame_idx]
                if isinstance(mask, dict) and 'mask' in mask:
                    mask = mask['mask']
                binary_mask = (mask > 0.5).astype(np.uint8)
                single_view[binary_mask > 0] = cv2.addWeighted(
                    single_view[binary_mask > 0], 0.7, 
                    np.full_like(single_view[binary_mask > 0], colors['single']), 0.3, 0
                )
            
            # Original frame with ground truth overlay
            gt_view = frame.copy()
            if frame_idx in ground_truth:
                mask = ground_truth[frame_idx]
                if isinstance(mask, dict) and 'mask' in mask:
                    mask = mask['mask']
                binary_mask = (mask > 0.5).astype(np.uint8)
                gt_view[binary_mask > 0] = cv2.addWeighted(
                    gt_view[binary_mask > 0], 0.7, 
                    np.full_like(gt_view[binary_mask > 0], colors['ground_truth']), 0.3, 0
                )
            
            # Original frame with multi-frame overlay
            multi_view = frame.copy()
            if frame_idx in multi_predictions:
                mask = multi_predictions[frame_idx]
                if isinstance(mask, dict) and 'mask' in mask:
                    mask = mask['mask']
                binary_mask = (mask > 0.5).astype(np.uint8)
                multi_view[binary_mask > 0] = cv2.addWeighted(
                    multi_view[binary_mask > 0], 0.7, 
                    np.full_like(multi_view[binary_mask > 0], colors['multi']), 0.3, 0
                )
            
            # Add labels
            cv2.putText(single_view, "Single-frame", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(gt_view, "Ground Truth", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(multi_view, "Multi-frame", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Add frame number
            cv2.putText(single_view, f"Frame: {frame_idx}", (10, height - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Combine the views
            comparison[:, :width] = single_view
            comparison[:, width:width*2] = gt_view
            comparison[:, width*2:] = multi_view
        else:
            # No ground truth, just single vs multi
            comparison = np.zeros((height, comparison_width, 3), dtype=np.uint8)
            
            # Original frame with single-frame overlay
            single_view = frame.copy()
            if frame_idx in single_predictions:
                mask = single_predictions[frame_idx]
                if isinstance(mask, dict) and 'mask' in mask:
                    mask = mask['mask']
                binary_mask = (mask > 0.5).astype(np.uint8)
                single_view[binary_mask > 0] = cv2.addWeighted(
                    single_view[binary_mask > 0], 0.7, 
                    np.full_like(single_view[binary_mask > 0], colors['single']), 0.3, 0
                )
            
            # Original frame with multi-frame overlay
            multi_view = frame.copy()
            if frame_idx in multi_predictions:
                mask = multi_predictions[frame_idx]
                if isinstance(mask, dict) and 'mask' in mask:
                    mask = mask['mask']
                binary_mask = (mask > 0.5).astype(np.uint8)
                multi_view[binary_mask > 0] = cv2.addWeighted(
                    multi_view[binary_mask > 0], 0.7, 
                    np.full_like(multi_view[binary_mask > 0], colors['multi']), 0.3, 0
                )
            
            # Add labels
            cv2.putText(single_view, "Single-frame", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(multi_view, "Multi-frame", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Add frame number
            cv2.putText(single_view, f"Frame: {frame_idx}", (10, height - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Combine the views
            comparison[:, :width] = single_view
            comparison[:, width:] = multi_view
        
        # Write the comparison frame
        out.write(comparison)
    
    # Release resources
    cap.release()
    out.release()
    
    print(f"Comparison video saved to: {output_path}")
    return output_path

def upload_masks_to_mdai(client, masks_data, project_id, dataset_id=None):
    """
    Upload multiple masks to MD.ai using batch processing with proper format.
    Deletes existing annotations first to avoid duplicates.
    
    Args:
        client: MD.ai client instance
        masks_data: List of dictionaries containing mask info
        project_id: MD.ai project ID
        dataset_id: MD.ai dataset ID (optional)
    """
    if dataset_id is None:
        dataset_id = DATASET_ID  # Use the global dataset ID if not provided
        
    try:
        print("\nPreparing annotations for MD.ai upload...")
        
        # Format annotations according to MD.ai schema
        annotations = []
        
        # Track which study/series pairs we need to clean up
        cleanup_pairs = set()
        
        for mask_info in masks_data:
            if mask_info is None:
                continue
                
            # Add to the cleanup set
            cleanup_pairs.add((mask_info['study_uid'], mask_info['series_uid']))
                
            annotation = {
                'labelId': LABEL_ID_FLUID_OF,
                'StudyInstanceUID': mask_info['study_uid'],
                'SeriesInstanceUID': mask_info['series_uid'],
                'frameNumber': int(mask_info['frame_number']),  # Ensure integer
                'data': mask_info['mask_data'],  # Use the converted mask data directly
                'groupId': LABEL_ID_MACHINE_GROUP
            }
            annotations.append(annotation)

        # Exit early if no valid annotations
        if not annotations:
            print("No valid annotations to upload")
            return []
    
            
        print(f"Uploading {len(annotations)} annotations to MD.ai...")
        
        # Try batch upload first using import_annotations
        try:
            failed_annotations = client.import_annotations(
                annotations=annotations,
                project_id=project_id,
                dataset_id=dataset_id
            )
            
            if failed_annotations:
                print(f"Some annotations failed to upload ({len(failed_annotations)} failures):")
                for failed in failed_annotations[:5]:  # Show first 5 failures
                    print(f"  Index {failed.get('index')}: {failed.get('reason')}")
                if len(failed_annotations) > 5:
                    print(f"  ... and {len(failed_annotations) - 5} more failures")
                    
                # Calculate successful uploads
                successful_count = len(annotations) - len(failed_annotations)
                print(f"Successfully uploaded {successful_count} out of {len(annotations)} annotations")
                
                # Return empty list as a placeholder for successful annotations
                # (MD.ai's import_annotations doesn't return the successful IDs)
                return [{'success': True} for _ in range(successful_count)]
            else:
                print(f"Successfully uploaded all {len(annotations)} annotations")
                return [{'success': True} for _ in range(len(annotations))]
        except Exception as e:
            print(f"Error with batch upload: {str(e)}")
            traceback.print_exc()
            print("Falling back to individual uploads...")

            # Fall back to individual uploads if batch fails
            successful_uploads = []
            for idx, annotation in enumerate(annotations):
                try:
                    print(f"\nUploading annotation {idx + 1}/{len(annotations)}")
                    
                    # Try alternative method with post_annotation if available
                    if hasattr(client, 'post_annotation'):
                        annotation_copy = annotation.copy()
                        annotation_copy['projectId'] = project_id
                        annotation_copy['datasetId'] = dataset_id
                        response = client.post_annotation(annotation_copy)
                    else:
                        # Try single-item import_annotations
                        response = not client.import_annotations(
                            annotations=[annotation],
                            project_id=project_id,
                            dataset_id=dataset_id
                        )
                    
                    if response:
                        successful_uploads.append(response if isinstance(response, dict) else {'success': True})
                        print(f"Successfully uploaded annotation {idx + 1}")
                    else:
                        print(f"Warning: Empty response for annotation {idx + 1}")
                except Exception as e:
                    print(f"Error uploading annotation {idx + 1}: {str(e)}")
                    continue

            print(f"\nUploaded {len(successful_uploads)} out of {len(annotations)} annotations individually")
            return successful_uploads

    except Exception as e:
        print(f"Error during upload process: {str(e)}")
        traceback.print_exc()
        return []

def prepare_mask_data(mask, frame_number, study_uid, series_uid):
    """
    Prepare mask data for MD.ai upload using the correct format for video frames
    """
    try:
        # Ensure the mask is binary (0 or 1 values)
        binary_mask = (mask > 0.5).astype(np.uint8)
        
        # Convert mask to MD.ai format using their utility function
        mask_data = mdai.common_utils.convert_mask_data(binary_mask)
        
        if not mask_data:
            print(f"Warning: No valid mask data for frame {frame_number}")
            return None

        return {
            'study_uid': study_uid,
            'series_uid': series_uid,
            'frame_number': int(frame_number),  # Ensure integer
            'mask_data': mask_data  # Store the converted mask data directly
        }

    except Exception as e:
        print(f"Error preparing mask data for frame {frame_number}: {str(e)}")
        traceback.print_exc()

def verify_uploads(client, responses):
    """
    Verify uploaded annotations exist in MD.ai
    """
    try:
        if not responses:
            print("No annotations to verify")
            return False

        success = True
        for response in responses:
            annotation_id = response.get('id')
            if annotation_id:
                try:
                    result = client.get_annotation(annotation_id)
                    if not result:
                        print(f"Could not verify annotation {annotation_id}")
                        success = False
                except Exception as e:
                    print(f"Error verifying annotation {annotation_id}: {str(e)}")
                    success = False

        return success

    except Exception as e:
        print(f"Error during verification: {str(e)}")
        return False
    

def visualize_flow(frame, flow, skip=8):
    """
    Visualize optical flow for debugging
    
    Args:
        frame: Input frame
        flow: Flow field (x and y displacements)
        skip: Spacing between displayed vectors
        
    Returns:
        Visualization image
    """
    h, w = frame.shape[:2]
    
    # Create empty visualization image
    vis = np.zeros((h * 2, w, 3), dtype=np.uint8)
    
    # Copy original frame to top half
    if len(frame.shape) == 2:  # Convert grayscale to color if needed
        vis[:h, :] = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    else:
        vis[:h, :] = frame.copy()
    
    # Create flow visualization
    # Calculate flow magnitude and angle
    flow_x = flow[..., 0]
    flow_y = flow[..., 1]
    magnitude, angle = cv2.cartToPolar(flow_x, flow_y)
    
    # Create HSV image for flow visualization
    hsv = np.zeros((h, w, 3), dtype=np.uint8)
    hsv[..., 0] = angle * 180 / np.pi / 2  # Hue: direction
    hsv[..., 1] = 255                       # Saturation: max
    hsv[..., 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)  # Value: magnitude
    
    # Convert HSV to BGR
    flow_vis = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    
    # Copy flow visualization to bottom half
    vis[h:, :] = flow_vis
    
    # Draw flow vectors on top half
    for y in range(0, h, skip):
        for x in range(0, w, skip):
            fx, fy = flow[y, x]
            
            # Only draw significant flow
            if np.sqrt(fx*fx + fy*fy) > 1:
                cv2.arrowedLine(vis[:h], (x, y), (int(x+fx), int(y+fy)), (0, 255, 0), 1, tipLength=0.3)
    
    # Add labels
    cv2.putText(vis, "Original frame", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
    cv2.putText(vis, "Flow visualization", (10, h+20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
    
    return vis

def diagnose_video_processing(video_path, output_video_path):
    """
    Diagnoses video processing by comparing input and output video properties.
    """
    print("\nDiagnosing video processing:")
    
    # Check input video
    in_cap = cv2.VideoCapture(video_path)
    in_frames = int(in_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    in_fps = in_cap.get(cv2.CAP_PROP_FPS)
    in_width = int(in_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    in_height = int(in_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    in_duration = in_frames / in_fps if in_fps > 0 else 0
    
    print(f"\nInput Video Properties:")
    print(f"Path: {video_path}")
    print(f"Frame Count: {in_frames}")
    print(f"FPS: {in_fps}")
    print(f"Resolution: {in_width}x{in_height}")
    print(f"Duration: {in_duration:.2f} seconds")
    
    # Check if output video exists
    if os.path.exists(output_video_path):
        out_cap = cv2.VideoCapture(output_video_path)
        out_frames = int(out_cap.get(cv2.CAP_PROP_FRAME_COUNT))
        out_fps = out_cap.get(cv2.CAP_PROP_FPS)
        out_duration = out_frames / out_fps if out_fps > 0 else 0
        
        print(f"\nOutput Video Properties:")
        print(f"Path: {output_video_path}")
        print(f"Frame Count: {out_frames}")
        print(f"FPS: {out_fps}")
        print(f"Duration: {out_duration:.2f} seconds")
        
        # Check for frame loss
        if out_frames < in_frames:
            print(f"\nWARNING: Frame loss detected!")
            print(f"Missing {in_frames - out_frames} frames")
        
        out_cap.release()
    else:
        print(f"\nWARNING: Output video not found at {output_video_path}")
    
    in_cap.release()
    return {
        'input_frames': in_frames,
        'input_fps': in_fps,
        'input_duration': in_duration,
        'output_frames': out_frames if 'out_frames' in locals() else 0,
        'output_fps': out_fps if 'out_fps' in locals() else 0,
        'output_duration': out_duration if 'out_duration' in locals() else 0
    }


def save_combined_video(video_path, output_video_path, initial_mask, frame_number, debug_dir, flow_processor, 
                       mdai_client=None, study_uid=None, series_uid=None,
                       overlay_color=(0, 255, 0),
                       overlay_alpha=0.3,
                       add_info=True,
                       upload_to_mdai=False,
                       batch_size=50):
    """
    Process video frames, save with mask overlay, and upload masks to MD.ai
    """
    # Keep diagnostics
    print("\nStarting video processing diagnostics...")
    diag_before = diagnose_video_processing(video_path, output_video_path)
    
    # Create fresh debug directory
    if os.path.exists(debug_dir):
        print(f"Removing existing debug directory: {debug_dir}")
        shutil.rmtree(debug_dir)  # This removes the entire directory and all its contents

    # Create a fresh debug directory
    os.makedirs(debug_dir, exist_ok=True)
    print(f"Created fresh debug directory: {debug_dir}")
    
    # Create flow visualization directory
    flow_vis_dir = os.path.join(debug_dir, 'flow_vis')
    os.makedirs(flow_vis_dir, exist_ok=True)
    
    # Initialize tracking vars for MD.ai uploads
    upload_stats = {
        'total_frames': 0,
        'processed_frames': 0,
        'uploaded_frames': 0,
        'failed_uploads': 0
    }
    
    try:
        # Setup directories and video capture
        mask_dir = os.path.join(os.path.dirname(output_video_path), "masks")
        os.makedirs(mask_dir, exist_ok=True)
        cap = cv2.VideoCapture(video_path)
        
        # Get video properties
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        print(f"\nProcessing settings:")
        print(f"Total frames to process: {total_frames}")
        print(f"Frame number to start from: {frame_number}")
        print(f"FPS setting: {fps}")

        print(f"\nCalling track_frames:")
        print(f"  start_frame: {frame_number}")  # Use frame_number instead of start_frame
        print(f"  end_frame: {0}")  # For backward tracking, end frame is 0
        print(f"  debug_dir: {debug_dir}")
        print(f"  forward: False")  # For backward tracking, forward is False
        
        # Process frames
        with tqdm(total=total_frames, desc="Processing video", unit="frame") as pbar:
            backward_frames = track_frames(cap, frame_number, 0, initial_mask, debug_dir, 
                                        forward=False, pbar=pbar, flow_processor=flow_processor)
            
            # Add debug prints before forward tracking
            print(f"\nCalling track_frames:")
            print(f"  start_frame: {frame_number}")  
            print(f"  end_frame: {total_frames - 1}")  
            print(f"  debug_dir: {debug_dir}")
            print(f"  forward: True")  

            forward_frames = track_frames(cap, frame_number, total_frames - 1, initial_mask, debug_dir, 
                                       forward=True, pbar=pbar, flow_processor=flow_processor)

        # Combine frames
        combined_frames = backward_frames[::-1] + forward_frames[1:]
        combined_frames_count = len(combined_frames)
        upload_stats['total_frames'] = combined_frames_count
        
        print(f"\nFrames collected:")
        print(f"Backward frames: {len(backward_frames)}")
        print(f"Forward frames: {len(forward_frames)}")
        print(f"Combined frames: {combined_frames_count}")

        # Setup video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

        # Process and save frames
        print("\nWriting video and preparing MD.ai annotations...")
        frames_written = 0
        
        # Write frames and prepare for MD.ai upload
        for frame_data in tqdm(combined_frames, desc="Saving frames", unit="frame"):
            try:
                # Unpack the frame data properly - respecting its structure
                if len(frame_data) == 6:  # New format with intermediate masks
                    frame_idx, frame, mask, flow, flow_mask, adjusted_mask = frame_data
                else:  # Handle older format if needed
                    frame_idx, frame, mask = frame_data[:3]
                    flow = None if len(frame_data) < 4 else frame_data[3]
                    flow_mask = mask.copy() if mask is not None else np.zeros_like(initial_mask)
                    adjusted_mask = mask.copy() if mask is not None else np.zeros_like(initial_mask)
                
                # Verify we have valid data
                if frame is None or mask is None:
                    print(f"Warning: Invalid frame data for frame {frame_idx}, skipping")
                    continue
                
                # Create overlay frame
                overlay_frame = frame.copy()
                mask_bool = mask > 0
                if np.any(mask_bool):
                    overlay_frame[mask_bool] = overlay_frame[mask_bool] * (1 - overlay_alpha) + \
                                          np.array(overlay_color, dtype=np.uint8) * overlay_alpha
                
                # Add frame information if requested
                if add_info:
                    cv2.putText(overlay_frame, f"Frame: {frame_idx}", 
                              (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                              1, (255, 255, 255), 2)
                    
                    timestamp = frame_idx / fps
                    cv2.putText(overlay_frame, f"Time: {timestamp:.2f}s", 
                              (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 
                              1, (255, 255, 255), 2)
                    
                    coverage = np.mean(mask) * 100
                    cv2.putText(overlay_frame, f"Coverage: {coverage:.1f}%", 
                              (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 
                              1, (255, 255, 255), 2)
                
                # Set mask type based on position relative to annotation frame
                if frame_idx < frame_number:
                    mask_type = "backward_predicted"
                elif frame_idx > frame_number:
                    mask_type = "forward_predicted"
                else:
                    mask_type = "annotation"
                    
                cv2.putText(overlay_frame, f"Type: {mask_type}", 
                          (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 
                          1, (255, 255, 255), 2)

                # Write frame to video
                out.write(overlay_frame)
                frames_written += 1

                # Save mask
                mask_path = os.path.join(mask_dir, f"mask_{frame_idx:04d}.png")
                cv2.imwrite(mask_path, (mask * 255).astype(np.uint8))

                # Create and save flow visualization if flow exists
                if flow is not None:
                    flow_vis = visualize_flow(frame, flow, skip=8)
                    flow_vis_path = os.path.join(flow_vis_dir, f'flow_vis_{frame_idx:04d}.png')
                    cv2.imwrite(flow_vis_path, flow_vis)

                # Save debug frame with proper intermediate masks
                debug_frame = debug_visualize(
                    frame=frame, 
                    initial_mask=initial_mask, 
                    flow_mask=flow_mask,
                    adjusted_mask=adjusted_mask, 
                    final_mask=mask, 
                    frame_number=frame_idx,
                    flow=flow
                )
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                debug_filename = f'debug_frame_{frame_idx:04d}_{timestamp}.png'
                cv2.imwrite(os.path.join(debug_dir, debug_filename), debug_frame)

            except Exception as e:
                print(f"Error processing frame {frame_idx}: {str(e)}")
                traceback.print_exc()  # Print full stack trace
                continue

        # Make sure to close the video writer properly
        out.release()
        print(f"\nFrames written to video: {frames_written}")
        
        # NEW TRACKING VALIDATION CODE - MOVED HERE BEFORE MD.AI UPLOAD
        print("\n==== STARTING TRACKING VALIDATION ====")
        print(f"Debug directory: {debug_dir}")
        print(f"Output video path: {output_video_path}")

        print("\nTracking validation:")
        if len(backward_frames) > 1:
            first_mask_size = np.sum(backward_frames[0][2])
            last_mask_size = np.sum(backward_frames[-1][2])
            print(f"Backward tracking: Mask changed from {last_mask_size} to {first_mask_size} pixels")

        if len(forward_frames) > 1:
            first_mask_size = np.sum(forward_frames[0][2])
            last_mask_size = np.sum(forward_frames[-1][2])
            print(f"Forward tracking: Mask changed from {first_mask_size} to {last_mask_size} pixels")

        # Create a tracking summary video
        tracking_folder = os.path.join(debug_dir, "tracking_sequence")
        os.makedirs(tracking_folder, exist_ok=True)

        for i, frame_data in enumerate(combined_frames):
            try:
                # Unpack the frame data safely
                if len(frame_data) >= 3:
                    frame_idx = frame_data[0]
                    frame = frame_data[1]
                    mask = frame_data[2]
                    
                    # Create simple overlay for tracking verification
                    if frame is not None and mask is not None:
                        overlay_frame = frame.copy()
                        mask_bool = mask > 0
                        if np.any(mask_bool):
                            overlay_frame[mask_bool] = overlay_frame[mask_bool] * (1 - overlay_alpha) + \
                                                      np.array(overlay_color, dtype=np.uint8) * overlay_alpha
                        
                        # Add frame info
                        cv2.putText(overlay_frame, f"Frame: {frame_idx}", (10, 30), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                        cv2.putText(overlay_frame, f"Coverage: {np.mean(mask)*100:.2f}%", (10, 60), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                        
                        # Save frame
                        cv2.imwrite(os.path.join(tracking_folder, f"tracking_{i:04d}.png"), overlay_frame)
            except Exception as e:
                print(f"Error creating tracking visualization for frame {i}: {str(e)}")
                continue

        # Create visualization video from flow visualizations
        flow_vis_video_path = os.path.join(os.path.dirname(output_video_path), "flow_visualization.mp4")
        
        # Check if we have flow visualizations
        if os.path.exists(flow_vis_dir) and len(os.listdir(flow_vis_dir)) > 0:
            create_debug_video(flow_vis_dir, flow_vis_video_path, fps=fps)
            print(f"Created flow visualization video: {flow_vis_video_path}")
        
        # Create video from tracking sequence
        tracking_video_path = os.path.join(os.path.dirname(output_video_path), "tracking_sequence.mp4")
        create_debug_video(tracking_folder, tracking_video_path, fps=fps)
        print(f"Created tracking verification video: {tracking_video_path}")
        
        # Create debug visualization video from all debug frames
        debug_video_path = os.path.join(os.path.dirname(output_video_path), "debug_visualization.mp4")
        print(f"Creating debug visualization from existing debug frames in: {debug_dir}")
        create_debug_video(debug_dir, debug_video_path, fps=fps)
        print(f"Created debug visualization video: {debug_video_path}")
        
        print(f"==== COMPLETED TRACKING VALIDATION ====")
        
        # Try the MD.ai upload after tracking validation is complete
        if mdai_client and study_uid and series_uid and upload_to_mdai:
            try:
                print("\n==== STARTING MD.AI UPLOAD ====")
                
                # Delete existing annotations first to avoid duplicates
                deleted_count = delete_existing_annotations(
                    client=mdai_client,
                    study_uid=study_uid,
                    series_uid=series_uid,
                    label_id=LABEL_ID_FLUID_OF,
                    group_id=LABEL_ID_MACHINE_GROUP
                )
                print(f"Deleted {deleted_count} existing annotations")
                
                print(f"Preparing mask data for upload to MD.ai...")
                
                # Prepare mask data for MD.ai upload
                masks_to_upload = []
                for frame_data in combined_frames:
                    # Unpack the frame data safely
                    if len(frame_data) >= 3:
                        frame_idx = frame_data[0]
                        mask = frame_data[2]
                        
                        # Only process mask if it has content
                        if mask is not None and np.sum(mask) > 0:
                            try:
                                # Create binary mask
                                binary_mask = (mask > 0.5).astype(np.uint8)
                                
                                # Convert to MD.ai format
                                mask_data = mdai.common_utils.convert_mask_data(binary_mask)
                                
                                if mask_data:  # Ensure we have valid mask data
                                    # Format annotation
                                    annotation = {
                                        'labelId': LABEL_ID_FLUID_OF,
                                        'StudyInstanceUID': study_uid,
                                        'SeriesInstanceUID': series_uid,
                                        'frameNumber': int(frame_idx),  # Ensure integer
                                        'data': mask_data,
                                        'groupId': LABEL_ID_MACHINE_GROUP
                                    }
                                    
                                    masks_to_upload.append(annotation)
                            except Exception as e:
                                print(f"Error preparing mask for frame {frame_idx}: {str(e)}")
                                continue
                        else:
                            print(f"Skipping empty mask for frame {frame_idx}")
                
                # Upload all frames to MD.ai
                if masks_to_upload:
                    print(f"Uploading {len(masks_to_upload)} masks to MD.ai...")
                    
                    # Try batch upload
                    try:
                        failed_annotations = mdai_client.import_annotations(
                            annotations=masks_to_upload,
                            project_id=PROJECT_ID,
                            dataset_id=DATASET_ID
                        )
                        
                        if failed_annotations:
                            print(f"Some annotations failed to upload ({len(failed_annotations)} failures)")
                            successful_count = len(masks_to_upload) - len(failed_annotations)
                        else:
                            print(f"Successfully uploaded all {len(masks_to_upload)} annotations")
                            successful_count = len(masks_to_upload)

                            try:
                                exam_number = find_exam_number(study_uid, annotations_json)
                                print(f"Annotations were uploaded to Exam #{exam_number}")
                            except Exception as e:
                                print(f"Could not determine exam number: {str(e)}")
                            
                        upload_stats['uploaded_frames'] = successful_count
                        upload_stats['failed_uploads'] = len(masks_to_upload) - successful_count
                        
                    except Exception as e:
                        print(f"Error in MD.ai upload: {str(e)}")
                        traceback.print_exc()
                else:
                    print("No valid masks to upload to MD.ai")
                    
                print("==== COMPLETED MD.AI UPLOAD ====")
                
            except Exception as e:
                print(f"Error in MD.ai upload: {str(e)}")
                traceback.print_exc()
                print("Continuing with tracking validation despite upload error")
        elif not upload_to_mdai:
            print("\nSkipping MD.ai upload as per configuration.")
            
        print("\nMD.ai Upload Statistics:")
        print(f"Total frames processed: {upload_stats['total_frames']}")
        print(f"Successfully uploaded: {upload_stats['uploaded_frames']}")
        print(f"Failed uploads: {upload_stats['failed_uploads']}")
        
        print("\nFinal video diagnostics...")
        diag_after = diagnose_video_processing(video_path, output_video_path)
        
        return {
            'frames_processed': combined_frames_count,
            'frames_written': frames_written,
            'input_diagnostics': diag_before,
            'output_diagnostics': diag_after,
            'mdai_upload_stats': upload_stats
        }

    except Exception as e:
        print(f"Error in save_combined_video: {str(e)}")
        traceback.print_exc()
        return None
    

def process_video_with_multi_frame_tracking_enhanced(video_path, annotations_df, study_uid, series_uid, 
                                         flow_processor, output_dir, annotations_json=None, mdai_client=None,
                                         label_id_fluid=None, label_id_machine=None,
                                         label_id_no_fluid=None, project_id=None, dataset_id=None,  
                                         upload_to_mdai=False, debug_mode=False):
    """
    Process a video using multi-frame tracking and optionally upload to MD.ai.
    
    Args:
        video_path: Path to the video file
        annotations_df: DataFrame containing annotations
        study_uid: Study Instance UID
        series_uid: Series Instance UID
        flow_processor: Optical flow processor
        output_dir: Directory to save outputs
        mdai_client: MD.ai client for uploads
        label_id_fluid: Label ID for fluid annotation
        label_id_machine: Label ID for machine group
        label_id_no_fluid: Label ID for no fluid annotations
        project_id: MD.ai project ID
        dataset_id: MD.ai dataset ID
        upload_to_mdai: Whether to upload to MD.ai
        debug_mode: Enable debug mode
        
    Returns:
        Dictionary with results
    """

    print("\n==== STARTING MULTI-FRAME TRACKING PROCESS ====")
    print(f"Video path: {video_path}")
    print(f"Annotations count: {len(annotations_df)}")
    print(f"Output directory: {output_dir}")
    print(f"Flow processor method: {flow_processor.method}")
    print(f"Debug mode: {debug_mode}")
    
    # Create a checkpoint file to verify this function was called
    checkpoint_file = os.path.join(output_dir, "multi_frame_tracking_started.txt")
    try:
        with open(checkpoint_file, "w") as f:
            f.write(f"Process started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Video: {video_path}\n")
            f.write(f"Study/Series: {study_uid}/{series_uid}\n")
    except Exception as e:
        print(f"Warning: Could not create checkpoint file: {str(e)}")

    try:
        exam_number = find_exam_number(study_uid, annotations_json)
        print(f"Processing Exam #{exam_number}")
        with open(checkpoint_file, "a") as f:
           f.write(f"Exam Number: {exam_number}\n")
    except Exception as e:
        print(f"Could not determine exam number: {e}")

    # Initialize multi-frame tracker
    tracker = MultiFrameTracker(flow_processor, output_dir, debug_mode=debug_mode)

    # Process annotations
    print("Calling tracker.process_annotations...")
    all_masks = tracker.process_annotations(annotations_df, video_path, study_uid, series_uid)
    print(f"Received {len(all_masks)} masks from process_annotations")
    
    # Update checkpoint
    try:
        with open(checkpoint_file, "a") as f:
            f.write(f"Process completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Generated {len(all_masks)} masks\n")
    except:
        pass
    
    # Upload to MD.ai if requested
    upload_results = None
    successful_uploads = 0
    annotations = []
    
    if upload_to_mdai and mdai_client and label_id_fluid and label_id_machine:
        print(f"\nPreparing annotations for MD.ai upload...")
        
        try:
            
            # Process each frame's mask
            fluid_count = 0
            clear_count = 0
            batch_size = 50
            current_batch = []
            
            for frame_idx, mask_info in all_masks.items():
                # Skip human annotations (already in MD.ai)
                if isinstance(mask_info, dict) and mask_info.get('is_annotation', False):
                    continue
                
                # Get the mask and type
                if isinstance(mask_info, dict):
                    mask = mask_info['mask']
                    mask_type = mask_info.get('type', '')
                else:
                    mask = mask_info
                    mask_type = 'unknown'
                
                # Handle clear frames differently
                is_clear = 'clear' in mask_type if mask_type else False
                
                # Create annotation
                try:
                    # For clear frames, create a special no fluid annotation
                    if is_clear and label_id_no_fluid:
                        annotation = {
                            'labelId': label_id_no_fluid,
                            'StudyInstanceUID': study_uid,
                            'SeriesInstanceUID': series_uid,
                            'frameNumber': int(frame_idx),
                            'groupId': label_id_machine
                        }
                        current_batch.append(annotation)
                        clear_count += 1
                    else:
                        # For fluid frames, check if mask has content
                        binary_mask = (mask > 0.5).astype(np.uint8)
                        if np.sum(binary_mask) > 0:
                            # Convert mask to MD.ai format
                            mask_data = mdai.common_utils.convert_mask_data(binary_mask)
                            if mask_data:
                                annotation = {
                                    'labelId': label_id_fluid,
                                    'StudyInstanceUID': study_uid,
                                    'SeriesInstanceUID': series_uid,
                                    'frameNumber': int(frame_idx),
                                    'data': mask_data,
                                    'groupId': label_id_machine
                                }
                                current_batch.append(annotation)
                                fluid_count += 1
                    
                    # Upload in batches
                    if len(current_batch) >= batch_size:
                        print(f"Uploading batch of {len(current_batch)} annotations...")
                        failures = mdai_client.import_annotations(
                            annotations=current_batch,
                            project_id=project_id,
                            dataset_id=dataset_id
                        )
                        successful_uploads += len(current_batch) - (len(failures) if failures else 0)
                        current_batch = []  # Reset batch
                    
                except Exception as e:
                    print(f"Error creating annotation for frame {frame_idx}: {str(e)}")
                    continue
            
            # Upload any remaining annotations
            if current_batch:
                print(f"Uploading final batch of {len(current_batch)} annotations...")
                failures = mdai_client.import_annotations(
                    annotations=current_batch,
                    project_id=project_id,
                    dataset_id=dataset_id
                )
                successful_uploads += len(current_batch) - (len(failures) if failures else 0)
            
            print(f"\nMD.ai Upload Summary:")
            print(f"Prepared {fluid_count} fluid annotations and {clear_count} clear annotations")
            print(f"Successfully uploaded {successful_uploads} annotations")
            
        except Exception as e:
            print(f"Error in MD.ai upload: {str(e)}")
            traceback.print_exc()

        try:
           exam_number = find_exam_number(study_uid, annotations_json)
           print(f"Annotations were uploaded to Exam #{exam_number}")
        except Exception as e:
           print(f"Could not determine exam number: {e}")
    
    # Count annotation types
    annotation_types = {}
    for frame, info in all_masks.items():
        if isinstance(info, dict) and 'type' in info:
            annotation_type = info['type']
            annotation_types[annotation_type] = annotation_types.get(annotation_type, 0) + 1
    
    print("==== MULTI-FRAME TRACKING PROCESS COMPLETED ====")

    # Return results with upload information
    return {
        'all_masks': all_masks,
        'annotated_frames': sum(1 for info in all_masks.values() if isinstance(info, dict) and info.get('is_annotation', False)),
        'predicted_frames': sum(1 for info in all_masks.values() if isinstance(info, dict) and not info.get('is_annotation', False)),
        'annotation_types': annotation_types,
        'output_video': os.path.join(output_dir, "multi_frame_tracking.mp4"),
        'md_ai_uploads': True if upload_to_mdai else False,
        'uploaded_count': successful_uploads,
        'exam_number': find_exam_number(study_uid, annotations_json)


    }

def process_videos_with_tracking():
    """
    Main processing loop for tracking free fluid in ultrasound videos.
    Modified to run ONLY multi-frame tracking for debugging.
    """
    # Create timestamp file in the root directory to ensure we can find it
    with open("multi_frame_test.txt", "w") as f:
        f.write(f"Test started at {datetime.now()}\n")
        f.write(f"TRACKING_MODE = {TRACKING_MODE}\n")
        f.write(f"MULTI_FRAME_AVAILABLE = {MULTI_FRAME_AVAILABLE}\n")
    
    print("\n==== MULTI-FRAME TRACKING TEST ====")
    print(f"TRACKING_MODE: {TRACKING_MODE}")
    print(f"MULTI_FRAME_AVAILABLE: {MULTI_FRAME_AVAILABLE}")
    
    # Determine which issue types to process based on debug mode
    issue_types_to_process = DEBUG_ISSUE_TYPES if DEBUG_MODE else list(LABEL_IDS.keys())
    print(f"Processing issue types: {issue_types_to_process}")
    
    # Create output directories if they don't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(MULTI_FRAME_DIR, exist_ok=True)
    
    # Initialize video counter
    videos_processed = 0
    
    # Initialize optical flow processor
    print("Initializing optical flow processor...")
    flow_processor = OpticalFlowProcessor(method=FLOW_METHOD[0])
    
    # Process each issue type
    for issue_type in issue_types_to_process:
        print(f"\nProcessing {issue_type} annotations...")
        
        # Filter annotations for this issue type
        type_annotations = matched_annotations[matched_annotations['issue_type'] == issue_type]
        
        # Limit processing to debug sample size if in debug mode
        if DEBUG_MODE:
            type_annotations = type_annotations.head(DEBUG_SAMPLE_SIZE)
        
        print(f"Found {len(type_annotations)} annotations for {issue_type}")
        
        # Process each annotation
        for idx, row in tqdm(type_annotations.iterrows(), total=len(type_annotations)):
            video_path = row['video_path']
            study_uid = row['StudyInstanceUID']
            series_uid = row['SeriesInstanceUID']
            
            # Find any frame annotation for this video
            video_annotations = free_fluid_annotations[
                (free_fluid_annotations['StudyInstanceUID'] == study_uid) &
                (free_fluid_annotations['SeriesInstanceUID'] == series_uid)
            ]

            if len(video_annotations) > 0:
                # Use the first available annotation
                row = video_annotations.iloc[0]
                frame_number = int(row['frameNumber'])
                
                # Get free fluid polygons from this specific annotation
                free_fluid_polygons = row['free_fluid_foreground']
                
                print(f"\nProcessing video {videos_processed + 1}/{len(type_annotations)}")
                print(f"Video: {video_path}")
                print(f"Frame number: {frame_number}")
            else:
                print("No annotations found for this video, skipping")
                continue
            
            # Continue only if we have valid polygons
            if not isinstance(free_fluid_polygons, list) or len(free_fluid_polygons) == 0:
                print("No valid polygons found, skipping")
                continue
            
            try:
                cap = cv2.VideoCapture(video_path)
                if not cap.isOpened():
                    print(f"Could not open video: {video_path}")
                    continue
                
                # Get video properties
                frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                
                print(f"Video dimensions: {frame_width}x{frame_height}")
                print(f"Total frames: {total_frames}")
                
                # Create initial mask from polygons
                initial_mask = polygons_to_mask(free_fluid_polygons, frame_height, frame_width)
                
                # Create output directory for this video
                multi_frame_output_dir = os.path.join(MULTI_FRAME_DIR, f"{study_uid}_{series_uid}")
                os.makedirs(multi_frame_output_dir, exist_ok=True)
                
                # Get all annotations for this video
                video_annotations = free_fluid_annotations[
                    (free_fluid_annotations['StudyInstanceUID'] == study_uid) &
                    (free_fluid_annotations['SeriesInstanceUID'] == series_uid)
                ].copy()
                
                print(f"Found {len(video_annotations)} annotations for this video")
                
                print("\n##############################################")
                print("# ABOUT TO CALL MULTI-FRAME TRACKING FUNCTION #")
                print("##############################################")
                print(f"Flow processor method: {flow_processor.method}")
                print(f"Study/Series: {study_uid}/{series_uid}")
                
                # Create detailed debug log
                with open(os.path.join(multi_frame_output_dir, "function_call_log.txt"), "w") as f:
                    f.write(f"About to call function at {datetime.now()}\n")
                    f.write(f"Video path: {video_path}\n")
                    f.write(f"Annotations count: {len(video_annotations)}\n")
                    f.write(f"Frame dimensions: {frame_width}x{frame_height}\n")
                    f.write(f"Initial mask sum: {np.sum(initial_mask)}\n")
                
                try:
                    # Call the multi-frame processing function
                  
                    
                    result = process_video_with_multi_frame_tracking_enhanced(
                        video_path=video_path,
                        annotations_df=video_annotations,
                        study_uid=study_uid,
                        series_uid=series_uid,
                        flow_processor=flow_processor,
                        output_dir=multi_frame_output_dir,
                        annotations_json=annotations_json,
                        mdai_client=mdai_client if 'mdai_client' in globals() else None,
                        label_id_fluid=LABEL_ID_FLUID_OF,
                        label_id_no_fluid=LABEL_ID_NO_FLUID,
                        label_id_machine=LABEL_ID_MACHINE_GROUP,
                        project_id=PROJECT_ID,
                        dataset_id=DATASET_ID,
                        upload_to_mdai=True,
                        debug_mode=True
                    )
                    
                    # Log successful completion
                    with open(os.path.join(multi_frame_output_dir, "function_call_log.txt"), "a") as f:
                        f.write(f"Function completed at {datetime.now()}\n")
                        f.write(f"Result type: {type(result)}\n")
                        if result:
                            f.write(f"Result keys: {result.keys()}\n")
                            f.write(f"Annotated frames: {result.get('annotated_frames', 0)}\n")
                            f.write(f"Predicted frames: {result.get('predicted_frames', 0)}\n")
                    
                    print("\nMulti-frame tracking completed successfully!")
                    print(f"Annotated frames: {result.get('annotated_frames', 0)}")
                    print(f"Predicted frames: {result.get('predicted_frames', 0)}")
                    
                    if result and 'md_ai_uploads' in result:
                        print(f"MD.ai upload status: {result['md_ai_uploads']}")
                        print(f"Total annotations uploaded: {result.get('uploaded_count', 0)}")
                    
                    # Verify output video exists
                    output_video = result.get('output_video', 'None')
                    if os.path.exists(output_video):
                        print(f"Confirmed: Output video created at {output_video} ({os.path.getsize(output_video)} bytes)")
                    else:
                        print(f"WARNING: Output video not found at expected path: {output_video}")
                    
                except Exception as e:
                    print(f"\n!!! ERROR IN MULTI-FRAME TRACKING FUNCTION CALL !!!")
                    print(f"Error type: {type(e).__name__}")
                    print(f"Error message: {str(e)}")
                    traceback.print_exc()
                    
                    # Log error details
                    with open(os.path.join(multi_frame_output_dir, "function_call_error.txt"), "w") as f:
                        f.write(f"Function call failed at {datetime.now()}\n")
                        f.write(f"Error type: {type(e).__name__}\n")
                        f.write(f"Error message: {str(e)}\n")
                        f.write(traceback.format_exc())
                
                # Increment counter regardless of success
                videos_processed += 1
                
                # Close video file
                cap.release()
                
            except Exception as e:
                print(f"Error processing video: {str(e)}")
                traceback.print_exc()
                continue
    
    print(f"\nProcessing completed. Total videos processed: {videos_processed}")


def get_annotations_for_study_series(mdai_client, project_id, dataset_id, study_uid, series_uid, label_id=None):
    """
    Get annotations for a specific study/series combination
    
    Args:
        mdai_client: MD.ai client instance
        project_id: MD.ai project ID
        dataset_id: MD.ai dataset ID
        study_uid: Study instance UID
        series_uid: Series instance UID
        label_id: Optional label ID to filter annotations
        
    Returns:
        List of annotations for the study/series
    """
    try:
        print(f"Fetching annotations for {study_uid}/{series_uid}")
        
        # First, get all annotations for the dataset
        result = mdai_client.project(
            project_id=project_id,
            dataset_id=dataset_id,
            path=DATA_DIR,
            annotations_only=True
        )
        
        # Find the latest annotations file
        annotation_files = [f for f in os.listdir(DATA_DIR) if f.startswith('mdai_ucsf_project') and f.endswith('.json')]
        
        if not annotation_files:
            print("No annotation files found")
            return []
        
        # Sort by modification time (newest first)
        annotation_files.sort(key=lambda f: os.path.getmtime(os.path.join(DATA_DIR, f)), reverse=True)
        annotations_file = os.path.join(DATA_DIR, annotation_files[0])
        
        # Load the JSON
        with open(annotations_file, 'r') as f:
            annotations_json = json.load(f)
        
        # Extract all annotations
        all_annotations = []
        for dataset in annotations_json.get('datasets', []):
            if 'annotations' in dataset:
                all_annotations.extend(dataset['annotations'])
        
        # Filter for the specific study/series
        filtered_annotations = [
            ann for ann in all_annotations
            if ann.get('StudyInstanceUID') == study_uid and ann.get('SeriesInstanceUID') == series_uid
        ]
        
        # Further filter by label_id if provided
        if label_id:
            filtered_annotations = [
                ann for ann in filtered_annotations
                if ann.get('labelId') == label_id
            ]
        
        print(f"Found {len(filtered_annotations)} annotations for {study_uid}/{series_uid}")
        return filtered_annotations
        
    except Exception as e:
        print(f"Error fetching annotations: {str(e)}")
        traceback.print_exc()
        return []


def evaluate_with_expert_feedback(video_paths, study_series_pairs, flow_processor, output_dir,
                                 mdai_client, project_id, dataset_id, ground_truth_label_id,
                                 algorithm_label_id):
    """
    Evaluates the algorithm against expert-refined ground truth annotations
    
    Args:
        video_paths: List of video paths to process
        study_series_pairs: List of (study_uid, series_uid) pairs
        flow_processor: Optical flow processor to use
        output_dir: Directory to save outputs
        mdai_client: MD.ai client
        project_id: MD.ai project ID
        dataset_id: MD.ai dataset ID
        ground_truth_label_id: Label ID for ground truth annotations
        algorithm_label_id: Label ID for algorithm annotations
        
    Returns:
        Dictionary with evaluation results
    """
    evaluation_results = {}
    
    for i, (video_path, (study_uid, series_uid)) in enumerate(zip(video_paths, study_series_pairs)):
        print(f"\nEvaluating video {i+1}/{len(video_paths)}")
        print(f"Study UID: {study_uid}")
        print(f"Series UID: {series_uid}")
        
        # Create output directory for this video
        video_output_dir = os.path.join(output_dir, f"{study_uid}_{series_uid}_evaluation")
        os.makedirs(video_output_dir, exist_ok=True)
        
        try:
            # Download expert-refined ground truth annotations
            ground_truth_annotations = get_annotations_for_study_series(
                mdai_client, project_id, dataset_id, study_uid, series_uid,
                label_id=ground_truth_label_id
            )
            
            if not ground_truth_annotations:
                print(f"No ground truth annotations found for {study_uid}/{series_uid}")
                evaluation_results[f"{study_uid}_{series_uid}"] = {
                    'error': 'No ground truth annotations found'
                }
                continue
            
            # Convert to mask dictionary for evaluation
            ground_truth_masks = {}
            for annotation in ground_truth_annotations:
                frame_number = annotation.get('frameNumber')
                if frame_number is not None and 'data' in annotation:
                    # Convert MD.ai format to binary mask
                    mask = mdai.common_utils.convert_to_mask(
                        annotation['data'],
                        int(annotation.get('height', 0)),
                        int(annotation.get('width', 0))
                    )
                    ground_truth_masks[frame_number] = mask
            
            print(f"Found {len(ground_truth_masks)} ground truth masks")
            
            # Run the algorithm to generate new predictions
            annotations_df = pd.DataFrame(ground_truth_annotations)
            
            # Initialize the multi-frame tracker
            tracker = MultiFrameTracker(flow_processor, video_output_dir, debug_mode=True)
            
            # Process using the existing workflow
            algorithm_masks = tracker.process_annotations(annotations_df, video_path, study_uid, series_uid)
            
            print(f"Generated {len(algorithm_masks)} algorithm masks")
            
            # Evaluate algorithm against ground truth
            metrics = evaluate_with_iou(algorithm_masks, ground_truth_masks)
            
            # Create visualization of results
            visualization_path = visualize_comparison(
                video_path, algorithm_masks, ground_truth_masks, 
                os.path.join(video_output_dir, "comparison.mp4")
            )
            
            # Store results
            evaluation_results[f"{study_uid}_{series_uid}"] = {
                'metrics': metrics,
                'ground_truth_count': len(ground_truth_masks),
                'algorithm_mask_count': len(algorithm_masks),
                'visualization_path': visualization_path
            }
            
            # Generate a report for this video
            report_path = os.path.join(video_output_dir, "evaluation_report.md")
            with open(report_path, 'w') as f:
                f.write(f"# Evaluation Report: {study_uid}/{series_uid}\n\n")
                f.write(f"- Ground Truth Masks: {len(ground_truth_masks)}\n")
                f.write(f"- Algorithm Masks: {len(algorithm_masks)}\n\n")
                
                f.write("## Evaluation Metrics\n\n")
                f.write(f"- Mean IoU: {metrics['mean_iou']:.4f}\n")
                f.write(f"- Median IoU: {metrics['median_iou']:.4f}\n")
                f.write(f"- Mean Dice: {metrics['mean_dice']:.4f}\n")
                f.write(f"- % Frames IoU > 0.7: {metrics['iou_over_0.7']*100:.1f}%\n\n")
                
                f.write("## Frame-by-Frame IoU\n\n")
                f.write("| Frame | IoU | Dice |\n")
                f.write("|-------|-----|------|\n")
                
                # Add a few example frames
                for frame in sorted(list(set(ground_truth_masks.keys()) & set(algorithm_masks.keys())))[:10]:
                    frame_iou = calculate_iou(algorithm_masks[frame], ground_truth_masks[frame])
                    frame_dice = calculate_dice(algorithm_masks[frame], ground_truth_masks[frame])
                    f.write(f"| {frame} | {frame_iou:.4f} | {frame_dice:.4f} |\n")
                
                if len(set(ground_truth_masks.keys()) & set(algorithm_masks.keys())) > 10:
                    f.write("| ... | ... | ... |\n")
            
            print(f"Evaluation report saved to: {report_path}")
            
        except Exception as e:
            print(f"Error evaluating video: {e}")
            traceback.print_exc()
            evaluation_results[f"{study_uid}_{series_uid}"] = {
                'error': str(e)
            }
    
    # Generate overall summary
    valid_results = [r for r in evaluation_results.values() if 'metrics' in r]
    
    if valid_results:
        summary = {
            'total_videos': len(video_paths),
            'successful_evaluations': len(valid_results),
            'overall_mean_iou': np.mean([r['metrics']['mean_iou'] for r in valid_results]),
            'overall_mean_dice': np.mean([r['metrics']['mean_dice'] for r in valid_results]),
            'best_video': max([(k, v['metrics']['mean_iou']) for k, v in evaluation_results.items() 
                             if 'metrics' in v], key=lambda x: x[1])[0],
            'worst_video': min([(k, v['metrics']['mean_iou']) for k, v in evaluation_results.items() 
                              if 'metrics' in v], key=lambda x: x[1])[0]
        }
        
        evaluation_results['summary'] = summary
        
        # Create overall report
        report_path = os.path.join(output_dir, "overall_evaluation_report.md")
        with open(report_path, 'w') as f:
            f.write("# Overall Evaluation Report\n\n")
            f.write(f"- Total Videos: {summary['total_videos']}\n")
            f.write(f"- Successful Evaluations: {summary['successful_evaluations']}\n")
            f.write(f"- Overall Mean IoU: {summary['overall_mean_iou']:.4f}\n")
            f.write(f"- Overall Mean Dice: {summary['overall_mean_dice']:.4f}\n")
            f.write(f"- Best Performing Video: {summary['best_video']}\n")
            f.write(f"- Worst Performing Video: {summary['worst_video']}\n\n")
            
            f.write("## Per-Video Results\n\n")
            f.write("| Video | Mean IoU | Mean Dice | IoU > 0.7 |\n")
            f.write("|-------|----------|-----------|----------|\n")
            
            for video_id, results in evaluation_results.items():
                if video_id != 'summary' and 'metrics' in results:
                    metrics = results['metrics']
                    f.write(f"| {video_id} | {metrics['mean_iou']:.4f} | {metrics['mean_dice']:.4f} | {metrics['iou_over_0.7']*100:.1f}% |\n")
        
        print(f"Overall evaluation report saved to: {report_path}")
    
    # Save results to file
    with open(os.path.join(output_dir, 'evaluation_results.json'), 'w') as f:
        # Convert numpy values to Python native types for JSON serialization
        json_results = convert_numpy_to_python(evaluation_results)
        json.dump(json_results, f, indent=2)
    
    return evaluation_results

def run_ground_truth_feedback_loop(target_videos, num_iterations=3):
    """
    Runs the complete feedback loop for ground truth creation and evaluation
    
    Args:
        target_videos: List of (video_path, study_uid, series_uid) tuples to process
        num_iterations: Number of feedback loop iterations to run
        
    Returns:
        Dictionary with overall results
    """
    results = {
        'iterations': []
    }
    
    base_output_dir = os.path.join(OUTPUT_DIR, "ground_truth_feedback_loop")
    os.makedirs(base_output_dir, exist_ok=True)
    
    # Initialize optical flow processor
    flow_processor = OpticalFlowProcessor(method=FLOW_METHOD[0])
    
    # Extract video paths and study/series pairs
    video_paths = [v[0] for v in target_videos]
    study_series_pairs = [(v[1], v[2]) for v in target_videos]
    
    for iteration in range(num_iterations):
        print(f"\n{'='*50}")
        print(f"STARTING ITERATION {iteration+1}/{num_iterations}")
        print(f"{'='*50}\n")
        
        # Create iteration output directory
        iter_output_dir = os.path.join(base_output_dir, f"iteration_{iteration+1}")
        os.makedirs(iter_output_dir, exist_ok=True)
        
        # Step 1: Create/update ground truth dataset
        print("\nStep 1: Creating/updating ground truth dataset...")
        gt_results = create_ground_truth_dataset(
            video_paths, study_series_pairs, flow_processor, 
            os.path.join(iter_output_dir, "ground_truth"),
            mdai_client, PROJECT_ID, DATASET_ID, LABEL_ID_GROUND_TRUTH
        )
        
        # Step 2: Wait for expert review
        print("\nStep 2: Expert review phase")
        print("Please have experts review and modify the ground truth annotations in MD.ai.")
        print("The ground truth annotations have been uploaded with label ID:", LABEL_ID_GROUND_TRUTH)
        
        if iteration < num_iterations - 1:  # Don't prompt on last iteration
            input("Press Enter once expert review is complete to continue to evaluation...")
        
        # Step 3: Evaluate algorithm against expert-refined ground truth
        print("\nStep 3: Evaluating algorithm against expert-refined ground truth...")
        eval_results = evaluate_with_expert_feedback(
            video_paths, study_series_pairs, flow_processor,
            os.path.join(iter_output_dir, "evaluation"),
            mdai_client, PROJECT_ID, DATASET_ID, 
            LABEL_ID_GROUND_TRUTH, LABEL_ID_FLUID_OF
        )
        
        # Store iteration results
        results['iterations'].append({
            'iteration_number': iteration + 1,
            'ground_truth_results': gt_results,
            'evaluation_results': eval_results
        })
        
        # Generate iteration report
        if 'summary' in eval_results:
            summary = eval_results['summary']
            print("\nIteration Summary:")
            print(f"- Overall Mean IoU: {summary['overall_mean_iou']:.4f}")
            print(f"- Overall Mean Dice: {summary['overall_mean_dice']:.4f}")
        
        # Optional: Implement algorithm improvements based on evaluation
        # This would be where you'd adjust algorithm parameters or implement new features
        # based on the evaluation results
        
    # Generate final report comparing all iterations
    if results['iterations']:
        create_feedback_loop_report(results, base_output_dir)
    
    return results

# ===== 8. MAIN EXECUTION =====
def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Ultrasound free fluid tracking using optical flow')
    
    # Add arguments for study and series UIDs
    parser.add_argument('--study', type=str, help='Specific StudyInstanceUID to process')
    parser.add_argument('--series', type=str, help='Specific SeriesInstanceUID to process')
    parser.add_argument('--issue', type=str, choices=['disappear_reappear', 'branching_fluid', 'multiple_distinct', 'no_fluid'],
                        help='Specific issue type to process')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    parser.add_argument('--upload', action='store_true', help='Upload results to MD.ai')

    parser.add_argument('--create-ground-truth', action='store_true', help='Create ground truth dataset')
    parser.add_argument('--ground-truth-videos', type=int, default=15, help='Number of videos per issue type for ground truth')
    parser.add_argument('--feedback-loop', action='store_true', help='Run ground truth feedback loop')
    parser.add_argument('--iterations', type=int, default=3, help='Number of iterations for feedback loop')

    parser.add_argument('--ground-truth-single-exam', type=str, help='Create ground truth for a single exam number (for debugging)')
    parser.add_argument('--ground-truth-single-study', type=str, help='Create ground truth for a single StudyInstanceUID (for debugging)')
    parser.add_argument('--ground-truth-single-series', type=str, help='Create ground truth for a single SeriesInstanceUID (for debugging)')
    
    
    return parser.parse_args()

if __name__ == "__main__":
    # Parse command line arguments
    args = parse_arguments()
    
    # Override debug setting if specified
    if args.debug:
        DEBUG_MODE = True

    # Initialize logging
    log_file = setup_logging(OUTPUT_DIR)
    print(f"All console output will be saved to: {log_file}")
    
    # Load environment variables
    load_dotenv('.env')
    ACCESS_TOKEN = os.getenv('MDAI_TOKEN')
    DATA_DIR = os.getenv('DATA_DIR')
    DOMAIN = os.getenv('DOMAIN')
    PROJECT_ID = os.getenv('PROJECT_ID')
    DATASET_ID = os.getenv('DATASET_ID')
    ANNOTATIONS = os.getenv('ANNOTATIONS')
    LABEL_ID_FREE_FLUID = os.environ["LABEL_ID_FREE_FLUID"] = "L_13yPql"
    LABEL_ID_NO_FLUID = os.getenv("LABEL_ID_NO_FLUID") 
    LABEL_IDS = {
        "disappear_reappear": os.getenv("LABEL_ID_DISAPPEAR_REAPPEAR"),
        "branching_fluid": os.getenv("LABEL_ID_BRANCHING_FLUID"),
        "multiple_distinct": os.getenv("LABEL_ID_MULTIPLE_DISTINCT"),
        "no_fluid": os.getenv("LABEL_ID_NO_FLUID"),
    }
    LABEL_ID_MACHINE_GROUP = os.getenv("LABEL_ID_MACHINE_GROUP")
    LABEL_ID_FLUID_OF = os.getenv("LABEL_ID_FLUID_OF")
    LABEL_ID_GROUND_TRUTH = os.getenv("LABEL_ID_GROUND_TRUTH")
    
    debug_print(f"ANNOTATIONS path: {ANNOTATIONS}")
    debug_print(f"LABEL_ID_FREE_FLUID: {LABEL_ID_FREE_FLUID}")
    debug_print(f"LABEL_ID_NO_FLUID: {LABEL_ID_NO_FLUID}")     
    print(f"LABEL_ID_GROUND_TRUTH: {os.getenv('LABEL_ID_GROUND_TRUTH')}")

    # Start MD.ai client
    try:
        mdai_client = mdai.Client(domain=DOMAIN, access_token=ACCESS_TOKEN)
        project = mdai_client.project(project_id=PROJECT_ID, path=DATA_DIR)
        print("Successfully connected to MD.ai")
    except Exception as e:
        print(f"Error connecting to MD.ai: {str(e)}")
        raise
    

    # Validate that they're loaded
    if None in [LABEL_ID_MACHINE_GROUP, LABEL_ID_FLUID_OF]:
        raise ValueError("Machine label IDs not properly set in .env file")
    
    try:
        print("Fetching latest annotations from MD.ai...")
    
        DATASET_ID = "D_V688LQ"  # PECARN Video dataset
        LABEL_ID_FREE_FLUID = "L_13yPql"  # Free fluid label ID
    
        # Fetch annotations for the specific dataset
        result = mdai_client.project(
            project_id=PROJECT_ID, 
            dataset_id=DATASET_ID,
            path=DATA_DIR,
            annotations_only=True
        )
    
        # Get the annotations file path
        annotation_files = [f for f in os.listdir(DATA_DIR) if f.startswith('mdai_ucsf_project') and f.endswith('.json')]
    
        if annotation_files:
            # Sort by modification time (newest first)
            annotation_files.sort(key=lambda f: os.path.getmtime(os.path.join(DATA_DIR, f)), reverse=True)
            ANNOTATIONS = os.path.join(DATA_DIR, annotation_files[0])
            mod_time = datetime.fromtimestamp(os.path.getmtime(ANNOTATIONS)).strftime('%Y-%m-%d %H:%M:%S')
            print(f"Using latest downloaded annotations file: {ANNOTATIONS} (last modified: {mod_time})")
        else:
            raise FileNotFoundError("No annotation files found in data directory after download")
        
    except Exception as e:
        print(f"Error fetching annotations from MD.ai: {str(e)}")
        print("Falling back to local annotations file...")

        ANNOTATIONS = "/Users/Shreya1/Documents/GitHub/goobusters/data/mdai_ucsf_project_x9N2LJBZ_annotations_dataset_D_V688LQ_labelgroup_G_7n3P09_2025-05-07-194248.json"
        print(f"Using hardcoded backup path: {ANNOTATIONS}")

    if not os.path.exists(ANNOTATIONS):
        raise FileNotFoundError(f"Annotations file not found at: {ANNOTATIONS}")

    # Load and normalize annotations
    with open(ANNOTATIONS, 'r') as f:
        annotations_json = json.load(f)
    
    datasets = annotations_json.get('datasets', [])
    all_annotations = []
    for dataset in datasets:
        if 'annotations' in dataset:
            all_annotations.extend(dataset['annotations'])
    
    annotations_df = json_normalize(all_annotations, sep='_')

    debug_print(f"Loaded annotations from file with timestamp: {os.path.getmtime(ANNOTATIONS)}")
    
    # After processing the JSON into a DataFrame
    print(f"All unique label IDs: {annotations_df['labelId'].unique()}")
    no_fluid_count = sum(annotations_df['labelId'] == LABEL_ID_NO_FLUID)
    print(f"Found {no_fluid_count} no fluid annotations")

    print("\n=== DEBUGGING FILE PATHS ===")
    print(f"ANNOTATIONS path: {ANNOTATIONS}")

    print(f"LABEL_ID_FREE_FLUID = {LABEL_ID_FREE_FLUID}")
    # Check all unique label IDs in the dataset to see if your expected ID exists
    print(f"Unique label IDs in dataset: {annotations_df['labelId'].unique()}")
    
    # Step 1: Filter free fluid annotations separately
    free_fluid_annotations = annotations_df[
        ((annotations_df['labelId'] == LABEL_ID_FREE_FLUID) |
         (annotations_df['labelId'] == LABEL_ID_NO_FLUID)) &
        (annotations_df['frameNumber'].notna())
    ].copy()
    free_fluid_annotations['frameNumber'] = free_fluid_annotations['frameNumber'].astype(int)
    
    print("Columns in free_fluid_annotations:", free_fluid_annotations.columns.tolist())

    no_fluid_anns = annotations_df[annotations_df['labelId'] == LABEL_ID_NO_FLUID]
    debug_print(f"Found {len(no_fluid_anns)} 'no fluid' annotations")
    if len(no_fluid_anns) > 0:
        debug_print(f"No fluid frames: {no_fluid_anns['frameNumber'].tolist()}")

    # Construct video paths for free fluid annotations
    BASE = project.get_dataset_by_id(DATASET_ID).images_dir

    print("\n=== DEBUGGING BASE PATH ===")
    print(f"BASE: {BASE}")
    print(f"BASE type: {type(BASE)}")
    print(f"BASE exists: {os.path.exists(BASE) if isinstance(BASE, str) else 'Not a string path'}") 

    # Instead of using apply, create a list first
    paths = []
    for idx, row in free_fluid_annotations.iterrows():
        path = os.path.join(BASE, row['StudyInstanceUID'], f"{row['SeriesInstanceUID']}.mp4")
        paths.append(path)
    
    # Then assign the list to the column
    free_fluid_annotations['video_path'] = paths    
    free_fluid_annotations['file_exists'] = free_fluid_annotations['video_path'].apply(os.path.exists)
    free_fluid_annotations = free_fluid_annotations[free_fluid_annotations['file_exists']]
    
    free_fluid_annotations.rename(columns={'data_foreground': 'free_fluid_foreground'}, inplace=True)
    
    print(f"Total Free Fluid Annotations: {len(free_fluid_annotations)}")
    
    # Step 2: Link issue-type annotations to free fluid annotations
    matched_annotations = pd.DataFrame()
    
    # Determine which issue types to process based on command-line args
    if args.issue:
        issue_types_to_process = [args.issue]
        print(f"Processing only issue type: {args.issue} (from command-line argument)")
    else:
        issue_types_to_process = DEBUG_ISSUE_TYPES if DEBUG_MODE else list(LABEL_IDS.keys())
        print(f"Processing issue types: {issue_types_to_process}")
    
    for issue_type, label_id in LABEL_IDS.items():
        # Skip issue types not in our process list
        if issue_type not in issue_types_to_process:
            print(f"Skipping {issue_type} (not in process list)")
            continue
            
        issue_annotations = annotations_df[annotations_df['labelId'] == label_id].copy()
        print(f"Total {issue_type} Annotations: {len(issue_annotations)}")
    
        # Merge without using 'frameNumber' as a key
        merged_annotations = pd.merge(
            issue_annotations,
            free_fluid_annotations[['StudyInstanceUID', 'SeriesInstanceUID', 'frameNumber', 'video_path', 'free_fluid_foreground']],
            on=['StudyInstanceUID', 'SeriesInstanceUID'],
            suffixes=('_complication', '_free_fluid'),
            how='inner'
        )
    
        # Inherit the frameNumber from free_fluid_annotations
        merged_annotations['frameNumber'] = merged_annotations['frameNumber_free_fluid'].fillna(merged_annotations['frameNumber_free_fluid'])
    
        # Filter out annotations with missing 'free_fluid_foreground'
        valid_annotations = merged_annotations[~merged_annotations['free_fluid_foreground'].isna()]
        
        # Check if specific study/series was requested
        if args.study and args.series:
            specific_annotations = valid_annotations[
                (valid_annotations['StudyInstanceUID'] == args.study) & 
                (valid_annotations['SeriesInstanceUID'] == args.series)
            ]
            
            if len(specific_annotations) > 0:
                print(f"Found specific requested study/series: {args.study}/{args.series}")
                sampled_annotations = specific_annotations
            else:
                print(f"Warning: Requested study/series not found among {issue_type} annotations")
                if DEBUG_MODE:
                    print("Falling back to default sampling")
                    sample_size = min(5, len(valid_annotations))
                    sampled_annotations = valid_annotations.sample(n=sample_size, random_state=42)
                else:
                    print("Skipping this issue type")
                    continue
        else:
            # Sample a subset of the valid annotations
            sample_size = min(5, len(valid_annotations))
            sampled_annotations = valid_annotations.sample(n=sample_size, random_state=42)
            
        sampled_annotations['issue_type'] = issue_type
        matched_annotations = pd.concat([matched_annotations, sampled_annotations])
    
    print(f"Total matched annotations: {len(matched_annotations)}")
    
    # Check for multi-frame tracking availability
    MULTI_FRAME_AVAILABLE = True
    try:
        # Import multi-frame tracking modules
        import multi_frame_tracking
        print("Successfully imported multi_frame_tracking module")
        
        from multi_frame_tracking.multi_frame_tracker import MultiFrameTracker, process_video_with_multi_frame_tracking
        print("Successfully imported MultiFrameTracker and process_video_with_multi_frame_tracking")
    except ImportError as e:
        print(f"Import error: {e}")
        traceback.print_exc()
        MULTI_FRAME_AVAILABLE = False
        print("WARNING: Multi-frame tracking module not found. Only single-frame tracking will be available.")
    except RecursionError as e:
        print(f"Recursion error: {e}")
        traceback.print_exc()
        MULTI_FRAME_AVAILABLE = False
        print("WARNING: RecursionError in multi-frame tracking module. Only single-frame tracking will be available.")
    
    # Override upload setting if specified in args
    if args.upload:
        print("Upload to MD.ai enabled via command-line argument")
        UPLOAD_TO_MDAI = True
    
    # Create debug tracking mode file
    with open(os.path.join(OUTPUT_DIR, "tracking_mode_debug.txt"), "w") as f:
        f.write(f"TRACKING_MODE = {TRACKING_MODE}\n")
        f.write(f"MULTI_FRAME_AVAILABLE = {MULTI_FRAME_AVAILABLE}\n")
        f.write(f"DEBUG_MODE = {DEBUG_MODE}\n")
        f.write(f"Started at {datetime.now()}\n")
        if args.study and args.series:
            f.write(f"Target study/series: {args.study}/{args.series}\n")
        if args.issue:
            f.write(f"Target issue type: {args.issue}\n")

    # MAIN BRANCHING LOGIC - This is where the if-else should be
    if args.create_ground_truth:
        print("\n==== CREATING GROUND TRUTH DATASET ====")
        
        # Get LABEL_ID_GROUND_TRUTH from environment
        LABEL_ID_GROUND_TRUTH = os.getenv("LABEL_ID_GROUND_TRUTH")
        if not LABEL_ID_GROUND_TRUTH:
            print("Error: LABEL_ID_GROUND_TRUTH not defined in .env file")
            sys.exit(1)
            
        print(f"Using ground truth label ID: {LABEL_ID_GROUND_TRUTH}")
        
        # Select videos for ground truth dataset
        gt_videos = []
        gt_study_series_pairs = []
        
        # Check if matched_annotations is available
        if 'matched_annotations' not in locals() or matched_annotations.empty:
            print("ERROR: No matched annotations available. Please run normal workflow first.")
            sys.exit(1)
        
        # Check if free_fluid_annotations is available
        if 'free_fluid_annotations' not in locals() or free_fluid_annotations.empty:
            print("ERROR: No free_fluid_annotations available. Please run normal workflow first.")
            sys.exit(1)
        
        # NEW: Check for single exam/study/series ground truth
        if args.ground_truth_single_exam or args.ground_truth_single_study or args.ground_truth_single_series:
            print("\n>>> SINGLE VIDEO GROUND TRUTH MODE <<<")
            
            # Filter to a single video
            single_gt_videos = []
            single_gt_study_series_pairs = []
            
            # First get all available videos
            all_gt_videos = []
            all_gt_study_series_pairs = []
            
            # Process each issue type to get all available videos
            for issue_type in LABEL_IDS.keys():
                issue_annotations = matched_annotations[matched_annotations['issue_type'] == issue_type]
                
                for _, row in issue_annotations.iterrows():
                    # Ensure video_path is properly set
                    if 'video_path' not in row or pd.isna(row['video_path']):
                        video_path = os.path.join(BASE, row['StudyInstanceUID'], f"{row['SeriesInstanceUID']}.mp4")
                    else:
                        video_path = row['video_path']
                        
                    if os.path.exists(video_path):
                        all_gt_videos.append(video_path)
                        all_gt_study_series_pairs.append((row['StudyInstanceUID'], row['SeriesInstanceUID']))
            
            # Process based on which argument was provided
            if args.ground_truth_single_exam:
                # Find video with this exam number
                print(f"Looking for Exam #{args.ground_truth_single_exam}...")
                for video, (study, series) in zip(all_gt_videos, all_gt_study_series_pairs):
                    try:
                        # Import here to avoid circular import
                        from multi_frame_tracking.utils import find_exam_number
                        exam_number = find_exam_number(study, annotations_json)
                        if exam_number == args.ground_truth_single_exam:
                            single_gt_videos.append(video)
                            single_gt_study_series_pairs.append((study, series))
                            print(f"Found Exam #{exam_number}: {study}/{series}")
                            break
                    except Exception as e:
                        print(f"Error checking exam number for {study}: {e}")
                        continue
            
            elif args.ground_truth_single_study and args.ground_truth_single_series:
                # Find video with specific study/series
                print(f"Looking for Study: {args.ground_truth_single_study}, Series: {args.ground_truth_single_series}")
                for video, (study, series) in zip(all_gt_videos, all_gt_study_series_pairs):
                    if study == args.ground_truth_single_study and series == args.ground_truth_single_series:
                        single_gt_videos.append(video)
                        single_gt_study_series_pairs.append((study, series))
                        print(f"Found Study/Series: {study}/{series}")
                        break
            
            elif args.ground_truth_single_study:
                # Find any video with this study (take first series)
                print(f"Looking for any video in Study: {args.ground_truth_single_study}")
                for video, (study, series) in zip(all_gt_videos, all_gt_study_series_pairs):
                    if study == args.ground_truth_single_study:
                        single_gt_videos.append(video)
                        single_gt_study_series_pairs.append((study, series))
                        print(f"Found Study: {study}/{series}")
                        break
            
            # Use single video if found
            if single_gt_videos:
                gt_videos = single_gt_videos
                gt_study_series_pairs = single_gt_study_series_pairs
                print(f"Processing single video for ground truth")
            else:
                print("ERROR: No video found matching the specified criteria")
                print("Available options (first 10):")
                for i, (study, series) in enumerate(all_gt_study_series_pairs[:10]):
                    try:
                        from multi_frame_tracking.utils import find_exam_number
                        exam_num = find_exam_number(study, annotations_json)
                        print(f"  Exam #{exam_num}: {study}/{series}")
                    except:
                        print(f"  Unknown exam: {study}/{series}")
                sys.exit(1)
        
        else:
            # NORMAL MODE: Process multiple videos per issue type
            print("\n>>> MULTI-VIDEO GROUND TRUTH MODE <<<")
            
            # Process each issue type to gather videos
            videos_per_type = args.ground_truth_videos
            print(f"Selecting {videos_per_type} videos per issue type...")
            
            for issue_type in LABEL_IDS.keys():
                # Filter annotations for this issue type
                issue_annotations = matched_annotations[matched_annotations['issue_type'] == issue_type]
                
                # Sample videos
                if len(issue_annotations) > 0:
                    sample_size = min(videos_per_type, len(issue_annotations))
                    samples = issue_annotations.sample(n=sample_size, random_state=42)
                    
                    print(f"Selected {len(samples)} videos for issue type '{issue_type}'")
                    
                    for _, row in samples.iterrows():
                        # Ensure video_path is properly set
                        if 'video_path' not in row or pd.isna(row['video_path']):
                            # Reconstruct video path
                            video_path = os.path.join(BASE, row['StudyInstanceUID'], f"{row['SeriesInstanceUID']}.mp4")
                        else:
                            video_path = row['video_path']
                            
                        if os.path.exists(video_path):
                            gt_videos.append(video_path)
                            gt_study_series_pairs.append((row['StudyInstanceUID'], row['SeriesInstanceUID']))
                        else:
                            print(f"WARNING: Video not found at {video_path}")
                else:
                    print(f"No annotations found for issue type '{issue_type}'")
        
        # Create output directory for ground truth
        ground_truth_dir = os.path.join(OUTPUT_DIR, "ground_truth_dataset")
        os.makedirs(ground_truth_dir, exist_ok=True)
        
        # Initialise flow processor if not already done
        if 'flow_processor' not in locals():
            flow_processor = OpticalFlowProcessor(method=FLOW_METHOD[0])
        
        # Run the ground truth creation
        print(f"Creating ground truth dataset for {len(gt_videos)} videos...")
        results = create_ground_truth_dataset(
            gt_videos,
            gt_study_series_pairs,
            flow_processor,
            ground_truth_dir,
            mdai_client,
            PROJECT_ID,
            DATASET_ID,
            LABEL_ID_GROUND_TRUTH,
            matched_annotations,
            free_fluid_annotations,  
            label_id_fluid=LABEL_ID_FREE_FLUID,
            label_id_no_fluid=LABEL_ID_NO_FLUID,
            label_id_machine=LABEL_ID_MACHINE_GROUP,
            annotations_json=annotations_json
        )
        
        print("\nGround truth dataset creation complete!")
        print(f"Total videos processed: {results['summary']['total_videos']}")
        print(f"Successfully processed: {results['summary']['successful_videos']}")
        print(f"Total annotations created: {results['summary']['total_annotations_created']}")
        print(f"Total annotations uploaded: {results['summary']['total_upload_success']}")
        print(f"Exam numbers processed: {', '.join(map(str, results['summary']['exam_numbers_processed']))}")
        print("\nPlease review and modify these annotations in MD.ai as needed.")
        
        # Exit after ground truth creation
        sys.exit(0)
    
    else:  
        # Run the normal processing function
        process_videos_with_tracking()
    
    debug_print(f"=== DEBUG LOG ENDED AT {time.ctime()} ===")
    debug_log.close()