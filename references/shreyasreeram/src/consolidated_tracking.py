#!/usr/bin/env python3
import os
import cv2
import numpy as np
import pandas as pd
from pandas import json_normalize
import json
import time
import traceback
import matplotlib.pyplot as plt
from tqdm import tqdm
import mdai
import argparse
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime
import gc
import psutil
import warnings
import logging
from dotenv import load_dotenv
import signal
import sys
import glob
import torch
import traceback
from pathlib import Path
import matplotlib.pyplot as plt
from multi_frame_tracking.multi_frame_tracker import MultiFrameTracker
from multi_frame_tracking.utils import convert_numpy_to_python  

# Import local modules
from src.multi_frame_tracking.multi_frame_tracker import MultiFrameTracker
from src.multi_frame_tracking.opticalflowprocessor import OpticalFlowProcessor
from src.multi_frame_tracking.utils import find_exam_number
from src.utils import (
    evaluate_with_iou, calculate_iou, calculate_dice, evaluate_tracking,
    save_evaluation_visualizations, create_evaluation_report,
    visualize_comparison, process_video_with_multi_frame_tracking_enhanced,
    get_annotations_for_study_series, create_ground_truth_dataset,
    polygons_to_mask, create_feedback_loop_report
)

os.environ['PYTHONWARNINGS'] = 'ignore'
warnings.filterwarnings('ignore')
logging.getLogger().setLevel(logging.ERROR)

def cleanup_memory():
    """Force garbage collection and print memory usage"""
    gc.collect()
    if 'psutil' in sys.modules:
        process = psutil.Process()
        memory_mb = process.memory_info().rss / 1024 / 1024
        print(f"[MEMORY] After cleanup: {memory_mb:.1f} MB")

def cleanup_debug_files(debug_dir):
    """Remove debug files after processing each video"""
    if os.path.exists(debug_dir):

        for file in os.listdir(debug_dir):
            if file.endswith('.png') or file.endswith('.mp4'):
                try:
                    os.remove(os.path.join(debug_dir, file))
                except:
                    pass

# Global checkpoint tracker
checkpoint_times = {}
last_checkpoint = None
start_time = time.time()

def checkpoint(name, details=""):
    """Add checkpoint with timestamp"""
    global last_checkpoint, start_time
    current_time = time.time()
    
    if last_checkpoint:
        duration = current_time - checkpoint_times[last_checkpoint]
        print(f"[CHECKPOINT] ✓ {last_checkpoint} completed in {duration:.2f}s")
    
    checkpoint_times[name] = current_time
    last_checkpoint = name
    
    elapsed = current_time - start_time
    timestamp = time.strftime("%H:%M:%S", time.localtime(current_time))
    print(f"[CHECKPOINT] {timestamp} (T+{elapsed:.1f}s) - Starting: {name}")
    if details:
        print(f"  Details: {details}")
    
    # Force flush output
    sys.stdout.flush()

def setup_timeout_monitor(timeout_minutes=30):
    """Set up a timeout monitor to prevent infinite loops"""
    def timeout_handler(signum, frame):
        print(f"\n!!! TIMEOUT: Script has been running for {timeout_minutes} minutes")
        print("!!! Current checkpoint:", last_checkpoint)
        print("!!! Process times:")
        for cp_name, cp_time in checkpoint_times.items():
            print(f"     {cp_name}: {cp_time - start_time:.1f}s")
        print("!!! Forcing script termination...")
        os._exit(1)
    
    # Set up timeout
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(timeout_minutes * 60)
    print(f"[MONITOR] Timeout monitor set for {timeout_minutes} minutes")


MULTI_FRAME_AVAILABLE = True  
TRACKING_MODE = 'multi'  
DEBUG_MODE = False 

# Load environment variables from .env 
dotenv_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '.env')
load_dotenv(dotenv_path)

# Initialise label IDs
LABEL_IDS = {
    'FLUID_OF': os.getenv('LABEL_ID_FLUID_OF', 'L_JykNe7'),
    'FREE_FLUID': os.getenv('LABEL_ID_FREE_FLUID', 'L_13yPql'),
    'NO_FLUID': os.getenv('LABEL_ID_NO_FLUID', 'L_75K42J'),
    'GROUND_TRUTH': os.getenv('LABEL_ID_GROUND_TRUTH', 'L_7DRjNJ'),
    'MACHINE_GROUP': os.getenv('LABEL_ID_MACHINE_GROUP', 'G_RJY6Qn')
}


LABEL_ID_FLUID_OF = LABEL_IDS['FLUID_OF']
LABEL_ID_FREE_FLUID = LABEL_IDS['FREE_FLUID']
LABEL_ID_NO_FLUID = LABEL_IDS['NO_FLUID']
LABEL_ID_GROUND_TRUTH = LABEL_IDS['GROUND_TRUTH']
LABEL_ID_MACHINE_GROUP = LABEL_IDS['MACHINE_GROUP']
label_id_fluid = LABEL_ID_FLUID_OF  
label_id_no_fluid = LABEL_ID_NO_FLUID

# Function to handle no-fluid annotations
def create_empty_mask(height, width):
    """Create an empty mask for no-fluid annotations"""
    return np.zeros((height, width), dtype=np.uint8)

def is_no_fluid_annotation(annotation):
    """Check if an annotation represents no fluid"""
    if isinstance(annotation, dict):
        return annotation.get('labelId') == LABEL_ID_NO_FLUID
    elif hasattr(annotation, 'labelId'):  # Handle pandas Series
        return annotation.labelId == LABEL_ID_NO_FLUID
    return False

def is_fluid_annotation(annotation):
    """Check if an annotation represents fluid"""
    if isinstance(annotation, dict):
        return annotation.get('labelId') == LABEL_ID_FREE_FLUID  
    elif hasattr(annotation, 'labelId'): 
        return annotation.labelId == LABEL_ID_FREE_FLUID  
    return False

# Print initial values --> sanity check 
print(f"\nInitialized label IDs:")
print(f"LABEL_ID_FLUID_OF: {LABEL_ID_FLUID_OF}")
print(f"LABEL_ID_FREE_FLUID: {LABEL_ID_FREE_FLUID}")
print(f"LABEL_ID_NO_FLUID: {LABEL_ID_NO_FLUID}")
print(f"LABEL_ID_GROUND_TRUTH: {LABEL_ID_GROUND_TRUTH}")
print(f"LABEL_ID_MACHINE_GROUP: {LABEL_ID_MACHINE_GROUP}")
print(f"label_id_fluid: {label_id_fluid}")
print(f"label_id_no_fluid: {label_id_no_fluid}\n")

# Create a dedicated debug log file
debug_log_path = f"debug_no_fluid_{int(time.time())}.log"
debug_log = open(debug_log_path, "w")

def debug_print(message):
    """Write debug messages to a separate file"""
    debug_log.write(f"{message}\n")
    debug_log.flush()  
debug_print(f"=== DEBUG LOG STARTED AT {time.ctime()} ===")


# Enable debug mode --> only if specified 

DEBUG_SAMPLE_SIZE = 5
DEBUG_ISSUE_TYPES = ["multiple_distinct"]

# Set debugging options for tracking
TARGET_FRAMES = []  
VERBOSE_DEBUGGING = False

# Set tracking mode: 'single', 'multi', or 'both'
TRACKING_MODE = 'multi'  # this can be changed to switch between tracking modes

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(PROJECT_DIR, "output")
MULTI_FRAME_DIR = os.path.join(OUTPUT_DIR, "multi_frame_output") 

FLOW_METHOD = ['dis'] # Only tested dis in this project
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
            if message.strip():  
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
        cv2.fillPoly(mask, [points], 1) # Using OpenCV'S fill function to convert to binary masks
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
    
    # Create empty visualisation image
    vis = np.zeros((h * 2, w, 3), dtype=np.uint8)
    
    # Copy original frame to top half
    if len(frame.shape) == 2:  # Convert grayscale to colour if needed
        vis[:h, :] = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    else:
        vis[:h, :] = frame.copy()
    
    # Calculate flow magnitude and angle
    flow_x = flow[..., 0]
    flow_y = flow[..., 1]
    magnitude, angle = cv2.cartToPolar(flow_x, flow_y)
    
    # Create HSV image for flow visualisation
    hsv = np.zeros((h, w, 3), dtype=np.uint8)
    hsv[..., 0] = angle * 180 / np.pi / 2  
    hsv[..., 1] = 255                       
    hsv[..., 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)  
    
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
        
        # Apply the difference visualisation
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
    
    # Add flow visualisation (Bottom row) if flow is provided
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
                # If visualisation format is different, just use what we have
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
        Visualisation showing frame differences and enhanced flow vectors
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
    
    # Apply colour map for better visualization
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
                    fx = flow[y, x, 0] * 3.0  
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
    
    # Create a combined visualisation
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

def calculate_tracking_metrics(algorithm_masks, ground_truth_masks):
    """
    Calculate various metrics to compare tracking approaches.
    Fixed version to handle unhashable type errors.
    
    Args:
        algorithm_masks: Dictionary of algorithm predictions {frame_idx: mask}
        ground_truth_masks: Dictionary of ground truth annotations {frame_idx: mask}
        
    Returns:
        Dictionary of metrics
    """
    print("\nCalculating tracking metrics...")
    
    metrics = {
        'per_frame': {},
        'summary': {}
    }
    
    # Ensure ground_truth_masks is a proper dictionary with integer keys
    if isinstance(ground_truth_masks, list):
        print("Converting ground truth list to dictionary...")
        gt_dict = {}
        for item in ground_truth_masks:
            if isinstance(item, dict) and 'frameNumber' in item:
                frame_num = int(item['frameNumber'])
                gt_dict[frame_num] = item
        ground_truth_masks = gt_dict
    
    # Convert any non-integer keys to integers
    clean_gt_masks = {}
    for key, value in ground_truth_masks.items():
        try:
            # Ensure key is an integer
            clean_key = int(key) if not isinstance(key, int) else key
            clean_gt_masks[clean_key] = value
        except (ValueError, TypeError) as e:
            print(f"Warning: Skipping invalid key {key}: {e}")
            continue
    
    ground_truth_masks = clean_gt_masks
    
    # Do the same for algorithm masks
    clean_algo_masks = {}
    for key, value in algorithm_masks.items():
        try:
            clean_key = int(key) if not isinstance(key, int) else key
            clean_algo_masks[clean_key] = value
        except (ValueError, TypeError) as e:
            print(f"Warning: Skipping invalid algorithm mask key {key}: {e}")
            continue
    
    algorithm_masks = clean_algo_masks
    
    # Get all frames that have ground truth
    ground_truth_frames = sorted(list(ground_truth_masks.keys()))
    
    if not ground_truth_frames:
        print("No ground truth frames found!")
        return metrics
        
    print(f"Found {len(ground_truth_frames)} ground truth frames")
    print(f"Ground truth frame range: {min(ground_truth_frames)} to {max(ground_truth_frames)}")
    print(f"Algorithm mask frames: {len(algorithm_masks)}")
    if algorithm_masks:
        algo_frames = sorted(list(algorithm_masks.keys()))
        print(f"Algorithm frame range: {min(algo_frames)} to {max(algo_frames)}")
    
    # Calculate IoU and Dice for each frame
    ious = []
    dices = []
    frames_with_high_iou = 0
    processed_frames = 0
    
    for frame_idx in ground_truth_frames:
        if frame_idx not in algorithm_masks:
            print(f"Warning: Frame {frame_idx} missing from algorithm predictions")
            continue
            
        # Get masks
        algo_mask = algorithm_masks[frame_idx]
        gt_mask = ground_truth_masks[frame_idx]
        
        # Handle dictionary format from MultiFrameTracker
        if isinstance(algo_mask, dict):
            if 'mask' in algo_mask:
                algo_mask = algo_mask['mask']
            elif algo_mask.get('type', '').lower() == 'clear' or algo_mask.get('is_no_fluid', False):
                print(f"Frame {frame_idx}: Algorithm predicts no fluid")
                # Create empty mask with same dimensions as ground truth
                if isinstance(gt_mask, dict):
                    if 'mask' in gt_mask:
                        gt_shape = gt_mask['mask'].shape
                    else:
                        # Try to extract shape from polygon data
                        try:
                            if 'data' in gt_mask and 'foreground' in gt_mask['data']:
                                # Default shape - adjustable 
                                gt_shape = (480, 640)  # Common ultrasound dimensions
                            else:
                                gt_shape = (480, 640)
                        except:
                            gt_shape = (480, 640)
                    algo_mask = np.zeros(gt_shape, dtype=np.uint8)
                else:
                    algo_mask = np.zeros_like(gt_mask, dtype=np.uint8)
        
        # Handle ground truth format
        if isinstance(gt_mask, dict):
            if 'mask' in gt_mask:
                gt_mask = gt_mask['mask']
            elif 'data' in gt_mask and 'foreground' in gt_mask['data']:
                # Convert polygon data to mask
                try:
                    foreground_data = gt_mask['data']['foreground']
                    if foreground_data and len(foreground_data) > 0:
                        
                        print(f"Frame {frame_idx}: Skipping polygon-format ground truth")
                        continue
                    else:
                        # Empty ground truth
                        gt_mask = np.zeros_like(algo_mask, dtype=np.uint8)
                except Exception as e:
                    print(f"Frame {frame_idx}: Error processing ground truth: {e}")
                    continue
            elif gt_mask.get('type', '').lower() == 'clear' or gt_mask.get('is_no_fluid', False):
                gt_mask = np.zeros_like(algo_mask, dtype=np.uint8)
        
        # Ensure both masks are numpy arrays
        if not isinstance(algo_mask, np.ndarray) or not isinstance(gt_mask, np.ndarray):
            print(f"Frame {frame_idx}: Invalid mask format - algo: {type(algo_mask)}, gt: {type(gt_mask)}")
            continue
        
        # Ensure same dimensions
        if algo_mask.shape != gt_mask.shape:
            print(f"Frame {frame_idx}: Shape mismatch - algo: {algo_mask.shape}, gt: {gt_mask.shape}")
            # Try to resize
            try:
                if len(algo_mask.shape) == 2 and len(gt_mask.shape) == 2:
                    import cv2
                    algo_mask = cv2.resize(algo_mask, (gt_mask.shape[1], gt_mask.shape[0]))
            except:
                print(f"Frame {frame_idx}: Could not resize masks, skipping")
                continue
        
        # Convert to binary masks
        try:
            algo_binary = (algo_mask > 0.5).astype(np.uint8)
            gt_binary = (gt_mask > 0.5).astype(np.uint8)
        except Exception as e:
            print(f"Frame {frame_idx}: Error converting to binary: {e}")
            continue
        
        # Calculate metrics
        try:
            iou = calculate_iou(algo_binary, gt_binary)
            dice = calculate_dice(algo_binary, gt_binary)
            
            if np.isnan(iou) or np.isnan(dice):
                print(f"Frame {frame_idx}: NaN metrics detected, skipping")
                continue
            
            ious.append(iou)
            dices.append(dice)
            processed_frames += 1
            
            if iou > 0.7:
                frames_with_high_iou += 1
                
            # Store per-frame metrics
            metrics['per_frame'][frame_idx] = {
                'iou': float(iou),
                'dice': float(dice)
            }
            
            # Print progress for some frames
            if processed_frames <= 10 or processed_frames % 10 == 0:
                print(f"Frame {frame_idx}: IoU = {iou:.4f}, Dice = {dice:.4f}")
                
        except Exception as e:
            print(f"Frame {frame_idx}: Error calculating metrics: {e}")
            continue
    
    # Calculate summary metrics
    if ious and len(ious) > 0:
        metrics['summary'] = {
            'mean_iou': float(np.mean(ious)),
            'median_iou': float(np.median(ious)),
            'mean_dice': float(np.mean(dices)),
            'median_dice': float(np.median(dices)),
            'frames_with_iou_over_0.7': float(frames_with_high_iou / len(ious)) if len(ious) > 0 else 0.0,
            'total_frames_evaluated': len(ious),
            'total_ground_truth_frames': len(ground_truth_frames),
            'processed_frames': processed_frames
        }
        
        print("\nFinal Metrics:")
        print(f"Processed frames: {processed_frames}")
        print(f"Mean IoU: {metrics['summary']['mean_iou']:.4f}")
        print(f"Mean Dice: {metrics['summary']['mean_dice']:.4f}")
        print(f"Frames with IoU > 0.7: {metrics['summary']['frames_with_iou_over_0.7']*100:.1f}%")
    else:
        print("WARNING: No valid metrics could be calculated")
        metrics['summary'] = {
            'mean_iou': 0.0,
            'median_iou': 0.0,
            'mean_dice': 0.0,
            'median_dice': 0.0,
            'frames_with_iou_over_0.7': 0.0,
            'total_frames_evaluated': 0,
            'total_ground_truth_frames': len(ground_truth_frames),
            'processed_frames': processed_frames,
            'error': 'No valid metrics could be calculated'
        }
    
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
        dataset_id = DATASET_ID  
        
    try:
        print("\nPreparing annotations for MD.ai upload...")
        
        # Format annotations according to MD.ai schema
        annotations = []
        
        
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
        
        # NEW TRACKING VALIDATION CODE 
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
                                         label_id_fluid=None, label_id_no_fluid=None, label_id_machine=None,
                                         project_id=None, dataset_id=None,  
                                         upload_to_mdai=False, debug_mode=False):
    """
    Process a video using multi-frame tracking and optionally upload to MD.ai.
    Enhanced version with improved feedback loop support.
    
    Args:
        video_path: Path to the video file
        annotations_df: DataFrame containing annotations
        study_uid: Study Instance UID
        series_uid: Series Instance UID
        flow_processor: Optical flow processor
        output_dir: Directory to save outputs
        annotations_json: JSON dictionary containing all annotations data
        mdai_client: MD.ai client for uploads
        label_id_fluid: Label ID for fluid annotation
        label_id_no_fluid: Label ID for no fluid annotation
        label_id_machine: Label ID for the machine group
        project_id: MD.ai project ID
        dataset_id: MD.ai dataset ID
        upload_to_mdai: Whether to upload to MD.ai
        debug_mode: Enable debug mode
        
    Returns:
        Dictionary with results
    """
    import os
    import cv2
    import numpy as np
    import pandas as pd
    import traceback
    from datetime import datetime
    import mdai
    
    print("\n==== STARTING ENHANCED MULTI-FRAME TRACKING PROCESS ====")
    print(f"Video path: {video_path}")
    print(f"Annotations count: {len(annotations_df)}")
    print(f"Output directory: {output_dir}")
    print(f"Flow processor method: {flow_processor.method}")
    print(f"Debug mode: {debug_mode}")
    print(f"Label IDs: fluid={label_id_fluid}, no_fluid={label_id_no_fluid}, machine={label_id_machine}")
    
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
        from src.multi_frame_tracking.utils import find_exam_number
        exam_number = find_exam_number(study_uid, annotations_json)
        print(f"Processing Exam #{exam_number}")
        with open(checkpoint_file, "a") as f:
           f.write(f"Exam Number: {exam_number}\n")
    except Exception as e:
        print(f"Could not determine exam number: {str(e)}")
        exam_number = "unknown"

    # Initialize multi-frame tracker
    from src.multi_frame_tracking.multi_frame_tracker import MultiFrameTracker
    tracker = MultiFrameTracker(flow_processor, output_dir, debug_mode=debug_mode)
    
    # Enable feedback loop mode
    tracker.feedback_loop_mode = True 
    tracker.learning_mode = True  # Also enable learning mode
    
    print(f"Feedback loop mode: {'enabled' if tracker.feedback_loop_mode else 'disabled'}")
    print(f"Learning mode: {'enabled' if tracker.learning_mode else 'disabled'}")

    # Debug: Check annotation dataframe structure
    print("\nAnnotation dataframe structure check:")
    print(f"Columns: {list(annotations_df.columns)}")
    print(f"First row sample: {annotations_df.iloc[0].to_dict() if len(annotations_df) > 0 else 'No annotations'}")
    
    # Check for ground truth vs. free fluid annotations
    has_ground_truth = any(row.get('labelId') == os.getenv("LABEL_ID_GROUND_TRUTH") for _, row in annotations_df.iterrows())
    print(f"Contains ground truth annotations: {has_ground_truth}")
    
    # Add automatic classification for "no fluid" annotations
    def is_no_fluid(row):
        if row.get('labelId') == label_id_no_fluid:
            return True
        
        # Check if there's polygon data in free_fluid_foreground
        if 'free_fluid_foreground' in row:
            polygons = row['free_fluid_foreground']
            if not polygons or len(polygons) == 0:
                return True
            
        return False
    
    # Mark no fluid annotations
    annotations_df['is_no_fluid'] = annotations_df.apply(is_no_fluid, axis=1)
    print(f"Identified {annotations_df['is_no_fluid'].sum()} no-fluid annotations")

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
            # Delete existing machine annotations first
            from src.multi_frame_tracking.utils import delete_existing_annotations
            
            from src.multi_frame_tracking.utils import delete_existing_annotations
            
            deleted_count = delete_existing_annotations(
                client=mdai_client,
                study_uid=study_uid,
                series_uid=series_uid,
                label_id=label_id_fluid,
                group_id=label_id_machine
            )
            
            print(f"Deleted {deleted_count} existing fluid annotations")
            
            # Also delete no-fluid annotations
            if label_id_no_fluid:
                deleted_no_fluid = delete_existing_annotations(
                    client=mdai_client,
                    study_uid=study_uid,
                    series_uid=series_uid,
                    label_id=label_id_no_fluid,
                    group_id=label_id_machine
                )
                print(f"Deleted {deleted_no_fluid} existing no-fluid annotations")
            
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
                is_clear = 'clear' in mask_type.lower() if mask_type else False
                
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
                        
                        # Add timestamp and source information
                        annotation['createdAt'] = datetime.now().isoformat()
                        
                        if isinstance(mask_info, dict) and 'source' in mask_info:
                            annotation['note'] = f"Source: {mask_info['source']}"
                            
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
                                
                                # Add timestamp and source information
                                annotation['createdAt'] = datetime.now().isoformat()
                                
                                if isinstance(mask_info, dict) and 'source' in mask_info:
                                    annotation['note'] = f"Source: {mask_info['source']}"
                                
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
           print(f"Could not determine exam number: {str(e)}")
    
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
        'exam_number': exam_number
    }
    

def process_videos_with_tracking():
    """
    Main processing loop for tracking free fluid in ultrasound videos.
    Modified with checkpoints to debug hanging issues.
    """
    checkpoint("FUNCTION_START", "process_videos_with_tracking started")
    
    # Create timestamp file in the root directory
    with open("multi_frame_test.txt", "w") as f:
        f.write(f"Test started at {datetime.now()}\n")
        f.write(f"TRACKING_MODE = {TRACKING_MODE}\n")
        f.write(f"MULTI_FRAME_AVAILABLE = {MULTI_FRAME_AVAILABLE}\n")
    
    checkpoint("TIMESTAMP_FILE", "Created timestamp file")
    
    print("\n==== MULTI-FRAME TRACKING TEST ====")
    print(f"TRACKING_MODE: {TRACKING_MODE}")
    print(f"MULTI_FRAME_AVAILABLE: {MULTI_FRAME_AVAILABLE}")

    if 'LABEL_IDS' not in globals():
        LABEL_IDS = {
        "disappear_reappear": os.getenv("LABEL_ID_DISAPPEAR_REAPPEAR"),
        "branching_fluid": os.getenv("LABEL_ID_BRANCHING_FLUID"),
        "multiple_distinct": os.getenv("LABEL_ID_MULTIPLE_DISTINCT")
    }
    
    # Determine which issue types to process
    issue_types_to_process = DEBUG_ISSUE_TYPES if DEBUG_MODE else list(LABEL_IDS.keys())
    print(f"Processing issue types: {issue_types_to_process}")
    
    checkpoint("ISSUE_TYPES", f"Processing {len(issue_types_to_process)} issue types")
    
    # Create output directories
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(MULTI_FRAME_DIR, exist_ok=True)
    
    checkpoint("DIRECTORIES", "Created output directories")
    
    # Initialize video counter
    videos_processed = 0
    
    # Initialize optical flow processor
    print("Initializing optical flow processor...")
    flow_processor = OpticalFlowProcessor(method=FLOW_METHOD[0])
    
    checkpoint("FLOW_PROCESSOR", f"Initialized flow processor with method: {FLOW_METHOD[0]}")
    
    # Process each issue type
    for issue_type in issue_types_to_process:
        checkpoint("ISSUE_TYPE_START", f"Starting issue type: {issue_type}")
        
        print(f"\nProcessing {issue_type} annotations...")
        
        # Filter annotations for this issue type
        type_annotations = matched_annotations[matched_annotations['issue_type'] == issue_type]
        
        # Limit processing to debug sample size if in debug mode
        if DEBUG_MODE:
            type_annotations = type_annotations.head(DEBUG_SAMPLE_SIZE)
        
        checkpoint("FILTERED_ANNOTATIONS", f"Found {len(type_annotations)} annotations for {issue_type}")
        
        print(f"Found {len(type_annotations)} annotations for {issue_type}")
        
        # Process each annotation
        for idx, row in tqdm(type_annotations.iterrows(), total=len(type_annotations)):
            checkpoint("VIDEO_START", f"Starting video {idx+1}/{len(type_annotations)} (Total: {videos_processed+1})")
            
            video_path = row['video_path']
            study_uid = row['StudyInstanceUID']
            series_uid = row['SeriesInstanceUID']
            
            checkpoint("VIDEO_DETAILS", f"Video: {os.path.basename(video_path)}")
            
            # Find any frame annotation for this video
            video_annotations = free_fluid_annotations[
                (free_fluid_annotations['StudyInstanceUID'] == study_uid) &
                (free_fluid_annotations['SeriesInstanceUID'] == series_uid)
            ]
            
            checkpoint("VIDEO_ANNOTATIONS", f"Found {len(video_annotations)} annotations for video")

            if len(video_annotations) > 0:
                # Use the first available annotation
                row = video_annotations.iloc[0]
                frame_number = int(row['frameNumber'])
                
                # Get free fluid polygons from this specific annotation
                free_fluid_polygons = row['free_fluid_foreground']
                
                checkpoint("ANNOTATION_DETAILS", f"Frame number: {frame_number}")
                
                print(f"\nProcessing video {videos_processed + 1}/{len(type_annotations)}")
                print(f"Video: {video_path}")
                print(f"Frame number: {frame_number}")
            else:
                checkpoint("NO_ANNOTATIONS", "No annotations found, skipping video")
                continue
            
            # Continue only if we have valid polygons
            if not isinstance(free_fluid_polygons, list) or len(free_fluid_polygons) == 0:
                checkpoint("NO_POLYGONS", "No valid polygons found, skipping video")
                continue
            
            try:
                checkpoint("VIDEO_OPEN", "Opening video file")
                cap = cv2.VideoCapture(video_path)
                if not cap.isOpened():
                    checkpoint("VIDEO_ERROR", f"Could not open video")
                    continue
                
                # Get video properties
                frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                
                checkpoint("VIDEO_PROPS", f"Resolution: {frame_width}x{frame_height}, Total frames: {total_frames}")
                
                print(f"Video dimensions: {frame_width}x{frame_height}")
                print(f"Total frames: {total_frames}")
                
                # Create initial mask from polygons
                initial_mask = polygons_to_mask(free_fluid_polygons, frame_height, frame_width)
                
                checkpoint("INITIAL_MASK", f"Created initial mask, sum: {np.sum(initial_mask)}")
                
                # Create output directory for this video
                multi_frame_output_dir = os.path.join(MULTI_FRAME_DIR, f"{study_uid}_{series_uid}")
                os.makedirs(multi_frame_output_dir, exist_ok=True)
                debug_dir = os.path.join(multi_frame_output_dir, 'debug')
                checkpoint("OUTPUT_DIR", f"Created output directory")
                
                # Get all annotations for this video
                video_annotations = free_fluid_annotations[
                    (free_fluid_annotations['StudyInstanceUID'] == study_uid) &
                    (free_fluid_annotations['SeriesInstanceUID'] == series_uid)
                ].copy()
                
                print(f"Found {len(video_annotations)} annotations for this video")
                
                checkpoint("ALL_ANNOTATIONS", f"Collected {len(video_annotations)} annotations")
                
                # Log function call details
                with open(os.path.join(multi_frame_output_dir, "function_call_log.txt"), "w") as f:
                    f.write(f"About to call function at {datetime.now()}\n")
                    f.write(f"Video path: {video_path}\n")
                    f.write(f"Annotations count: {len(video_annotations)}\n")
                    f.write(f"Frame dimensions: {frame_width}x{frame_height}\n")
                    f.write(f"Initial mask sum: {np.sum(initial_mask)}\n")
                
                checkpoint("PRE_FUNCTION_CALL", "About to call process_video_with_multi_frame_tracking_enhanced")
                
                print("\n##############################################")
                print("# ABOUT TO CALL MULTI-FRAME TRACKING FUNCTION #")
                print("##############################################")
                print(f"Flow processor method: {flow_processor.method}")
                print(f"Study/Series: {study_uid}/{series_uid}")
                
                try:
                    # CRITICAL POINT - Add a timeout for this specific function call
                    checkpoint("DURING_FUNCTION_CALL", "Inside process_video_with_multi_frame_tracking_enhanced call")
                    
                    # Start a timer for this function call
                    function_start_time = time.time()
                    
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
                        debug_mode=False
                    )
                    
                    function_duration = time.time() - function_start_time
                    checkpoint("POST_FUNCTION_CALL", f"Multi-frame function completed in {function_duration:.2f}s")
                    
                    # Log successful completion
                    with open(os.path.join(multi_frame_output_dir, "function_call_log.txt"), "a") as f:
                        f.write(f"Function completed at {datetime.now()}\n")
                        f.write(f"Duration: {function_duration:.2f}s\n")
                        f.write(f"Result type: {type(result)}\n")
                        if result:
                            f.write(f"Result keys: {result.keys()}\n")
                            f.write(f"Annotated frames: {result.get('annotated_frames', 0)}\n")
                            f.write(f"Predicted frames: {result.get('predicted_frames', 0)}\n")
                    
                    checkpoint("RESULT_LOGGED", "Results logged successfully")
                    
                    print("\nMulti-frame tracking completed successfully!")
                    print(f"Annotated frames: {result.get('annotated_frames', 0)}")
                    print(f"Predicted frames: {result.get('predicted_frames', 0)}")
                    
                    if result and 'md_ai_uploads' in result:
                        print(f"MD.ai upload status: {result['md_ai_uploads']}")
                        print(f"Total annotations uploaded: {result.get('uploaded_count', 0)}")
                    
                    # Verify output video exists
                    output_video = result.get('output_video', 'None')
                    if output_video and os.path.exists(output_video):
                        checkpoint("VIDEO_VERIFIED", f"Output video created: {os.path.getsize(output_video)} bytes")
                        print(f"Confirmed: Output video created at {output_video} ({os.path.getsize(output_video)} bytes)")
                    else:
                        checkpoint("VIDEO_MISSING", f"WARNING: Output video not found")
                        print(f"WARNING: Output video not found at expected path: {output_video}")
                    
                except Exception as e:
                    checkpoint("FUNCTION_ERROR", f"ERROR in multi-frame function: {str(e)}")
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
                checkpoint("VIDEO_COMPLETE", f"Completed video {videos_processed}")
                
                # Close video file
                cap.release()
                cleanup_memory()  # Add this
                cleanup_debug_files(debug_dir) 
                checkpoint("VIDEO_RELEASED", "Video capture released")
                
            except Exception as e:
                checkpoint("VIDEO_PROCESSING_ERROR", f"Error processing video: {str(e)}")
                print(f"Error processing video: {str(e)}")
                traceback.print_exc()
                continue
        
        checkpoint("ISSUE_TYPE_COMPLETE", f"Completed issue type: {issue_type}")
    
    checkpoint("ALL_COMPLETE", "All processing completed")
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
    

def preprocess_ground_truth_annotations(ground_truth_annotations, video_path, study_uid, series_uid):
    """
    Apply the same preprocessing to ground truth that normal annotations get
    """
    print("Preprocessing ground truth annotations...")
    
    # Convert list of annotation dicts to DataFrame (same as normal processing)
    gt_df = pd.DataFrame(ground_truth_annotations)
    
    # Extract foreground data from the nested structure (same as normal processing)
    # This mimics what happens during normal annotation loading
    
    # Extract the foreground polygons from data field
    def extract_foreground(row):
        if 'data' in row and isinstance(row['data'], dict):
            # Check if this is for exam 91
            study_uid = row.get('StudyInstanceUID')
            if study_uid == "1.2.826.0.1.3680043.8.498.21582572478922879563110991046360588727":
                frame_number = row.get('frameNumber', 'unknown')
                foreground = row['data'].get('foreground', [])
                print(f"DEBUG Exam 91: Frame {frame_number}, foreground len: {len(foreground)}")
            return row['data'].get('foreground', [])
        return []
    
    gt_df['data_foreground'] = gt_df.apply(extract_foreground, axis=1)
    
    # Rename to match what tracker expects (same as normal processing)
    gt_df.rename(columns={'data_foreground': 'free_fluid_foreground'}, inplace=True)
    
    # Add video path and other required fields
    gt_df['video_path'] = video_path
    gt_df['StudyInstanceUID'] = study_uid
    gt_df['SeriesInstanceUID'] = series_uid
    
    # Filter out annotations without valid foreground data
    gt_df = gt_df[gt_df['free_fluid_foreground'].apply(lambda x: x and len(x) > 0)]
    
    print(f"Preprocessed {len(gt_df)} ground truth annotations")
    print(f"Columns: {list(gt_df.columns)}")
    
    # Debug: check a sample
    if len(gt_df) > 0:
        sample = gt_df.iloc[0]
        print(f"Sample frame {sample['frameNumber']}:")
        print(f"  free_fluid_foreground type: {type(sample['free_fluid_foreground'])}")
        print(f"  Has polygon data: {len(sample['free_fluid_foreground']) > 0}")
    
    return gt_df

def preprocess_ground_truth_for_tracker(ground_truth_annotations, video_path, study_uid, series_uid, feedback_loop=True, input_sampling_rate=10):
    """
    Apply the same preprocessing to ground truth that normal annotations get
    
    Args:
        ground_truth_annotations: List of ground truth annotations
        video_path: Path to the video file
        study_uid: Study instance UID
        series_uid: Series instance UID
        feedback_loop: Whether this is being used in feedback loop mode
        input_sampling_rate: Take every Nth frame as INPUT for tracking (NOT for evaluation)
    """
    print("Preprocessing ground truth annotations...")
    
    # Convert list of annotation dicts to DataFrame
    gt_df = pd.DataFrame(ground_truth_annotations)
    
    # Extract foreground data (same as before)
    def extract_foreground(row):
        if 'data' in row and isinstance(row['data'], dict):
            return row['data'].get('foreground', [])
        return []
    
    gt_df['data_foreground'] = gt_df.apply(extract_foreground, axis=1)
    gt_df.rename(columns={'data_foreground': 'free_fluid_foreground'}, inplace=True)
    
    # Add video path and other required fields
    gt_df['video_path'] = video_path
    gt_df['StudyInstanceUID'] = study_uid
    gt_df['SeriesInstanceUID'] = series_uid
    
    # Filter out annotations without valid foreground data
    gt_df = gt_df[gt_df['free_fluid_foreground'].apply(lambda x: x and len(x) > 0)]
    
    # CRITICAL: Apply input sampling for sparse INPUT testing
    if feedback_loop and input_sampling_rate > 1:
        # Sort by frame number
        gt_df = gt_df.sort_values('frameNumber')
        frame_numbers = gt_df['frameNumber'].unique()
        
        # Take every Nth frame for INPUT
        sampled_frames = frame_numbers[::input_sampling_rate]
        print(f"\n🔍 INPUT SAMPLING DEBUG:")
        print(f"Input sampling rate: {input_sampling_rate}")
        print(f"Original frames available: {len(frame_numbers)}")
        print(f"Frames given to tracker as INPUT: {len(sampled_frames)}")
        print(f"Input frame numbers: {sorted(sampled_frames)}")
        print(f"Average gap between input frames: {np.mean(np.diff(sorted(sampled_frames))):.1f}")
        
        # Filter to only use sampled frames as input
        gt_df = gt_df[gt_df['frameNumber'].isin(sampled_frames)]
    
    print(f"Final preprocessed annotations for tracker INPUT: {len(gt_df)}")
    
    return gt_df

def extract_algorithm_masks_for_evaluation(algorithm_masks, ground_truth_masks):
    """
    Function to extract algorithm masks and align them properly with ground truth,
    handling MD.ai 1-indexed frames correctly.
    
    Args:
        algorithm_masks: Dictionary of masks from the tracking process
        ground_truth_masks: Dictionary of ground truth masks to align with
        
    Returns:
        Dictionary of aligned algorithm masks suitable for evaluation
    """
    import numpy as np
    
    print("\n=== ADVANCED ALGORITHM MASK EXTRACTION WITH INDEX ALIGNMENT ===")
    print(f"Algorithm masks before processing: {len(algorithm_masks)}")
    print(f"Ground truth masks: {len(ground_truth_masks)}")
    
    # Get frame indices from both sources
    algo_frames = sorted(list(algorithm_masks.keys()))
    gt_frames = sorted(list(ground_truth_masks.keys()))
    
    print(f"Algorithm frame range: {min(algo_frames)} to {max(algo_frames)}")
    print(f"Ground truth frame range: {min(gt_frames)} to {max(gt_frames)}")
    
    # Check for MD.ai 1-indexing vs 0-indexing
    mdai_offset = 0
    # If ground truth starts at 1 but algorithm has frame 0, we might have an offset
    if 1 in gt_frames and 0 in algo_frames and 1 not in algo_frames:
        print("Detected potential MD.ai 1-indexing vs code 0-indexing mismatch")
        mdai_offset = 1
        print(f"Will apply +{mdai_offset} offset to algorithm frames")
    
    # Create aligned algorithm masks
    aligned_masks = {}
    skipped_frames = 0
    processed_frames = 0
    
    # Process each algorithm mask, applying the offset if needed
    for frame_idx, mask_info in sorted(algorithm_masks.items()):
        # Extract mask data
        if isinstance(mask_info, dict):
            mask = mask_info.get('mask')
            mask_type = mask_info.get('type', 'unknown')
            
            if mask is None:
                skipped_frames += 1
                continue
        else:
            mask = mask_info
            mask_type = 'direct'
        
        # Skip clear frames (no fluid)
        if isinstance(mask_info, dict) and 'clear' in mask_type.lower():
            if np.sum(mask) == 0:
                # This is a valid clear frame with no content
                # For evaluation, include empty masks for clear frames
                aligned_masks[frame_idx + mdai_offset] = np.zeros_like(mask)
                processed_frames += 1
                continue
        
        # Include ALL masks, even if they have no content
        if isinstance(mask, np.ndarray):
            # Ensure mask is binary
            if mask.dtype != np.uint8:
                mask = (mask > 0.5).astype(np.uint8)
            
            # Apply MD.ai offset adjustment when saving
            aligned_idx = frame_idx + mdai_offset
            aligned_masks[aligned_idx] = mask
            processed_frames += 1
            
            # Debug output for first few processed frames
            if processed_frames <= 20 or np.sum(mask) > 0:
                print(f"✓ Processed frame {frame_idx} → {aligned_idx}: type={mask_type}, sum={np.sum(mask)}")
                
            # AGGRESSIVE: If this frame is in the middle of ground truth range but not in ground truth keys,
            # also try other nearby offsets
            if gt_frames and min(gt_frames) < aligned_idx < max(gt_frames) and aligned_idx not in ground_truth_masks:
                for offset_adjust in [-1, 1, -2, 2]:
                    alt_idx = aligned_idx + offset_adjust
                    if alt_idx in ground_truth_masks and alt_idx not in aligned_masks:
                        aligned_masks[alt_idx] = mask
                        print(f"✦ Added alternate frame {alt_idx} from {frame_idx} to match ground truth")
        else:
            skipped_frames += 1
    
    print(f"\nAlignment summary:")
    print(f"  - Original algorithm frames: {len(algorithm_masks)}")
    print(f"  - Frames processed: {processed_frames}")
    print(f"  - Frames skipped: {skipped_frames}")
    print(f"  - Final aligned masks: {len(aligned_masks)}")
    
    # Verify overlap with ground truth
    common_frames = set(aligned_masks.keys()) & set(ground_truth_masks.keys())
    print(f"Frames matching ground truth: {len(common_frames)} out of {len(ground_truth_masks)}")
    
    # If poor overlap, try other alignments
    if len(common_frames) < min(len(aligned_masks), len(ground_truth_masks)) * 0.5:
        print("WARNING: Poor overlap with ground truth. Trying alternative alignments...")
        
        best_offset = mdai_offset
        best_overlap = len(common_frames)
        
        # Try different offsets
        for test_offset in range(-3, 4):
            if test_offset == mdai_offset:
                continue  # Skip the one we already tried
                
            test_aligned = {frame_idx + test_offset: mask for frame_idx, mask in algorithm_masks.items() 
                         if isinstance(mask, np.ndarray) and np.sum(mask > 0.5) > 0}
            
            test_common = set(test_aligned.keys()) & set(ground_truth_masks.keys())
            print(f"  Offset {test_offset}: {len(test_common)} common frames")
            
            if len(test_common) > best_overlap:
                best_offset = test_offset
                best_overlap = len(test_common)
        
        # If we found a better offset, use it
        if best_offset != mdai_offset:
            print(f"Using improved offset {best_offset} with {best_overlap} common frames")
            
            # Recreate aligned masks with new offset
            aligned_masks = {}
            for frame_idx, mask_info in algorithm_masks.items():
                # Extract mask
                if isinstance(mask_info, dict):
                    mask = mask_info.get('mask')
                    if mask is None:
                        continue
                else:
                    mask = mask_info
                
                # Only include masks with content
                if isinstance(mask, np.ndarray) and np.sum(mask > 0.5) > 0:
                    # Ensure mask is binary
                    if mask.dtype != np.uint8:
                        mask = (mask > 0.5).astype(np.uint8)
                    
                    # Apply new best offset
                    aligned_idx = frame_idx + best_offset
                    aligned_masks[aligned_idx] = mask
            
            # Verify final overlap
            common_frames = set(aligned_masks.keys()) & set(ground_truth_masks.keys())
            print(f"Final frames matching ground truth: {len(common_frames)} out of {len(ground_truth_masks)}")
    
    return aligned_masks

def extract_algorithm_masks_only(algorithm_masks, ground_truth_indices=None):
    """
    Extract ONLY algorithm-generated masks for evaluation (no ground truth)
    Make sure to properly handle different frame indexing schemes to maximize overlap with ground truth
    
    Args:
        algorithm_masks: Dictionary of algorithm masks
        ground_truth_indices: Optional list of ground truth frame indices to target specifically
    """
    print("\n=== ENHANCED ALGORITHM MASK EXTRACTION FOR EVALUATION ===")
    algorithm_masks_clean = {}
    
    # Track different mask types for debugging
    mask_types = {}
    original_frames = []
    
    # First extract all valid masks using the original indexing
    for frame_idx, mask_info in algorithm_masks.items():
        # Skip annotations (they're ground truth)
        if isinstance(mask_info, dict) and mask_info.get('is_annotation', False):
            continue
        
        # Extract the actual mask
        if isinstance(mask_info, dict):
            mask = mask_info.get('mask')
            mask_type = mask_info.get('type', '')
            
            # Count mask types
            mask_types[mask_type] = mask_types.get(mask_type, 0) + 1
            
            if mask is None:
                continue
        else:
            mask = mask_info
            mask_type = 'direct'
        
        # Ensure it's a numpy array with content
        if isinstance(mask, np.ndarray):
            # Convert to binary if needed
            if mask.dtype != np.uint8:
                mask = (mask > 0.5).astype(np.uint8)
                
            # Add the original frame index
            algorithm_masks_clean[frame_idx] = mask
            original_frames.append(frame_idx)
                
            # Debug output for the first few frames
            if len(original_frames) < 20 or np.sum(mask) > 0:
                print(f"✓ Algorithm frame {frame_idx} → mask sum: {np.sum(mask)} ({mask_type})")
    
   
    # First, try the standard +1 offset from MD.ai
    mdai_frames = []
    for frame_idx in original_frames:
        # Add the +1 offset version (standard MD.ai offset)
        mdai_frame_idx = frame_idx + 1
        
        # Don't override existing frames
        if mdai_frame_idx not in algorithm_masks_clean:
            algorithm_masks_clean[mdai_frame_idx] = algorithm_masks_clean[frame_idx].copy()
            mdai_frames.append(mdai_frame_idx)
            
            # Debug output for the first few frames
            if len(mdai_frames) < 20:
                print(f"+ Added MD.ai offset frame {mdai_frame_idx} (from {frame_idx})")
    
    # If ground truth indices are provided, ensure we have masks for those specific indices
    gt_mapped_frames = []
    if ground_truth_indices:
        print(f"\n=== TARGETING {len(ground_truth_indices)} GROUND TRUTH FRAMES ===")
        
        # For each ground truth index, find the closest algorithm frame and copy its mask
        for gt_idx in ground_truth_indices:
            if gt_idx in algorithm_masks_clean:
                # Already have this frame
                continue
                
            # First look for original or offset frames close to this index
            closest_frame = None
            min_distance = float('inf')
            
            for alg_idx in original_frames:
                # Check for small offsets around algorithm frame
                for offset in [0, -1, 1, -2, 2]:
                    if alg_idx + offset == gt_idx:
                        # Found a match with an offset
                        closest_frame = alg_idx
                        min_distance = abs(offset)
                        break
                
                # If an exact match was found, no need to keep searching
                if min_distance == 0:
                    break
                    
                # Otherwise check if this is the closest so far
                distance = abs(alg_idx - gt_idx)
                if distance < min_distance:
                    min_distance = distance
                    closest_frame = alg_idx
            
            # If we found a reasonably close frame, use its mask
            if closest_frame is not None and min_distance <= 5:  # Only use frames that are within 5 indices
                algorithm_masks_clean[gt_idx] = algorithm_masks_clean[closest_frame].copy()
                gt_mapped_frames.append(gt_idx)
                
                print(f"* Mapped ground truth frame {gt_idx} from algorithm frame {closest_frame} (distance {min_distance})")
    
    # Print detailed summary of extracted masks
    print(f"\nExtracted {len(algorithm_masks_clean)} algorithm masks for evaluation")
    print(f"  - Original frames: {len(original_frames)}")
    print(f"  - MD.ai offset frames: {len(mdai_frames)}")
    if ground_truth_indices:
        print(f"  - Ground truth mapped frames: {len(gt_mapped_frames)}")
        print(f"  - Ground truth coverage: {len(set(ground_truth_indices) & set(algorithm_masks_clean.keys()))} / {len(ground_truth_indices)}")
    
    # Print mask type distribution
    print("\nMask types found:")
    for mask_type, count in mask_types.items():
        print(f"  - {mask_type}: {count}")
    
    # Print frame range information
    if original_frames:
        print(f"\nFrame range (original): {min(original_frames)} to {max(original_frames)}")
    if mdai_frames:
        print(f"Frame range (MD.ai): {min(mdai_frames)} to {max(mdai_frames)}")
    
    # Check for overlapping frames with 1-offset between original and MD.ai frames
    overlap_check = set([f+1 for f in original_frames]) & set(original_frames)
    if overlap_check:
        print(f"\nWARNING: Found {len(overlap_check)} overlapping frames with 1-offset!")
        print(f"Overlap examples: {sorted(list(overlap_check)[:5])}")
    
    print("=== END ENHANCED ALGORITHM MASK EXTRACTION ===\n")
    
    return algorithm_masks_clean
def create_sparse_annotations(ground_truth_annotations, sampling_rate=10, min_frames=1):
    """
    Create a sparse subset of annotations by sampling frames at regular intervals.
    
    Args:
        ground_truth_annotations: Original ground truth annotations (list or dictionary)
        sampling_rate: Take every Nth frame (default: 10)
        min_frames: Minimum number of frames to include (default: 1)
        
    Returns:
        Tuple of (sparse_annotations, sampling_info)
        - sparse_annotations: Dictionary of sampled annotations
        - sampling_info: Dictionary containing sampling statistics
    """
    if not ground_truth_annotations:
        return {}, {"error": "No annotations provided"}
        
    print(f"\n=== CREATING SPARSE ANNOTATIONS (sampling rate: {sampling_rate}) ===")
    
    # Convert list to dictionary if needed
    if isinstance(ground_truth_annotations, list):
        # Create dictionary with frame numbers as keys
        annotations_dict = {}
        for ann in ground_truth_annotations:
            frame_num = int(ann.get('frameNumber', 0))
            if frame_num not in annotations_dict:
                annotations_dict[frame_num] = ann
        ground_truth_annotations = annotations_dict
    
    # Get all frame indices
    frame_indices = sorted(list(ground_truth_annotations.keys()))
    total_frames = len(frame_indices)
    
    # Calculate step size based on sampling rate
    step = sampling_rate
    
    # Sample frames
    sampled_indices = frame_indices[::step]
    
    # Ensure we have at least min_frames
    if len(sampled_indices) < min_frames:
        print(f"Warning: Sampling rate {sampling_rate} would result in fewer than {min_frames} frames.")
        print(f"Adjusting sampling to ensure {min_frames} frames are included.")
        step = max(1, total_frames // min_frames)
        sampled_indices = frame_indices[::step]
    
    # Create sparse annotations dictionary
    sparse_annotations = {idx: ground_truth_annotations[idx] for idx in sampled_indices}
    
    # Create sampling information
    sampling_info = {
        "total_frames": total_frames,
        "sampled_frames": len(sparse_annotations),
        "sampling_rate": sampling_rate,
        "actual_step_size": step,
        "sampled_frame_numbers": sorted(list(sparse_annotations.keys())),
        "sampling_percentage": (len(sparse_annotations) / total_frames) * 100
    }
    
    print("\nSampling Summary:")
    print(f"Total frames available: {total_frames}")
    print(f"Frames sampled: {len(sparse_annotations)}")
    print(f"Sampling rate used: {sampling_rate}")
    print(f"Actual step size: {step}")
    print(f"Sampled frame numbers: {sorted(list(sparse_annotations.keys()))}")
    print(f"Sampling percentage: {sampling_info['sampling_percentage']:.1f}%")
    
    return sparse_annotations, sampling_info


def evaluate_with_expert_feedback(video_paths, study_series_pairs, flow_processor, output_dir,
                                 mdai_client, project_id, dataset_id, ground_truth_label_id,
                                 algorithm_label_id, label_id_no_fluid=None, 
                                 label_id_machine=None, annotations_json=None, args=None,
                                 shared_params=None, learning_mode=False, iteration_number=1,
                                 use_genuine_evaluation=False, 
                                 sampling_rate=10,              
                                 input_sampling_rate=None,      
                                 evaluation_sampling_rate=None,
                                 include_method_comparison=False): 
    """
    Evaluates the algorithm against expert-refined ground truth annotations
    with support for iterative learning and feedback loop, and optional method comparison
    
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
        label_id_no_fluid: Label ID for "no fluid" annotations
        label_id_machine: Label ID for the machine group
        annotations_json: Path to annotations JSON file
        args: Command line arguments
        shared_params: SharedParams object for tracking parameters
        learning_mode: Whether to enable learning from corrections
        iteration_number: Current iteration number in the feedback loop
        use_genuine_evaluation: Whether to use the genuine evaluation approach
        sampling_rate: Legacy sampling rate (for backward compatibility)
        input_sampling_rate: Sampling rate for tracker INPUT
        evaluation_sampling_rate: Sampling rate for evaluation
        include_method_comparison: Whether to run single vs multi-frame comparison
        
    Returns:
        Dictionary with evaluation results
    """
    
    # Handle parameter compatibility
    if input_sampling_rate is None:
        input_sampling_rate = sampling_rate
    if evaluation_sampling_rate is None:
        evaluation_sampling_rate = 1  # Default to dense evaluation
    
    # Debug output at the beginning
    print("\n" + "="*60)
    print("=== LABEL ID DEBUG IN evaluate_with_expert_feedback ===")
    print(f"ground_truth_label_id: {ground_truth_label_id}")
    print(f"algorithm_label_id: {algorithm_label_id}")
    print(f"label_id_no_fluid: {label_id_no_fluid}")
    print(f"label_id_machine: {label_id_machine}")
    print(f"include_method_comparison: {include_method_comparison}")  
    print("="*60 + "\n")
    
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
            
            # ADD DEBUG CODE HERE
            print("\n=== DEBUGGING GROUND TRUTH ANNOTATIONS ===")
            print(f"Found {len(ground_truth_annotations)} ground truth annotations")
            if ground_truth_annotations:
                # Check the first annotation
                first_annotation = ground_truth_annotations[0]
                print(f"\nFirst annotation structure:")
                print(f"Keys: {list(first_annotation.keys())}")
                print(f"Label ID: {first_annotation.get('labelId')}")
                print(f"Frame Number: {first_annotation.get('frameNumber')}")
                print(f"Has 'data': {'data' in first_annotation}")
                
                if 'data' in first_annotation:
                    print(f"Data type: {type(first_annotation['data'])}")
                    print(f"Data content preview: {str(first_annotation['data'])[:200]}...")
                    if isinstance(first_annotation['data'], dict):
                        print(f"Data keys: {list(first_annotation['data'].keys())}")
                
            print("=== END GROUND TRUTH DEBUG ===\n")
            
            if not ground_truth_annotations:
                print(f"No ground truth annotations found for {study_uid}/{series_uid}")
                evaluation_results[f"{study_uid}_{series_uid}"] = {
                    'error': 'No ground truth annotations found'
                }
                continue
            
            # Convert video dimensions
            cap = cv2.VideoCapture(video_path)
            video_height = None
            video_width = None
            
            if cap.isOpened():
                ret, frame = cap.read()
                if ret:
                    video_height, video_width = frame.shape[:2]
                    print(f"Video dimensions: {video_width}x{video_height}")
                cap.release()
            else:
                print("Could not open video to get dimensions")
                # Use fallback dimensions
                video_height, video_width = 840, 1580
                print(f"Using fallback dimensions: {video_width}x{video_height}")
            
            # UPDATED MASK CONVERSION SECTION - Handle both list and DataFrame inputs
            ground_truth_masks = {}
            no_fluid_ground_truth_frames = set()  # Track no-fluid frames from ground truth
            
            print("\n=== PROCESSING GROUND TRUTH ANNOTATIONS ===")
            
            # Function to process a single annotation
            def process_annotation(annotation):
                try:
                    frame_num = int(annotation.get('frameNumber', 0))
                    
                    # Check if this is a no-fluid annotation
                    is_no_fluid = False
                    if 'data' in annotation and isinstance(annotation['data'], dict):
                        data = annotation['data']
                        if not data.get('foreground') and not data.get('free_fluid_foreground'):
                            is_no_fluid = True
                            print(f"Frame {frame_num}: No fluid annotation")
                    
                    # ENHANCED: Also check labelId for explicit no-fluid
                    if annotation.get('labelId') == label_id_no_fluid:
                        is_no_fluid = True
                        print(f"Frame {frame_num}: Explicit no-fluid label")
                    
                    if is_no_fluid:
                        # For no-fluid frames, create an empty mask
                        mask = np.zeros((video_height, video_width), dtype=np.uint8)
                        ground_truth_masks[frame_num] = {
                            'mask': mask,
                            'is_no_fluid': True,
                            'type': 'no_fluid'
                        }
                        no_fluid_ground_truth_frames.add(frame_num)  
                    else:
                        # Try to get polygons from different possible fields
                        polygons = None
                        if 'free_fluid_foreground' in annotation:
                            polygons = annotation['free_fluid_foreground']
                        elif 'data' in annotation and isinstance(annotation['data'], dict):
                            polygons = annotation['data'].get('foreground') or annotation['data'].get('free_fluid_foreground')
                        
                        if polygons and isinstance(polygons, list) and len(polygons) > 0:
                            mask = polygons_to_mask(polygons, video_height, video_width)
                            ground_truth_masks[frame_num] = mask
                            print(f"Frame {frame_num}: Created mask from {len(polygons)} polygons")
                        else:
                            print(f"Frame {frame_num}: No valid polygons found")
                except Exception as e:
                    print(f"Error processing annotation: {str(e)}")
            
            # Process annotations based on input type
            if isinstance(ground_truth_annotations, pd.DataFrame):
                print("Processing DataFrame annotations...")
                for _, row in ground_truth_annotations.iterrows():
                    process_annotation(row)
            else:
                print("Processing list annotations...")
                for annotation in ground_truth_annotations:
                    process_annotation(annotation)
            
            print(f"\nProcessed {len(ground_truth_masks)} ground truth frames")
            print(f"No-fluid ground truth frames: {len(no_fluid_ground_truth_frames)}")
            if no_fluid_ground_truth_frames:
                print(f"No-fluid frame numbers: {sorted(list(no_fluid_ground_truth_frames))}")
            
            # Convert annotations to DataFrame if needed
            if not isinstance(ground_truth_annotations, pd.DataFrame):
                print("Converting annotations to DataFrame...")
                annotations_df = pd.DataFrame(ground_truth_annotations)
            else:
                annotations_df = ground_truth_annotations
            
            # Preprocess for tracker
            annotations_df = preprocess_ground_truth_for_tracker(
                ground_truth_annotations, video_path, study_uid, series_uid, 
                feedback_loop=True, input_sampling_rate=input_sampling_rate
            )
            
            print(f"🔍 After preprocessing: DataFrame has {len(annotations_df)} rows")
            print(f"🔍 Input sampling rate used: {input_sampling_rate}")
            
            # Initialize the multi-frame tracker with feedback loop settings
            from src.multi_frame_tracking.multi_frame_tracker import MultiFrameTracker
            tracker = MultiFrameTracker(flow_processor, video_output_dir, debug_mode=False, shared_params=shared_params)
            
            # Enable feedback loop and learning mode if specified
            if learning_mode:
                tracker.feedback_loop_mode = True
                tracker.learning_mode = True
                print(f"✓ Feedback loop and learning mode enabled for iteration {iteration_number}")
            
            # Run multi-frame tracking (existing logic)
            multi_frame_results = tracker.process_annotations(annotations_df, video_path, study_uid, series_uid)
            
            # NEW: METHOD COMPARISON SECTION
            single_frame_results = None
            comparison_metrics = None
            
            if include_method_comparison:
                print(f"\n🔬 Running method comparison for {study_uid}/{series_uid}")
                
                try:
                    # Import the single frame tracker
                    from src.multi_frame_tracking.tracking_comparison import SingleFrameTracker, compare_single_vs_multi_frame
                    
                    # Run single-frame tracking
                    single_tracker = SingleFrameTracker(flow_processor, video_output_dir, shared_params=shared_params)
                    single_frame_results = single_tracker.process_single_annotation(
                        annotations_df, video_path, study_uid, series_uid
                    )
                    
                    # Compare the two methods
                    comparison_metrics = compare_single_vs_multi_frame(single_frame_results, multi_frame_results)
                    
                    # Print quick summary
                    print(f"   Single-frame predictions: {comparison_metrics.get('single_frame_count', 0)}")
                    print(f"   Multi-frame predictions: {comparison_metrics.get('multi_frame_count', 0)}")
                    print(f"   Method agreement IoU: {comparison_metrics.get('mean_iou', 0.0):.4f}")
                    
                except Exception as e:
                    print(f"⚠️  Error in method comparison: {str(e)}")
                    single_frame_results = None
                    comparison_metrics = {'error': str(e)}
            
            # Use multi_frame_results as the main algorithm_masks for the rest of the evaluation
            algorithm_masks = multi_frame_results
            
            # ENFORCE NO-FLUID CONSTRAINTS BEFORE ANY FURTHER PROCESSING
            print(f"\n🚫 ENFORCING NO-FLUID CONSTRAINTS")
            if no_fluid_ground_truth_frames:
                print(f"🚫 Ground truth no-fluid frames: {sorted(list(no_fluid_ground_truth_frames))}")
                
                # Create empty mask template
                empty_mask = np.zeros((video_height, video_width), dtype=np.uint8)
                violations_fixed = 0
                
                # Force all ground truth no-fluid frames to be empty in algorithm results
                for frame_idx in no_fluid_ground_truth_frames:
                    if frame_idx in algorithm_masks:
                        current_mask = algorithm_masks[frame_idx]
                        
                        # Check if current mask has content
                        current_sum = 0
                        if isinstance(current_mask, dict):
                            mask_data = current_mask.get('mask', empty_mask)
                            current_sum = np.sum(mask_data)
                        else:
                            current_sum = np.sum(current_mask)
                        
                        if current_sum > 0:
                            print(f"🚫 VIOLATION FIXED: Frame {frame_idx} had mask sum {current_sum}, now 0")
                            violations_fixed += 1
                    
                    # FORCE empty mask regardless
                    algorithm_masks[frame_idx] = {
                        'mask': empty_mask.copy(),
                        'type': 'enforced_no_fluid',
                        'source': 'no_fluid_constraint_enforcement',
                        'is_annotation': False,
                        'is_no_fluid': True,
                        'immutable': True
                    }
                
                print(f"🚫 Fixed {violations_fixed} no-fluid constraint violations")
            else:
                print("🚫 No ground truth no-fluid frames found")
            
            # Get frame ranges for alignment
            if algorithm_masks and ground_truth_masks:
                algo_frames = sorted(algorithm_masks.keys())
                gt_frames = sorted(ground_truth_masks.keys())
                print(f"\n=== FRAME ALIGNMENT ===")
                print(f"Algorithm frames: {len(algo_frames)} (range: {min(algo_frames)} to {max(algo_frames)})")
                print(f"Ground truth frames: {len(gt_frames)} (range: {min(gt_frames)} to {max(gt_frames)})")
                
                # Calculate optimal frame offset
                frame_offset = min(gt_frames) - min(algo_frames)
                print(f"Calculated frame offset: {frame_offset}")
                
                # Apply offset to algorithm masks
                algorithm_masks_aligned = {}
                for frame_idx, mask_info in algorithm_masks.items():
                    aligned_idx = frame_idx + frame_offset
                    if aligned_idx in ground_truth_masks:
                        algorithm_masks_aligned[aligned_idx] = mask_info
                
                algorithm_masks = algorithm_masks_aligned
                print(f"Aligned frames: {len(algorithm_masks)}")
            
            # Debug the tracker output
            print("\n=== DEBUGGING MULTIFRAME TRACKER OUTPUT ===")
            print(f"Total algorithm_masks returned: {len(algorithm_masks)}")
            
            # Enhanced debugging for algorithm masks
            print("\nAnalyzing algorithm masks structure:")
            mask_types = {}
            mask_content = {
                'total': 0,
                'with_content': 0,
                'empty': 0,
                'invalid': 0
            }
            
            for frame_idx, mask_info in algorithm_masks.items():
                try:
                    if isinstance(mask_info, dict):
                        mask_type = mask_info.get('type', 'unknown')
                        mask_types[mask_type] = mask_types.get(mask_type, 0) + 1
                        
                        if 'mask' in mask_info:
                            mask = mask_info['mask']
                            if isinstance(mask, np.ndarray):
                                mask_content['total'] += 1
                                if np.sum(mask) > 0:
                                    mask_content['with_content'] += 1
                                else:
                                    mask_content['empty'] += 1
                            else:
                                mask_content['invalid'] += 1
                    elif isinstance(mask_info, np.ndarray):
                        mask_types['direct_array'] = mask_types.get('direct_array', 0) + 1
                        mask_content['total'] += 1
                        if np.sum(mask_info) > 0:
                            mask_content['with_content'] += 1
                        else:
                            mask_content['empty'] += 1
                    else:
                        print(f"Warning: Unexpected mask type for frame {frame_idx}: {type(mask_info)}")
                        mask_content['invalid'] += 1
                except Exception as e:
                    print(f"Error analyzing mask for frame {frame_idx}: {str(e)}")
                    mask_content['invalid'] += 1
            
            print("\nMask types distribution:")
            for mask_type, count in mask_types.items():
                print(f"  {mask_type}: {count}")
            
            print("\nMask content analysis:")
            for key, value in mask_content.items():
                print(f"  {key}: {value}")
            
            print("\n=== END DEBUGGING ===")
            
            # Check what's being marked as annotations
            annotation_count = 0
            prediction_count = 0
            clear_frame_count = 0
            
            for frame_idx, mask_info in algorithm_masks.items():
                if isinstance(mask_info, dict):
                    is_annotation = mask_info.get('is_annotation', False)
                    mask_type = mask_info.get('type', 'unknown')
                    is_clear = 'clear' in mask_type.lower() if mask_type else False
                    
                    if is_clear:
                        clear_frame_count += 1
                    elif is_annotation:
                        annotation_count += 1
                    else:
                        prediction_count += 1
            
            print(f"\nSummary:")
            print(f"Annotations: {annotation_count}")
            print(f"Predictions: {prediction_count}")
            print(f"Clear frames: {clear_frame_count}")
            
            # Upload algorithm predictions to MD.ai (with option to skip)
            if algorithm_masks and not (args and hasattr(args, 'no_upload') and args.no_upload):
                print("Uploading algorithm predictions to MD.ai...")
                
                # Prepare annotations for upload
                annotations_to_upload = []
                for frame_idx, mask_info in algorithm_masks.items():
                    # Get the mask and type
                    if isinstance(mask_info, dict):
                        mask = mask_info['mask']
                        mask_type = mask_info.get('type', '')
                        is_no_fluid = mask_info.get('is_no_fluid', False)
                    else:
                        mask = mask_info
                        mask_type = 'unknown'
                        is_no_fluid = False
                    
                    # Handle clear/no-fluid frames
                    is_clear = 'clear' in mask_type.lower() if mask_type else False
                    
                    try:
                        if is_clear or is_no_fluid:
                            # Create no-fluid annotation
                            annotation = {
                                'labelId': label_id_no_fluid,  
                                'StudyInstanceUID': study_uid,
                                'SeriesInstanceUID': series_uid,
                                'frameNumber': int(frame_idx),
                                'groupId': label_id_machine,
                                'note': f'No fluid frame (iteration {iteration_number})'
                            }
                            annotations_to_upload.append(annotation)
                            print(f"✓ Added no-fluid annotation for frame {frame_idx}")
                        else:
                            # For fluid frames, check if mask has content
                            binary_mask = (mask > 0.5).astype(np.uint8)
                            if np.sum(binary_mask) > 0:
                                # Check - Don't upload fluid if it's a no-fluid ground truth frame
                                if frame_idx in no_fluid_ground_truth_frames:
                                    print(f"🚫 BLOCKED: Preventing fluid annotation on no-fluid ground truth frame {frame_idx}")
                                    # Convert to no-fluid annotation instead
                                    annotation = {
                                        'labelId': label_id_no_fluid,
                                        'StudyInstanceUID': study_uid,
                                        'SeriesInstanceUID': series_uid,
                                        'frameNumber': int(frame_idx),
                                        'groupId': label_id_machine,
                                        'note': f'Corrected to no-fluid (iteration {iteration_number})'
                                    }
                                    annotations_to_upload.append(annotation)
                                    print(f"✓ Added corrected no-fluid annotation for frame {frame_idx}")
                                else:
                                    # Convert mask to MD.ai format
                                    mask_data = mdai.common_utils.convert_mask_data(binary_mask)
                                    if mask_data:
                                        annotation = {
                                            'labelId': algorithm_label_id,
                                            'StudyInstanceUID': study_uid,
                                            'SeriesInstanceUID': series_uid,
                                            'frameNumber': int(frame_idx),
                                            'data': mask_data,
                                            'groupId': label_id_machine,
                                            'note': f'Fluid annotation (iteration {iteration_number})'
                                        }
                                        annotations_to_upload.append(annotation)
                                        print(f"✓ Added fluid annotation for frame {frame_idx}")
                    
                    except Exception as e:
                        print(f"Error creating annotation for frame {frame_idx}: {str(e)}")
                        continue
                
                # FINAL VALIDATION BEFORE UPLOAD
                print(f"\n🔍 FINAL VALIDATION: Checking {len(annotations_to_upload)} annotations before upload")
                validated_annotations = []
                validation_fixes = 0
                
                for annotation in annotations_to_upload:
                    frame_idx = annotation['frameNumber']
                    
                    # Check if this frame should be no-fluid according to ground truth
                    if frame_idx in no_fluid_ground_truth_frames:
                        if annotation['labelId'] == algorithm_label_id:
                            print(f"🚫 FINAL VALIDATION FIX: Frame {frame_idx} should be no-fluid but was marked as fluid")
                            validation_fixes += 1
                            # Force to no-fluid
                            annotation = {
                                'labelId': label_id_no_fluid,
                                'StudyInstanceUID': annotation['StudyInstanceUID'],
                                'SeriesInstanceUID': annotation['SeriesInstanceUID'],
                                'frameNumber': annotation['frameNumber'],
                                'groupId': annotation['groupId'],
                                'note': f'Final validation correction to no-fluid (iteration {iteration_number})'
                            }
                    
                    validated_annotations.append(annotation)
                
                if validation_fixes > 0:
                    print(f"🚫 FINAL VALIDATION: Fixed {validation_fixes} violations")
                else:
                    print("✅ FINAL VALIDATION: No violations found")
                
                annotations_to_upload = validated_annotations
                
                # Upload to MD.ai
                if annotations_to_upload:
                    try:
                        failed_annotations = mdai_client.import_annotations(
                            annotations=annotations_to_upload,
                            project_id=project_id,
                            dataset_id=dataset_id
                        )
                        successful_count = len(annotations_to_upload) - (len(failed_annotations) if failed_annotations else 0)
                        print(f"Successfully uploaded {successful_count} algorithm predictions as {algorithm_label_id}")
                        
                        try:
                            exam_number = find_exam_number(study_uid, annotations_json)
                            print(f"Annotations were uploaded to Exam #{exam_number}")
                        except Exception as e:
                            print(f"Could not determine exam number: {e}")
                        
                    except Exception as e:
                        print(f"Error uploading algorithm predictions: {str(e)}")
                        traceback.print_exc()
                else:
                    print("No valid algorithm predictions to upload")
            else:
                # Handle case where upload is skipped
                if args and hasattr(args, 'no_upload') and args.no_upload:
                    print("Skipping MD.ai upload (--no-upload flag specified)")
                    print(f"Would have uploaded {len(algorithm_masks) if algorithm_masks else 0} algorithm predictions")
                else:
                    print("No algorithm masks generated")
            
            # DATA LEAKAGE CHECK
            print("\n🔍 DATA LEAKAGE CHECK:")
            annotation_frames = []
            prediction_frames = []
            for frame_idx, mask_info in algorithm_masks.items():
                if isinstance(mask_info, dict):
                    if mask_info.get('is_annotation', False):
                        annotation_frames.append(frame_idx)
                    else:
                        prediction_frames.append(frame_idx)
            print(f"Frames marked as 'annotation' (memorized): {len(annotation_frames)}")
            print(f"Frames marked as 'prediction' (tracked): {len(prediction_frames)}")
            
            # Check if evaluation is using memorized data
            gt_frames = set(ground_truth_masks.keys())
            annotation_frame_set = set(annotation_frames)
            overlap = gt_frames & annotation_frame_set
            print(f"⚠️  Overlap between ground truth and memorized frames: {len(overlap)}/{len(gt_frames)}")
            if len(overlap) > 0:
                print("🚨 DATA LEAKAGE DETECTED: Excluding memorized frames from evaluation!")
            
            # EXTRACT MASKS FOR EVALUATION (EXCLUDING MEMORIZED FRAMES)
            print("\n=== EXTRACTING MASKS FOR EVALUATION (EXCLUDING MEMORIZED FRAMES) ===")
            
            # Extract only PREDICTED masks (exclude memorized annotations)
            algorithm_masks_clean = {}
            for frame_idx, mask_info in algorithm_masks.items():
                # CRITICAL: Skip frames that were used as input annotations
                if isinstance(mask_info, dict) and mask_info.get('is_annotation', False):
                    print(f"Excluding memorized frame {frame_idx} from evaluation")
                    continue
                
                # Extract the actual predicted mask
                if isinstance(mask_info, dict):
                    mask = mask_info.get('mask')
                    if mask is None:
                        continue
                else:
                    mask = mask_info
                
                if isinstance(mask, np.ndarray):
                    algorithm_masks_clean[frame_idx] = mask
            
            print(f"Extracted {len(algorithm_masks_clean)} PREDICTED masks for evaluation")
            print(f"Excluded {len(annotation_frames)} memorized frames from evaluation")
            ground_truth_masks_clean = ground_truth_masks  
            
            # Find common frames for evaluation (excluding memorised frames)
            common_frames = set(algorithm_masks_clean.keys()) & set(ground_truth_masks_clean.keys())
            print(f"Common frames for GENUINE evaluation: {len(common_frames)}")
            
            # Try additional offsets if needed
            if len(common_frames) < min(len(algorithm_masks_clean), len(ground_truth_masks_clean)) // 2:
               print("Testing additional offsets:")
               for offset in [-1, 1, 2]:
                  shifted_algo = {k + offset: v for k, v in algorithm_masks_clean.items()}
                  common_with_offset = set(shifted_algo.keys()) & set(ground_truth_masks_clean.keys())
                  print(f"  With offset {offset}: {len(common_with_offset)} common frames")
        
                  # If a better match is found, update the algorithm masks
                  if len(common_with_offset) > len(common_frames):
                     print(f"  Found better match with offset {offset}")
                     algorithm_masks_clean = shifted_algo
                     common_frames = common_with_offset
            
            # Decide which evaluation approach to use
            if use_genuine_evaluation:
                print("\n=== USING GENUINE EVALUATION WITH SPARSE ANNOTATIONS (NO DATA LEAKAGE) ===")
                genuine_output_dir = os.path.join(video_output_dir, "genuine_evaluation")
                os.makedirs(genuine_output_dir, exist_ok=True)
                
                # Create required directories
                ground_truth_dir = os.path.join(genuine_output_dir, "ground_truth")
                algorithm_dir = os.path.join(genuine_output_dir, "algorithm")
                os.makedirs(ground_truth_dir, exist_ok=True)
                os.makedirs(algorithm_dir, exist_ok=True)
                
                # Get frame indices and apply sampling (ONLY ON PREDICTED FRAMES)
                gt_frames = sorted(list(ground_truth_masks_clean.keys()))
                algo_frames = sorted(list(algorithm_masks_clean.keys()))  
                common_frames_for_sampling = sorted(list(set(gt_frames) & set(algo_frames)))
                
                print(f"Ground truth masks available: {len(ground_truth_masks_clean)}")
                print(f"Algorithm PREDICTED masks available: {len(algorithm_masks_clean)}")
                print(f"Common frames before sampling: {len(common_frames_for_sampling)}")
                print(f"Evaluation sampling rate: {evaluation_sampling_rate}")
                
                # Apply sampling rate logic for evaluation
                if evaluation_sampling_rate > 1 and len(common_frames_for_sampling) > evaluation_sampling_rate:
                    sampled_frames = common_frames_for_sampling[::evaluation_sampling_rate]
                    
                    # Always include first and last frame
                    if common_frames_for_sampling[0] not in sampled_frames:
                        sampled_frames.insert(0, common_frames_for_sampling[0])
                    if common_frames_for_sampling[-1] not in sampled_frames:
                        sampled_frames.append(common_frames_for_sampling[-1])
                    
                    sampled_frames = sorted(list(set(sampled_frames)))
                    print(f"Frames after evaluation sampling (rate {evaluation_sampling_rate}): {len(sampled_frames)}")
                    print(f"Selected frames for evaluation: {sampled_frames}")
                else:
                    sampled_frames = common_frames_for_sampling
                    print(f"Using all frames for evaluation (sampling rate {evaluation_sampling_rate} >= frame count or rate = 1)")
                
                # Save masks for the sampled frames only
                print("Saving masks for genuine evaluation...")
                saved_gt_count = 0
                saved_algo_count = 0
                
                for frame_idx in sampled_frames:
                    try:
                        # Save ground truth mask
                        if frame_idx in ground_truth_masks_clean:
                            gt_mask = ground_truth_masks_clean[frame_idx]
                            
                            # Handle no-fluid frames
                            if isinstance(gt_mask, dict) and gt_mask.get('is_no_fluid', False):
                                mask = np.zeros((video_height, video_width), dtype=np.uint8)
                            elif isinstance(gt_mask, dict) and 'mask' in gt_mask:
                                mask = (gt_mask['mask'] > 0.5).astype(np.uint8) * 255
                            else:
                                mask = (gt_mask > 0.5).astype(np.uint8) * 255
                            
                            mask_path = os.path.join(ground_truth_dir, f"{int(frame_idx):04d}.png")
                            cv2.imwrite(mask_path, mask)
                            saved_gt_count += 1
                        
                        # Save algorithm mask (PREDICTED ONLY)
                        if frame_idx in algorithm_masks_clean:
                            algo_mask = algorithm_masks_clean[frame_idx]
                            
                            # Handle different mask formats
                            if isinstance(algo_mask, dict):
                                if algo_mask.get('is_no_fluid', False) or 'clear' in algo_mask.get('type', '').lower():
                                    mask = np.zeros((video_height, video_width), dtype=np.uint8)
                                elif 'mask' in algo_mask:
                                    mask = (algo_mask['mask'] > 0.5).astype(np.uint8) * 255
                                else:
                                    mask = np.zeros((video_height, video_width), dtype=np.uint8)
                            else:
                                mask = (algo_mask > 0.5).astype(np.uint8) * 255
                            
                            mask_path = os.path.join(algorithm_dir, f"{int(frame_idx):04d}.png")
                            cv2.imwrite(mask_path, mask)
                            saved_algo_count += 1
                            
                    except Exception as e:
                        print(f"Error saving masks for frame {frame_idx}: {str(e)}")
                        continue
                
                print(f"Saved {saved_gt_count} ground truth masks")
                print(f"Saved {saved_algo_count} algorithm PREDICTED masks")
                
                # Import the evaluation function directly
                from src.validation.sparse_validation import create_sparse_validation_set
                
                try:
                    print("\nRunning genuine evaluation...")
                    print(f"Ground truth directory: {ground_truth_dir}")
                    print(f"Algorithm directory: {algorithm_dir}")
                    print(f"Number of ground truth masks: {len(os.listdir(ground_truth_dir))}")
                    print(f"Number of algorithm masks: {len(os.listdir(algorithm_dir))}")
                    
                    genuine_results = create_sparse_validation_set(
                        ground_truth_dir=ground_truth_dir,
                        algorithm_results_dir=algorithm_dir,
                        num_frames=evaluation_sampling_rate,
                        output_dir=genuine_output_dir
                    )
                    
                    # Extract metrics from the results
                    metrics = genuine_results.get("summary", {})
                    print("\nGenuine evaluation metrics:")
                    print(f"Mean IoU: {metrics.get('mean_iou', 0.0):.4f}")
                    print(f"Mean Dice: {metrics.get('mean_dice', 0.0):.4f}")
                    
                except Exception as e:
                    print(f"\nError in genuine evaluation: {str(e)}")
                    traceback.print_exc()
                    genuine_results = {
                        "error": str(e),
                        "metrics": {
                            "mean_iou": 0.0,
                            "median_iou": 0.0,
                            "mean_dice": 0.0,
                            "iou_over_0.7": 0.0
                        }
                    }
                    metrics = genuine_results["metrics"]
                
                # Add genuine evaluation details to results
                evaluation_results[f"{study_uid}_{series_uid}_genuine"] = genuine_results
                print(f"Genuine evaluation completed with sampling rate {evaluation_sampling_rate}")
            
            # STANDARD EVALUATION (UPDATED)
            if common_frames:
                # Create algorithm and ground truth subsets (PREDICTED FRAMES ONLY)
                algo_subset = {}
                gt_subset = {}
                
                print(f"Creating evaluation subsets from {len(common_frames)} common PREDICTED frames")
                
                for frame_idx in common_frames:
                    # Get algorithm mask (these are now prediction-only)
                    algo_mask = algorithm_masks_clean[frame_idx]
                    if isinstance(algo_mask, dict) and 'mask' in algo_mask:
                        algo_subset[frame_idx] = algo_mask['mask']
                    else:
                        algo_subset[frame_idx] = algo_mask
                    
                    # Get ground truth mask
                    gt_mask = ground_truth_masks_clean[frame_idx]
                    if isinstance(gt_mask, dict) and 'mask' in gt_mask:
                        gt_subset[frame_idx] = gt_mask['mask']
                    else:
                        gt_subset[frame_idx] = gt_mask
                
                print(f"Evaluating on {len(algo_subset)} PREDICTED frames (no memorized data)")
                metrics = evaluate_with_iou(algo_subset, gt_subset)
                print(f"GENUINE evaluation completed on {len(common_frames)} predicted frames")
            else:
                print("No common PREDICTED frames found for evaluation!")
                metrics = {
                    'mean_iou': 0.0,
                    'median_iou': 0.0,
                    'mean_dice': 0.0,
                    'iou_over_0.7': 0.0
                }
            
            # NEW: SINGLE-FRAME GROUND TRUTH EVALUATION (if method comparison enabled)
            if include_method_comparison and single_frame_results:
                print(f"\n📊 EVALUATING SINGLE-FRAME VS GROUND TRUTH")
                
                # Extract only PREDICTED frames from single-frame results (exclude memorized)
                single_frame_masks_clean = {}
                single_frame_annotation_frames = []
                
                for frame_idx, mask_info in single_frame_results.items():
                    # Skip frames that were used as input annotations
                    if isinstance(mask_info, dict) and mask_info.get('is_annotation', False):
                        single_frame_annotation_frames.append(frame_idx)
                        print(f"Excluding single-frame memorized frame {frame_idx} from evaluation")
                        continue
                    
                    # Extract the actual predicted mask
                    if isinstance(mask_info, dict):
                        mask = mask_info.get('mask')
                        if mask is None:
                            continue
                    else:
                        mask = mask_info
                    
                    if isinstance(mask, np.ndarray):
                        single_frame_masks_clean[frame_idx] = mask
                
                print(f"Single-frame: {len(single_frame_masks_clean)} predicted frames for evaluation")
                print(f"Single-frame: {len(single_frame_annotation_frames)} memorized frames excluded")
                
                # Find common frames between single-frame predictions and ground truth
                single_common_frames = set(single_frame_masks_clean.keys()) & set(ground_truth_masks_clean.keys())
                print(f"Single-frame: {len(single_common_frames)} common frames with ground truth")
                
                # Evaluate single-frame against ground truth
                if single_common_frames:
                    single_algo_subset = {}
                    single_gt_subset = {}
                    
                    for frame_idx in single_common_frames:
                        # Get single-frame mask
                        single_mask = single_frame_masks_clean[frame_idx]
                        single_algo_subset[frame_idx] = single_mask
                        
                        # Get ground truth mask
                        gt_mask = ground_truth_masks_clean[frame_idx]
                        if isinstance(gt_mask, dict) and 'mask' in gt_mask:
                            single_gt_subset[frame_idx] = gt_mask['mask']
                        else:
                            single_gt_subset[frame_idx] = gt_mask
                    
                    print(f"Evaluating single-frame on {len(single_algo_subset)} predicted frames")
                    single_frame_metrics = evaluate_with_iou(single_algo_subset, single_gt_subset)
                    
                    print(f"🎯 SINGLE-FRAME vs Ground Truth:")
                    print(f"   Mean IoU: {single_frame_metrics.get('mean_iou', 0.0):.4f}")
                    print(f"   Mean Dice: {single_frame_metrics.get('mean_dice', 0.0):.4f}")
                    print(f"   Frames IoU > 0.7: {single_frame_metrics.get('iou_over_0.7', 0.0)*100:.1f}%")
                    
                else:
                    print("No common frames between single-frame predictions and ground truth!")
                    single_frame_metrics = {
                        'mean_iou': 0.0,
                        'median_iou': 0.0,
                        'mean_dice': 0.0,
                        'iou_over_0.7': 0.0
                    }
                
                # Add single-frame metrics to comparison results
                if comparison_metrics and 'error' not in comparison_metrics:
                    comparison_metrics['single_frame_vs_gt'] = single_frame_metrics
                    comparison_metrics['multi_frame_vs_gt'] = metrics  # This is your existing multi-frame metrics
                    
                    # Print comparison summary
                    print(f"\n🏆 PERFORMANCE COMPARISON:")
                    multi_iou = metrics.get('mean_iou', 0.0)
                    single_iou = single_frame_metrics.get('mean_iou', 0.0)
                    improvement = ((multi_iou - single_iou) / single_iou * 100) if single_iou > 0 else 0
                    
                    print(f"   Multi-frame Mean IoU: {multi_iou:.4f}")
                    print(f"   Single-frame Mean IoU: {single_iou:.4f}")
                    print(f"   Multi-frame improvement: {improvement:+.1f}%")
                    
                    # Add improvement metrics
                    comparison_metrics['performance_improvement'] = {
                        'multi_frame_iou': multi_iou,
                        'single_frame_iou': single_iou,
                        'iou_improvement_percent': improvement,
                        'multi_frame_dice': metrics.get('mean_dice', 0.0),
                        'single_frame_dice': single_frame_metrics.get('mean_dice', 0.0)
                    }
            
            # EVALUATION SUMMARY
            print(f"\n📊 EVALUATION SUMMARY:")
            print(f"Total algorithm frames: {len(algorithm_masks)}")
            print(f"Memorized frames (excluded): {len(annotation_frames)}")
            print(f"Predicted frames (evaluated): {len(algorithm_masks_clean)}")
            print(f"Ground truth frames: {len(ground_truth_masks_clean)}")
            print(f"Common frames for evaluation: {len(common_frames)}")
            print(f"🎯 GENUINE Mean IoU: {metrics.get('mean_iou', 0.0):.4f}")
            print(f"🎯 GENUINE Mean Dice: {metrics.get('mean_dice', 0.0):.4f}")
            
            # Create visualization of results
            visualization_path = visualize_comparison(
                video_path, algorithm_masks_clean, ground_truth_masks_clean, 
                os.path.join(video_output_dir, "comparison.mp4")
            )
            
            # Update tracking parameters with metrics if in learning mode
            if learning_mode and shared_params:
                print("\nUpdating tracking parameters based on performance metrics...")
                improved = shared_params.update_from_feedback(metrics)
                print(f"Parameters {'improved' if improved else 'adjusted'} based on feedback")
                
                # Add the learning results to metrics
                metrics['learning'] = {
                    'params_updated': True,
                    'params_improved': improved,
                    'params_version': shared_params.version,
                    'window_size': shared_params.tracking_params['window_size'],
                    'flow_quality_threshold': shared_params.tracking_params['flow_quality_threshold'],
                    'border_constraint_weight': shared_params.tracking_params['border_constraint_weight']
                }
            
            # Store results with method comparison if available
            result_data = {
                'metrics': metrics,
                'multi_frame_results': multi_frame_results,
                'ground_truth_count': len(ground_truth_masks_clean),
                'algorithm_mask_count': len(algorithm_masks_clean),
                'memorized_frames_excluded': len(annotation_frames),
                'visualization_path': visualization_path,
                'iteration': iteration_number,
                'learning_mode': learning_mode
            }
            
            # NEW: Add method comparison results if available
            if include_method_comparison:
                result_data['single_frame_results'] = single_frame_results
                result_data['method_comparison'] = comparison_metrics
                
                # Print method comparison summary
                if comparison_metrics and 'error' not in comparison_metrics:
                    print(f"\n🔬 METHOD COMPARISON SUMMARY:")
                    print(f"Single-frame predictions: {comparison_metrics.get('single_frame_count', 0)}")
                    print(f"Multi-frame predictions: {comparison_metrics.get('multi_frame_count', 0)}")
                    print(f"Method agreement IoU: {comparison_metrics.get('mean_iou', 0.0):.4f}")
                    print(f"Method agreement Dice: {comparison_metrics.get('mean_dice', 0.0):.4f}")
                    
                    # Print performance improvement if available
                    if 'performance_improvement' in comparison_metrics:
                        perf = comparison_metrics['performance_improvement']
                        print(f"Multi-frame vs Ground Truth IoU: {perf.get('multi_frame_iou', 0.0):.4f}")
                        print(f"Single-frame vs Ground Truth IoU: {perf.get('single_frame_iou', 0.0):.4f}")
                        print(f"Performance improvement: {perf.get('iou_improvement_percent', 0.0):+.1f}%")
            
            evaluation_results[f"{study_uid}_{series_uid}"] = result_data
            
            # Generate a report for this video
            report_path = os.path.join(video_output_dir, "evaluation_report.md")
            try:
                with open(report_path, 'w') as f:
                    f.write(f"# Evaluation Report: {study_uid}/{series_uid}\n\n")
                    f.write(f"- Ground Truth Masks: {len(ground_truth_masks_clean)}\n")
                    f.write(f"- Algorithm Masks (Total): {len(algorithm_masks)}\n")
                    f.write(f"- Algorithm Masks (Predicted): {len(algorithm_masks_clean)}\n")
                    f.write(f"- Memorized Frames (Excluded): {len(annotation_frames)}\n")
                    f.write(f"- Common Frames: {len(common_frames)}\n")
                    f.write(f"- Method Comparison: {'Enabled' if include_method_comparison else 'Disabled'}\n\n")
                    
                    f.write("## Data Leakage Prevention\n\n")
                    f.write(f"- Memorized annotation frames excluded: {len(annotation_frames)}\n")
                    f.write(f"- Only predicted frames used for evaluation: {len(algorithm_masks_clean)}\n")
                    f.write(f"- Overlap between GT and memorized: {len(overlap)}\n\n")
                    
                    f.write("## No-Fluid Constraint Enforcement\n\n")
                    f.write(f"- Ground truth no-fluid frames: {len(no_fluid_ground_truth_frames)}\n")
                    f.write(f"- No-fluid frames enforced: {sorted(list(no_fluid_ground_truth_frames))}\n\n")
                    
                    f.write("## Evaluation Metrics (GENUINE)\n\n")
                    f.write(f"- Mean IoU: {metrics.get('mean_iou', 0.0):.4f}\n")
                    f.write(f"- Median IoU: {metrics.get('median_iou', 0.0):.4f}\n")
                    f.write(f"- Mean Dice: {metrics.get('mean_dice', 0.0):.4f}\n")
                    f.write(f"- % Frames IoU > 0.7: {metrics.get('iou_over_0.7', 0.0)*100:.1f}%\n\n")
                    
                 
                    if include_method_comparison and comparison_metrics:
                        f.write("## Method Comparison\n\n")
                        if 'error' in comparison_metrics:
                            f.write(f"- Error: {comparison_metrics['error']}\n\n")
                        else:
                            f.write(f"- Single-frame predictions: {comparison_metrics.get('single_frame_count', 0)}\n")
                            f.write(f"- Multi-frame predictions: {comparison_metrics.get('multi_frame_count', 0)}\n")
                            f.write(f"- Method agreement IoU: {comparison_metrics.get('mean_iou', 0.0):.4f}\n")
                            f.write(f"- Method agreement Dice: {comparison_metrics.get('mean_dice', 0.0):.4f}\n")
                            
                            # Add performance comparison
                            if 'performance_improvement' in comparison_metrics:
                                perf = comparison_metrics['performance_improvement']
                                f.write(f"\n### Performance Against Ground Truth\n\n")
                                f.write(f"- Multi-frame vs GT IoU: {perf.get('multi_frame_iou', 0.0):.4f}\n")
                                f.write(f"- Single-frame vs GT IoU: {perf.get('single_frame_iou', 0.0):.4f}\n")
                                f.write(f"- Multi-frame improvement: {perf.get('iou_improvement_percent', 0.0):+.1f}%\n")
                                f.write(f"- Multi-frame vs GT Dice: {perf.get('multi_frame_dice', 0.0):.4f}\n")
                                f.write(f"- Single-frame vs GT Dice: {perf.get('single_frame_dice', 0.0):.4f}\n\n")
                    
                    if 'error' in metrics:
                        f.write(f"**Error in metrics calculation:** {metrics['error']}\n\n")
                    
                    f.write("## Frame-by-Frame IoU (Sample)\n\n")
                    f.write("| Frame | IoU | Dice | Type |\n")
                    f.write("|-------|-----|------|------|\n")
                    
                    # Add a few example frames
                    try:
                        for frame in sorted(list(common_frames))[:10]:
                            try:
                                frame_iou = calculate_iou(algorithm_masks_clean[frame], ground_truth_masks_clean[frame])
                                frame_dice = calculate_dice(algorithm_masks_clean[frame], ground_truth_masks_clean[frame])
                                f.write(f"| {frame} | {frame_iou:.4f} | {frame_dice:.4f} | Predicted |\n")
                            except Exception as e:
                                f.write(f"| {frame} | Error: {str(e)} | | |\n")
                        
                        if len(common_frames) > 10:
                            f.write("| ... | ... | ... | ... |\n")
                    except Exception as e:
                        f.write(f"Error processing frame-by-frame details: {str(e)}\n")
            except Exception as e:
                print(f"Error generating evaluation report: {str(e)}")
            
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
        try:
            summary = {
                'total_videos': len(video_paths),
                'successful_evaluations': len(valid_results),
                'overall_mean_iou': np.mean([r['metrics'].get('mean_iou', 0.0) for r in valid_results]),
                'overall_mean_dice': np.mean([r['metrics'].get('mean_dice', 0.0) for r in valid_results]),
                'total_memorized_frames_excluded': sum([r.get('memorized_frames_excluded', 0) for r in valid_results]),
                'method_comparison_enabled': include_method_comparison,  # NEW
                'best_video': max([(k, v['metrics'].get('mean_iou', 0.0)) for k, v in evaluation_results.items() 
                                if 'metrics' in v], key=lambda x: x[1])[0],
                'worst_video': min([(k, v['metrics'].get('mean_iou', 0.0)) for k, v in evaluation_results.items() 
                                    if 'metrics' in v], key=lambda x: x[1])[0]
            }
            
            # Add method comparison summary if enabled
            if include_method_comparison:
                comparison_results = [r.get('method_comparison', {}) for r in valid_results 
                                    if 'method_comparison' in r and 'error' not in r.get('method_comparison', {})]
                if comparison_results:
                    summary['method_comparison_summary'] = {
                        'videos_compared': len(comparison_results),
                        'avg_method_agreement_iou': np.mean([c.get('mean_iou', 0.0) for c in comparison_results]),
                        'avg_single_frame_count': np.mean([c.get('single_frame_count', 0) for c in comparison_results]),
                        'avg_multi_frame_count': np.mean([c.get('multi_frame_count', 0) for c in comparison_results])
                    }
                    
                    # Add performance improvement summary
                    perf_results = [c.get('performance_improvement', {}) for c in comparison_results 
                                   if 'performance_improvement' in c]
                    if perf_results:
                        summary['performance_improvement_summary'] = {
                            'avg_multi_frame_iou': np.mean([p.get('multi_frame_iou', 0.0) for p in perf_results]),
                            'avg_single_frame_iou': np.mean([p.get('single_frame_iou', 0.0) for p in perf_results]),
                            'avg_improvement_percent': np.mean([p.get('iou_improvement_percent', 0.0) for p in perf_results])
                        }
                
        except Exception as e:
            print(f"Error generating evaluation summary: {str(e)}")
            summary = {
                'total_videos': len(video_paths),
                'successful_evaluations': len(valid_results),
                'overall_mean_iou': 0.0,
                'overall_mean_dice': 0.0,
                'total_memorized_frames_excluded': 0,
                'method_comparison_enabled': include_method_comparison,
                'error': str(e)
            }
        
        evaluation_results['summary'] = summary
        
        # Create overall report
        report_path = os.path.join(output_dir, "overall_evaluation_report.md")
        try:
            with open(report_path, 'w') as f:
                f.write("# Overall Evaluation Report (Data Leakage Prevention & No-Fluid Enforcement)\n\n")
                f.write(f"- Total Videos: {summary['total_videos']}\n")
                f.write(f"- Successful Evaluations: {summary['successful_evaluations']}\n")
                f.write(f"- Total Memorized Frames Excluded: {summary.get('total_memorized_frames_excluded', 0)}\n")
                f.write(f"- Method Comparison: {'Enabled' if summary.get('method_comparison_enabled', False) else 'Disabled'}\n")
                f.write(f"- Overall Mean IoU (GENUINE): {summary.get('overall_mean_iou', 0.0):.4f}\n")
                f.write(f"- Overall Mean Dice (GENUINE): {summary.get('overall_mean_dice', 0.0):.4f}\n")
                f.write(f"- Best Performing Video: {summary.get('best_video', 'N/A')}\n")
                f.write(f"- Worst Performing Video: {summary.get('worst_video', 'N/A')}\n\n")
                
                # Add method comparison summary section
                if include_method_comparison and 'method_comparison_summary' in summary:
                    mc_summary = summary['method_comparison_summary']
                    f.write("## Method Comparison Summary\n\n")
                    f.write(f"- Videos with successful comparison: {mc_summary.get('videos_compared', 0)}\n")
                    f.write(f"- Average method agreement IoU: {mc_summary.get('avg_method_agreement_iou', 0.0):.4f}\n")
                    f.write(f"- Average single-frame predictions per video: {mc_summary.get('avg_single_frame_count', 0.0):.1f}\n")
                    f.write(f"- Average multi-frame predictions per video: {mc_summary.get('avg_multi_frame_count', 0.0):.1f}\n\n")
                    
                    # Add performance improvement summary
                    if 'performance_improvement_summary' in summary:
                        perf_summary = summary['performance_improvement_summary']
                        f.write("## Performance Improvement Summary\n\n")
                        f.write(f"- Average Multi-frame IoU vs GT: {perf_summary.get('avg_multi_frame_iou', 0.0):.4f}\n")
                        f.write(f"- Average Single-frame IoU vs GT: {perf_summary.get('avg_single_frame_iou', 0.0):.4f}\n")
                        f.write(f"- Average Multi-frame Improvement: {perf_summary.get('avg_improvement_percent', 0.0):+.1f}%\n\n")
                
                f.write("## Data Leakage Prevention Summary\n\n")
                f.write("This evaluation excludes all frames that were used as input annotations (memorized frames)\n")
                f.write("to ensure genuine algorithm performance assessment.\n\n")
                
                f.write("## No-Fluid Constraint Enforcement\n\n")
                f.write("All frames marked as 'no fluid' in ground truth are enforced to have empty masks\n")
                f.write("in the algorithm output, preventing incorrect fluid annotations on clear frames.\n\n")
                
                f.write("## Per-Video Results (GENUINE)\n\n")
                header = "| Video | Mean IoU | Mean Dice | IoU > 0.7 | Excluded Frames | Method Comparison |"
                if include_method_comparison:
                    header += " Multi-IoU | Single-IoU | Improvement |"
                header += "\n"
                f.write(header)
                
                separator = "|-------|----------|-----------|-----------|------------------|-------------------|"
                if include_method_comparison:
                    separator += "----------|------------|-------------|"
                separator += "\n"
                f.write(separator)
                
                for video_id, results in evaluation_results.items():
                    if video_id != 'summary' and 'metrics' in results:
                        metrics = results['metrics']
                        excluded = results.get('memorized_frames_excluded', 0)
                        has_comparison = 'method_comparison' in results
                        comparison_status = "✓" if has_comparison else "✗"
                        
                        try:
                            row = f"| {video_id} | {metrics.get('mean_iou', 0.0):.4f} | {metrics.get('mean_dice', 0.0):.4f} | {metrics.get('iou_over_0.7', 0.0)*100:.1f}% | {excluded} | {comparison_status} |"
                            
                            if include_method_comparison and has_comparison:
                                comp = results.get('method_comparison', {})
                                if 'performance_improvement' in comp:
                                    perf = comp['performance_improvement']
                                    row += f" {perf.get('multi_frame_iou', 0.0):.4f} | {perf.get('single_frame_iou', 0.0):.4f} | {perf.get('iou_improvement_percent', 0.0):+.1f}% |"
                                else:
                                    row += " N/A | N/A | N/A |"
                            elif include_method_comparison:
                                row += " N/A | N/A | N/A |"
                            
                            row += "\n"
                            f.write(row)
                        except Exception as e:
                            f.write(f"| {video_id} | Error: {str(e)} | | | | | | | |\n")
            
            print(f"Overall evaluation report saved to: {report_path}")
        except Exception as e:
            print(f"Error creating evaluation report: {str(e)}")
    
    # Save results to file
    try:
        with open(os.path.join(output_dir, 'evaluation_results.json'), 'w') as f:
            
            json_results = convert_numpy_to_python(evaluation_results)
            json.dump(json_results, f, indent=2)
    except Exception as e:
        print(f"Error saving evaluation results to JSON: {str(e)}")
    
    return evaluation_results

def run_ground_truth_feedback_loop(target_videos, num_iterations=3, matched_annotations=None, 
                                 free_fluid_annotations=None, annotations_json=None, 
                                 mdai_client=None, project_id=None, dataset_id=None,
                                 label_id_ground_truth=None, label_id_fluid=None,
                                 label_id_no_fluid=None, label_id_machine=None,
                                 flow_processor=None, exam_id=None, learning_mode=False,
                                 params_file=None, use_genuine_evaluation=False, sampling_rate=10, input_sampling_rate= None, evaluation_sampling_rate=None, include_method_comparison=False):
    """
    Runs the complete feedback loop for ground truth creation and evaluation
    """

    if input_sampling_rate is None:
        input_sampling_rate = sampling_rate  # Default to legacy sampling rate
    if evaluation_sampling_rate is None:
        evaluation_sampling_rate = 1         # Default to dense evaluation (every frame)
    
    print(f"\n🧪 FEEDBACK LOOP CONFIGURATION:")
    print(f"Input sampling rate: {input_sampling_rate}")
    print(f"Evaluation sampling rate: {evaluation_sampling_rate}")
    
    print("\n" + "="*80)
    print("=== FEEDBACK LOOP WITH IMPROVEMENTS ===")
    print(f"Exam ID: {exam_id or 'ALL'}")
    print(f"Learning mode: {'ENABLED' if learning_mode else 'DISABLED'}")
    print(f"Parameters file: {params_file or 'Using defaults'}")
    print(f"Genuine evaluation: {'ENABLED' if use_genuine_evaluation else 'DISABLED'}")
    if use_genuine_evaluation:
        print(f"Sparse sampling rate: {sampling_rate} (using every {sampling_rate}th frame)")
    print(f"Number of iterations: {num_iterations}")
    print("="*80 + "\n")
    
    # Check for global no-fluid labels
    global_no_fluid_label = "L_7BGg21"  
    no_fluid_exams = set()
    
    if annotations_json:
        print("\nChecking for global no-fluid labels...")
        for dataset in annotations_json.get('datasets', []):
            for annotation in dataset.get('annotations', []):
                if annotation.get('labelId') == global_no_fluid_label:
                    study_uid = annotation.get('StudyInstanceUID')
                    if study_uid:
                        try:
                            exam_number = find_exam_number(study_uid, annotations_json)
                            no_fluid_exams.add(exam_number)
                            print(f"Found global no-fluid label for exam #{exam_number}")
                        except Exception as e:
                            print(f"Could not determine exam number for study {study_uid}: {e}")
    
    if no_fluid_exams:
        print(f"\nFound {len(no_fluid_exams)} exams marked as globally no-fluid: {sorted(list(no_fluid_exams))}")
        
        # If we're processing a specific exam that's marked as no-fluid
        if exam_id and int(exam_id) in no_fluid_exams:
            print(f"\nExam #{exam_id} is marked as globally no-fluid")
            print("Creating empty masks for all frames...")
            
            # Get the video info
            for video_path, study_uid, series_uid in target_videos:
                try:
                    video_exam_id = find_exam_number(study_uid, annotations_json)
                    if str(video_exam_id) == str(exam_id):
                        # Create empty masks for all frames
                        cap = cv2.VideoCapture(video_path)
                        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        cap.release()
                        
                        # Create output directory
                        output_dir = os.path.join(base_output_dir, "iteration_1")
                        os.makedirs(output_dir, exist_ok=True)
                        
                        # Save empty masks for all frames
                        empty_mask = np.zeros((frame_height, frame_width), dtype=np.uint8)
                        for frame_idx in range(total_frames):
                            mask_path = os.path.join(output_dir, f"frame_{frame_idx:04d}_mask.png")
                            cv2.imwrite(mask_path, empty_mask)
                            
                            # Also create the MD.ai annotation for this frame
                            if mdai_client and not os.getenv('SKIP_UPLOADS'):
                                try:
                                    annotation = {
                                        'labelId': label_id_machine,
                                        'StudyInstanceUID': study_uid,
                                        'SeriesInstanceUID': series_uid,
                                        'SOPInstanceUID': f"{series_uid}_{frame_idx}",
                                        'frameNumber': frame_idx + 1,  # MD.ai uses 1-based frame numbers
                                        'data': {'vertices': []},  # Empty polygon for no-fluid
                                        'type': 'polygon'
                                    }
                                    
                                    mdai_client.import_annotations(
                                        annotations=[annotation],
                                        project_id=project_id,
                                        dataset_id=dataset_id
                                    )
                                except Exception as e:
                                    print(f"Error uploading no-fluid annotation for frame {frame_idx}: {e}")
                        
                        print(f"Created {total_frames} empty masks for no-fluid exam")
                        return {
                            'status': 'success',
                            'message': f'Created empty masks for all frames in no-fluid exam #{exam_id}',
                            'frames_processed': total_frames
                        }
                except Exception as e:
                    print(f"Error processing no-fluid exam: {e}")
    
    # Continue with normal processing for non-no-fluid exams
    results = {
        'iterations': [],
        'exam_id': exam_id,
        'learning_mode': learning_mode,
        'params_file': params_file
    }
    
    # Create output directory
    base_output_dir = os.path.join(OUTPUT_DIR, f"feedback_loop_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    if exam_id:
        base_output_dir = os.path.join(OUTPUT_DIR, f"feedback_loop_exam_{exam_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(base_output_dir, exist_ok=True)
    
    # Filter target videos for specific exam if needed
    if exam_id:
        print(f"Filtering videos for exam ID {exam_id}")
        filtered_videos = []
        for video_path, study_uid, series_uid in target_videos:
            # Extract exam number from path
            video_exam_id = find_exam_number(study_uid, annotations_json)
            if video_exam_id == exam_id:
                print(f"✓ Found matching video: {video_path}")
                filtered_videos.append((video_path, study_uid, series_uid))
        
        if not filtered_videos:
            print(f"❌ No videos found for exam ID {exam_id}")
            return {'status': 'error', 'message': f'No videos found for exam ID {exam_id}'}
        
        target_videos = filtered_videos
    
    # Initialise shared parameters
    shared_params = None
    if learning_mode:
        try:
            from src.multi_frame_tracking.multi_frame_tracker import SharedParams
            shared_params = SharedParams(params_file=params_file)
            print(f"✓ Initialized SharedParams (version {shared_params.version})")
        except Exception as e:
            print(f"❌ Error initializing SharedParams: {str(e)}")
            print("Falling back to standard tracking without learning")
            learning_mode = False
    
    # Initialise optical flow processor 
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
        
        # Step 1: Create/update ground truth dataset (ONLY ON FIRST ITERATION)
        if iteration == 0:
            # Check if ground truth already exists for this exam
            existing_gt = get_annotations_for_study_series(
                mdai_client, project_id, dataset_id, study_series_pairs[0][0], study_series_pairs[0][1],
                label_id=label_id_ground_truth
            )
    
            if existing_gt:
                print(f"\nStep 1: Found {len(existing_gt)} existing ground truth annotations, skipping creation")
                gt_results = {"status": "skipped", "message": "Ground truth already exists"}
            else:
                print("\nStep 1: Creating initial ground truth dataset...")
                gt_results = create_ground_truth_dataset(
                    video_paths, study_series_pairs, flow_processor, 
                    os.path.join(iter_output_dir, "ground_truth"),
                    mdai_client, project_id, dataset_id, label_id_ground_truth,
                    matched_annotations,       
                    free_fluid_annotations,     
                    label_id_fluid=label_id_fluid,
                    label_id_no_fluid=label_id_no_fluid,
                    label_id_machine=label_id_machine,
                    annotations_json=annotations_json
                )
           
            # Step 2: Wait for expert review (ONLY ON FIRST ITERATION)
            print("\nStep 2: Expert review phase")
            print("Please have experts review and modify the ground truth annotations in MD.ai.")
            print("The ground truth annotations have been uploaded with label ID:", label_id_ground_truth)
            
            if num_iterations > 1 and iteration == 0:
                
                print("Proceeding automatically to evaluation phase...")
               
                time.sleep(3) 
        else:
            print(f"\nStep 1: Skipping ground truth creation for iteration {iteration+1}")
            print("Using existing expert-corrected ground truth annotations...")
            gt_results = {"status": "skipped", "message": "Using existing ground truth"}
        
        # Step 3: Run enhanced evaluation with feedback loop processing
        print("\nStep 3: Evaluating algorithm with feedback loop...")
        
        try:
            eval_results = evaluate_with_expert_feedback(
                video_paths, study_series_pairs, flow_processor,
                os.path.join(iter_output_dir, "evaluation"),
                mdai_client, project_id, dataset_id, 
                label_id_ground_truth,   # Expert corrected annotations
                label_id_fluid,          # Original fluid annotations
                label_id_no_fluid,       # No fluid label ID
                label_id_machine,        # Machine group label ID
                annotations_json, 
                args=args,
                shared_params=shared_params,
                learning_mode=learning_mode,
                iteration_number=iteration+1,
                include_method_comparison=include_method_comparison,
                use_genuine_evaluation=use_genuine_evaluation,
                sampling_rate=sampling_rate,
                input_sampling_rate=input_sampling_rate,
                evaluation_sampling_rate=evaluation_sampling_rate
               )
            
            print(f"Evaluation completed successfully for iteration {iteration+1}")
        except Exception as e:
            print(f"Error during evaluation for iteration {iteration+1}: {str(e)}")
            traceback.print_exc()
            eval_results = {f"{study_series_pairs[0][0]}_{study_series_pairs[0][1]}": {"error": str(e)}}
            print("Continuing with next iteration despite evaluation error")
        
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
            
            # Save updated parameters if in learning mode
            if learning_mode and shared_params:
                params_path = os.path.join(iter_output_dir, f"tracking_params_v{shared_params.version}.json")
                shared_params.save_to_file(params_path)
                print(f"✓ Saved updated parameters to {params_path}")
                
                # Update results with parameter changes
                results['iterations'][-1]['params_version'] = shared_params.version
                results['iterations'][-1]['params_file'] = params_path
        
    # Generate final report comparing all iterations
    if results['iterations']:
        create_feedback_loop_report(results, base_output_dir)
    
    # Save final results
    final_results_path = os.path.join(base_output_dir, "feedback_loop_results.json")
    with open(final_results_path, 'w') as f:
        json.dump(convert_numpy_to_python(results), f, indent=2)
    print(f"✓ Saved final results to {final_results_path}")
    
    return results


def create_adaptive_sampling_report(results, output_dir):
    """Create a detailed report of the adaptive sampling experiment"""
    report_path = os.path.join(output_dir, "adaptive_sampling_report.md")
    
    with open(report_path, 'w') as f:
        f.write("# Adaptive Sampling Parameter Learning Report\n\n")
        f.write(f"**Exam ID:** {results['exam_id']}\n")
        f.write(f"**Sampling Rates Tested:** {results['sampling_rates']}\n")
        f.write(f"**Iterations per Rate:** {results['iterations_per_rate']}\n\n")
        
        # Parameter evolution summary
        f.write("## Parameter Evolution Summary\n\n")
        f.write("| Iteration | Sampling Rate | IoU | Window Size | Flow Quality | Flow Noise |\n")
        f.write("|-----------|---------------|-----|-------------|--------------|------------|\n")
        
        for param_data in results['parameter_evolution']:
            f.write(f"| {param_data['iteration']} | {param_data['sampling_rate']} | {param_data['iou']:.4f} | ")
            f.write(f"{param_data['params_after']['window_size']} | ")
            f.write(f"{param_data['params_after']['flow_quality_threshold']:.3f} | ")
            f.write(f"{param_data['params_after']['flow_noise_threshold']:.3f} |\n")
        
        # Performance by sampling rate
        f.write("\n## Performance by Sampling Rate\n\n")
        for rate, rate_data in results['rate_results'].items():
            f.write(f"### Sampling Rate: {rate}\n")
            if rate_data['iterations']:
                first_iou = rate_data['iterations'][0]['iou']
                last_iou = rate_data['iterations'][-1]['iou']
                f.write(f"- Initial IoU: {first_iou:.4f}\n")
                f.write(f"- Final IoU: {last_iou:.4f}\n")
                f.write(f"- Change: {last_iou - first_iou:+.4f}\n")
            f.write("\n")
        
        # Learning insights
        f.write("## Key Findings\n\n")
        f.write("### Parameter Learning Evidence\n")
        
        # Check if parameters actually changed
        if len(results['parameter_evolution']) > 1:
            first_params = results['parameter_evolution'][0]['params_before']
            last_params = results['parameter_evolution'][-1]['params_after']
            
            param_changes = []
            for key in first_params:
                if first_params[key] != last_params[key]:
                    param_changes.append(f"- {key}: {first_params[key]} → {last_params[key]}")
            
            if param_changes:
                f.write("Parameters that changed during learning:\n")
                f.write("\n".join(param_changes))
                f.write("\n\n")
            else:
                f.write("No parameter changes detected during learning.\n\n")
        
        # Compensation effect analysis
        f.write("### Compensation Effect Analysis\n")
        rate_performance = {}
        for rate, rate_data in results['rate_results'].items():
            if rate_data['iterations']:
                avg_iou = np.mean([it['iou'] for it in rate_data['iterations']])
                rate_performance[rate] = avg_iou
        
        if rate_performance:
            best_rate = min(rate_performance.keys())
            worst_rate = max(rate_performance.keys())
            f.write(f"- Best performance (Rate {best_rate}): {rate_performance[best_rate]:.4f}\n")
            f.write(f"- Worst performance (Rate {worst_rate}): {rate_performance[worst_rate]:.4f}\n")
            f.write(f"- Performance gap: {rate_performance[best_rate] - rate_performance[worst_rate]:.4f}\n")
    
    print(f"Detailed report saved to: {report_path}")

# ===== 8. MAIN EXECUTION =====
def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Ultrasound free fluid tracking using optical flow')
    
    #input and evaluation sampling rates
    parser.add_argument('--input-sampling-rate', type=int, default=10, 
                       help='Sampling rate for tracker INPUT (every Nth frame given to tracker)')
    parser.add_argument('--eval-sampling-rate', type=int, default=1, 
                       help='Sampling rate for evaluation (every Nth frame evaluated, 1=all frames)')
    
    
    parser.add_argument('--sampling-rate', type=int, default=10, 
                       help='Legacy: applies to both input and evaluation')
    
    # Core functionality flags
    parser.add_argument('--create-ground-truth', action='store_true',
                       help='Create ground truth dataset')
    parser.add_argument('--feedback-loop', action='store_true',
                       help='Run ground truth feedback loop')
    parser.add_argument('--iterations', type=int, default=3, help='Number of iterations for feedback loop')
    parser.add_argument('--debug', action='store_true',
                       help='Enable debug mode')
    parser.add_argument('--upload', action='store_true',
                       help='Upload results to MD.ai')
    parser.add_argument('--no-upload', action='store_true',
                       help='Skip uploading annotations to MD.ai')
    parser.add_argument('--compare-methods', action='store_true', help='Enable single vs multi-frame method comparison')
    
    
    
    # Add arguments for study and series UIDs
    parser.add_argument('--study', type=str, help='Specific StudyInstanceUID to process')
    parser.add_argument('--series', type=str, help='Specific SeriesInstanceUID to process')
    parser.add_argument('--issue', type=str, choices=['disappear_reappear', 'branching_fluid', 'multiple_distinct', 'no_fluid'],
                        help='Specific issue type to process')
    
    parser.add_argument('--ground-truth-videos', type=int, default=15, help='Number of videos per issue type for ground truth')
    parser.add_argument('--exam-id', type=str, help='Run feedback loop on a specific exam ID')
    parser.add_argument('--learning-mode', action='store_true', help='Enable learning mode in feedback loop')
    parser.add_argument('--params-file', type=str, help='Path to parameters JSON file for feedback loop')
    parser.add_argument('--genuine-evaluation', action='store_true', help='Use genuine evaluation with sparse annotations')
    
    parser.add_argument('--ground-truth-single-exam', type=str, help='Create ground truth for a single exam number (for debugging)')
    parser.add_argument('--ground-truth-single-study', type=str, help='Create ground truth for a single StudyInstanceUID (for debugging)')
    parser.add_argument('--ground-truth-single-series', type=str, help='Create ground truth for a single SeriesInstanceUID (for debugging)')
    parser.add_argument('--all-issues', action='store_true', 
                    help='Process all issue types (disappear_reappear, branching_fluid, multiple_distinct)')
    
    parser.add_argument('--images-dir', type=str, help='Specific path to the images directory')
    
    
    return parser.parse_args()
def quick_test():
    """Run a quick test to make sure everything works"""
    try:
        print("Testing imports...")
        from src.multi_frame_tracking.opticalflowprocessor import OpticalFlowProcessor
        from src.multi_frame_tracking.multi_frame_tracker import MultiFrameTracker
        from src.multi_frame_tracking.utils import track_frames
        print("✓ All imports successful!")
        
        print("Testing basic functionality...")
        flow_processor = OpticalFlowProcessor(method='dis')
        print("✓ Flow processor created")
        
        print("All tests passed! Ready to run.")
        return True
    except Exception as e:
        print(f"✗ Test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":

    setup_timeout_monitor(30)

    if not quick_test():
        print("❌ STOPPING: Fix the import errors above before proceeding.")
        sys.exit(1)
    
    print("✅ Tests passed! Proceeding with main processing...")

    # Parse command line arguments
    args = parse_arguments()

    
    
    # EARLY BYPASS FOR FEEDBACK LOOP - Skip all MD.ai downloads
    if args.feedback_loop:
        print("\n==== MANUAL DOWNLOAD MODE FOR FEEDBACK LOOP ====")
        print("Bypassing MD.ai downloads, using local files...")
        
        # Clear any potential caches
        import importlib
        import sys
        
        # Clear Python module cache
        importlib.invalidate_caches()
        
        # Force refresh of file system state
        import os
        if hasattr(os, 'sync'):
            os.sync()
        
        print("✓ Cleared system caches")
        
        # Override debug setting if specified
        if args.debug:
            DEBUG_MODE = True

      
        log_file = setup_logging(OUTPUT_DIR)
        print(f"All console output will be saved to: {log_file}")
        
        # Load environment variables
        load_dotenv('.env')
        ACCESS_TOKEN = os.getenv('MDAI_TOKEN')
        DATA_DIR = os.getenv('DATA_DIR')
        DOMAIN = os.getenv('DOMAIN')
        PROJECT_ID = os.getenv('PROJECT_ID')
        DATASET_ID = os.getenv('DATASET_ID')
        LABEL_ID_FREE_FLUID = os.getenv("LABEL_ID_FREE_FLUID")
        LABEL_IDS = {
            "disappear_reappear": os.getenv("LABEL_ID_DISAPPEAR_REAPPEAR"),
            "branching_fluid": os.getenv("LABEL_ID_BRANCHING_FLUID"),
            "multiple_distinct": os.getenv("LABEL_ID_MULTIPLE_DISTINCT"),
        }
        LABEL_ID_MACHINE_GROUP = os.getenv("LABEL_ID_MACHINE_GROUP")
        LABEL_ID_FLUID_OF = os.getenv("LABEL_ID_FLUID_OF")
        LABEL_ID_GROUND_TRUTH = os.getenv("LABEL_ID_GROUND_TRUTH")
        LABEL_ID_NO_FLUID = os.getenv("LABEL_ID_NO_FLUID")
        
        # Validate that they're loaded
        if None in [LABEL_ID_MACHINE_GROUP, LABEL_ID_FLUID_OF]:
            raise ValueError("Machine label IDs not properly set in .env file")
        
        
        try:
            domain_value = os.getenv('DOMAIN')
            if not domain_value:
                domain_value = "ucsf.md.ai"  # Hardcode as fallback
                print(f"WARNING: Using hardcoded domain {domain_value}")
            
            access_token_value = os.getenv('MDAI_TOKEN')
            if not access_token_value:
                print("ERROR: MDAI_TOKEN not found in environment variables")
                sys.exit(1)
                
            print(f"Using domain: {domain_value}")
            mdai_client = mdai.Client(domain=domain_value, access_token=access_token_value)
            print("Successfully connected to MD.ai client")
        except Exception as e:
            print(f"Error connecting to MD.ai: {str(e)}")
            sys.exit(1)

        print("Skipping project object creation to avoid corrupted file issues")

        # Get project
        project = mdai_client.project(
         project_id=PROJECT_ID, 
           dataset_id=DATASET_ID,
           path=DATA_DIR) 

        # DEBUG: 
        print("\n=== FILE SYSTEM DEBUG ===")
        print(f"DATA_DIR: {DATA_DIR}")
        print(f"Checking directory: {os.path.abspath(DATA_DIR)}")
        print(f"Directory exists: {os.path.exists(DATA_DIR)}")
        
        # Force a fresh directory read
        import time
        time.sleep(0.1)  
        
        if os.path.exists(DATA_DIR):
          
            all_files = []
            for item in os.listdir(DATA_DIR):
                item_path = os.path.join(DATA_DIR, item)
                if os.path.isfile(item_path):
                    all_files.append(item)
            
            print(f"Total files in directory: {len(all_files)}")
            
           
            annotation_files = [f for f in all_files if 'annotations' in f and f.endswith('.json')]
            print(f"\nAnnotation files found ({len(annotation_files)}):")
            for f in sorted(annotation_files):
                file_path = os.path.join(DATA_DIR, f)
                file_size = os.path.getsize(file_path)
                print(f"  - {f} ({file_size:,} bytes)")
                
            # Show specifically what we're looking for
            dataset_files = [f for f in all_files 
                           if f.startswith('mdai_ucsf_project_x9N2LJBZ_annotations_dataset_D_V688LQ_') 
                           and f.endswith('.json')]
            print(f"\nDataset files ({len(dataset_files)}):")
            for f in sorted(dataset_files):
                file_path = os.path.join(DATA_DIR, f)
                file_size = os.path.getsize(file_path)
                print(f"  - {f} ({file_size:,} bytes)")
        print("=" * 50)
        
       
        recent_file = "mdai_ucsf_project_x9N2LJBZ_annotations_dataset_D_V688LQ_2025-05-13-202256.json"
        ANNOTATIONS = os.path.join(DATA_DIR, recent_file)
        
        print(f"\nFirst checking for most recent file: {recent_file}")
        print(f"Full path: {ANNOTATIONS}")
        print(f"Exists: {os.path.exists(ANNOTATIONS)}")
        
        if os.path.exists(ANNOTATIONS):
            print(f"✓ Found most recent file: {recent_file}")
            # Test if it can be parsed
            try:
                with open(ANNOTATIONS, 'r') as f:
                    test_data = json.load(f)
                print("✓ File is valid and readable")
                file_size = os.path.getsize(ANNOTATIONS)
                print(f"✓ File size: {file_size:,} bytes")
            except json.JSONDecodeError as e:
                print(f"✗ File is corrupted: {e}")
                ANNOTATIONS = None
            except Exception as e:
                print(f"✗ Error reading file: {e}")
                ANNOTATIONS = None
        else:
            print(f"✗ Most recent file not found")
            ANNOTATIONS = None
        
    
        if ANNOTATIONS is None:
            print("\nTrying alternative files...")
            
            # List all available files and let user know what we're trying
            available_files = os.listdir(DATA_DIR)
            
            # Try the 05-07 labelgroup file as second choice
            labelgroup_file = "mdai_ucsf_project_x9N2LJBZ_annotations_dataset_D_V688LQ_labelgroup_G_7n3P09_2025-05-07-194248.json"
            test_path = os.path.join(DATA_DIR, labelgroup_file)
            
            print(f"Trying labelgroup file: {labelgroup_file}")
            if os.path.exists(test_path):
                try:
                    with open(test_path, 'r') as f:
                        json.load(f)
                    ANNOTATIONS = test_path
                    print(f"✓ Using labelgroup file: {labelgroup_file}")
                except json.JSONDecodeError as e:
                    print(f"✗ Labelgroup file corrupted: {e}")
                except Exception as e:
                    print(f"✗ Error with labelgroup file: {e}")
            
           # most recent working file
            if ANNOTATIONS is None:
                dataset_files = [f for f in available_files 
                               if f.startswith('mdai_ucsf_project_x9N2LJBZ_annotations_dataset_D_V688LQ_') 
                               and f.endswith('.json')]
                
                if dataset_files:
                    # Sort by modification time (newest first)
                    dataset_files.sort(key=lambda f: os.path.getmtime(os.path.join(DATA_DIR, f)), reverse=True)
                    
                    for filename in dataset_files:
                        if filename == recent_file:  
                            continue
                        test_path = os.path.join(DATA_DIR, filename)
                        print(f"Trying {filename}...")
                        try:
                            with open(test_path, 'r') as f:
                                json.load(f)
                            ANNOTATIONS = test_path
                            print(f"✓ Using file: {filename}")
                            break
                        except json.JSONDecodeError as e:
                            print(f"✗ Skipping corrupted file: {filename} - {e}")
                            continue
                        except Exception as e:
                            print(f"✗ Error with {filename}: {e}")
                            continue
        
        if ANNOTATIONS is None:
            print("\nERROR: Could not find any valid annotations file!")
            print("Please check your file paths and try downloading annotations again.")
            sys.exit(1)
        
        print(f"\n*** FINAL CHOICE: {os.path.basename(ANNOTATIONS)} ***")
        print(f"Full path: {ANNOTATIONS}")
        print(f"File exists: {os.path.exists(ANNOTATIONS)}")
        print(f"File size: {os.path.getsize(ANNOTATIONS):,} bytes")
        
        # Load annotations directly
        try:
            with open(ANNOTATIONS, 'r') as f:
                annotations_json = json.load(f)
            print("✓ Successfully loaded annotations")
        except json.JSONDecodeError as e:
            print(f"ERROR: Could not parse annotations file: {e}")
            print("The file appears to be corrupted. Please re-download from MD.ai.")
            sys.exit(1)
        except Exception as e:
            print(f"ERROR: Unexpected error loading file: {e}")
            sys.exit(1)
        
        # Extract annotations from JSON
        datasets = annotations_json.get('datasets', [])
        all_annotations = []
        for dataset in datasets:
            if 'annotations' in dataset:
                all_annotations.extend(dataset['annotations'])
        
        annotations_df = json_normalize(all_annotations, sep='_')
        
        # Debug: Show what annotations we have
        print(f"\n=== ANNOTATION DEBUG ===")
        print(f"Total annotations loaded: {len(annotations_df)}")
        print(f"Unique label IDs found: {sorted(annotations_df['labelId'].unique())}")
        print(f"Looking for ground truth label: {LABEL_ID_GROUND_TRUTH}")
        print(f"Looking for free fluid label: {LABEL_ID_FREE_FLUID}")
        print("=" * 40)
        
        
        class SimpleProject:
            def __init__(self, data_dir, dataset_id):
                self.data_dir = data_dir
                self.dataset_id = dataset_id
                
            def get_dataset_by_id(self, dataset_id):
                # Return a simple object with the images_dir property
                class SimpleDataset:
                    def __init__(self, images_dir):
                        self.images_dir = images_dir
                
                # Construct the images directory path based on the pattern
                images_dir = os.path.join(self.data_dir, f"mdai_ucsf_project_{PROJECT_ID}_images_dataset_{dataset_id}")
                return SimpleDataset(images_dir)
        
        # Check for environment variable first
        env_base = os.environ.get("MDAI_IMAGES_DIR")
        if env_base and os.path.exists(env_base):
            # Use the environment variable if it's set
            BASE = env_base
            print(f"Using BASE path from environment: {BASE}")
        else:
            # Try to find the most recent images directory
            import glob
            pattern = os.path.join(DATA_DIR, "mdai_ucsf_project_*_images_dataset_*")
            matching_dirs = sorted(glob.glob(pattern), key=os.path.getmtime, reverse=True)
            
            if matching_dirs:
                BASE = matching_dirs[0]
                print(f"Using most recent BASE path: {BASE}")
            else:
                # Fall back to the project object as a last resort
                project = SimpleProject(DATA_DIR, DATASET_ID)
                print("Created simple project interface for BASE path")
                
                BASE = os.path.join(DATA_DIR, f"mdai_ucsf_project_{PROJECT_ID}_images_dataset_{DATASET_ID}")
                print(f"Using fallback BASE path: {BASE}")
                
        print(f"Final BASE path: {BASE}")
        print(f"BASE exists: {os.path.exists(BASE)}")

        if os.path.exists(BASE):
           print("✓ Images directory found!")
    # Show what's in the BASE directory
           base_contents = os.listdir(BASE)
           print(f"BASE directory contains {len(base_contents)} items")
           print("First 5 study directories:")
           for item in sorted(base_contents)[:5]:
              print(f"  {item}")
        else:
            print("ERROR: Images directory not found at expected path")
            sys.exit(1)


        print("Skipping zip extraction since images are already extracted")   
        
        # Process free fluid annotations
        # First check if we have free fluid annotations in this file
        free_fluid_annotations = annotations_df[
            ((annotations_df['labelId'] == LABEL_ID_FREE_FLUID) |
             (annotations_df['labelId'] == LABEL_ID_NO_FLUID)) &
            (annotations_df['frameNumber'].notna())
        ].copy()
        
   
        ground_truth_annotations = annotations_df[
            (annotations_df['labelId'] == LABEL_ID_GROUND_TRUTH) &
            (annotations_df['frameNumber'].notna())
        ].copy()
        
        print(f"Found {len(ground_truth_annotations)} ground truth annotations")
        print(f"Found {len(free_fluid_annotations)} free fluid annotations in main file")
        
     
        if len(free_fluid_annotations) == 0 and len(ground_truth_annotations) > 0:
            print("\nNo free fluid annotations found in ground truth file.")
            print("Looking for original annotations file with free fluid data...")
         
            available_files = os.listdir(DATA_DIR)
            original_files = [f for f in available_files 
                            if f.startswith('mdai_ucsf_project_x9N2LJBZ_annotations_dataset_D_V688LQ_') 
                            and f.endswith('.json')
                            and 'labelgroup' not in f
                            and '2025-05-13' not in f]  # Skip corrupted files
            
            if original_files:
                # Sort by date, try older files first
                original_files.sort(key=lambda f: os.path.getmtime(os.path.join(DATA_DIR, f)), reverse=False)
                
                for filename in original_files:
                    test_path = os.path.join(DATA_DIR, filename)
                    try:
                        print(f"Checking {filename} for free fluid annotations...")
                        with open(test_path, 'r') as f:
                            test_annotations_json = json.load(f)
                        
                        # Extract annotations from this file
                        test_datasets = test_annotations_json.get('datasets', [])
                        test_all_annotations = []
                        for dataset in test_datasets:
                            if 'annotations' in dataset:
                                test_all_annotations.extend(dataset['annotations'])
                        
                        test_annotations_df = json_normalize(test_all_annotations, sep='_')
                        
                        # Check for free fluid annotations
                        test_free_fluid = test_annotations_df[
                            ((test_annotations_df['labelId'] == LABEL_ID_FREE_FLUID) |
                             (test_annotations_df['labelId'] == LABEL_ID_NO_FLUID)) &
                            (test_annotations_df['frameNumber'].notna())
                        ].copy()
                        
                        if len(test_free_fluid) > 0:
                            print(f"✓ Found {len(test_free_fluid)} free fluid annotations in {filename}")
                            free_fluid_annotations = test_free_fluid
                            break
                        else:
                            print(f"  No free fluid annotations in {filename}")
                            
                    except Exception as e:
                        print(f"  Error reading {filename}: {e}")
                        continue
            
            if len(free_fluid_annotations) == 0:
                print("ERROR: Could not find any free fluid annotations in any file!")
                print("The feedback loop needs original free fluid annotations to find videos.")
                sys.exit(1)
        
        if len(free_fluid_annotations) > 0:
            free_fluid_annotations['frameNumber'] = free_fluid_annotations['frameNumber'].astype(int)
        
        if len(ground_truth_annotations) > 0:
            ground_truth_annotations['frameNumber'] = ground_truth_annotations['frameNumber'].astype(int)
        
        # Create video paths
        paths = []
        for idx, row in free_fluid_annotations.iterrows():
            path = os.path.join(BASE, row['StudyInstanceUID'], f"{row['SeriesInstanceUID']}.mp4")
            paths.append(path)
        
        free_fluid_annotations['video_path'] = paths    
        free_fluid_annotations['file_exists'] = free_fluid_annotations['video_path'].apply(os.path.exists)
        free_fluid_annotations = free_fluid_annotations[free_fluid_annotations['file_exists']]
        free_fluid_annotations.rename(columns={'data_foreground': 'free_fluid_foreground'}, inplace=True)
        
        print(f"Total Free Fluid Annotations: {len(free_fluid_annotations)}")
        
       
        if args.feedback_loop and (args.ground_truth_single_exam or args.exam_id):
            # Use exam_id if provided, otherwise use ground_truth_single_exam
            exam_id_to_use = args.exam_id if args.exam_id else args.ground_truth_single_exam
            print(f"\n=== FEEDBACK LOOP: FINDING EXAM #{exam_id_to_use} ===")
    
            target_exam = exam_id_to_use
            target_videos = []
    
            print(f"Looking for ground truth annotations for Exam #{target_exam}")
            print(f"Total ground truth annotations available: {len(ground_truth_annotations)}")
    
            # First, let's test one ground truth annotation to make sure our BASE path works
            if len(ground_truth_annotations) > 0:
                test_row = ground_truth_annotations.iloc[0]
                test_video_path = os.path.join(BASE, test_row['StudyInstanceUID'], f"{test_row['SeriesInstanceUID']}.mp4")
                print(f"\nTesting BASE path with first ground truth annotation:")
                print(f"Test path: {test_video_path}")
                print(f"Test path exists: {os.path.exists(test_video_path)}")
    
            # Find all videos with ground truth annotations for this exam
           
            for _, row in ground_truth_annotations.iterrows():
                try:
                    # Check if this annotation belongs to the target exam
                    exam_number = find_exam_number(row['StudyInstanceUID'], annotations_json)
                    if exam_number == target_exam:
                        study_uid = row['StudyInstanceUID']
                        series_uid = row['SeriesInstanceUID']
                
                        
                        video_path = os.path.join(BASE, study_uid, f"{series_uid}.mp4")
                
                        print(f"Checking video for Exam #{exam_number}:")
                        print(f"  Study UID: {study_uid}")
                        print(f"  Series UID: {series_uid}")
                        print(f"  Video path: {video_path}")
                        print(f"  Exists: {os.path.exists(video_path)}")
                
                        if os.path.exists(video_path):
                            
                            video_info = (video_path, study_uid, series_uid)
                            if video_info not in target_videos:
                                target_videos.append(video_info)
                                print(f"✓ Found video for Exam #{exam_number}")
                                try:
                                    file_size = os.path.getsize(video_path)
                                    print(f"  Video size: {file_size:,} bytes")
                                except:
                                    pass
                        else:
                            print(f"✗ Video not found")
                            # Debug what's in the study directory
                            study_dir = os.path.join(BASE, study_uid)
                            print(f"  Study directory: {study_dir}")
                            print(f"  Study dir exists: {os.path.exists(study_dir)}")
                            if os.path.exists(study_dir):
                                try:
                                    contents = os.listdir(study_dir)
                                    print(f"  Study dir contains {len(contents)} items:")
                                    for item in sorted(contents)[:5]:
                                        print(f"    {item}")
                                    if len(contents) > 5:
                                        print(f"    ... and {len(contents)-5} more")
                                except Exception as e:
                                    print(f"  Error listing study dir: {e}")
                except Exception as e:
                    print(f"Error checking annotation for {row.get('StudyInstanceUID', 'unknown')}: {e}")
                    traceback.print_exc()
                    continue
            
            print(f"\nFound {len(target_videos)} video(s) for Exam #{target_exam}")
            
            if len(target_videos) == 0:
                print(f"ERROR: No ground truth videos found for Exam #{target_exam}")
                print("This could be because:")
                print("1. The exam number doesn't exist in the annotations")
                print("2. No ground truth annotations were created for this exam")
                print("3. The video files are missing")
                
             
                print("\nDebugging: Available exam numbers in ground truth annotations:")
                available_exams = set()
                for _, row in ground_truth_annotations.head(20).iterrows():
                    try:
                        exam_num = find_exam_number(row['StudyInstanceUID'], annotations_json)
                        available_exams.add(exam_num)
                    except:
                        pass
                print(f"Sample exam numbers found: {sorted(list(available_exams))}")
                sys.exit(1)
            
            
            matched_annotations = pd.DataFrame()  
            
            # Also prepare the ground truth annotations for these specific videos
            feedback_ground_truth = ground_truth_annotations[
                ground_truth_annotations['StudyInstanceUID'].isin([v[1] for v in target_videos])
            ].copy()
            
            # Add video paths 
            feedback_ground_truth['video_path'] = feedback_ground_truth.apply(
                lambda row: os.path.join(BASE, row['StudyInstanceUID'], f"{row['SeriesInstanceUID']}.mp4"),
                axis=1
            )
            
            print(f"Prepared {len(feedback_ground_truth)} ground truth annotations for feedback loop")

        else:
            # Original matching logic for non-feedback loop cases
            print("\n=== STANDARD MATCHING LOGIC ===")
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
                
                # Sample a subset of the valid annotations
                sample_size = min(5, len(valid_annotations))
                sampled_annotations = valid_annotations.sample(n=sample_size, random_state=42)
                    
                sampled_annotations['issue_type'] = issue_type
                matched_annotations = pd.concat([matched_annotations, sampled_annotations])
            
            print(f"Total matched annotations: {len(matched_annotations)}")
            
            
            target_videos = []
        
        # Check for multi-frame tracking availability
        MULTI_FRAME_AVAILABLE = True
        try:
            import multi_frame_tracking
            from multi_frame_tracking.multi_frame_tracker import MultiFrameTracker
            print("Successfully imported multi_frame_tracking module")
        except ImportError as e:
            print(f"Import error: {e}")
            MULTI_FRAME_AVAILABLE = False
            print("WARNING: Multi-frame tracking module not found.")
        
        if not target_videos:
            print("ERROR: Please specify --exam-id or --ground-truth-single-exam for feedback loop")
            print("Example: --exam-id 64")
            sys.exit(1)
        
        # Run the feedback loop
        print(f"Running feedback loop evaluation on {len(target_videos)} video(s)")
        
        # Initialise flow processor
        flow_processor = OpticalFlowProcessor(method=FLOW_METHOD[0])
        
        # Run the feedback loop
        results = run_ground_truth_feedback_loop(
            target_videos=target_videos, 
            num_iterations=args.iterations,
            matched_annotations=matched_annotations,           
            free_fluid_annotations=free_fluid_annotations,     
            annotations_json=annotations_json,                 
            mdai_client=mdai_client,                         
            project_id=PROJECT_ID,                           
            dataset_id=DATASET_ID,                            
            label_id_ground_truth=LABEL_ID_GROUND_TRUTH,      
            label_id_fluid=LABEL_ID_FLUID_OF,              
            label_id_no_fluid=LABEL_ID_NO_FLUID,              
            label_id_machine=LABEL_ID_MACHINE_GROUP,          
            flow_processor=flow_processor,                      
            exam_id=args.exam_id,                             
            learning_mode=args.learning_mode,                   
            params_file=args.params_file,
            use_genuine_evaluation=args.genuine_evaluation,
            sampling_rate=args.sampling_rate,
            include_method_comparison=args.compare_methods,
            input_sampling_rate=getattr(args, 'input_sampling_rate', args.sampling_rate),      
    evaluation_sampling_rate=getattr(args, 'eval_sampling_rate', 1)
        )
        
        print("\nFeedback loop completed!")
        print("Check the evaluation reports in: output/ground_truth_feedback_loop/")
        sys.exit(0)

    if args.images_dir:
       os.environ["MDAI_IMAGES_DIR"] = args.images_dir
       print(f"Using images directory: {args.images_dir}")

    
    

    elif args.create_ground_truth:
         print("\n==== CREATING GROUND TRUTH DATASET ====")
         
         # Check if exam_id is specified
         if args.exam_id:
             print(f"\n*** FILTERING TO ONLY PROCESS EXAM #{args.exam_id} ***")
         
         load_dotenv('.env')
         
         # Load environment variables properly
         ACCESS_TOKEN = os.getenv('MDAI_TOKEN')
         DATA_DIR = os.getenv('DATA_DIR')
         DOMAIN = os.getenv('DOMAIN')
         PROJECT_ID = os.getenv('PROJECT_ID')
         DATASET_ID = os.getenv('DATASET_ID')
    
         # Get LABEL_ID_GROUND_TRUTH from environment
         LABEL_ID_GROUND_TRUTH = os.getenv("LABEL_ID_GROUND_TRUTH")
         if not LABEL_ID_GROUND_TRUTH:
            print("Error: LABEL_ID_GROUND_TRUTH not defined in .env file")
            sys.exit(1)

    # Initialise MD.ai client
    try:
        domain_value = os.getenv('DOMAIN')
        if not domain_value:
            domain_value = "ucsf.md.ai"  
            print(f"WARNING: Using hardcoded domain {domain_value}")
        
        access_token_value = os.getenv('MDAI_TOKEN')
        if not access_token_value:
            print("ERROR: MDAI_TOKEN not found in environment variables")
            sys.exit(1)
            
        print(f"Using domain: {domain_value}")
        mdai_client = mdai.Client(domain=domain_value, access_token=access_token_value)
        print("Successfully connected to MD.ai client")
    except Exception as e:
        print(f"Error connecting to MD.ai: {str(e)}")
        sys.exit(1)
    
    # Get project
    print("\n=== DETERMINING BASE PATH ===")
    try:
        # First check if MDAI_IMAGES_DIR environment variable is set
        env_base = os.environ.get("MDAI_IMAGES_DIR")
        if env_base and os.path.exists(env_base):
            BASE = env_base
            print(f"Using BASE path from environment: {BASE}")
        else:
            # Try to get the BASE path from MD.ai project
            project = mdai_client.project(
                project_id=PROJECT_ID, 
                dataset_id=DATASET_ID,
                path=DATA_DIR
            )
            
            try:
                BASE = project.get_dataset_by_id(DATASET_ID).images_dir
                print(f"Using project-generated BASE path: {BASE}")
            except Exception as e:
                print(f"Error getting BASE from project: {e}")
                # Fallback to looking for most recent directory
                import glob
                pattern = os.path.join(DATA_DIR, f"mdai_ucsf_project_{PROJECT_ID}_images_dataset_{DATASET_ID}*")
                matching_dirs = glob.glob(pattern)
                
                if matching_dirs:
                    # Sort by modification time (newest first)
                    matching_dirs.sort(key=lambda x: os.path.getmtime(x), reverse=True)
                    BASE = matching_dirs[0]
                    print(f"Using most recent BASE path: {BASE}")
                else:
                    # Last resort fallback
                    BASE = os.path.join(DATA_DIR, f"mdai_ucsf_project_{PROJECT_ID}_images_dataset_{DATASET_ID}")
                    print(f"Using fallback BASE path: {BASE}")

        
        if not os.path.exists(BASE):
            print(f"ERROR: No valid BASE directory found")
            print(f"Checked {BASE} but it doesn't exist")
            print("Please download images or specify the correct path")
            sys.exit(1)
        print(f"BASE path confirmed: {BASE}")
        print(f"BASE exists: {os.path.exists(BASE)}")
        
        # Debug: Show what's inside BASE directory
        print("Listing first 5 items in BASE directory:")
        for item in sorted(os.listdir(BASE))[:5]:
            print(f"  {item}")

        print("\n=== STARTING ANNOTATIONS LOADING ===")
        
        # Set DATA_DIR directly to the correct path
        if 'DATA_DIR' not in locals() and 'DATA_DIR' not in globals():
            DATA_DIR = "/Users/Shreya1/Documents/GitHub/goobusters/data"
            print(f"Using hardcoded DATA_DIR: {DATA_DIR}")
            
        # Make sure other critical variables are defined
        if 'PROJECT_ID' not in locals() and 'PROJECT_ID' not in globals():
            PROJECT_ID = os.getenv('PROJECT_ID', 'x9N2LJBZ')
            print(f"Using fallback PROJECT_ID: {PROJECT_ID}")
            
        if 'DATASET_ID' not in locals() and 'DATASET_ID' not in globals():
            DATASET_ID = os.getenv('DATASET_ID', 'D_V688LQ')
            print(f"Using fallback DATASET_ID: {DATASET_ID}")
            
        if 'LABEL_ID_GROUND_TRUTH' not in locals() and 'LABEL_ID_GROUND_TRUTH' not in globals():
            LABEL_ID_GROUND_TRUTH = os.getenv('LABEL_ID_GROUND_TRUTH', 'L_7DRjNJ')
            print(f"Using fallback LABEL_ID_GROUND_TRUTH: {LABEL_ID_GROUND_TRUTH}")
            
        if 'LABEL_ID_FLUID_OF' not in locals() and 'LABEL_ID_FLUID_OF' not in globals():
            LABEL_ID_FLUID_OF = os.getenv('LABEL_ID_FLUID_OF', 'L_13yPql')
            print(f"Using fallback LABEL_ID_FLUID_OF: {LABEL_ID_FLUID_OF}")
            
        if 'LABEL_ID_NO_FLUID' not in locals() and 'LABEL_ID_NO_FLUID' not in globals():
            LABEL_ID_NO_FLUID = os.getenv('LABEL_ID_NO_FLUID', 'L_wzKV5r')
            print(f"Using fallback LABEL_ID_NO_FLUID: {LABEL_ID_NO_FLUID}")
            
        if 'LABEL_ID_MACHINE_GROUP' not in locals() and 'LABEL_ID_MACHINE_GROUP' not in globals():
            LABEL_ID_MACHINE_GROUP = os.getenv('LABEL_ID_MACHINE_GROUP', 'G_7n3P09')
            print(f"Using fallback LABEL_ID_MACHINE_GROUP: {LABEL_ID_MACHINE_GROUP}")
        
      
        print("Looking for annotation files in DATA_DIR:")
        annotation_files = [f for f in os.listdir(DATA_DIR) if f.startswith('mdai_ucsf_project') and f.endswith('.json')]
        print(f"Found {len(annotation_files)} annotation files")
        
        # If no files found, try looking in various locations
        if not annotation_files:
            print("No annotation files found in DATA_DIR. Trying alternate locations...")
            
            # Check current directory
            alt_dirs = ['.', '..', os.path.join('..', 'data')]
            for alt_dir in alt_dirs:
                if os.path.exists(alt_dir):
                    print(f"Checking {alt_dir}...")
                    alt_files = [f for f in os.listdir(alt_dir) if f.startswith('mdai_ucsf_project') and f.endswith('.json')]
                    if alt_files:
                        print(f"Found {len(alt_files)} annotation files in {alt_dir}")
                        DATA_DIR = alt_dir
                        annotation_files = alt_files
                        break
            
            
            specific_file = "mdai_ucsf_project_x9N2LJBZ_annotations_dataset_D_V688LQ_2025-05-21-155628.json"
            possible_locations = [DATA_DIR, '.', '..', os.path.join('..', 'data')]
            
            for location in possible_locations:
                specific_path = os.path.join(location, specific_file)
                if os.path.exists(specific_path):
                    print(f"Found the specific annotation file: {specific_path}")
                    DATA_DIR = location
                    annotation_files = [specific_file]
                    break
        for i, file in enumerate(annotation_files):
            file_path = os.path.join(DATA_DIR, file)
            file_size = os.path.getsize(file_path) / (1024 * 1024)  # size in MB
            print(f"  {i+1}. {file} - {file_size:.2f} MB")

        if annotation_files:
          
            annotation_files.sort(key=lambda f: os.path.getmtime(os.path.join(DATA_DIR, f)), reverse=True) 

            annotations_file = os.path.join(DATA_DIR, annotation_files[0])
            print(f"\nUsing most recent file: {os.path.basename(annotations_file)}")
           
            # Load the JSON with file size check
            file_size_mb = os.path.getsize(annotations_file) / (1024 * 1024)
            print(f"Annotations file size: {file_size_mb:.2f} MB")

            # Memory usage before loading
            try:
                import psutil
                process = psutil.Process()
                memory_mb = process.memory_info().rss / (1024 * 1024)
                print(f"Memory before loading JSON: {memory_mb:.1f} MB")
            except:
               print("Could not get memory usage")
           
            
            print("Starting JSON loading process...")
            try:
                import signal
               
                def timeout_handler(signum, frame):
                    raise TimeoutError("JSON loading timed out")
               
        
                signal.signal(signal.SIGALRM, timeout_handler)
                signal.alarm(60)  # 60 second timeout
               
                # Load the JSON file
                print("Opening annotations file...")
                with open(annotations_file, 'r') as f:
                    print("Reading JSON content...")
                    annotations_json = json.load(f)
               
                # Cancel timeout
                signal.alarm(0)
               
                print("JSON loaded successfully!")
                print(f"Top-level keys: {list(annotations_json.keys())}")
               
                # Check memory after loading
                try:
                    memory_mb = process.memory_info().rss / (1024 * 1024)
                    print(f"Memory after loading JSON: {memory_mb:.1f} MB")
                except:
                    print("Could not check memory usage after loading")
               
                # Extract all annotations from the JSON
                print("Extracting annotations from JSON...")
                datasets = annotations_json.get('datasets', [])
                all_annotations = []
                for dataset in datasets:
                    if 'annotations' in dataset:
                        all_annotations.extend(dataset['annotations'])
               
                print(f"Extracted {len(all_annotations)} annotations")
               
                # Convert to DataFrame
                from pandas import json_normalize
                annotations_df = json_normalize(all_annotations, sep='_')
                print(f"Created DataFrame with {len(annotations_df)} rows")
               
                # Now process free fluid annotations
                free_fluid_annotations = annotations_df[
                    ((annotations_df['labelId'] == LABEL_ID_FREE_FLUID) |  
                     (annotations_df['labelId'] == LABEL_ID_NO_FLUID)) &
                    (annotations_df['frameNumber'].notna())
                ].copy()
                
                print(f"Found {len(free_fluid_annotations)} free fluid annotations")
                print(f"Using LABEL_ID_FREE_FLUID: {LABEL_ID_FREE_FLUID}")  
                print(f"Using LABEL_ID_NO_FLUID: {LABEL_ID_NO_FLUID}")

                # Extract foreground data from annotations
                def extract_foreground(row):
                    print(f"\nDEBUG: Extracting foreground from row:")
                    print(f"  Row type: {type(row)}")
                    print(f"  Row keys: {row.index.tolist()}")
                    print(f"  Data field type: {type(row.get('data'))}")
                    print(f"  Data field value: {row.get('data')}")
                    print(f"  Data_foreground field type: {type(row.get('data_foreground'))}")
                    print(f"  Data_foreground field value: {row.get('data_foreground')}")
                    
                    # First try data_foreground field
                    if 'data_foreground' in row and row['data_foreground']:
                        print("  Using data_foreground field")
                        return row['data_foreground']
                    
                    # Then try data.foreground
                    if 'data' in row and isinstance(row['data'], dict):
                        print("  Using data.foreground field")
                        return row['data'].get('foreground', [])
                    
                    print("  No valid foreground data found")
                    return []

                print("\nExtracting foreground data from annotations...")
                free_fluid_annotations['free_fluid_foreground'] = free_fluid_annotations.apply(extract_foreground, axis=1)
                print(f"Extracted foreground data for {len(free_fluid_annotations)} annotations")

                # Add video paths and check existence
                print("\nAdding video paths and checking existence...")
                BASE = os.path.join(DATA_DIR, os.getenv('IMAGES_DIR', 'mdai_ucsf_project_x9N2LJBZ_images_dataset_D_V688LQ_2025-05-22-193253'))
                print(f"Using BASE path: {BASE}")

                # Create video paths and check existence
                free_fluid_annotations['video_path'] = free_fluid_annotations.apply(
                    lambda row: os.path.join(BASE, row['StudyInstanceUID'], f"{row['SeriesInstanceUID']}.mp4"), 
                    axis=1
                )

                # Check if files exist
                free_fluid_annotations['file_exists'] = free_fluid_annotations['video_path'].apply(os.path.exists)
                
                # Print the total number of annotations being checked
                print(f"Checking existence of {len(free_fluid_annotations)} video files...")
                
                # Now we can safely count the existing files
                existing_files = sum(free_fluid_annotations['file_exists'])
                print(f"Found {existing_files} existing video files")

                # Count by label type
                if len(free_fluid_annotations) > 0 and 'labelId' in free_fluid_annotations.columns:
                    label_counts = free_fluid_annotations['labelId'].value_counts().to_dict()
                    print("\nAnnotation counts by label:")
                    for label_id, count in label_counts.items():
                        label_type = "FLUID" if label_id == LABEL_ID_FLUID_OF else "NO FLUID"
                        print(f"  - Label {label_id} ({label_type}): {count} annotations")

                # Separate fluid and no-fluid annotations
                print("\nBefore filtering:")
                print(f"Total annotations: {len(free_fluid_annotations)}")

               
                print("\n=== PROCESSING FREE FLUID ANNOTATIONS ===")
                
                # First create target_videos list like in feedback loop
                print("\nCreating video paths...")
                
                # Find the most recent images directory
                def find_latest_images_dir():
                    base_pattern = f"mdai_ucsf_project_{PROJECT_ID}_images_dataset_{DATASET_ID}_*"
                    base_path = os.path.join(DATA_DIR, base_pattern)
                    matching_dirs = glob.glob(base_path)
                    if not matching_dirs:
                        return None
                    # Sort by modification time, most recent first
                    return max(matching_dirs, key=os.path.getmtime)
                
                # Get BASE path from environment or find latest
                env_base = os.environ.get("MDAI_IMAGES_DIR")
                if env_base:
                    BASE = env_base
                    print(f"Using BASE path from environment: {BASE}")
                else:
                    latest_dir = find_latest_images_dir()
                    if latest_dir:
                        BASE = latest_dir
                        print(f"Using most recent BASE path: {BASE}")
                    else:
                        # Fallback to project path
                        class SimpleProject:
                            def __init__(self, data_dir, dataset_id):
                                self.data_dir = data_dir
                            
                            def get_dataset_by_id(self, dataset_id):
                                images_dir = os.path.join(self.data_dir, f"mdai_ucsf_project_{PROJECT_ID}_images_dataset_{dataset_id}")
                                return SimpleDataset(images_dir)
                        
                        class SimpleDataset:
                            def __init__(self, images_dir):
                                self.images_dir = images_dir
                        
                        project = SimpleProject(DATA_DIR, DATASET_ID)
                        BASE = project.get_dataset_by_id(DATASET_ID).images_dir
                        print(f"Using BASE path from project: {BASE}")

                print(f"free_fluid_annotations size before creating target_videos: {len(free_fluid_annotations)}")

                if len(free_fluid_annotations) == 0:
                   print("❌ free_fluid_annotations is empty! Recreating from original data...")
    
    # Recreate from the original DataFrame
                free_fluid_annotations = annotations_df[
                  ((annotations_df['labelId'] == LABEL_ID_FREE_FLUID) |
                  (annotations_df['labelId'] == LABEL_ID_NO_FLUID)) &
                   (annotations_df['frameNumber'].notna())
                ].copy()
    
    # Re-process the basic fields
                free_fluid_annotations['frameNumber'] = free_fluid_annotations['frameNumber'].astype(int)
                free_fluid_annotations.rename(columns={'data_foreground': 'free_fluid_foreground'}, inplace=True)
    
                print(f"✓ Recreated {len(free_fluid_annotations)} annotations")
                
                
                # Create target_videos list from free fluid annotations
                target_videos = []
                for _, row in free_fluid_annotations.iterrows():
                    study_uid = row['StudyInstanceUID']
                    series_uid = row['SeriesInstanceUID']
                    video_path = os.path.join(BASE, study_uid, f"{series_uid}.mp4")
                    target_videos.append((video_path, study_uid, series_uid))
                
                print(f"Found {len(target_videos)} initial video paths")
                
                # Filter target videos for specific exam 
                if args.exam_id:
                    print(f"\n=== FILTERING FOR EXAM {args.exam_id} ===")
                    filtered_videos = []
                    for video_path, study_uid, series_uid in target_videos:
                        try:
                            video_exam_id = find_exam_number(study_uid, annotations_json)
                            if str(video_exam_id) == str(args.exam_id):
                                print(f"✓ Found matching video: {video_path}")
                                filtered_videos.append((video_path, study_uid, series_uid))
                        except Exception as e:
                            print(f"Error checking exam number for {study_uid}: {str(e)}")
                    
                    if not filtered_videos:
                        print(f"❌ No videos found for exam ID {args.exam_id}")
                        sys.exit(1)
                    
                    target_videos = filtered_videos
                
             
                video_paths = [v[0] for v in target_videos]
                study_series_pairs = [(v[1], v[2]) for v in target_videos]
                
              
                existing_videos = []
                for video_path, study_uid, series_uid in target_videos:
                    if os.path.exists(video_path):
                        existing_videos.append((video_path, study_uid, series_uid))
                    else:
                        print(f"Video not found: {video_path}")
                
                if not existing_videos:
                    print("❌ No video files found on disk")
                    sys.exit(1)
                
                
                video_paths = [v[0] for v in existing_videos]
                study_series_pairs = [(v[1], v[2]) for v in existing_videos]
                
                print(f"\nAfter filtering:")
                print(f"  - Videos found: {len(video_paths)}")
                print(f"  - Study-series pairs: {len(study_series_pairs)}")
                
                # Filter free fluid annotations to match the filtered videos
                free_fluid_annotations = free_fluid_annotations[
                    free_fluid_annotations['StudyInstanceUID'].isin([s[0] for s in study_series_pairs])
                ]
                print(f"  - Free fluid annotations: {len(free_fluid_annotations)}")
                
                if len(free_fluid_annotations) > 0 and len(video_paths) > 0:

                    from src.multi_frame_tracking.opticalflowprocessor import OpticalFlowProcessor
                    flow_processor = OpticalFlowProcessor(method='dis')  
                    print("✓ Initialized flow processor")
                    result = create_ground_truth_dataset(
                        video_paths=video_paths,
                        study_series_pairs=study_series_pairs,
                        flow_processor=flow_processor,
                        output_dir=os.path.join(OUTPUT_DIR, "ground_truth"),
                        mdai_client=mdai_client,
                        project_id=PROJECT_ID,
                        dataset_id=DATASET_ID,
                        ground_truth_label_id=LABEL_ID_GROUND_TRUTH,
                        matched_annotations=free_fluid_annotations,  
                        free_fluid_annotations=free_fluid_annotations,
                        label_id_fluid=LABEL_ID_FREE_FLUID, 
                        label_id_no_fluid=LABEL_ID_NO_FLUID,
                        label_id_machine=LABEL_ID_MACHINE_GROUP,
                        annotations_json=annotations_json,
                        args=args,
                        label_ids=LABEL_IDS,
                        base_path=BASE,
                        upload=args.upload
                    )
                else:
                    print("Error: No valid free fluid annotations or video files found. Cannot create ground truth dataset.")
                    sys.exit(1)
            
            except TimeoutError as e:
                print(f"ERROR: {str(e)}")
                print("The JSON loading process timed out. The file may be too large or corrupted.")
                print("Try using a smaller JSON file or increase the timeout.")
                sys.exit(1)
            except json.JSONDecodeError as e:
                print(f"ERROR: Invalid JSON format: {str(e)}")
                print(f"Error details: {e}")
                print("The annotations file appears to be corrupted.")
                print("Try using a different file.")
                sys.exit(1)
            except Exception as e:
                print(f"ERROR during JSON loading or processing: {str(e)}")
                traceback.print_exc()
                sys.exit(1)
        else:
            print("No annotation files found in DATA_DIR")
            sys.exit(1)
       
    except Exception as e:
        print(f"ERROR setting BASE path: {str(e)}")
        traceback.print_exc()
        sys.exit(1)
   
    # Exit after ground truth creation
    sys.exit(0)


else:
     
    load_dotenv('.env')
    ACCESS_TOKEN = os.getenv('MDAI_TOKEN')
    DATA_DIR = os.getenv('DATA_DIR')
    DOMAIN = os.getenv('DOMAIN')
    PROJECT_ID = os.getenv('PROJECT_ID')
    DATASET_ID = os.getenv('DATASET_ID')

    PROJECT_ID = os.getenv('PROJECT_ID')
    DATASET_ID = os.getenv('DATASET_ID')
    
    # Define LABEL_IDS before it's used anywhere
    LABEL_IDS = {
        "disappear_reappear": os.getenv("LABEL_ID_DISAPPEAR_REAPPEAR"),
        "branching_fluid": os.getenv("LABEL_ID_BRANCHING_FLUID"),
        "multiple_distinct": os.getenv("LABEL_ID_MULTIPLE_DISTINCT")
    }
    
    mdai_client = mdai.Client(domain=DOMAIN, access_token=ACCESS_TOKEN)
    project = mdai_client.project(
        project_id=PROJECT_ID, 
        dataset_id=DATASET_ID,
        path=DATA_DIR
    ) 
    BASE = project.get_dataset_by_id(DATASET_ID).images_dir
    print(f"\nBASE path: {BASE}")
    print(f"BASE exists: {os.path.exists(BASE)}")
    
    
    if __name__ == "__main__":
        process_videos_with_tracking()
  


debug_print(f"=== DEBUG LOG ENDED AT {time.ctime()} ===")
debug_log.close()



