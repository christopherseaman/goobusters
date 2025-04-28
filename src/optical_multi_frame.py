# Import libraries and set constants
# === Imports ===
import sys
import os
import logging
import json
import shutil
import traceback
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv
import mdai
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from pandas import json_normalize
from multi_frame_tracking.opticalflowprocessor import OpticalFlowProcessor
from optical_mdai_import import save_combined_video


sys.setrecursionlimit(10000)


try:
    # Print Python path for debugging
    print("Python path:", sys.path)
    
    # First import just the module to debug any issues
    import multi_frame_tracking
    print("Successfully imported multi_frame_tracking module")
    
    # Then try to import the specific classes
    from multi_frame_tracking.multi_frame_tracker import MultiFrameTracker, process_video_with_multi_frame_tracking
    print("Successfully imported MultiFrameTracker and process_video_with_multi_frame_tracking")
    
    MULTI_FRAME_AVAILABLE = True
    print("Multi-frame tracking module loaded successfully.")
except ImportError as e:
    print(f"Import error: {e}")
    import traceback
    traceback.print_exc()
    MULTI_FRAME_AVAILABLE = False
    print("WARNING: Multi-frame tracking module not found. Only single-frame tracking will be available.")
except RecursionError as e:
    print(f"Recursion error: {e}")
    import traceback
    traceback.print_exc()
    MULTI_FRAME_AVAILABLE = False
    print("WARNING: RecursionError in multi-frame tracking module. Only single-frame tracking will be available.")


# Enable debug mode
DEBUG_MODE = True
DEBUG_SAMPLE_SIZE = 2
DEBUG_ISSUE_TYPES = ["multiple_distinct"]

# Set tracking mode: 'single', 'multi', or 'both'
TRACKING_MODE = 'multi'  # this can be changed to switch between tracking modesa

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
    
    # Redirect stdout and stderr
    #sys.stdout = PrintToLogger()
    #sys.stderr = PrintToLogger()
    
    logging.info(f"Logging started, output to: {log_file}")
    return log_file


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
if not os.path.exists(ANNOTATIONS):
    ANNOTATIONS = "/Users/Shreya1/Documents/GitHub/goobusters/data/mdai_ucsf_project_x9N2LJBZ_annotations_2025-04-01-035131.json"
LABEL_ID_FREE_FLUID = os.getenv("LABEL_ID_FREE_FLUID")
LABEL_IDS = {
    "disappear_reappear": os.getenv("LABEL_ID_DISAPPEAR_REAPPEAR"),
    "branching_fluid": os.getenv("LABEL_ID_BRANCHING_FLUID"),
    "multiple_distinct": os.getenv("LABEL_ID_MULTIPLE_DISTINCT"),
}
LABEL_ID_MACHINE_GROUP = os.getenv("LABEL_ID_MACHINE_GROUP")  # L_1A4xv7
LABEL_ID_FLUID_OF = "L_7BGQ0l"       # L_JykNe7

# Validate that they're loaded
if None in [LABEL_ID_MACHINE_GROUP, LABEL_ID_FLUID_OF]:
    raise ValueError("Machine label IDs not properly set in .env file")

# Validate access token
if ACCESS_TOKEN is None:
    raise ValueError("ACCESS_TOKEN is not set.")
print("ACCESS_TOKEN is set")

# Start MD.ai client
try:
    mdai_client = mdai.Client(domain=DOMAIN, access_token=ACCESS_TOKEN)
    project = mdai_client.project(project_id=PROJECT_ID, path=DATA_DIR)
    print("Successfully connected to MD.ai")
except Exception as e:
    print(f"Error connecting to MD.ai: {str(e)}")
    raise

# Load and normalise annotations
with open(ANNOTATIONS, 'r') as f:
    annotations_json = json.load(f)

datasets = annotations_json.get('datasets', [])
all_annotations = []
for dataset in datasets:
    if 'annotations' in dataset:
        all_annotations.extend(dataset['annotations'])

annotations_df = json_normalize(all_annotations, sep='_')

# Step 1: Filter free fluid annotations separately
free_fluid_annotations = annotations_df[
    (annotations_df['labelId'] == LABEL_ID_FREE_FLUID) &
    (annotations_df['frameNumber'].notna())
].copy()
free_fluid_annotations['frameNumber'] = free_fluid_annotations['frameNumber'].astype(int)
print("Columns in free_fluid_annotations:", free_fluid_annotations.columns)
# Construct video paths for free fluid annotations
BASE = project.get_dataset_by_id(DATASET_ID).images_dir
free_fluid_annotations['video_path'] = free_fluid_annotations.apply(
    lambda row: os.path.join(BASE, row['StudyInstanceUID'], f"{row['SeriesInstanceUID']}.mp4"), axis=1
)
free_fluid_annotations['file_exists'] = free_fluid_annotations['video_path'].apply(os.path.exists)
free_fluid_annotations = free_fluid_annotations[free_fluid_annotations['file_exists']]

free_fluid_annotations.rename(columns={'data_foreground': 'free_fluid_foreground'}, inplace=True)

working_videos = free_fluid_annotations[free_fluid_annotations['file_exists']]
if not working_videos.empty:
    sample_video = working_videos.iloc[0]
    print(f"\nWORKING VIDEO PATH FOUND:")
    print(f"Path: {sample_video['video_path']}")
    print(f"Study UID: {sample_video['StudyInstanceUID']}")
    print(f"Series UID: {sample_video['SeriesInstanceUID']}")
    print(f"Frame Number: {sample_video['frameNumber']}")


print(f"Total Free Fluid Annotations: {len(free_fluid_annotations)}")

# Step 2: Link issue-type annotations to free fluid annotations
matched_annotations = pd.DataFrame()

for issue_type, label_id in LABEL_IDS.items():
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

    print(merged_annotations['free_fluid_foreground'])

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
    

def track_frames(cap, start_frame, end_frame, initial_mask, debug_dir, forward=True, pbar=None, flow_processor=None):
    """
    Track frames using optical flow with enhanced error handling and diagnostics.
    
    Args:
        cap: Video capture object
        start_frame: Starting frame number
        end_frame: Ending frame number
        initial_mask: Initial mask to track
        debug_dir: Directory for debug outputs
        forward: If True, track forward; if False, track backward
        pbar: Progress bar object
        flow_processor: Optical flow processor object
    
    Returns:
        List of tuples (frame_idx, frame, mask, flow, flow_mask, adjusted_mask)
    """
    # Define target frames for detailed analysis
    TARGET_FRAMES = [102, 103, 104, 105, 106, 107]  # Add specific frame numbers you want to analyse
    VERBOSE_DEBUGGING = True 
    
    frames = []
    step = 1 if forward else -1
    frame_idx = start_frame
    direction = "forward" if forward else "backward"
    consecutive_errors = 0
    max_consecutive_errors = 3
    total_frames_processed = 0
    total_frames_skipped = 0
    
    print(f"\nStarting {direction} tracking from frame {start_frame} to {end_frame}")
    print(f"Initial frame shape: {initial_mask.shape}")
    
    # Set the video capture to the starting frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, prev_frame = cap.read()
    if not ret:
        print(f"Failed to read starting frame {start_frame}.")
        return frames
        
    try:
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        mask = initial_mask.astype(float)
        
        # Save initial frame and mask for debugging
        if DEBUG_MODE:
            debug_path = os.path.join(debug_dir, f'initial_frame_{frame_idx:04d}.png')
            cv2.imwrite(debug_path, prev_frame)
            debug_path = os.path.join(debug_dir, f'initial_mask_{frame_idx:04d}.png')
            cv2.imwrite(debug_path, (mask * 255).astype(np.uint8))
    except Exception as e:
        print(f"Error initializing first frame: {str(e)}")
        return frames

    while (forward and frame_idx <= end_frame) or (not forward and frame_idx >= end_frame):
        try:
            # Print progress every 10 frames
            if total_frames_processed % 10 == 0:
                print(f"\nProcessing frame {frame_idx}")
                print(f"Frames processed: {total_frames_processed}")
                print(f"Frames skipped: {total_frames_skipped}")
                print(f"Consecutive errors: {consecutive_errors}")
            
            # Read the current frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                print(f"Failed to read frame {frame_idx}. Ending tracking.")
                break

            # Convert to grayscale
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Initialize for first frame
            if frame_idx == start_frame:
                flow = None
                flow_mask = np.zeros_like(mask)
                adjusted_mask = np.zeros_like(mask)
                new_mask = mask.copy()
                print_mask_stats(mask, frame_idx)  
                frames.append((frame_idx, frame, new_mask, flow, flow_mask, adjusted_mask))  
                total_frames_processed += 1
            else:
              
                try:
                    # Get flow field
                    flow = flow_processor.apply_optical_flow(prev_gray, frame_gray, mask)
                    if flow is None:
                        print(f"Flow computation returned None for frame {frame_idx}")
                        consecutive_errors += 1
                        total_frames_skipped += 1
                        frame_idx += step
                        if consecutive_errors >= max_consecutive_errors:
                            print(f"Too many consecutive errors, stopping at frame {frame_idx}")
                            break
                        continue
                        
                    # Calculate flow mask
                    flow_mask = flow_processor.warp_mask(mask, flow)
                    
                    # Verify flow mask is valid
                    if flow_mask is None or np.isnan(flow_mask).any():
                        print(f"Invalid flow mask at frame {frame_idx}")
                        consecutive_errors += 1
                        total_frames_skipped += 1
                        frame_idx += step
                        continue
                        
                    # Calculate flow metrics for debugging
                    mean_flow = np.mean(np.abs(flow))
                    if mean_flow < 0.01:
                        print(f"Warning: Very small flow detected at frame {frame_idx}: {mean_flow}")
                    
                    # Calculate binary mask areas and IoU
                    binary_mask = (mask > 0.5).astype(np.uint8)
                    binary_flow_mask = (flow_mask > 0.5).astype(np.uint8)
                    mask_area = np.sum(binary_mask)
                    flow_mask_area = np.sum(binary_flow_mask)
                    area_ratio = flow_mask_area / mask_area if mask_area > 0 else 0
                    
                    # Calculate IoU
                    intersection = np.sum(np.logical_and(binary_mask, binary_flow_mask))
                    union = np.sum(np.logical_or(binary_mask, binary_flow_mask))
                    iou = intersection / union if union > 0 else 0
                    
                    # Enhanced debugging for target frames
                    if frame_idx in TARGET_FRAMES or VERBOSE_DEBUGGING:
                        print(f"\n===== DETAILED ANALYSIS FOR FRAME {frame_idx} =====")
                        print(f"Before adjustment: mask sum = {np.sum(mask)}, flow_mask sum = {np.sum(flow_mask)}")
                        print(f"Mask ratio: flow_mask/mask = {np.sum(flow_mask)/np.sum(mask):.4f}")
                        print(f"Binary mask areas - Original: {mask_area}, Flow: {flow_mask_area}, Ratio: {area_ratio:.4f}")
                        print(f"Mask IoU: {iou:.4f}")
                        
                        # Detailed analysis of the flow
                        if np.any(mask > 0):
                            masked_flow = flow.copy()
                            masked_flow[mask <= 0] = 0
                            mean_x = np.mean(masked_flow[..., 0][mask > 0])
                            mean_y = np.mean(masked_flow[..., 1][mask > 0])
                            print(f"Average flow vector in mask: x={mean_x:.2f}, y={mean_y:.2f}")
                            
                            # Calculate flow magnitude statistics
                            flow_mag = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
                            masked_mag = flow_mag * (mask > 0)
                            if np.any(masked_mag > 0):
                                max_mag = np.max(masked_mag[masked_mag > 0])
                                mean_mag = np.mean(masked_mag[masked_mag > 0])
                                print(f"Flow magnitude in mask: max={max_mag:.2f}, mean={mean_mag:.2f}")
                    else:
                        
                        print(f"Before adjustment: mask sum = {np.sum(mask)}, flow_mask sum = {np.sum(flow_mask)}")
                        print(f"Binary mask areas - Original: {mask_area}, Flow: {flow_mask_area}, Ratio: {area_ratio:.4f}")
                        print(f"Mask IoU: {iou:.4f}")
                        
                    # Blend the masks with modified weights
                    adjusted_mask = flow_mask
                    blended_mask = (0.3 * mask + 0.7 * adjusted_mask).astype(float)
                   
                    # Enhanced debugging for target frames
                    if frame_idx in TARGET_FRAMES or VERBOSE_DEBUGGING:
                        print(f"After blending (0.3*mask + 0.7*flow_mask): new_mask sum = {np.sum(blended_mask)}")
                        print(f"Blend ratio: blended/original = {np.sum(blended_mask)/np.sum(mask):.4f}")
                        
                        # Save a visualization of the masks before and after blending
                        if DEBUG_MODE:
                            debug_compare = np.zeros((mask.shape[0], mask.shape[1]*3), dtype=np.uint8)
                            debug_compare[:, :mask.shape[1]] = (mask * 255).astype(np.uint8)
                            debug_compare[:, mask.shape[1]:mask.shape[1]*2] = (flow_mask * 255).astype(np.uint8)
                            debug_compare[:, mask.shape[1]*2:] = (blended_mask * 255).astype(np.uint8)
                            cv2.imwrite(os.path.join(debug_dir, f'mask_blending_{frame_idx:04d}.png'), debug_compare)
                    else:
                        # Basic diagnostic output for non-target frames
                        print(f"After blending: new_mask sum = {np.sum(blended_mask)}")
                   
                    new_mask = np.clip(blended_mask, 0, 1)

                    if np.sum(new_mask) > np.sum(mask) * 1.05:  # 5% growth limit
                        # Enhanced debugging for target frames
                        if frame_idx in TARGET_FRAMES or VERBOSE_DEBUGGING:
                            growth_factor = np.sum(new_mask)/np.sum(mask)
                            print(f"GROWTH CONSTRAINT APPLIED - Ratio before constraint: {growth_factor:.4f}")
                            print(f"Threshold used: {1.0 - (np.sum(mask) / np.sum(new_mask)):.4f}")
                        else:
                            # Basic diagnostic output for non-target frames
                            print(f"Applying growth constraint. Ratio: {np.sum(new_mask)/np.sum(mask):.2f}")
                        
                        # Scale back to the original size
                        scale_factor = np.sum(mask) / np.sum(new_mask)
                        # Applying a threshold that gets stricter as the mask grows
                        threshold = 1.0 - scale_factor
                        new_mask = (new_mask > threshold).astype(float)
                        print(f"Frame {frame_idx} - Mask constrained: growth factor {1/scale_factor:.2f}x, threshold {threshold:.2f}")

                    # Enhanced debugging for target frames
                    if frame_idx in TARGET_FRAMES or VERBOSE_DEBUGGING:
                        print(f"Final mask sum = {np.sum(new_mask)}")
                        print(f"Final ratio: {np.sum(new_mask)/np.sum(mask):.4f}")
                        
                        # Calculate binary area of final mask
                        binary_new_mask = (new_mask > 0.5).astype(np.uint8)
                        final_mask_area = np.sum(binary_new_mask)
                        final_area_ratio = final_mask_area / mask_area if mask_area > 0 else 0
                        print(f"Final binary mask area: {final_mask_area}, Ratio: {final_area_ratio:.4f}")
                        
                        print(f"Mask coverage: {np.mean(new_mask) * 100:.2f}%")
                        print(f"===== END DETAILED ANALYSIS FOR FRAME {frame_idx} =====\n")
                    else:
                        # Basic diagnostic output for non-target frames
                        print(f"Final mask sum = {np.sum(new_mask)}")
                        binary_new_mask = (new_mask > 0.5).astype(np.uint8)
                        final_mask_area = np.sum(binary_new_mask)
                        final_area_ratio = final_mask_area / mask_area if mask_area > 0 else 0
                        print(f"Final binary mask area: {final_mask_area}, Ratio: {final_area_ratio:.4f}")
                    
               
                    print_mask_stats(new_mask, frame_idx)
                    
                   
                    consecutive_errors = 0
                    total_frames_processed += 1
                    
                   
                    frames.append((frame_idx, frame, new_mask, flow, flow_mask, adjusted_mask))
                    
                except Exception as e:
                    print(f"Error computing flow at frame {frame_idx}: {str(e)}")
                    traceback.print_exc()
                    consecutive_errors += 1
                    total_frames_skipped += 1
                    if consecutive_errors >= max_consecutive_errors:
                        print(f"Too many consecutive errors, stopping at frame {frame_idx}")
                        break
                    frame_idx += step
                    continue

            # Save debug visualisation with flow
            if DEBUG_MODE:
                print(f"DEBUG_MODE is True, creating debug frame for frame {frame_idx}")
                try:
                    debug_frame = debug_visualize(
                        frame=frame, 
                        initial_mask=initial_mask, 
                        flow_mask=flow_mask if frame_idx != start_frame else np.zeros_like(mask), 
                        adjusted_mask=adjusted_mask if frame_idx != start_frame else np.zeros_like(mask), 
                        final_mask=new_mask, 
                        frame_number=frame_idx,
                        flow=flow
                    )
                    os.makedirs(debug_dir, exist_ok=True)
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    debug_filename = f'debug_frame_{frame_idx:04d}_{timestamp}.png'
                    cv2.imwrite(os.path.join(debug_dir, debug_filename), debug_frame)
                except Exception as e:
                    print(f"Error saving debug frame {frame_idx}: {str(e)}")

            # Update for next iteration
            prev_gray = frame_gray.copy()
            mask = new_mask.copy()
            frame_idx += step

            # Update progress bar
            if pbar:
                pbar.update(1)

        except Exception as e:
            print(f"Unexpected error processing frame {frame_idx}: {str(e)}")
            traceback.print_exc()
            consecutive_errors += 1
            total_frames_skipped += 1
            if consecutive_errors >= max_consecutive_errors:
                break
            frame_idx += step
            continue

    # Print final statistics
    print(f"\n{direction.capitalize()} tracking completed:")
    print(f"Total frames processed: {total_frames_processed}")
    print(f"Total frames skipped: {total_frames_skipped}")
    print(f"Final frame count: {len(frames)}")
    
    return frames

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

def process_videos_with_tracking():
    """
    Main processing loop for tracking free fluid in ultrasound videos.
    This function processes all matched annotations with both single-frame and multi-frame tracking
    based on the TRACKING_MODE setting.
    """
    print(f"\nProcessing videos with tracking mode: {TRACKING_MODE}")
    
    # Determine which issue types to process based on debug mode
    issue_types_to_process = DEBUG_ISSUE_TYPES if DEBUG_MODE else list(LABEL_IDS.keys())
    print(f"Processing issue types: {issue_types_to_process}")
    
    # Create output directories if they don't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(MULTI_FRAME_DIR, exist_ok=True)
    
    # Initialise video counter
    videos_processed = 0
    
    # Initialise optical flow processor
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
            
            # Find the annotation for frame 102
            frame_102_annotation = free_fluid_annotations[
                (free_fluid_annotations['StudyInstanceUID'] == study_uid) &
                (free_fluid_annotations['SeriesInstanceUID'] == series_uid) &
                (free_fluid_annotations['frameNumber'] == 102)
            ]
            
            if len(frame_102_annotation) > 0:
                # Use this annotation
                row = frame_102_annotation.iloc[0]
                frame_number = 102
                
                # Get free fluid polygons from this specific annotation
                free_fluid_polygons = row['free_fluid_foreground']
                
                print(f"\nProcessing video {videos_processed + 1}/{len(type_annotations)}")
                print(f"Video: {video_path}")
                print(f"Frame number: {frame_number}")
            else:
                print("No annotation found for frame 102, skipping this video")
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
                
                # Create a specific output directory for this video
                video_output_dir = os.path.join(OUTPUT_DIR, f"{study_uid}_{series_uid}")
                os.makedirs(video_output_dir, exist_ok=True)
                if not os.path.exists(video_output_dir):
                    print(f"ERROR: Could not create directory: {video_output_dir}")
                    continue
                else:
                    print(f"Using output directory: {video_output_dir}")
                
                # Store results from both tracking methods
                single_frame_results = {}  # Store single-frame results here
                multi_frame_results = {}   # Store multi-frame results here
                
                # Check which tracking modes to apply
                if TRACKING_MODE in ['single', 'both']:
                    print("\nPerforming single-frame tracking...")
                    single_output_dir = os.path.join(video_output_dir, "single_frame")
                    os.makedirs(single_output_dir, exist_ok=True)
                    
                    # Create debug directory for single-frame tracking
                    single_debug_dir = os.path.join(single_output_dir, "debug")
                    os.makedirs(single_debug_dir, exist_ok=True)
                    
                    # Output video path for single-frame tracking
                    single_output_video = os.path.join(single_output_dir, "single_frame_tracking.mp4")
                    
                    # Use save_combined_video for single-frame tracking
                    single_frame_result = save_combined_video(
                        video_path=video_path,
                        output_video_path=single_output_video,
                        initial_mask=initial_mask,
                        frame_number=frame_number,
                        debug_dir=single_debug_dir,
                        flow_processor=flow_processor,
                        mdai_client=None,  #upload to md.ai or not
                        study_uid=study_uid,
                        series_uid=series_uid,
                        overlay_color=(0, 0, 255),  # Red for single-frame tracking
                        overlay_alpha=0.3,
                        add_info=True,
                        upload_to_mdai=False  #upload to md.ai or not 
                    )
                    
                    # Extract frame results from the output
                    if single_frame_result:
                        print(f"Single-frame tracking completed successfully")
                        
                        # Collect the frames from backward and forward tracking for comparison
                        cap = cv2.VideoCapture(video_path)
                        
                        # Get frame for the initial frame
                        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
                        ret, frame = cap.read()
                        if ret:
                            single_frame_results[frame_number] = {"mask": initial_mask, "type": "annotation"}
                            
                        # Process backward frames
                        backward_frames = track_frames(
                            cap=cap,
                            start_frame=frame_number,
                            end_frame=0,
                            initial_mask=initial_mask,
                            debug_dir=single_debug_dir,
                            forward=False,
                            flow_processor=flow_processor
                        )
                        
                        # Add backward frames to results
                        for frame_data in backward_frames:
                            frame_idx, _, mask, _, _, _ = frame_data
                            if frame_idx != frame_number:  # Skip the initial frame
                                single_frame_results[frame_idx] = {"mask": mask, "type": "predicted_backward"}
                        
                        # Process forward frames
                        forward_frames = track_frames(
                            cap=cap,
                            start_frame=frame_number,
                            end_frame=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))-1,
                            initial_mask=initial_mask,
                            debug_dir=single_debug_dir,
                            forward=True,
                            flow_processor=flow_processor
                        )
                        
                        # Add forward frames to results
                        for frame_data in forward_frames:
                            frame_idx, _, mask, _, _, _ = frame_data
                            if frame_idx != frame_number:  # Skip the initial frame
                                single_frame_results[frame_idx] = {"mask": mask, "type": "predicted_forward"}
                        
                        cap.release()
                    else:
                        print(f"Single-frame tracking failed")
                
                if TRACKING_MODE in ['multi', 'both'] and MULTI_FRAME_AVAILABLE:
                    # Multi-frame tracking
                    print("\n===== ATTEMPTING TO USE MULTI-FRAME TRACKING =====")
                    print(f"TRACKING_MODE: {TRACKING_MODE}")
                    print(f"MULTI_FRAME_AVAILABLE: {MULTI_FRAME_AVAILABLE}")
                    multi_frame_output_dir = os.path.join(MULTI_FRAME_DIR, f"{study_uid}_{series_uid}")
                    os.makedirs(multi_frame_output_dir, exist_ok=True)
                    if not os.path.exists(multi_frame_output_dir):
                        print(f"ERROR: Could not create directory: {multi_frame_output_dir}")
                        continue
                    else:
                        print(f"Using output directory: {multi_frame_output_dir}")
                    
                    # Filter annotations for this video to include all frames
                    video_annotations = free_fluid_annotations[
                        (free_fluid_annotations['StudyInstanceUID'] == study_uid) &
                        (free_fluid_annotations['SeriesInstanceUID'] == series_uid)
                    ].copy()
                    
                    if len(video_annotations) > 0:
                        print(f"Found {len(video_annotations)} annotations for this video")
                        
                        # Call the multi-frame processing function
                        result = process_video_with_multi_frame_tracking(
                            video_path=video_path,
                            annotations_df=video_annotations,
                            study_uid=study_uid,
                            series_uid=series_uid,
                            flow_processor=flow_processor,
                            output_dir=multi_frame_output_dir,
                            mdai_client=mdai_client if 'mdai_client' in globals() else None,
                            label_id_fluid=LABEL_ID_FREE_FLUID,
                            label_id_machine=LABEL_ID_MACHINE_GROUP,
                            project_id=PROJECT_ID,
                            dataset_id=DATASET_ID,
                            upload_to_mdai=False,
                            debug_mode=True  
                        )
                        
                        # Store multi-frame results
                        multi_frame_results = result.get('all_masks', {})
                        
                        print(f"Multi-frame tracking completed. Results:")
                        print(f"  Annotated frames: {result.get('annotated_frames', 0)}")
                        print(f"  Predicted frames: {result.get('predicted_frames', 0)}")
                        print(f"  Annotation types: {result.get('annotation_types', {})}")
                        print(f"  Output video: {result.get('output_video', 'None')}")
                        
                        # Verify the output video exists
                        output_video = result.get('output_video', 'None')
                        if os.path.exists(output_video):
                            print(f"Confirmed: Output video created at {output_video} ({os.path.getsize(output_video)} bytes)")
                        else:
                            print(f"WARNING: Output video not found at expected path: {output_video}")
                    else:
                        print("No annotations found for this video, skipping multi-frame tracking")
                
                # Create comparison video if both tracking methods were used
                if TRACKING_MODE == 'both' and single_frame_results and multi_frame_results:
                    print("\nCreating comparison video...")
                    comparison_path = os.path.join(multi_frame_output_dir, "comparison.mp4")
                    create_comparison_video(
                        video_path=video_path,
                        single_predictions=single_frame_results,
                        multi_predictions=multi_frame_results,
                        ground_truth=None,  
                        output_path=comparison_path
                    )
                    print(f"Comparison video saved to: {comparison_path}")
                    
                    # Calculate metrics
                    print("\nCalculating tracking metrics...")
                    metrics = calculate_tracking_metrics(
                        single_predictions=single_frame_results,
                        multi_predictions=multi_frame_results
                    )
                    
                    # Print metrics
                    print("\nTracking Metrics Comparison:")
                    print(f"  Temporal consistency: Single={metrics['single']['temporal_consistency']:.4f}, Multi={metrics['multi']['temporal_consistency']:.4f}")
                    print(f"  Failure count: Single={metrics['single']['failure_count']}, Multi={metrics['multi']['failure_count']}")
                    print(f"  Area stability: Single={metrics['single']['area_stability']:.4f}, Multi={metrics['multi']['area_stability']:.4f}")
                    
                    # Save metrics to file
                    metrics_path = os.path.join(multi_frame_output_dir, "metrics.json")
                    with open(metrics_path, 'w') as f:
                        json.dump(metrics, f, indent=2)
                    print(f"Metrics saved to: {metrics_path}")
                    
                
                # Increment counter
                videos_processed += 1
                
                # Close the video
                cap.release()
                
            except Exception as e:
                print(f"Error processing video: {str(e)}")
                traceback.print_exc()
                continue
    
    print(f"\nProcessing completed. Total videos processed: {videos_processed}")

if __name__ == "__main__":
   
    process_videos_with_tracking()