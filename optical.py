# Import libraries and set constants
from dotenv import load_dotenv
import os
import mdai
import cv2
import pandas as pd
import numpy as np
from tqdm import tqdm
import torch
from opticalflowprocessor import OpticalFlowProcessor

DEBUG = True
# Set the optical flow method to use
# farneback, deepflow, dis
FLOW_METHOD = ['farneback', 'deepflow', 'dis', 'raft'  ]

load_dotenv('dot.env')

ACCESS_TOKEN = os.getenv('MDAI_TOKEN')
DATA_DIR = os.getenv('DATA_DIR')
DOMAIN = os.getenv('DOMAIN')
PROJECT_ID = os.getenv('PROJECT_ID')
DATASET_ID = os.getenv('DATASET_ID')
ANNOTATIONS = os.path.join(DATA_DIR, os.getenv('ANNOTATIONS'))
LABEL_ID = os.getenv('LABEL_ID')

# Define thresholds to filter out low-confidence points``
MASK_MIN_SIZE = 100
INTENSITY_THRESHOLD = 30

if ACCESS_TOKEN is None:
    print("ACCESS_TOKEN is not set, please set ACCESS_TOKEN in dot.env")
    exit()
else:
    print("ACCESS_TOKEN is set")
print(f"Using optical flow method: {FLOW_METHOD}")
print(f"DATA_DIR={DATA_DIR}")
print(f"DOMAIN={DOMAIN}")
print(f"PROJECT_ID={PROJECT_ID}")
print(f"DATASET_ID={DATASET_ID}")
print(f"ANNOTATIONS={ANNOTATIONS}")
print(f"LABEL_ID={LABEL_ID}")

# Start MD.ai client
mdai_client = mdai.Client(domain=DOMAIN, access_token=ACCESS_TOKEN)

# Download the dataset from MD.ai (or use cached version)
project = mdai_client.project(project_id=PROJECT_ID, path=DATA_DIR)

# Load the annotations
annotations_data = mdai.common_utils.json_to_dataframe(ANNOTATIONS)
annotations_df = pd.DataFrame(annotations_data['annotations'])
labels = annotations_df['labelId'].unique()

# Create the label map, LABEL_ID => 1, others in labels => 0
labels_dict = {LABEL_ID: 1}
project.set_labels_dict(labels_dict)

# Get the dataset
dataset = project.get_dataset_by_id(DATASET_ID)
dataset.classes_dict = project.classes_dict 

# Ensure BASE is set after preparing the dataset
BASE = dataset.images_dir

# Filter annotations for the free fluid label
free_fluid_annotations = annotations_df[annotations_df['labelId'] == LABEL_ID].copy()

# Function to construct the video path
def construct_video_path(base_dir, study_uid, series_uid):
    return os.path.join(base_dir, study_uid, f"{series_uid}.mp4")

# Add video paths to the dataframe using .loc to avoid the SettingWithCopyWarning
free_fluid_annotations['video_path'] = free_fluid_annotations.apply(
    lambda row: construct_video_path(BASE, row['StudyInstanceUID'], row['SeriesInstanceUID']), axis=1)

# Check if video files exist and add the result to the dataframe using .loc
free_fluid_annotations['file_exists'] = free_fluid_annotations['video_path'].apply(os.path.exists)

# Count the number of annotations with and without corresponding video files
num_with_files = free_fluid_annotations['file_exists'].sum()
num_without_files = len(free_fluid_annotations) - num_with_files

print(f"Annotations with corresponding video files: {num_with_files}")
print(f"Annotations without corresponding video files: {num_without_files}")

# # Select five random annotations with corresponding video files
if DEBUG:
    matched_annotations = free_fluid_annotations[free_fluid_annotations['file_exists']].sample(n=5, random_state=42)
else:
    matched_annotations = free_fluid_annotations[free_fluid_annotations['file_exists']]

# Display function
def polygons_to_mask(polygons, height, width):
    mask = np.zeros((height, width), dtype=np.uint8)
    for polygon in polygons:
        points = np.array(polygon, dtype=np.int32)
        cv2.fillPoly(mask, [points], 1)
    return mask

# Parameters for Lucas-Kanade optical flow
lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

# Parameters for ShiTomasi corner detection
feature_params = dict(maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)

def trim_and_threshold_mask(frame, mask, intensity_threshold, min_size):
    # Invert the frame so that dark areas (potential fluid) have high values
    inverted_frame = 255 - frame
    # Create a binary mask based on the intensity threshold
    intensity_mask = inverted_frame > intensity_threshold

    # Combine the optical flow mask with the intensity mask
    trimmed_mask = (mask > 0) & intensity_mask
    
    # Remove small connected components
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(trimmed_mask.astype(np.uint8), connectivity=8)
    for i in range(1, num_labels):  # Start from 1 to skip the background
        if stats[i, cv2.CC_STAT_AREA] < min_size:
            trimmed_mask[labels == i] = 0
    
    return trimmed_mask.astype(float)

def compute_temporal_gradient(prev_frame, current_frame):
    # Compute the absolute difference between the current and previous frames
    gradient = cv2.absdiff(current_frame, prev_frame)
    # Convert to grayscale if needed
    if len(gradient.shape) == 3:
        gradient = cv2.cvtColor(gradient, cv2.COLOR_BGR2GRAY)
    return gradient

def adjust_mask_confidence(mask, gradient):
    # Normalize the gradient to the range [0, 1]
    gradient_normalized = cv2.normalize(gradient, None, 0, 1, cv2.NORM_MINMAX)
    # Scale the mask by the normalized gradient
    adjusted_mask = mask * gradient_normalized
    return adjusted_mask

def blend_masks(original_mask, adjusted_mask, alpha=0.7):
    return alpha * original_mask + (1 - alpha) * adjusted_mask

def temporal_smooth(prev_mask, curr_mask, alpha=0.8):
    return alpha * prev_mask + (1 - alpha) * curr_mask

def debug_visualize(frame, initial_mask, flow_mask, adjusted_mask, final_mask, frame_number):
    # Create a 2x3 grid of images
    grid = np.zeros((frame.shape[0]*2, frame.shape[1]*3, 3), dtype=np.uint8)
    
    # Original frame (Top Left)
    grid[:frame.shape[0], :frame.shape[1]] = frame
    cv2.putText(grid, "Original Frame", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    # Frame with final mask applied (Bottom Left)
    mask_overlay = frame.copy()
    mask_overlay[final_mask > 0] = mask_overlay[final_mask > 0] * 0.5 + np.array([0, 255, 0], dtype=np.uint8) * 0.5
    grid[frame.shape[0]:, :frame.shape[1]] = mask_overlay
    cv2.putText(grid, "Frame with Mask", (10, frame.shape[0]+30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    # Initial mask (Top Center)
    initial_overlay = np.zeros_like(frame)
    initial_overlay[:,:,0] = initial_mask * 255  # Blue color for initial mask
    grid[:frame.shape[0], frame.shape[1]:frame.shape[1]*2] = initial_overlay
    cv2.putText(grid, "Initial Mask", (frame.shape[1]+10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    # Flow mask (Top Right)
    flow_overlay = np.zeros_like(frame)
    flow_overlay[:,:,2] = flow_mask * 255
    grid[:frame.shape[0], frame.shape[1]*2:] = flow_overlay
    cv2.putText(grid, "Flow Mask", (frame.shape[1]*2+10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    # Adjusted mask (Bottom Center)
    adjusted_overlay = np.zeros_like(frame)
    adjusted_overlay[:,:,0] = adjusted_mask * 255
    grid[frame.shape[0]:, frame.shape[1]:frame.shape[1]*2] = adjusted_overlay
    cv2.putText(grid, "Adjusted Mask", (frame.shape[1]+10, frame.shape[0]+30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    # Final mask (Bottom Right)
    final_overlay = np.zeros_like(frame)
    final_overlay[:,:,1] = final_mask * 255
    grid[frame.shape[0]:, frame.shape[1]*2:] = final_overlay
    cv2.putText(grid, "Final Mask", (frame.shape[1]*2+10, frame.shape[0]+30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    # Add frame number
    cv2.putText(grid, f"Frame: {frame_number}", (10, grid.shape[0]-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    return grid

def create_debug_video(debug_dir, output_video_path, fps=30):
    images = [img for img in os.listdir(debug_dir) if img.endswith(".png")]
    images.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))  # Sort by frame number
    
    if not images:
        print(f"No debug frames found in {debug_dir}")
        return
    
    frame = cv2.imread(os.path.join(debug_dir, images[0]))
    height, width, layers = frame.shape

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    for image in tqdm(images, desc="Creating debug video"):
        video.write(cv2.imread(os.path.join(debug_dir, image)))

    video.release()
    print(f"Debug video saved at {output_video_path}")


def remove_small_regions(mask, min_size):
    # Find all connected components
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    # Remove small components
    for i in range(1, num_labels):  # Start from 1 to skip the background
        if stats[i, cv2.CC_STAT_AREA] < min_size:
            mask[labels == i] = 0
    return mask

def track_frames(cap, start_frame, end_frame, initial_mask, debug_dir, forward=True, pbar=None, flow_processor=None):
    frames = []
    step = 1 if forward else -1
    frame_idx = start_frame
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, prev_frame = cap.read()
    if not ret:
        return frames
    
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    mask = initial_mask.astype(float)
    
    while (forward and frame_idx <= end_frame) or (not forward and frame_idx >= 0):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        if frame_idx == start_frame:
            new_mask = mask
            flow_mask = np.zeros_like(mask)
            adjusted_mask = np.zeros_like(mask)
        else:
            # Apply optical flow
            flow_mask = flow_processor.apply_optical_flow(prev_gray, frame_gray, mask)
            gradient = compute_temporal_gradient(prev_gray, frame_gray)
            adjusted_mask = adjust_mask_confidence(flow_mask, gradient)
            blended_mask = blend_masks(flow_mask, adjusted_mask)
            smooth_mask = temporal_smooth(mask, blended_mask)
            new_mask = trim_and_threshold_mask(frame_gray, smooth_mask, INTENSITY_THRESHOLD, MASK_MIN_SIZE)
        
        if DEBUG:
            # Create debug visualization
            debug_frame = debug_visualize(frame, mask, flow_mask, adjusted_mask, new_mask, frame_idx)
            
            # Save debug frame
            os.makedirs(debug_dir, exist_ok=True)
            cv2.imwrite(os.path.join(debug_dir, f'debug_frame_{frame_idx:04d}.png'), debug_frame)
        
        frames.append((frame_idx, frame, new_mask))
        
        prev_gray = frame_gray.copy()
        mask = new_mask
        frame_idx += step
        
        if pbar:
            pbar.update(1)
    
    return frames

def save_combined_video(video_path, output_video_path, initial_mask, frame_number, debug_dir, flow_processor):
    save_dir = os.path.dirname(output_video_path)
    mask_dir = os.path.join(save_dir, "masks")
    os.makedirs(mask_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    with tqdm(total=total_frames, desc="Processing video", unit="frame") as pbar:
        backward_frames = track_frames(cap, frame_number, 0, initial_mask, debug_dir, forward=False, pbar=pbar, flow_processor=flow_processor)
        forward_frames = track_frames(cap, frame_number, total_frames - 1, initial_mask, debug_dir, forward=True, pbar=pbar, flow_processor=flow_processor)
    
    combined_frames = backward_frames[::-1] + forward_frames[1:]
    
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))
    
    print("Writing video...")
    for frame_idx, frame, mask in tqdm(combined_frames, desc="Saving frames", unit="frame"):
        out.write(frame)
        
        mask_filename = os.path.join(mask_dir, f"mask_{frame_idx:04d}.png")
        cv2.imwrite(mask_filename, (mask * 255).astype(np.uint8))
    
    cap.release()
    out.release()
    print(f"Video saved at {output_video_path}")

def track_and_save_masks_as_video(annotation, output_dir, flow_processor):
    video_id = annotation['SeriesInstanceUID']
    video_path = annotation['video_path']
    frame_number = int(annotation['frameNumber'])
    foreground = annotation['data']['foreground']

    print(f"Processing Video: {video_id}; Frame: {frame_number}...")
    
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    ret, frame = cap.read()
    if not ret:
        print(f"Failed to read the frame number {frame_number} from the video.")
        return
    cap.release()
    
    initial_mask = polygons_to_mask(foreground, frame.shape[0], frame.shape[1])
    print(f"Initial mask shape: {initial_mask.shape}")
    print(f"Initial mask min: {initial_mask.min()}, max: {initial_mask.max()}")
    cv2.imwrite(os.path.join(output_dir, f'initial_mask_frame_{frame_number}.png'), initial_mask * 255)
    
    debug_dir = os.path.join(output_dir, 'debug')
    output_video_path = os.path.join(output_dir, f'masked_{video_id}.mp4')
    save_combined_video(video_path, output_video_path, initial_mask, frame_number, debug_dir, flow_processor)
    
    # Create debug video
    debug_video_path = os.path.join(output_dir, f'debug_{video_id}.mp4')
    create_debug_video(debug_dir, debug_video_path)

# In the main loop:
for method in FLOW_METHOD:
    output_base_dir = os.path.join('output', method)
    os.makedirs(output_base_dir, exist_ok=True)
    flow_processor = OpticalFlowProcessor(method)
    for index, annotation in matched_annotations.iterrows():
        output_dir = os.path.join(output_base_dir, f'annotation_{index}')
        os.makedirs(output_dir, exist_ok=True)
        track_and_save_masks_as_video(annotation, output_dir, flow_processor)

print("Tracking and saving videos completed.")
