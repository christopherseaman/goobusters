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
from metrics import apply_binary_threshold  # Removed calculate_jaccard_index

# Enables debug mode
DEBUG = True

# Optical flow methods to use
FLOW_METHOD = ['pwc', 'farneback']  # Hybrid model of PWC-Net and Farneback

# Load environment variables
load_dotenv('.env')

# Environment variables for MD.ai access
ACCESS_TOKEN = os.getenv('MDAI_TOKEN')
DATA_DIR = os.getenv('DATA_DIR')
DOMAIN = os.getenv('DOMAIN')
PROJECT_ID = os.getenv('PROJECT_ID')
DATASET_ID = os.getenv('DATASET_ID')
ANNOTATIONS = os.path.join(DATA_DIR, os.getenv('ANNOTATIONS'))
LABEL_ID = os.getenv('LABEL_ID')

# Define thresholds
MASK_MIN_SIZE = 100
INTENSITY_THRESHOLD = 30

# Initialize MD.ai client
mdai_client = mdai.Client(domain=DOMAIN, access_token=ACCESS_TOKEN)
project = mdai_client.project(project_id=PROJECT_ID, path=DATA_DIR)

# Load annotations and dataset
annotations_data = mdai.common_utils.json_to_dataframe(ANNOTATIONS)
annotations_df = pd.DataFrame(annotations_data['annotations'])
labels = annotations_df['labelId'].unique()
labels_dict = {LABEL_ID: 1}
project.set_labels_dict(labels_dict)
dataset = project.get_dataset_by_id(DATASET_ID)
dataset.classes_dict = project.classes_dict
BASE = dataset.images_dir

# Initialize OpticalFlowProcessor for PWC-Net and Farneback methods
pwc_processor = OpticalFlowProcessor('pwc-net')
farneback_processor = OpticalFlowProcessor('farneback')

# Helper functions
def polygons_to_mask(polygons, height, width):
    mask = np.zeros((height, width), dtype=np.uint8)
    for polygon in polygons:
        points = np.array(polygon, dtype=np.int32)
        cv2.fillPoly(mask, [points], 1)
    return mask

def debug_visualize(frame, mask, flow, frame_number, gt_mask=None, threshold=0.5):
    """Visualize optical flow without Jaccard Index calculation."""
    h, w = frame.shape[:2]
    flow_x, flow_y = flow[..., 0], flow[..., 1]
    hsv = np.zeros((h, w, 3), dtype=np.uint8)
    mag, ang = cv2.cartToPolar(flow_x, flow_y)
    hsv[..., 0] = ang * 180 / np.pi / 2
    hsv[..., 1] = 255
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    flow_rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    
    grid = np.zeros((h*2, w*2, 3), dtype=np.uint8)
    grid[:h, :w] = frame
    mask_overlay = frame.copy()
    mask_overlay[mask > 0] = mask_overlay[mask > 0] * 0.5 + np.array([0, 255, 0], dtype=np.uint8) * 0.5
    grid[:h, w:] = mask_overlay
    grid[h:, :w] = flow_rgb

    # Apply binary threshold
    binary_pred_mask = apply_binary_threshold(flow, threshold)
    final_mask = np.zeros_like(frame)
    final_mask[:, :, 1] = binary_pred_mask * 255
    grid[h:, w:] = final_mask
    
    cv2.putText(grid, f"Frame: {frame_number}", (10, grid.shape[0]-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    return grid

def construct_video_path(base_dir, study_uid, series_uid):
    return os.path.join(base_dir, study_uid, f"{series_uid}.mp4")

free_fluid_annotations = annotations_df[annotations_df['labelId'] == LABEL_ID].copy()
free_fluid_annotations['video_path'] = free_fluid_annotations.apply(
    lambda row: construct_video_path(BASE, row['StudyInstanceUID'], row['SeriesInstanceUID']), axis=1
)

free_fluid_annotations['file_exists'] = free_fluid_annotations['video_path'].apply(lambda path: os.path.exists(path))

if DEBUG:
    matched_annotations = free_fluid_annotations[free_fluid_annotations['file_exists']].sample(n=5, random_state=42)
else:
    matched_annotations = free_fluid_annotations[free_fluid_annotations['file_exists']]

def apply_hybrid_optical_flow(prev_frame, curr_frame, mask, pwc_processor, farneback_processor, blend_alpha=0.6):
    """Applies hybrid optical flow, blending PWC-Net and Farneback flows."""
    pwc_flow = pwc_processor.apply_optical_flow(prev_frame, curr_frame, mask)
    farneback_flow = farneback_processor.apply_optical_flow(prev_frame, curr_frame, mask)
    mask = (mask > 0).astype(np.float32)
    mask = cv2.GaussianBlur(mask, (15, 15), 0)
    combined_flow = blend_alpha * (pwc_flow * mask) + (1 - blend_alpha) * (farneback_flow * (1 - mask))
    return combined_flow

def track_and_save_masks_as_video(annotation, output_dir, pwc_processor, farneback_processor):
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

    # Creating the ground-truth mask from annotation polygons in each frame
    gt_mask = polygons_to_mask(foreground, frame.shape[0], frame.shape[1])

    # Process frames without Jaccard index calculation
    debug_dir = os.path.join(output_dir, 'debug')
    output_video_path = os.path.join(output_dir, f'masked_{video_id}.mp4')
    save_combined_video(video_path, output_video_path, gt_mask, frame_number, debug_dir, pwc_processor, farneback_processor)

def save_combined_video(video_path, output_video_path, gt_mask, frame_number, debug_dir, pwc_processor, farneback_processor):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))
    
    print("Writing video...")
    for i in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            break
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if i == 0:
            # For the first frame, initialize prev_frame and binary_pred_mask
            prev_frame = frame_gray
            prev_mask = gt_mask.astype(float)  # Set initial mask to gt_mask for the first frame
            binary_pred_mask = gt_mask  # Initialize binary_pred_mask as gt_mask for the first frame
        else:
            # Calculate flow and apply hybrid mask
            flow = apply_hybrid_optical_flow(prev_frame, frame_gray, prev_mask, pwc_processor, farneback_processor)
            binary_pred_mask = apply_binary_threshold(flow, threshold=0.5)

            # Visualization with debug overlay
            debug_frame = debug_visualize(frame, binary_pred_mask, flow, frame_number=i, gt_mask=gt_mask)
            os.makedirs(debug_dir, exist_ok=True)
            cv2.imwrite(os.path.join(debug_dir, f'debug_frame_{i:04d}.png'), debug_frame)
            out.write(frame)
        
        # Update previous frame and mask for next iteration
        prev_frame = frame_gray
        prev_mask = binary_pred_mask
    
    cap.release()
    out.release()
    print(f"Video saved at {output_video_path}")

# Main loop
output_base_dir = 'output_hybrid_test'
os.makedirs(output_base_dir, exist_ok=True)

for index, annotation in matched_annotations.iterrows():
    output_dir = os.path.join(output_base_dir, f'annotation_{index}')
    os.makedirs(output_dir, exist_ok=True)
    track_and_save_masks_as_video(annotation, output_dir, pwc_processor, farneback_processor)

print("Tracking and saving videos completed.")