# Import libraries and set constants
from dotenv import load_dotenv
import os
import mdai

import cv2
import pandas as pd
import numpy as np
from tqdm import tqdm
from opticalflowprocessor import OpticalFlowProcessor
import json
from pandas import json_normalize
from datetime import datetime
import traceback
import shutil
from pathlib import Path
import urllib.parse

# Enable debug mode
DEBUG = True

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(PROJECT_DIR, "output")


# Load environment variables
load_dotenv('.env')
ACCESS_TOKEN = os.getenv('MDAI_TOKEN')
DATA_DIR = os.getenv('DATA_DIR')
DOMAIN = os.getenv('DOMAIN')
PROJECT_ID = os.getenv('PROJECT_ID')
DATASET_ID = os.getenv('DATASET_ID')
ANNOTATIONS = os.path.join(DATA_DIR, os.getenv('ANNOTATIONS'))
LABEL_ID_FREE_FLUID = os.getenv("LABEL_ID_FREE_FLUID")
LABEL_IDS = {
    "disappear_reappear": os.getenv("LABEL_ID_DISAPPEAR_REAPPEAR"),
    "branching_fluid": os.getenv("LABEL_ID_BRANCHING_FLUID"),
    "multiple_distinct": os.getenv("LABEL_ID_MULTIPLE_DISTINCT"),
}
LABEL_ID_MACHINE_GROUP = os.getenv("LABEL_ID_MACHINE_GROUP")  # L_1A4xv7
LABEL_ID_FLUID_OF = os.getenv("LABEL_ID_FLUID_OF")          # L_JykNe7

# Validate that they're loaded
if None in [LABEL_ID_MACHINE_GROUP, LABEL_ID_FLUID_OF]:
    raise ValueError("Machine label IDs not properly set in .env file")

FLOW_METHOD = ['dis', 'farneback']
MASK_MIN_SIZE = 100
INTENSITY_THRESHOLD = 30

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

# Load and normalize annotations
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
# Rename 'data_foreground' in free_fluid_annotations to avoid conflicts
free_fluid_annotations.rename(columns={'data_foreground': 'free_fluid_foreground'}, inplace=True)


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
    coverage = np.mean(mask) * 100
    print(f"Frame {frame_num} - Mask coverage: {coverage:.2f}%")

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
        List of tuples (frame_idx, frame, mask, flow)  # Note: Added flow to return tuple
    """
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
        if DEBUG:
            debug_path = os.path.join(debug_dir, f'initial_frame_{frame_idx:04d}.png')
            cv2.imwrite(debug_path, prev_frame)
            debug_path = os.path.join(debug_dir, f'initial_mask_{frame_idx:04d}.png')
            cv2.imwrite(debug_path, (mask * 255).astype(np.uint8))
    except Exception as e:
        print(f"Error initializing first frame: {str(e)}")
        return frames

    while (forward and frame_idx <= end_frame) or (not forward and frame_idx >= 0):
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
                print_mask_stats(mask, frame_idx)  # Add coverage stats
                frames.append((frame_idx, frame, new_mask, flow))  # Added flow
                total_frames_processed += 1
            else:
                # Apply optical flow
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
                        
                    # Blend the masks with modified weights - CHECK 
                    adjusted_mask = flow_mask
                    blended_mask = (0.5 * mask + 0.5 * adjusted_mask).astype(float)  
                    new_mask = np.clip(blended_mask, 0, 1)
                    
                    # Print coverage stats
                    print_mask_stats(new_mask, frame_idx)
                    
                    # Reset error counter on success
                    consecutive_errors = 0
                    total_frames_processed += 1
                    
                    # Store frame and flow
                    frames.append((frame_idx, frame, new_mask, flow))
                    
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

            # Save debug visualization with flow
            if DEBUG:
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
                    cv2.imwrite(os.path.join(debug_dir, f'debug_frame_{frame_idx:04d}.png'), debug_frame)
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

def upload_masks_to_mdai(client, masks_data, project_id):
    """
    Upload multiple masks to MD.ai using batch processing
    
    Args:
        client: MD.ai client instance
        masks_data: List of dictionaries containing mask info
        project_id: MD.ai project ID
    """
    try:
        print("\nPreparing annotations for MD.ai upload...")
        
        # Format annotations according to MD.ai schema
        annotations = []
        for mask_info in masks_data:
            annotation = {
                'labelId': LABEL_ID_FLUID_OF,
                'StudyInstanceUID': mask_info['study_uid'],
                'SeriesInstanceUID': mask_info['series_uid'],
                'frameNumber': int(mask_info['frame_number']),
                'data': {
                    'foreground': mask_info['polygons']
                },
                'groupId': LABEL_ID_MACHINE_GROUP
            }
            annotations.append(annotation)

        print(f"Uploading {len(annotations)} annotations to MD.ai...")
        
        # Upload annotations one by one
        successful_uploads = []
        for idx, annotation in enumerate(annotations):
            try:
                print(f"\nUploading annotation {idx + 1}/{len(annotations)}")
                response = client.post_annotation(annotation)
                if response:
                    successful_uploads.append(response)
                    print(f"Successfully uploaded annotation {idx + 1}")
                else:
                    print(f"Warning: Empty response for annotation {idx + 1}")
            except Exception as e:
                print(f"Error uploading annotation {idx + 1}: {str(e)}")
                continue

        print(f"\nUploaded {len(successful_uploads)} out of {len(annotations)} annotations")
        return successful_uploads

    except Exception as e:
        print(f"Error during upload process: {str(e)}")
        traceback.print_exc()
        return None

def prepare_mask_data(mask, frame_number, study_uid, series_uid):
    """
    Prepare mask data for MD.ai upload
    """
    try:
        # Convert mask to polygons
        contours, _ = cv2.findContours(
            (mask * 255).astype(np.uint8),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        # Convert contours to MD.ai polygon format
        polygons = []
        for contour in contours:
            epsilon = 0.001 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            polygon = approx.squeeze().tolist()
            if isinstance(polygon, np.ndarray) and len(polygon.shape) == 2:
                polygons.append(polygon.tolist())

        if not polygons:
            print(f"Warning: No valid polygons found for frame {frame_number}")
            return None

        return {
            'study_uid': study_uid,
            'series_uid': series_uid,
            'frame_number': frame_number,
            'polygons': polygons
        }

    except Exception as e:
        print(f"Error preparing mask data for frame {frame_number}: {str(e)}")
        return None

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

# Modify save_combined_video to include diagnostics
def save_combined_video(video_path, output_video_path, initial_mask, frame_number, debug_dir, flow_processor, 
                       mdai_client, study_uid, series_uid,
                       overlay_color=(0, 255, 0),
                       overlay_alpha=0.3,
                       add_info=True,
                       batch_size=50):
    """
    Process video frames, save with mask overlay, and upload masks to MD.ai
    """
    # Keep diagnostics
    print("\nStarting video processing diagnostics...")
    diag_before = diagnose_video_processing(video_path, output_video_path)
    
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
        
        # Process frames
        with tqdm(total=total_frames, desc="Processing video", unit="frame") as pbar:
            backward_frames = track_frames(cap, frame_number, 0, initial_mask, debug_dir, 
                                        forward=False, pbar=pbar, flow_processor=flow_processor)
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
        for frame_idx, frame, mask, flow in tqdm(combined_frames, desc="Saving frames", unit="frame"):
            try:
                # Create overlay frame
                overlay_frame = frame.copy()
                overlay_frame[mask > 0] = overlay_frame[mask > 0] * (1 - overlay_alpha) + \
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

                # Write frame to video
                out.write(overlay_frame)
                frames_written += 1

                # Save mask
                mask_path = os.path.join(mask_dir, f"mask_{frame_idx:04d}.png")
                cv2.imwrite(mask_path, (mask * 255).astype(np.uint8))

                # Save debug frame
                debug_frame = debug_visualize(
                    frame=frame, 
                    initial_mask=initial_mask, 
                    flow_mask=mask,
                    adjusted_mask=mask, 
                    final_mask=mask, 
                    frame_number=frame_idx,
                    flow=flow
                )
                cv2.imwrite(os.path.join(debug_dir, f'debug_frame_{frame_idx:04d}.png'), debug_frame)

            except Exception as e:
                print(f"Error processing frame {frame_idx}: {str(e)}")
                continue

        # Upload all frames to MD.ai
        batch_result = batch_upload_to_mdai(
            mdai_client, 
            combined_frames,
            PROJECT_ID,
            study_uid,
            series_uid,
            batch_size
        )
        
        if batch_result:
            upload_stats['uploaded_frames'] = batch_result['successful']
            upload_stats['failed_uploads'] = batch_result['failed']

        # Cleanup
        cap.release()
        out.release()
        
        print(f"\nFrames written to video: {frames_written}")
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
def visualize_flow(frame, flow, skip=16):
    """
    Visualizes optical flow vectors over the frame.
    
    Args:
        frame: Original frame
        flow: Optical flow array (h, w, 2)
        skip: Show flow arrows every 'skip' pixels
    """
    h, w = flow.shape[:2]
    frame_vis = frame.copy()
    
    # Convert flow to polar coordinates for better visualization
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    
    # Normalize magnitude for visualization
    mag_norm = cv2.normalize(mag, None, 0, 1, cv2.NORM_MINMAX)
    
    # Draw flow vectors
    for y in range(0, h, skip):
        for x in range(0, w, skip):
            if mag_norm[y, x] > 0.1:  # Only draw significant motion
                fx = flow[y, x, 0]
                fy = flow[y, x, 1]
                # Draw arrow
                cv2.arrowedLine(frame_vis, (x, y), 
                              (int(x + fx), int(y + fy)),
                              (0, 255, 0), 1, tipLength=0.3)
    
    # Create heatmap
    hsv = np.zeros((h, w, 3), dtype=np.uint8)
    hsv[..., 0] = ang * 180 / np.pi / 2  # Hue according to direction
    hsv[..., 1] = 255  # Full saturation
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)  # Value according to magnitude
    
    flow_heatmap = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    
    # Combine visualizations
    combined_vis = np.zeros((h * 2, w, 3), dtype=np.uint8)
    combined_vis[:h] = frame_vis
    combined_vis[h:] = flow_heatmap
    
    # Add labels
    cv2.putText(combined_vis, "Flow Vectors", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(combined_vis, "Flow Heatmap", (10, h + 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    return combined_vis

def debug_visualize(frame, initial_mask, flow_mask, adjusted_mask, final_mask, frame_number, flow=None):
    """
    Enhanced debug visualization with flow vectors and mask coverage.
    """
    h, w = frame.shape[:2]
    grid = np.zeros((h * 2, w * 3, 3), dtype=np.uint8)  # Extended width for flow visualization
    
    # Original frame (Top Left)
    grid[:h, :w] = frame
    cv2.putText(grid, "Original Frame", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    # Flow Mask (Top Middle)
    if flow_mask is not None:
        flow_viz = frame.copy()
        flow_viz[flow_mask > 0] = flow_viz[flow_mask > 0] * 0.7 + np.array([0, 0, 255], dtype=np.uint8) * 0.3
        grid[:h, w:w*2] = flow_viz
    cv2.putText(grid, "Optical Flow Mask", (w + 10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    # Flow Visualization (Top Right)
    if flow is not None:
        flow_viz = frame.copy()
        # Draw flow vectors
        for y in range(0, h, 20):  # Sample every 20 pixels
            for x in range(0, w, 20):
                if final_mask[y, x] > 0:  # Only draw vectors in mask region
                    dx, dy = flow[y, x]
                    cv2.arrowedLine(flow_viz, (x, y), 
                                  (int(x + dx), int(y + dy)), 
                                  (255, 0, 0), 2)
        grid[:h, w*2:w*3] = flow_viz
    cv2.putText(grid, "Flow Vectors", (w*2 + 10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    # Initial Mask (Bottom Left)
    if initial_mask is not None:
        initial_viz = frame.copy()
        initial_viz[initial_mask > 0] = initial_viz[initial_mask > 0] * 0.7 + np.array([255, 0, 0], dtype=np.uint8) * 0.3
        grid[h:h*2, :w] = initial_viz
    cv2.putText(grid, "Initial Segmentation Mask", (10, h + 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    # Final Mask (Bottom Middle)
    if final_mask is not None:
        final_viz = frame.copy()
        final_viz[final_mask > 0] = final_viz[final_mask > 0] * 0.7 + np.array([0, 255, 0], dtype=np.uint8) * 0.3
        grid[h:h*2, w:w*2] = final_viz
    cv2.putText(grid, "Final Tracked Mask", (w + 10, h + 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    # Coverage Stats (Bottom Right)
    stats_viz = np.zeros((h, w, 3), dtype=np.uint8)
    coverage = np.mean(final_mask) * 100
    cv2.putText(stats_viz, f"Frame: {frame_number}", (10, h//4), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(stats_viz, f"Coverage: {coverage:.2f}%", (10, h//2), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    grid[h:h*2, w*2:w*3] = stats_viz
    
    return grid
# Function to calculate flow metrics

def prepare_mdai_annotation(mask, frame_number, study_uid, series_uid, label_id, group_id):
    """
    Prepare a single annotation in MD.ai format using mdai.common_utils
    """
    try:

        binary_mask = (mask > 0).astype(np.uint8)
        # Convert mask to MD.ai format using built-in function
        mask_data = mdai.common_utils.convert_mask_data(binary_mask)

        if not mask_data:
            print(f"Warning: No valid mask data for frame {frame_number}")
            return None

        # Create MD.ai annotation format
        annotation = {
            'labelId': label_id,
            'StudyInstanceUID': study_uid,
            'SeriesInstanceUID': series_uid,
            'frameNumber': int(frame_number),
            'data': mask_data,  # Use the converted mask data
            'groupId': group_id,
            'type': 'annotation'
        }
        
        return annotation

    except Exception as e:
        print(f"Error preparing annotation for frame {frame_number}: {str(e)}")
        return None

    except Exception as e:
        print(f"Error preparing annotation for frame {frame_number}: {str(e)}")
        traceback.print_exc()
        return None
    
def batch_upload_to_mdai(client, masks_data, project_id, dataset_id, study_uid, series_uid, batch_size=50):
    """
    Uploads multiple annotations in a batch to MD.ai

    Args:
        client: MD.ai client instance
        masks_data: List of tuples (frame_idx, mask, sop_instance_uid)
        project_id: MD.ai project ID
        dataset_id: MD.ai dataset ID
        study_uid: Study Instance UID
        series_uid: Series Instance UID
        batch_size: Number of annotations per batch

    Returns:
        dict: Summary of upload results
    """
    print(f"\nPreparing batch upload of {len(masks_data)} masks to MD.ai...")

    annotations = []
    successful_uploads = 0

    # Process each frame's mask data
    for frame_idx, mask, sop_instance_uid in masks_data:
        try:
            annotation = {
                "labelId": LABEL_ID_FLUID_OF,
                "SOPInstanceUID": sop_instance_uid,  # Ensure correct UID for DICOM frame
                "data": mdai.common_utils.convert_mask_data(mask),
                "note": f"Generated from frame {frame_idx}"
            }
            annotations.append(annotation)

            # Upload in batches
            if len(annotations) >= batch_size:
                failed_annotations = client.import_annotations(annotations, project_id, dataset_id)
                if failed_annotations:
                    print(f"Some annotations failed: {failed_annotations}")
                else:
                    successful_uploads += len(annotations)
                annotations = []  # Reset batch

        except Exception as e:
            print(f"Error preparing annotation for frame {frame_idx}: {str(e)}")

    # Upload any remaining annotations
    if annotations:
        failed_annotations = client.import_annotations(annotations, project_id, dataset_id)
        if failed_annotations:
            print(f"Some final annotations failed: {failed_annotations}")
        else:
            successful_uploads += len(annotations)

    return {
        "total_uploaded": successful_uploads,
        "failed_annotations": failed_annotations if "failed_annotations" in locals() else [],
        "total_attempted": len(masks_data),
    }
    
def test_single_mask_upload(client, mask, frame_number, study_uid, series_uid):
    """
    Test uploading a single mask to MD.ai with detailed logging
    """
    print("\n=== Starting MD.ai Single Mask Upload Test ===")
    
    try:
        # 1. Print input information
        print("\nInput Parameters:")
        print(f"Frame number: {frame_number}")
        print(f"Study UID: {study_uid}")
        print(f"Series UID: {series_uid}")
        
        # 2. Check mask properties
        print("\nMask Properties:")
        print(f"Shape: {mask.shape}")
        print(f"Data type: {mask.dtype}")
        print(f"Unique values: {np.unique(mask)}")
        
        # 3. Prepare annotation
        print("\nPreparing annotation...")
        annotation = prepare_mdai_annotation(mask, frame_number, study_uid, series_uid)
        
        if annotation:
            print("Annotation prepared successfully")
            print("Annotation keys:", annotation.keys())
            
            # 4. Try upload
            print("\nAttempting upload...")
            try:
                response = client.post_annotation(annotation)
                print("\nUpload Response:", response)
                
                if response and 'id' in response:
                    print(f"Success! Annotation ID: {response['id']}")
                    return True
                else:
                    print("Upload failed - no annotation ID in response")
                    return False
                    
            except Exception as e:
                print(f"Upload error: {str(e)}")
                return False
        else:
            print("Failed to prepare annotation")
            return False
            
    except Exception as e:
        print(f"Test error: {str(e)}")
        return False


def calculate_flow_metrics(flow, mask):
    """
    Calculate metrics to evaluate flow quality in the masked region.
    """
    if mask is None or flow is None:
        return {}
        
    # Only consider flow within the mask
    mask_bool = mask > 0
    if not np.any(mask_bool):
        return {}
        
    flow_magnitude = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
    
    metrics = {
        'mean_magnitude': float(np.mean(flow_magnitude[mask_bool])),
        'max_magnitude': float(np.max(flow_magnitude[mask_bool])),
        'std_magnitude': float(np.std(flow_magnitude[mask_bool])),
        'coherence': float(np.mean(np.abs(np.mean(flow[mask_bool], axis=0)) / 
                                 (np.mean(np.abs(flow[mask_bool]), axis=0) + 1e-6)))
    }
    
    return metrics

# Function to create a debug video from saved debug frames
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


def create_study_to_exam_mapping(dicom_metadata):
    """
    Creates a mapping of StudyInstanceUID to Exam Number from DICOM metadata.
    Args:
        dicom_metadata (dict): The DICOM metadata JSON content.
    Returns:
        dict: Mapping of StudyInstanceUID to Exam Number.
    """
    study_to_exam = {}
    for dataset in dicom_metadata.get('datasets', []):
        for study in dataset.get('studies', []):
            study_uid = study.get('StudyInstanceUID')
            exam_number = study.get('number')
            if study_uid and exam_number:
                study_to_exam[study_uid] = exam_number

    print(f"Created Study-to-Exam mapping with {len(study_to_exam)} entries.")
    return study_to_exam



def track_and_save_masks(annotation, output_dir, flow_processor, study_to_exam_map, series_to_study_map):
    video_id = annotation['SeriesInstanceUID']
    video_path = annotation['video_path']
    frame_number = annotation.get('frameNumber_free_fluid')

    # Validate frame number
    if pd.isna(frame_number):
        print(f"Skipping annotation {annotation['id']} due to missing frame number.")
        return

    # Use 'free_fluid_foreground' instead of 'data_foreground'
    data_foreground = annotation.get('free_fluid_foreground')

    if not isinstance(data_foreground, list) or len(data_foreground) == 0:
        print(f"Skipping annotation {annotation['id']} due to invalid 'free_fluid_foreground'.")
        return

    # Link SeriesInstanceUID to StudyInstanceUID
    study_uid = series_to_study_map.get(video_id, "Unknown")
    if study_uid == "Unknown":
        print(f"Warning: Study UID not found for Series UID: {video_id}")
    exam_number = study_to_exam_map.get(study_uid, "Unknown")

    # Open the video file to get the frame dimensions
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    if not ret:
        print(f"Failed to read the first frame of the video {video_id}.")
        cap.release()
        return
    frame_height, frame_width = frame.shape[:2]
    cap.release()

    # Create the initial mask
    initial_mask = polygons_to_mask(data_foreground, frame_height, frame_width)
    os.makedirs(output_dir, exist_ok=True)
    cv2.imwrite(os.path.join(output_dir, f'initial_mask_frame_{frame_number}.png'), initial_mask * 255)

    # Define output paths
    output_video_path = os.path.join(output_dir, f'masked_{video_id}.mp4')
    debug_dir = os.path.join(output_dir, 'debug')
    os.makedirs(debug_dir, exist_ok=True)

    save_combined_video(video_path, output_video_path, initial_mask, frame_number, debug_dir, flow_processor)

    # Create metadata.json
    metadata = {
        "annotation_id": annotation['id'],
        "video_id": video_id,
        "frame_number": frame_number,
        "output_video_path": output_video_path,
        "exam_number": exam_number,
        "StudyInstanceUID": study_uid
    }

    metadata_path = os.path.join(output_dir, "metadata.json")
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=4)

    print(f"Metadata for annotation {annotation['id']} saved to {metadata_path}")

def add_exam_numbers_to_metadata(base_dir, dicom_metadata_path):
    """
    Adds the Exam Number from the DICOM metadata file to each annotation's metadata.json.
    
    Args:
        base_dir (str): The base directory containing annotation folders.
        dicom_metadata_path (str): Path to the DICOM metadata JSON file.
    """
    # Load the DICOM metadata JSON file
    with open(dicom_metadata_path, 'r') as f:
        dicom_metadata = json.load(f)

    # Create a mapping of StudyInstanceUID to Exam Number
    study_uid_to_exam = {
        entry['StudyInstanceUID']: entry['number']
        for entry in dicom_metadata['studies']
    }

    # Traverse through all annotation directories
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if file == "metadata.json":
                metadata_path = os.path.join(root, file)
                
                # Load metadata.json
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)

                # Add ExamNumber if StudyInstanceUID matches
                study_uid = metadata.get('StudyInstanceUID')
                if study_uid in study_uid_to_exam:
                    metadata['ExamNumber'] = study_uid_to_exam[study_uid]
                else:
                    print(f"Warning: No Exam Number found for StudyInstanceUID {study_uid}")
                
                # Save updated metadata.json
                with open(metadata_path, 'w') as f:
                    json.dump(metadata, f, indent=4)

                print(f"Updated metadata.json for {study_uid}")

def create_series_to_study_mapping(dicom_metadata):
    """
    Creates a mapping of SeriesInstanceUID to StudyInstanceUID from DICOM metadata.
    Args:# Function to process and save the masks and metadata
def track_and_save_masks(annotation, output_dir, flow_processor, study_to_exam_map):
    video_id = annotation['SeriesInstanceUID']
    study_uid = annotation.get('StudyInstanceUID')  # Link to StudyInstanceUID
    video_path = annotation['video_path']
    frame_number = annotation.get('frameNumber_free_fluid')

    # Validate frame number
    if pd.isna(frame_number):
        print(f"Skipping annotation {annotation['id']} due to missing frame number.")
        return

    # Use 'free_fluid_foreground' instead of 'data_foreground'
    data_foreground = annotation.get('free_fluid_foreground')

    # Check if 'free_fluid_foreground' is valid
    if not isinstance(data_foreground, list) or len(data_foreground) == 0:
        print(f"Skipping annotation {annotation['id']} due to invalid 'free_fluid_foreground'.")
        return

    # Ensure the polygon data is a list of lists of integers
    try:
        data_foreground = [[list(map(int, point)) for point in polygon] for polygon in data_foreground]
    except Exception as e:
        print(f"Error processing polygon data for annotation {annotation['id']}: {e}")
        return
        dicom_metadata (dict): The DICOM metadata JSON content.
    Returns:
        dict: Mapping of SeriesInstanceUID to StudyInstanceUID.
    """
    series_to_study = {}
    for dataset in dicom_metadata.get('datasets', []):
        for study in dataset.get('studies', []):
            study_uid = study.get('StudyInstanceUID')
            series_list = study.get('series', [])
            for series in series_list:
                series_uid = series.get('SeriesInstanceUID')
                if series_uid and study_uid:
                    series_to_study[series_uid] = study_uid

    print(f"Created Series-to-Study mapping with {len(series_to_study)} entries.")
    return series_to_study

# Add these function definitions with your other functions
def cleanup_output_directory(base_dir):
    """
    Cleans up the output directory structure by removing invalid directories
    and organizing the remaining ones.
    """
    valid_methods = ['farneback', 'dis', 'raft', 'deepflow']
    valid_issue_types = ['disappear_reappear', 'branching_fluid', 'multiple_distinct']
    
    for method in os.listdir(base_dir):
        method_path = os.path.join(base_dir, method)
        if not os.path.isdir(method_path):
            continue
            
        if method not in valid_methods:
            print(f"Removing invalid method directory: {method}")
            import shutil
            shutil.rmtree(method_path)
            continue
            
        # Clean up issue type directories
        for issue_dir in os.listdir(method_path):
            issue_path = os.path.join(method_path, issue_dir)
            if not os.path.isdir(issue_path):
                continue
                
            if issue_dir not in valid_issue_types:
                print(f"Removing invalid issue directory: {issue_dir} in {method}")
                import shutil
                shutil.rmtree(issue_path)
                continue

def ensure_metadata_consistency(base_dir):
    """
    Ensures metadata.json files are consistent across all valid directories.
    """
    metadata_template = {
        "annotation_id": None,
        "video_id": None,
        "frame_number": None,
        "output_video_path": None,
        "exam_number": None,
        "StudyInstanceUID": None
    }
    
    for root, dirs, files in os.walk(base_dir):
        if "annotation_" in os.path.basename(root):
            metadata_path = os.path.join(root, "metadata.json")
            
            # Check if metadata exists and is valid
            if os.path.exists(metadata_path):
                try:
                    with open(metadata_path, 'r') as f:
                        metadata = json.load(f)
                    # Ensure all required fields exist
                    for key in metadata_template.keys():
                        if key not in metadata:
                            print(f"Warning: Missing key {key} in {metadata_path}")
                except Exception as e:
                    print(f"Error reading metadata at {metadata_path}: {e}")
            else:
                print(f"Missing metadata.json in {root}")

def verify_output_structure(base_dir):
    """Verifies the output directory structure and prints the findings."""
    print("\nVerifying output structure:")
    if os.path.exists(base_dir):
        print(f"Output directory found at: {base_dir}")
        methods_found = [d for d in os.listdir(base_dir) 
                        if os.path.isdir(os.path.join(base_dir, d))]
        print(f"Methods found: {methods_found}")
        
        for method in methods_found:
            method_dir = os.path.join(base_dir, method)
            issues_found = [d for d in os.listdir(method_dir) 
                          if os.path.isdir(os.path.join(method_dir, d))]
            print(f"\nMethod '{method}' contains issue types: {issues_found}")
            
            for issue in issues_found:
                issue_dir = os.path.join(method_dir, issue)
                annotations = [d for d in os.listdir(issue_dir) 
                             if os.path.isdir(os.path.join(issue_dir, d))]
                print(f"  - Issue '{issue}' has {len(annotations)} annotation directories")
    else:
        print(f"Warning: Output directory not found at {base_dir}")

def reorganize_output_directory(base_dir):
    """
    Complete reorganization of the output directory.
    """
    print("\nStarting output directory cleanup...")
    cleanup_output_directory(base_dir)
    
    print("\nChecking metadata consistency...")
    ensure_metadata_consistency(base_dir)
    
    print("\nVerifying final structure...")
    verify_output_structure(base_dir)

def consolidate_metadata_to_csv(base_dir, output_csv):
    """
    Consolidates all metadata JSON files in the given directory into a single CSV,
    organizing by optical flow method and issue type.
    
    Args:
        base_dir (str): The base directory containing subdirectories for each method
        output_csv (str): The path to save the consolidated CSV file
    """
    metadata_list = []
    
    # Walk through the directory structure
    for method in ['farneback', 'dis', 'raft']:
        method_dir = os.path.join(base_dir, method)
        if not os.path.exists(method_dir):
            print(f"Warning: Method directory not found: {method_dir}")
            continue
            
        for issue_type in ['disappear_reappear', 'branching_fluid', 'multiple_distinct']:
            issue_dir = os.path.join(method_dir, issue_type)
            if not os.path.exists(issue_dir):
                print(f"Warning: Issue type directory not found: {issue_dir}")
                continue
                
            # Walk through annotation directories
            for annotation_dir in os.listdir(issue_dir):
                full_annotation_dir = os.path.join(issue_dir, annotation_dir)
                if not os.path.isdir(full_annotation_dir):
                    continue
                    
                metadata_path = os.path.join(full_annotation_dir, "metadata.json")
                if not os.path.exists(metadata_path):
                    print(f"Warning: No metadata.json found in {full_annotation_dir}")
                    continue
                    
                try:
                    with open(metadata_path, 'r') as f:
                        metadata = json.load(f)
                        # Add method and issue type to metadata
                        metadata['optical_flow_method'] = method
                        metadata['issue_type'] = issue_type
                        
                        # Add video file existence check
                        if 'output_video_path' in metadata:
                            metadata['video_exists'] = os.path.exists(metadata['output_video_path'])
                            
                        # Add relative path for easier access
                        metadata['relative_video_path'] = os.path.relpath(
                            metadata['output_video_path'], base_dir
                        ) if 'output_video_path' in metadata else None
                        
                        metadata_list.append(metadata)
                        print(f"Processed metadata from: {metadata_path}")
                except Exception as e:
                    print(f"Error processing {metadata_path}: {e}")
    
    if not metadata_list:
        print("No metadata files found!")
        return
        
    # Create DataFrame
    df = pd.DataFrame(metadata_list)
    
    # Add success metrics if available
    if 'video_exists' in df.columns:
        success_rate = (df['video_exists'].sum() / len(df)) * 100
        print(f"\nProcessing Success Rate: {success_rate:.2f}%")
    
    # Basic statistics
    print("\nSummary:")
    print(f"Total entries: {len(df)}")
    if 'optical_flow_method' in df.columns:
        print("\nEntries by method:")
        print(df['optical_flow_method'].value_counts())
    if 'issue_type' in df.columns:
        print("\nEntries by issue type:")
        print(df['issue_type'].value_counts())
    
    # Save to CSV
    df.to_csv(output_csv, index=False)
    print(f"\nConsolidated metadata saved to: {output_csv}")
    
    return df



def validate_videos(metadata_df):
    """
    Validates video files referenced in the metadata DataFrame and adds validation information.
    
    Args:
        metadata_df (pd.DataFrame): DataFrame containing the consolidated metadata
        
    Returns:
        pd.DataFrame: Updated DataFrame with video validation information
    """
    def check_video(row):
        if not os.path.exists(row['output_video_path']):
            return {
                'video_exists': False,
                'video_size_mb': 0,
                'video_duration': 0,
                'frame_count': 0
            }
            
        try:
            # Get video file size
            size_mb = os.path.getsize(row['output_video_path']) / (1024 * 1024)
            
            # Get video properties
            cap = cv2.VideoCapture(row['output_video_path'])
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            duration = frame_count / fps if fps > 0 else 0
            cap.release()
            
            return {
                'video_exists': True,
                'video_size_mb': round(size_mb, 2),
                'video_duration': round(duration, 2),
                'frame_count': frame_count
            }
        except Exception as e:
            print(f"Error checking video {row['output_video_path']}: {e}")
            return {
                'video_exists': False,
                'video_size_mb': 0,
                'video_duration': 0,
                'frame_count': 0
            }

    # Apply validation to each row
    validation_results = metadata_df.apply(check_video, axis=1)
    
    # Update DataFrame with validation results
    for key in ['video_exists', 'video_size_mb', 'video_duration', 'frame_count']:
        metadata_df[key] = validation_results.apply(lambda x: x[key])
    
    # Print summary statistics
    print("\nVideo Validation Summary:")
    print(f"Total videos: {len(metadata_df)}")
    print(f"Existing videos: {metadata_df['video_exists'].sum()}")
    print(f"Average video duration: {metadata_df['video_duration'].mean():.2f} seconds")
    print(f"Average video size: {metadata_df['video_size_mb'].mean():.2f} MB")
    
    # Group statistics by method
    if 'optical_flow_method' in metadata_df.columns:
        print("\nSuccess rate by method:")
        success_by_method = metadata_df.groupby('optical_flow_method')['video_exists'].mean() * 100
        print(success_by_method)
    
    return metadata_df
# Example usage
#base_dir = os.path.join(os.getcwd(), "outputs") 
#output_csv = os.path.join(os.getcwd(), "consolidated_metadata.csv")  
#consolidate_metadata_to_csv(base_dir, output_csv)

def create_enhanced_summary(metadata_df, summary_csv):
    """
    Creates detailed summary statistics for optical flow tracking results.
    
    Args:
        metadata_df (pd.DataFrame): DataFrame containing the validated metadata
        summary_csv (str): Path to save the summary Excel file
    """
    # First, check what columns are available
    available_columns = metadata_df.columns
    print("\nAvailable columns for summary:", available_columns.tolist())

    # Basic summary with only existing columns
    agg_dict = {
        'video_exists': ['sum', 'mean'],
        'video_size_mb': ['mean', 'std', 'min', 'max'],
        'video_duration': ['mean', 'std', 'min', 'max'],
        'frame_count': ['mean', 'min', 'max']
    }

    # Remove any columns that don't exist from the aggregation dictionary
    agg_dict = {k: v for k, v in agg_dict.items() if k in available_columns}

    # Create the basic summary
    basic_summary = metadata_df.groupby(['optical_flow_method', 'issue_type']).agg(agg_dict).round(2)

    # Success rates by method
    success_rates = metadata_df.groupby('optical_flow_method').agg({
        'video_exists': lambda x: (x.sum() / len(x) * 100)
    }).round(2)
    
    # Success rates by issue type
    issue_success = metadata_df.groupby('issue_type').agg({
        'video_exists': lambda x: (x.sum() / len(x) * 100)
    }).round(2)

    # Save summaries to Excel with multiple sheets
    with pd.ExcelWriter(summary_csv.replace('.csv', '.xlsx')) as writer:
        basic_summary.to_excel(writer, sheet_name='Detailed Summary')
        success_rates.to_excel(writer, sheet_name='Success by Method')
        issue_success.to_excel(writer, sheet_name='Success by Issue Type')
        
        # Add overall statistics
        overall_stats = pd.DataFrame({
            'Total Videos': len(metadata_df),
            'Successful Videos': metadata_df['video_exists'].sum(),
            'Overall Success Rate': (metadata_df['video_exists'].sum() / len(metadata_df) * 100),
            'Average Duration': metadata_df['video_duration'].mean(),
            'Average Size (MB)': metadata_df['video_size_mb'].mean(),
            'Total Frames': metadata_df['frame_count'].sum()
        }, index=[0]).round(2)
        
        overall_stats.to_excel(writer, sheet_name='Overall Statistics')

    print("\nSummary Statistics:")
    print(f"Total number of videos: {len(metadata_df)}")
    print(f"Successfully processed: {metadata_df['video_exists'].sum()}")
    print(f"Average duration: {metadata_df['video_duration'].mean():.2f} seconds")
    print(f"Average size: {metadata_df['video_size_mb'].mean():.2f} MB")
    
    return basic_summary

def consolidate_videos_and_metadata(base_dir, output_dir, create_symlinks=True):
    """
    Consolidates metadata and creates organized video links.
    
    Args:
        base_dir: Base directory containing the output structure
        output_dir: Directory to store consolidated data
        create_symlinks: Whether to create symbolic links to videos
    """
    metadata_list = []
    organized_dir = os.path.join(output_dir, 'organized_videos')
    os.makedirs(organized_dir, exist_ok=True)
    
    # Walk through the directory structure
    for method in ['farneback', 'dis']:
        for issue_type in ['disappear_reappear', 'branching_fluid', 'multiple_distinct']:
            method_dir = os.path.join(base_dir, method)
            if not os.path.exists(method_dir):
                continue
                
            issue_dir = os.path.join(method_dir, issue_type)
            if not os.path.exists(issue_dir):
                continue
            
            # Create organized directory structure
            target_dir = os.path.join(organized_dir, method, issue_type)
            os.makedirs(target_dir, exist_ok=True)
            
            # Process each annotation directory
            for annotation_dir in os.listdir(issue_dir):
                full_annotation_dir = os.path.join(issue_dir, annotation_dir)
                if not os.path.isdir(full_annotation_dir):
                    continue
                    
                metadata_path = os.path.join(full_annotation_dir, "metadata.json")
                if not os.path.exists(metadata_path):
                    continue
                    
                try:
                    with open(metadata_path, 'r') as f:
                        metadata = json.load(f)
                        
                    # Get video path
                    video_path = metadata.get('output_video_path')
                    if video_path and os.path.exists(video_path):
                        # Create organized filename
                        org_filename = f"{method}_{issue_type}_{metadata['annotation_id']}.mp4"
                        org_video_path = os.path.join(target_dir, org_filename)
                        
                        # Create symbolic link or copy
                        if create_symlinks:
                            if os.path.exists(org_video_path):
                                os.remove(org_video_path)
                            os.symlink(os.path.abspath(video_path), org_video_path)
                        else:
                            shutil.copy2(video_path, org_video_path)
                        
                        # Update metadata
                        metadata.update({
                            'optical_flow_method': method,
                            'issue_type': issue_type,
                            'organized_video_path': org_video_path,
                            'relative_video_path': os.path.relpath(org_video_path, output_dir),
                            'video_exists': True,
                            'video_size_mb': round(os.path.getsize(video_path) / (1024 * 1024), 2)
                        })
                        
                        metadata_list.append(metadata)
                        print(f"Processed: {org_filename}")
                        
                except Exception as e:
                    print(f"Error processing {metadata_path}: {e}")
    
    if not metadata_list:
        print("No metadata files found!")
        return
        
    # Create DataFrame
    df = pd.DataFrame(metadata_list)
    
    # Add validation columns
    df['duration'] = df.apply(lambda row: get_video_duration(row['organized_video_path']), axis=1)
    
    # Save to Excel with multiple sheets
    excel_path = os.path.join(output_dir, 'consolidated_metadata.xlsx')
    
    with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
        # Main data sheet
        df.to_excel(writer, sheet_name='Videos', index=False)
        
        # Summary by method
        method_summary = df.groupby('optical_flow_method').agg({
            'video_exists': 'sum',
            'video_size_mb': 'mean',
            'duration': 'mean'
        }).round(2)
        method_summary.to_excel(writer, sheet_name='By Method')
        
        # Summary by issue type
        issue_summary = df.groupby('issue_type').agg({
            'video_exists': 'sum',
            'video_size_mb': 'mean',
            'duration': 'mean'
        }).round(2)
        issue_summary.to_excel(writer, sheet_name='By Issue Type')
    
    # Save a CSV version
    csv_path = os.path.join(output_dir, 'consolidated_metadata.csv')
    df.to_csv(csv_path, index=False)
    
    print(f"\nResults saved to:")
    print(f"Excel: {excel_path}")
    print(f"CSV: {csv_path}")
    print(f"Organized videos: {organized_dir}")
    
    return df

def get_video_duration(video_path):
    """Get video duration using OpenCV"""
    try:
        cap = cv2.VideoCapture(video_path)
        frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        duration = frames / fps if fps > 0 else 0
        cap.release()
        return round(duration, 2)
    except:
        return 0

# Main processing loop

# Main processing loop

# Load DICOM metadata with error handling
try:
    print("\nLoading DICOM metadata...")
    dicom_metadata_path = "/Users/Shreya1/Documents/GitHub/goobusters/data/mdai_ucsf_project_x9N2LJBZ_dicom_metadata_dataset_D_V688LQ_2024-12-12-040335.json"
    with open(dicom_metadata_path, 'r') as f:
        dicom_metadata = json.load(f)
    print("Successfully loaded DICOM metadata")
except Exception as e:
    print(f"Error loading DICOM metadata: {str(e)}")
    raise

# Create the mappings
study_to_exam_map = create_study_to_exam_mapping(dicom_metadata)
series_to_study_map = create_series_to_study_mapping(dicom_metadata)

# Initialize processing statistics
stats = {
    'processed': 0,
    'failed': 0,
    'uploaded': 0,
    'errors': []
}

# Main processing loop
for method in FLOW_METHOD:
    print(f"\nProcessing method: {method}")
    output_base_dir = os.path.join(OUTPUT_DIR, method)
    os.makedirs(output_base_dir, exist_ok=True)
    
    try:
        flow_processor = OpticalFlowProcessor(method)
    except Exception as e:
        print(f"Error initializing flow processor for {method}: {str(e)}")
        continue
    
    for issue_type in LABEL_IDS.keys():
        print(f"\nProcessing issue type: {issue_type}")
        output_issue_dir = os.path.join(output_base_dir, issue_type)
        os.makedirs(output_issue_dir, exist_ok=True)
        issue_annotations = matched_annotations[matched_annotations['issue_type'] == issue_type]
        
        print(f"Found {len(issue_annotations)} annotations for {issue_type}")
        
        for index, annotation in tqdm(issue_annotations.iterrows(), 
                                    desc=f"Processing {method}-{issue_type}",
                                    total=len(issue_annotations)):
            try:
                output_dir = os.path.join(output_issue_dir, f'annotation_{index}')
                os.makedirs(output_dir, exist_ok=True)
                
                # Get study and series UIDs
                study_uid = annotation['StudyInstanceUID']
                series_uid = annotation['SeriesInstanceUID']
                
                # Validate video path and frame number
                video_path = annotation['video_path']
                if not os.path.exists(video_path):
                    raise FileNotFoundError(f"Video file not found: {video_path}")
                
                frame_number = annotation.get('frameNumber_free_fluid')
                if pd.isna(frame_number):
                    raise ValueError(f"Missing frame number for annotation {annotation['id']}")

                # Validate foreground data
                data_foreground = annotation.get('free_fluid_foreground')
                if not isinstance(data_foreground, list) or len(data_foreground) == 0:
                    raise ValueError(f"Invalid foreground data for annotation {annotation['id']}")

                # Initialize video and create mask
                cap = cv2.VideoCapture(video_path)
                ret, frame = cap.read()
                if not ret:
                    raise RuntimeError(f"Failed to read video: {video_path}")
                frame_height, frame_width = frame.shape[:2]
                cap.release()

                # Create initial mask
                initial_mask = polygons_to_mask(data_foreground, frame_height, frame_width)
                mask_path = os.path.join(output_dir, f'initial_mask_frame_{frame_number}.png')
                cv2.imwrite(mask_path, initial_mask * 255)

                # Define output paths
                output_video_path = os.path.join(output_dir, f'masked_{series_uid}.mp4')
                debug_dir = os.path.join(output_dir, 'debug')
                os.makedirs(debug_dir, exist_ok=True)

                # Process video and upload masks
                result = save_combined_video(
                    video_path=video_path,
                    output_video_path=output_video_path,
                    initial_mask=initial_mask,
                    frame_number=frame_number,
                    debug_dir=debug_dir,
                    flow_processor=flow_processor,
                    mdai_client=mdai_client,
                    study_uid=study_uid,
                    series_uid=series_uid,
                    overlay_color=(0, 255, 0),
                    overlay_alpha=0.3,
                    add_info=True
                )

                if result:
                    stats['processed'] += 1
                    if result.get('mdai_upload_stats', {}).get('uploaded_frames', 0) > 0:
                        stats['uploaded'] += 1

                    # Create metadata with additional information
                    metadata = {
                        "annotation_id": annotation['id'],
                        "video_id": series_uid,
                        "frame_number": frame_number,
                        "output_video_path": output_video_path,
                        "exam_number": study_to_exam_map.get(study_uid, "Unknown"),
                        "StudyInstanceUID": study_uid,
                        "machine_label_id": LABEL_ID_MACHINE_GROUP,
                        "fluid_of_label_id": LABEL_ID_FLUID_OF,
                        "processing_method": method,
                        "issue_type": issue_type,
                        "processing_timestamp": datetime.now().isoformat(),
                        "processing_stats": result
                    }

                    # Save metadata
                    metadata_path = os.path.join(output_dir, "metadata.json")
                    with open(metadata_path, 'w') as f:
                        json.dump(metadata, f, indent=4)

                    print(f"\nSuccessfully processed annotation {annotation['id']}")
                    print(f"Frames processed: {result.get('frames_processed', 0)}")
                    print(f"Frames uploaded: {result.get('mdai_upload_stats', {}).get('uploaded_frames', 0)}")

            except Exception as e:
                stats['failed'] += 1
                error_msg = f"Error processing annotation {annotation['id']}: {str(e)}"
                stats['errors'].append(error_msg)
                print(f"\n{error_msg}")
                traceback.print_exc()
                continue

print("\nProcessing Summary:")
print(f"Total processed: {stats['processed']}")
print(f"Successfully uploaded to MD.ai: {stats['uploaded']}")
print(f"Failed: {stats['failed']}")
if stats['errors']:
    print("\nErrors encountered:")
    for error in stats['errors'][:5]:  # Show first 5 errors
        print(f"- {error}")
    if len(stats['errors']) > 5:
        print(f"...and {len(stats['errors']) - 5} more errors")

# Create evaluation directory and consolidate results
print("\nCreating consolidated metadata and organizing videos...")
EVAL_DIR = os.path.join(PROJECT_DIR, "evaluation")
os.makedirs(EVAL_DIR, exist_ok=True)

try:
    df = consolidate_videos_and_metadata(OUTPUT_DIR, EVAL_DIR)
    print("\nMetadata consolidation completed successfully!")
    print(f"Evaluation materials saved to: {EVAL_DIR}")
    print(f"Excel file: {os.path.join(EVAL_DIR, 'consolidated_metadata.xlsx')}")
    print(f"CSV file: {os.path.join(EVAL_DIR, 'consolidated_metadata.csv')}")
    print(f"Organized videos: {os.path.join(EVAL_DIR, 'organized_videos')}")
except Exception as e:
    print(f"\nError during metadata consolidation: {str(e)}")
    traceback.print_exc()