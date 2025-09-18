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

# Enable debug mode
DEBUG = True

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

FLOW_METHOD = ['farneback', 'deepflow', 'dis']
MASK_MIN_SIZE = 100
INTENSITY_THRESHOLD = 30

# Validate access token
if ACCESS_TOKEN is None:
    raise ValueError("ACCESS_TOKEN is not set.")
print("ACCESS_TOKEN is set")

# Start MD.ai client
mdai_client = mdai.Client(domain=DOMAIN, access_token=ACCESS_TOKEN)
project = mdai_client.project(project_id=PROJECT_ID, path=DATA_DIR)

# Load and normalize annotations
annotations_file_path = ANNOTATIONS
with open(annotations_file_path, 'r') as f:
    annotations_json = json.load(f)

datasets = annotations_json.get('datasets', [])
all_annotations = []
for dataset in datasets:
    if 'annotations' in dataset:
        all_annotations.extend(dataset['annotations'])

annotations_df = json_normalize(all_annotations, sep='_')

# Filter "free fluid" annotations
free_fluid_annotations = annotations_df[
    (annotations_df['labelId'] == LABEL_ID_FREE_FLUID) &
    (annotations_df['frameNumber'].notna())
].copy()
free_fluid_annotations['frameNumber'] = free_fluid_annotations['frameNumber'].astype(int)

# Construct video paths
BASE = project.get_dataset_by_id(DATASET_ID).images_dir
free_fluid_annotations['video_path'] = free_fluid_annotations.apply(
    lambda row: os.path.join(BASE, row['StudyInstanceUID'], f"{row['SeriesInstanceUID']}.mp4"), axis=1
)
free_fluid_annotations['file_exists'] = free_fluid_annotations['video_path'].apply(os.path.exists)
free_fluid_annotations = free_fluid_annotations[free_fluid_annotations['file_exists']]

print(f"Free fluid annotations with valid video files: {len(free_fluid_annotations)}")

# Pair "free fluid" annotations with each complication type
matched_annotations = pd.DataFrame()
for issue_type, label_id in LABEL_IDS.items():
    issue_annotations = annotations_df[annotations_df['labelId'] == label_id].copy()
    merged_annotations = pd.merge(
        issue_annotations,
        free_fluid_annotations[['StudyInstanceUID', 'SeriesInstanceUID', 'frameNumber', 'video_path']],
        on=['StudyInstanceUID', 'SeriesInstanceUID'],
        suffixes=('_complication', '_free_fluid'),
        how='inner'
    )
    merged_annotations['file_exists'] = merged_annotations['video_path'].apply(os.path.exists)
    merged_annotations = merged_annotations[merged_annotations['file_exists']]

    sample_size = min(5, len(merged_annotations))
    sampled_annotations = merged_annotations.sample(n=sample_size, random_state=42)
    sampled_annotations['issue_type'] = issue_type
    matched_annotations = pd.concat([matched_annotations, sampled_annotations])

print(f"Total matched annotations: {len(matched_annotations)}")

# Define helper functions
def polygons_to_mask(polygons, height, width):
    mask = np.zeros((height, width), dtype=np.uint8)
    for polygon in polygons:
        points = np.array(polygon, dtype=np.int32)
        cv2.fillPoly(mask, [points], 1)
    return mask

def save_combined_video(video_path, output_video_path, initial_mask, frame_number, debug_dir, flow_processor):
    # Ensure directories exist
    save_dir = os.path.dirname(output_video_path)
    mask_dir = os.path.join(save_dir, "masks")
    os.makedirs(mask_dir, exist_ok=True)

    # Open the video file
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Define the video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

    print("Writing combined video...")

    # Set progress bar
    with tqdm(total=total_frames, desc="Processing video", unit="frame") as pbar:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = cap.read()
        if not ret:
            print(f"Error: Failed to read frame {frame_number}.")
            cap.release()
            return

        # Initialize mask and previous frame
        prev_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mask = initial_mask.astype(float)

        # Process frames forward from the starting frame
        while ret:
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Apply optical flow to update the mask
            flow_mask = flow_processor.apply_optical_flow(prev_gray, frame_gray, mask)
            blended_mask = (0.7 * mask + 0.3 * flow_mask).astype(float)

            # Create a mask overlay on the frame
            mask_overlay = frame.copy()
            mask_overlay[blended_mask > 0] = [0, 255, 0]

            # Write the frame with overlay to the output video
            out.write(mask_overlay)

            # Save the mask as an image
            mask_filename = os.path.join(mask_dir, f"mask_{frame_number:04d}.png")
            cv2.imwrite(mask_filename, (blended_mask * 255).astype(np.uint8))

            # Update for the next iteration
            prev_gray = frame_gray.copy()
            mask = blended_mask
            frame_number += 1
            pbar.update(1)

            # Read the next frame
            ret, frame = cap.read()

    # Release resources
    cap.release()
    out.release()
    print(f"Video saved at {output_video_path}")

def track_and_save_masks_as_video(annotation, output_dir, flow_processor):
    video_id = annotation['SeriesInstanceUID']
    video_path = annotation['video_path']
    print(f"Annotation columns: {annotation.index}")

    # Retrieve frame number for free fluid annotation
    frame_number = annotation.get('frameNumber_free_fluid')

    # Safely access the 'data' field
    data_field = annotation['data']

    # Initialize foreground to None
    foreground = None

    # Check if the data field is a valid dictionary, not NaN or float
    if isinstance(data_field, dict):
        foreground = data_field.get('foreground', None)
    elif isinstance(data_field, str):
        try:
            # Attempt to parse JSON string
            data_dict = json.loads(data_field)
            foreground = data_dict.get('foreground', None)
        except json.JSONDecodeError:
            print(f"Warning: Failed to parse 'data' field for annotation {annotation}")
    else:
        print(f"Warning: Unexpected data type for 'data' field: {type(data_field)}")

    # Check if foreground data is available
    if foreground is None or not isinstance(foreground, list):
        print(f"Warning: No valid foreground data for annotation {annotation['id']}. Skipping this annotation.")
        return

    print(f"Processing Video: {video_id}; Frame: {frame_number}...")

    # Open the video file and set to the specified frame
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    ret, frame = cap.read()
    if not ret:
        print(f"Failed to read the frame number {frame_number} from the video.")
        return
    cap.release()

    # Create the initial mask from the foreground data
    try:
        initial_mask = polygons_to_mask(foreground, frame.shape[0], frame.shape[1])
    except TypeError:
        print(f"Error: Invalid foreground data for annotation {annotation['id']}. Skipping this annotation.")
        return

    output_video_path = os.path.join(output_dir, f'masked_{video_id}.mp4')
    debug_dir = os.path.join(output_dir, 'debug')

    # Continue with the rest of the processing (e.g., saving video)
    save_combined_video(video_path, output_video_path, initial_mask, frame_number, debug_dir, flow_processor)
    print(f"Saved masked video for {video_id} at {output_video_path}")

    save_combined_video(video_path, output_video_path, initial_mask, frame_number, debug_dir, flow_processor)

# Main processing loop
for method in FLOW_METHOD:
    output_base_dir = os.path.join('output', method)
    os.makedirs(output_base_dir, exist_ok=True)
    flow_processor = OpticalFlowProcessor(method)

    for issue_type in LABEL_IDS.keys():
        output_issue_dir = os.path.join(output_base_dir, issue_type)
        os.makedirs(output_issue_dir, exist_ok=True)
        issue_annotations = matched_annotations[matched_annotations['issue_type'] == issue_type]

        for index, annotation in issue_annotations.iterrows():
            output_dir = os.path.join(output_issue_dir, f'annotation_{index}')
            os.makedirs(output_dir, exist_ok=True)
            track_and_save_masks_as_video(annotation, output_dir, flow_processor)

print("Tracking and saving videos completed.")