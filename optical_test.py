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

def track_frames(cap, start_frame, end_frame, initial_mask, debug_dir, forward=True, pbar=None, flow_processor=None):
    frames = []
    step = 1 if forward else -1
    frame_idx = start_frame

    # Set the video capture to the starting frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, prev_frame = cap.read()
    if not ret:
        print(f"Failed to read starting frame {start_frame}.")
        return frames

    # Convert the previous frame to grayscale
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    mask = initial_mask.astype(float)

    while (forward and frame_idx <= end_frame) or (not forward and frame_idx >= 0):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            print(f"Failed to read frame {frame_idx}. Ending tracking.")
            break

        # Convert the current frame to grayscale
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Initialize masks for the first frame
        if frame_idx == start_frame:
            flow_mask = np.zeros_like(mask)
            adjusted_mask = np.zeros_like(mask)
            new_mask = mask
        else:
            # Apply optical flow using the chosen method
            try:
                flow_mask = flow_processor.apply_optical_flow(prev_gray, frame_gray, mask)
            except Exception as e:
                print(f"Error applying optical flow at frame {frame_idx}: {e}")
                break

            # Blend the flow mask with the previous mask
            adjusted_mask = flow_mask
            blended_mask = (0.7 * mask + 0.3 * adjusted_mask).astype(float)
            new_mask = np.clip(blended_mask, 0, 1)

        # Save the debug frame if in DEBUG mode
        if DEBUG:
            debug_frame = debug_visualize(frame, initial_mask, flow_mask, adjusted_mask, new_mask, frame_idx)
            os.makedirs(debug_dir, exist_ok=True)
            cv2.imwrite(os.path.join(debug_dir, f'debug_frame_{frame_idx:04d}.png'), debug_frame)

        # Append the current frame, mask, and index to the list
        frames.append((frame_idx, frame, new_mask))

        # Update the previous frame and mask for the next iteration
        prev_gray = frame_gray.copy()
        mask = new_mask
        frame_idx += step

        # Update progress bar if provided
        if pbar:
            pbar.update(1)

    return frames
   
def save_combined_video(video_path, output_video_path, initial_mask, frame_number, debug_dir, flow_processor):
    mask_dir = os.path.join(os.path.dirname(output_video_path), "masks")
    os.makedirs(mask_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    with tqdm(total=total_frames, desc="Processing video", unit="frame") as pbar:
        backward_frames = track_frames(cap, frame_number, 0, initial_mask, debug_dir, forward=False, pbar=pbar, flow_processor=flow_processor)
        forward_frames = track_frames(cap, frame_number, total_frames - 1, initial_mask, debug_dir, forward=True, pbar=pbar, flow_processor=flow_processor)

    combined_frames = backward_frames[::-1] + forward_frames[1:]

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

    print("Writing video...")
    for frame_idx, frame, mask in tqdm(combined_frames, desc="Saving frames", unit="frame"):
        out.write(frame)

        mask_filename = os.path.join(mask_dir, f"mask_{frame_idx:04d}.png")
        cv2.imwrite(mask_filename, (mask * 255).astype(np.uint8))

        # Create and save debug frame
        debug_frame = debug_visualize(frame, initial_mask, mask, mask, mask, frame_idx)
        cv2.imwrite(os.path.join(debug_dir, f'debug_frame_{frame_idx:04d}.png'), debug_frame)

    cap.release()
    out.release()
    print(f"Video saved at {output_video_path}")

# Function to create detailed debug visualization
def debug_visualize(frame, initial_mask, flow_mask, adjusted_mask, final_mask, frame_number):
    # Create a 2x3 grid of images
    grid = np.zeros((frame.shape[0] * 2, frame.shape[1] * 3, 3), dtype=np.uint8)

    # Original frame (Top Left)
    grid[:frame.shape[0], :frame.shape[1]] = frame
    cv2.putText(grid, "Original Frame", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Frame with final mask applied (Bottom Left)
    mask_overlay = frame.copy()
    mask_overlay[final_mask > 0] = mask_overlay[final_mask > 0] * 0.5 + np.array([0, 255, 0], dtype=np.uint8) * 0.5
    grid[frame.shape[0]:, :frame.shape[1]] = mask_overlay
    cv2.putText(grid, "Frame with Mask", (10, frame.shape[0] + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Initial mask (Top Center)
    initial_overlay = np.zeros_like(frame)
    initial_overlay[:, :, 0] = initial_mask * 255  # Blue color for initial mask
    grid[:frame.shape[0], frame.shape[1]:frame.shape[1] * 2] = initial_overlay
    cv2.putText(grid, "Initial Mask", (frame.shape[1] + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Flow mask (Top Right)
    flow_overlay = np.zeros_like(frame)
    flow_overlay[:, :, 2] = flow_mask * 255
    grid[:frame.shape[0], frame.shape[1] * 2:] = flow_overlay
    cv2.putText(grid, "Flow Mask", (frame.shape[1] * 2 + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Adjusted mask (Bottom Center)
    adjusted_overlay = np.zeros_like(frame)
    adjusted_overlay[:, :, 0] = adjusted_mask * 255
    grid[frame.shape[0]:, frame.shape[1]:frame.shape[1] * 2] = adjusted_overlay
    cv2.putText(grid, "Adjusted Mask", (frame.shape[1] + 10, frame.shape[0] + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Final mask (Bottom Right)
    final_overlay = np.zeros_like(frame)
    final_overlay[:, :, 1] = final_mask * 255
    grid[frame.shape[0]:, frame.shape[1] * 2:] = final_overlay
    cv2.putText(grid, "Final Mask", (frame.shape[1] * 2 + 10, frame.shape[0] + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Add frame number
    cv2.putText(grid, f"Frame: {frame_number}", (10, grid.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    return grid

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


# Function to create a mask from annotation polygons
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

# Main processing loop

# Load DICOM metadata 
dicom_metadata_path = "/Users/Shreya1/Documents/GitHub/goobusters/data/mdai_ucsf_project_x9N2LJBZ_dicom_metadata_dataset_D_V688LQ_2024-12-12-040335.json"
with open(dicom_metadata_path, 'r') as f:
    dicom_metadata = json.load(f)

# Create the mappings
study_to_exam_map = create_study_to_exam_mapping(dicom_metadata)
series_to_study_map = create_series_to_study_mapping(dicom_metadata)

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
            track_and_save_masks(annotation, output_dir, flow_processor, study_to_exam_map, series_to_study_map)

print("Tracking and saving videos completed.")

def consolidate_metadata_to_csv(base_dir, output_csv):
    """
    Consolidates all metadata JSON files in the given directory into a single CSV.
    
    Args:
        base_dir (str): The base directory containing metadata.json files.
        output_csv (str): The path to save the consolidated CSV file.
    """
    metadata_list = []

    # Traverse through all subdirectories in the base directory
    for root, dirs, files in os.walk(base_dir):
        print(f"Scanning directory: {root}")  # Debug: Print each directory
        for file in files:
            print(f"Found file: {file}")  # Debug: Print each file
            if file.lower() == "metadata.json":  # Case-insensitive match
                metadata_path = os.path.join(root, file)
                try:
                    with open(metadata_path, 'r') as f:
                        metadata = json.load(f)
                        if metadata:  # Ensure metadata is not empty
                            metadata_list.append(metadata)
                            print(f"Appended metadata from: {metadata_path}")
                        else:
                            print(f"Empty metadata file skipped: {metadata_path}")
                except json.JSONDecodeError as e:
                    print(f"Error decoding {metadata_path}: {e}")
                except Exception as e:
                    print(f"Error reading {metadata_path}: {e}")

    # Check if any metadata was found
    if metadata_list:
        # Get all unique keys from metadata
        all_keys = set().union(*[m.keys() for m in metadata_list])

        # Ensure all dictionaries have the same keys
        metadata_list = [{key: metadata.get(key, None) for key in all_keys} for metadata in metadata_list]

        # Convert the list of dictionaries into a DataFrame
        metadata_df = pd.DataFrame(metadata_list)

        # Remove duplicate rows
        metadata_df.drop_duplicates(inplace=True)

        # Save the DataFrame to a CSV
        metadata_df.to_csv(output_csv, index=False)
        print(f"Consolidated metadata saved to {output_csv}")
    else:
        print(f"No metadata.json files found in the directory: {base_dir}")

# Example usage
#base_dir = os.path.join(os.getcwd(), "outputs") 
#output_csv = os.path.join(os.getcwd(), "consolidated_metadata.csv")  
#consolidate_metadata_to_csv(base_dir, output_csv)

base_dir = "/Users/Shreya1/Documents/GitHub/goobusters/outputs"  
output_csv = "/Users/Shreya1/Documents/GitHub/goobusters/consolidated_metadata.csv"  
consolidate_metadata_to_csv(base_dir, output_csv)

