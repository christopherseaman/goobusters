# ground_truth_utils.py
import os
import json
import numpy as np
import pandas as pd
import traceback
import matplotlib.pyplot as plt
import time
from src.multi_frame_tracking.multi_frame_tracker import MultiFrameTracker
import mdai
import cv2
from .shared_utils import process_video_with_multi_frame_tracking_enhanced, get_annotations_for_study_series


DATA_DIR = os.getenv('DATA_DIR')

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
    
def delete_existing_annotations(client, study_uid, series_uid, label_id, group_id=None):
    """
    Delete existing annotations for a study/series with a specific label ID
    
    This can be moved from your existing utils.py file if you already have it
    """
    # Your existing implementation...
    pass

#Function for creating ground truth dataset (will run if flag is specified)
LABEL_ID_FREE_FLUID = os.environ["LABEL_ID_FREE_FLUID"] = "L_13yPql"
LABEL_ID_NO_FLUID = os.getenv("LABEL_ID_NO_FLUID") 
LABEL_ID_MACHINE_GROUP = os.getenv("LABEL_ID_MACHINE_GROUP")


def create_ground_truth_dataset(video_paths, study_series_pairs, flow_processor, output_dir, 
                              mdai_client, project_id, dataset_id, ground_truth_label_id,
                              matched_annotations=None, free_fluid_annotations=None, 
                              label_id_fluid=None, label_id_no_fluid=None, 
                              label_id_machine=None, annotations_json=None, args=None,
                              label_ids=None, base_path=None, upload=False):
    """
    Creates a ground truth dataset following the exact same annotation processing 
    pattern as the normal workflow
    """
    
    print("\n=== STARTING GROUND TRUTH DATASET CREATION ===")
    print(f"Number of videos to process: {len(video_paths)}")
    print(f"Output directory: {output_dir}")
    
    # Check for global no-fluid labels
    global_no_fluid_label = "L_7BGg21"  # The label ID for global no-fluid exams
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
    
    # Add a max frames limit to prevent infinite processing
    MAX_FRAMES_TO_PROCESS = 150  # Adjust this value based on your needs
    MAX_PROCESSING_TIME = 600  # Maximum processing time in seconds (10 minutes)
    
    # Initialize time tracking variables
    start_time = time.time()
    last_progress_time = time.time()
    frames_processed = 0

    # Determine upload status from both args and parameter
    should_upload = upload or (args and hasattr(args, 'upload') and args.upload)

    print("\n=== UPLOAD CONFIGURATION ===")
    print(f"Upload parameter: {upload}")
    print(f"Args upload value: {args.upload if args and hasattr(args, 'upload') else 'Not set'}")
    print(f"Final should_upload value: {should_upload}")
    print(f"MD.ai client: {'Available' if mdai_client else 'Not available'}")
    print(f"Project ID: {project_id}")
    print(f"Dataset ID: {dataset_id}")
    print("===========================\n")
    
    # Verify we have the necessary data
    if matched_annotations is None or free_fluid_annotations is None or matched_annotations.empty or free_fluid_annotations.empty:
        print("ERROR: Missing required annotation data")
        return {"error": "Missing required annotation data"}
        
    print("✓ Required annotation data available")
    print(f"Matched annotations: {len(matched_annotations)} rows")
    print(f"Free fluid annotations: {len(free_fluid_annotations)} rows")
    
    # CRITICAL: Adjust frame numbers from MD.ai (1-based) to our system (0-based)
    print("\nAdjusting frame numbers from MD.ai 1-based to 0-based indexing...")
    
    if 'frameNumber' in matched_annotations.columns:
        matched_annotations['original_mdai_frame'] = matched_annotations['frameNumber'].copy()
        matched_annotations['frameNumber'] = matched_annotations['frameNumber'].astype(int) - 1
        print(f"Adjusted {len(matched_annotations)} frame numbers in matched_annotations")
        print("Sample adjustments:")
        for i in range(min(5, len(matched_annotations))):
            print(f"  MD.ai frame {matched_annotations['original_mdai_frame'].iloc[i]} -> System frame {matched_annotations['frameNumber'].iloc[i]}")
    
    if 'frameNumber' in free_fluid_annotations.columns:
        free_fluid_annotations['original_mdai_frame'] = free_fluid_annotations['frameNumber'].copy()
        free_fluid_annotations['frameNumber'] = free_fluid_annotations['frameNumber'].astype(int) - 1
        print(f"Adjusted {len(free_fluid_annotations)} frame numbers in free_fluid_annotations")
        print("Sample adjustments:")
        for i in range(min(5, len(free_fluid_annotations))):
            print(f"  MD.ai frame {free_fluid_annotations['original_mdai_frame'].iloc[i]} -> System frame {free_fluid_annotations['frameNumber'].iloc[i]}")

    # Store the original frame numbers for later use when uploading back to MD.ai
    if 'original_mdai_frame' not in free_fluid_annotations.columns:
        free_fluid_annotations['original_mdai_frame'] = free_fluid_annotations['frameNumber'] + 1
    
    # Verify video paths and try to find them if they don't exist
    updated_video_paths = []
    for i, (study_uid, series_uid) in enumerate(study_series_pairs):
        valid_path = None
        
        # Check if the provided path exists
        if i < len(video_paths) and os.path.exists(video_paths[i]):
            valid_path = video_paths[i]
        else:
            # Try to find the video file using our utility function
            found_path = find_video_file(study_uid, series_uid)
            if found_path:
                valid_path = found_path
                print(f"Found video file for {study_uid}/{series_uid} at: {found_path}")
            
        if valid_path:
            updated_video_paths.append(valid_path)
        else:
            print(f"WARNING: Could not find video file for {study_uid}/{series_uid}")
    
    # If we found valid paths, update video_paths
    if updated_video_paths:
        print(f"Updated {len(updated_video_paths)} video paths")
        video_paths = updated_video_paths
    else:
        print("WARNING: No valid video paths found")
        return {"error": "No valid video paths found"}
        
    print("\nDEBUG: Starting annotation processing")
    # OVERRIDE the label_id_fluid parameter
    label_id_fluid = os.getenv("LABEL_ID_FREE_FLUID")
    print(f"OVERRIDE: Using LABEL_ID_FREE_FLUID: {label_id_fluid}")
    print(f"Debug: label_id_fluid = {label_id_fluid}")    
    # Import the find_exam_number function
    from src.multi_frame_tracking.utils import find_exam_number
    
    # Define a default LABEL_IDS if not provided
    LABEL_IDS = label_ids or {}
    
    # Set global BASE variable if provided
    if base_path:
        global BASE
        BASE = base_path
        print(f"Using provided BASE path: {BASE}")
    
    # CRITICAL NEW APPROACH: Let's extract annotations directly from the JSON for Exam #64
    if args and hasattr(args, 'ground_truth_single_exam') and args.ground_truth_single_exam:
        target_exam = args.ground_truth_single_exam
        print(f"\n*** FILTERING TO ONLY PROCESS EXAM #{target_exam} ***")
        
        target_studies = []
        
        # Search through all datasets in annotations_json
        if annotations_json:
            print(f"Looking for studies in exam #{target_exam} directly in annotations_json...")
            
            try:
                datasets = annotations_json.get('datasets', [])
                for dataset in datasets:
                    studies = dataset.get('studies', [])
                    for study in studies:
                        study_uid = study.get('StudyInstanceUID')
                        if not study_uid:
                            continue
                            
                        try:
                            exam_number = find_exam_number(study_uid, annotations_json)
                            if str(exam_number) == str(target_exam):
                                target_studies.append(study)
                                print(f"Found study {study_uid} for exam #{target_exam}")
                                
                                # Look for annotations directly
                                series_list = study.get('seriesList', [])
                                for series in series_list:
                                    series_uid = series.get('SeriesInstanceUID')
                                    if series_uid:
                                        print(f"  Series: {series_uid}")
                                    
                                    # Look for annotations with polygons
                                    annotations_found = False
                                    
                                    # Try to iterate through all annotations in annotations_json
                                    all_annotations = []
                                    for dataset in annotations_json.get('datasets', []):
                                        if 'annotations' in dataset:
                                            all_annotations.extend(dataset['annotations'])
                                    
                                    series_annotations = [
                                        ann for ann in all_annotations
                                        if ann.get('StudyInstanceUID') == study_uid and 
                                           ann.get('SeriesInstanceUID') == series_uid and
                                           ann.get('labelId') == label_id_fluid
                                    ]
                                    
                                    # Check if video file exists in any data directory
                                    video_path = find_video_file(study_uid, series_uid)
                                    if video_path:
                                        print(f"  ✓ Found corresponding video file at: {video_path}")
                                        # This will make sure we actually keep this study/series
                                    
                                    if series_annotations:
                                        print(f"  Found {len(series_annotations)} free fluid annotations")
                                        
                                        # Check for polygons
                                        for ann in series_annotations[:10]:  # Check first 10
                                            data = ann.get('data', {})
                                            foreground = data.get('foreground', [])
                                            
                                            if isinstance(foreground, list) and len(foreground) > 0:
                                                print(f"  Found valid polygons in frame {ann.get('frameNumber')}")
                                                print(f"  First polygon: {foreground[0]}")
                                                annotations_found = True
                                                break
                                    
                                    if not annotations_found:
                                        print(f"  No valid polygons found for this series")
                        except Exception as e:
                            print(f"Error checking exam number for {study_uid}: {e}")
                            continue
                
                if not target_studies:
                    print(f"ERROR: No studies found for exam #{target_exam}")
                    return {"error": f"No studies found for exam #{target_exam}"}
                    
                print(f"Found {len(target_studies)} studies for exam #{target_exam}")
            except Exception as e:
                print(f"Error searching annotations_json: {e}")
                traceback.print_exc()
        
        # Filter video paths based on study/series UIDs
        filtered_pairs = []
        filtered_paths = []
        
        for i, ((study_uid, series_uid), video_path) in enumerate(zip(study_series_pairs, video_paths)):
            try:
                exam_number = find_exam_number(study_uid, annotations_json)
                if str(exam_number) == str(target_exam):
                    if (study_uid, series_uid) not in filtered_pairs:
                        filtered_pairs.append((study_uid, series_uid))
                        filtered_paths.append(video_path)
                        print(f"✓ Adding video for exam #{exam_number}: {study_uid}/{series_uid}")
            except Exception as e:
                print(f"Error checking exam number for {study_uid}: {e}")
                
        if not filtered_pairs:
            print(f"WARNING: No matching videos found using annotation approach for exam #{target_exam}")
            print(f"Trying to find videos for all studies in this exam...")
            
            # Search for all studies with this exam number in annotations
            all_studies_with_exam = []
            try:
                datasets = annotations_json.get('datasets', [])
                for dataset in datasets:
                    studies = dataset.get('studies', [])
                    for study in studies:
                        study_uid = study.get('StudyInstanceUID')
                        if not study_uid:
                            continue
                            
                        try:
                            exam_number = find_exam_number(study_uid, annotations_json)
                            if str(exam_number) == str(target_exam):
                                all_studies_with_exam.append(study_uid)
                                print(f"Found study {study_uid} for exam #{target_exam}")
                        except Exception as e:
                            print(f"Error checking exam number for {study_uid}: {e}")
                            continue
            except Exception as e:
                print(f"Error searching for studies: {e}")
            
            # Try to find videos for all these studies
            for study_uid in all_studies_with_exam:
                # Search for all data directories
                data_dirs = sorted([d for d in os.listdir('data') 
                                if os.path.isdir(os.path.join('data', d)) and 
                                (d.startswith('mdai_') or d == 'images')], 
                                key=lambda d: os.path.getmtime(os.path.join('data', d)), 
                                reverse=True)
                
                for data_dir in data_dirs:
                    study_path = os.path.join('data', data_dir, study_uid)
                    if os.path.isdir(study_path):
                        print(f"Checking directory: {study_path}")
                        for file in os.listdir(study_path):
                            if file.endswith('.mp4'):
                                series_uid = file.replace('.mp4', '')
                                video_path = os.path.join(study_path, file)
                                filtered_pairs.append((study_uid, series_uid))
                                filtered_paths.append(video_path)
                                print(f"  ✓ Added video: {video_path}")
                
                # If we didn't find any videos yet, try using find_video_file
                if not filtered_paths:
                    # Get all series UIDs for this study
                    series_uids = set()
                    for dataset in annotations_json.get('datasets', []):
                        if 'annotations' in dataset:
                            for ann in dataset['annotations']:
                                if ann.get('StudyInstanceUID') == study_uid:
                                    series_uid = ann.get('SeriesInstanceUID')
                                    if series_uid:
                                        series_uids.add(series_uid)
                    
                    # Try to find video for each series
                    for series_uid in series_uids:
                        video_path = find_video_file(study_uid, series_uid)
                        if video_path:
                            filtered_pairs.append((study_uid, series_uid))
                            filtered_paths.append(video_path)
                            print(f"  ✓ Found video using utility: {video_path}")
            
        if not filtered_pairs:
            print(f"ERROR: No matching videos found for exam #{target_exam}")
            return {"error": f"No videos found for exam #{target_exam}"}
        
        # Replace the original lists with the filtered ones
        study_series_pairs = filtered_pairs
        video_paths = filtered_paths
        print(f"Filtered to {len(study_series_pairs)} videos for exam #{target_exam}")
        
        # CRITICAL: Instead of using the existing filtered data, let's create fresh annotations from scratch
        print("Creating fresh annotation data for this exam...")
        
        # Initialize arrays to store the data
        study_uids = []
        series_uids = []
        frame_numbers = []
        video_paths_list = []
        polygon_data = []
        issue_types = []
        
        # Process each study/series pair
        for (study_uid, series_uid), video_path in zip(study_series_pairs, video_paths):
            # Find all annotations for this study/series
            all_annotations = []
            for dataset in annotations_json.get('datasets', []):
                if 'annotations' in dataset:
                    all_annotations.extend(dataset['annotations'])
            
            study_series_annotations = [
                ann for ann in all_annotations
                if ann.get('StudyInstanceUID') == study_uid and 
                   ann.get('SeriesInstanceUID') == series_uid
            ]
            
            # Find issue type
            issue_type = "unknown"
            
            # Print available label IDs for debugging
            print(f"Available label IDs: {label_ids}")
            
            if label_ids:
                for ann in study_series_annotations:
                    labelId = ann.get('labelId')
                    for it_name, it_id in label_ids.items():
                        if labelId == it_id:
                            issue_type = it_name
                            print(f"Found issue type: {issue_type} (labelId: {labelId})")
                            break
                    if issue_type != "unknown":
                        break
            
            # Find annotations with valid polygons
            free_fluid_anns = [
                ann for ann in study_series_annotations
                if ann.get('labelId') == label_id_fluid
            ]
            
            print(f"Found {len(free_fluid_anns)} free fluid annotations")
            valid_polygon_count = 0
            
            for ann in free_fluid_anns:
                if 'data' in ann and isinstance(ann['data'], dict):
                    foreground = ann['data'].get('foreground', [])
                    if isinstance(foreground, list) and len(foreground) > 0:
                        valid_polygon_count += 1
                        # Add this annotation to our data
                        study_uids.append(study_uid)
                        series_uids.append(series_uid)
                        frame_numbers.append(int(ann.get('frameNumber', 0)))
                        video_paths_list.append(video_path)
                        polygon_data.append(foreground)
                        issue_types.append(issue_type)
            
            print(f"Found {valid_polygon_count} valid polygons")
        
        # Create a new DataFrame
        if study_uids:
            import pandas as pd
            # Create the DataFrame
            new_df = pd.DataFrame({
                'StudyInstanceUID': study_uids,
                'SeriesInstanceUID': series_uids,
                'frameNumber': frame_numbers,
                'video_path': video_paths_list,
                'free_fluid_foreground': polygon_data,
                'issue_type': issue_types
            })
            
            print(f"Created fresh DataFrame with {len(new_df)} rows of annotation data")
            print("Sample row:")
            if len(new_df) > 0:
                sample_row = new_df.iloc[0]
                print(f"Study UID: {sample_row['StudyInstanceUID']}")
                print(f"Series UID: {sample_row['SeriesInstanceUID']}")
                print(f"Frame number: {sample_row['frameNumber']}")
                print(f"Issue type: {sample_row['issue_type']}")
                print(f"Polygon data type: {type(sample_row['free_fluid_foreground'])}")
                print(f"Polygon data length: {len(sample_row['free_fluid_foreground'])}")
                print(f"First polygon sample: {str(sample_row['free_fluid_foreground'][0])[:100]}...")
            
            # Replace both matched_annotations and free_fluid_annotations
            matched_annotations = new_df
            free_fluid_annotations = new_df
            
            print(f"Using freshly created annotation data for exam #{target_exam}")
        else:
            print("ERROR: Could not create fresh annotation data - no valid polygons found")
            return {"error": "No valid polygons found for any video"}
    
    # Continue with original function
    print(f"Starting to process {len(matched_annotations)} matched annotations")
    os.makedirs(output_dir, exist_ok=True)
    
    if matched_annotations is None or free_fluid_annotations is None:
        raise ValueError("Both matched_annotations and free_fluid_annotations must be provided")
    
    results = {
        'processed_videos': [],
        'failed_videos': [],
        'total_annotations_created': 0,
        'total_upload_success': 0,
        'total_upload_failures': 0
    }
    
    print(f"Creating ground truth dataset for {len(video_paths)} videos...")
    
    for i, (video_path, (study_uid, series_uid)) in enumerate(zip(video_paths, study_series_pairs)):
        print(f"\nProcessing video {i+1}/{len(video_paths)}")
        print(f"Study UID: {study_uid}")
        print(f"Series UID: {series_uid}")

        # Find exam number for this study
        exam_number = "unknown"
        if annotations_json:
            try:
                exam_number = find_exam_number(study_uid, annotations_json)
                print(f"Exam Number: {exam_number}")
                
                # Check if this is a globally no-fluid exam
                if exam_number in no_fluid_exams:
                    print(f"This exam is marked as globally no-fluid")
                    # Create empty masks for all frames
                    cap = cv2.VideoCapture(video_path)
                    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    cap.release()
                    
                    # Create output directory
                    video_output_dir = os.path.join(output_dir, f"exam_{exam_number}_{study_uid}_{series_uid}")
                    os.makedirs(video_output_dir, exist_ok=True)
                    
                    # Save empty masks for all frames
                    empty_mask = np.zeros((frame_height, frame_width), dtype=np.uint8)
                    for frame_idx in range(total_frames):
                        mask_path = os.path.join(video_output_dir, f"frame_{frame_idx:04d}_mask.png")
                        cv2.imwrite(mask_path, empty_mask)
                    
                    print(f"Created {total_frames} empty masks for no-fluid exam")
                    continue
                    
            except Exception as e:
                print(f"Could not determine exam number: {e}")
        
        # Create output directory for this video
        video_output_dir = os.path.join(output_dir, f"exam_{exam_number}_{study_uid}_{series_uid}")
        os.makedirs(video_output_dir, exist_ok=True)

        try:
            # Get all annotations for this video
            video_annotations = free_fluid_annotations[
                (free_fluid_annotations['StudyInstanceUID'] == study_uid) &
                (free_fluid_annotations['SeriesInstanceUID'] == series_uid)
            ]
            
            if len(video_annotations) == 0:
                print(f"No annotations found for {study_uid}/{series_uid}, skipping")
                results['failed_videos'].append({
                    'video': video_path,
                    'study_uid': study_uid,
                    'series_uid': series_uid,
                    'exam_number': exam_number,
                    'error': 'No annotations found'
                })
                continue
            
            # First, identify no-fluid frames
            no_fluid_frames = []
            if 'labelId' in video_annotations.columns:
                no_fluid_frames = video_annotations[
                    video_annotations['labelId'] == label_id_no_fluid
                ]['frameNumber'].tolist()
                
                print(f"\nFound {len(no_fluid_frames)} no-fluid frames: {sorted(no_fluid_frames)}")
                
                # If we have multiple no-fluid frames, all frames between them should be no-fluid
                if len(no_fluid_frames) >= 2:
                    no_fluid_frames = sorted(no_fluid_frames)
                    start_frame = no_fluid_frames[0]
                    end_frame = no_fluid_frames[-1]
                    print(f"All frames between {start_frame} and {end_frame} should be no-fluid")
                    
                    # Add these frames to video_annotations with no-fluid label
                    for frame in range(start_frame, end_frame + 1):
                        if frame not in no_fluid_frames:
                            new_row = pd.Series({
                                'StudyInstanceUID': study_uid,
                                'SeriesInstanceUID': series_uid,
                                'frameNumber': frame,
                                'labelId': label_id_no_fluid,
                                'is_no_fluid': True,
                                'free_fluid_foreground': []  # Empty polygon list for no fluid
                            })
                            video_annotations = video_annotations.append(new_row, ignore_index=True)
                    
                    print(f"Added no-fluid annotations for all frames between {start_frame} and {end_frame}")
            
            # Now look for fluid annotations
            valid_frame = None
            frame_number = None
            free_fluid_polygons = None
            
            print(f"\nDEBUGGING ANNOTATIONS for {study_uid}/{series_uid}:")
            print(f"Total annotations for this video: {len(video_annotations)}")
            print(f"Column names: {video_annotations.columns.tolist()}")
            
            # Check if exam number is 91
            is_exam_91 = exam_number == 91
            if is_exam_91:
                print(f"\n*** SPECIAL HANDLING FOR EXAM 91 ***")
                print(f"Checking all {len(video_annotations)} annotations in detail...")
                
                # Get all unique label IDs
                if 'labelId' in video_annotations.columns:
                    all_label_ids = video_annotations['labelId'].unique().tolist()
                    print(f"All label IDs in this video: {all_label_ids}")
                    
                # Get the environment variables for labels
                fluid_label_id = os.environ.get('LABEL_ID_FLUID_OF', label_id_fluid)
                no_fluid_label_id = os.environ.get('LABEL_ID_NO_FLUID')
                print(f"Expected fluid label ID: {fluid_label_id}")
                print(f"Expected no-fluid label ID: {no_fluid_label_id}")
                
                # Try to filter to only fluid annotations if we have more than one label type
                if len(all_label_ids) > 1 and fluid_label_id in all_label_ids:
                    fluid_annotations = video_annotations[video_annotations['labelId'] == fluid_label_id]
                    print(f"Found {len(fluid_annotations)} annotations with fluid label ID")
                    if len(fluid_annotations) > 0:
                        video_annotations = fluid_annotations
                
            # Check if label_id is correct
            if 'labelId' in video_annotations.columns:
                label_ids = video_annotations['labelId'].unique().tolist()
                print(f"Label IDs in annotations: {label_ids}")
                print(f"Expected fluid label ID: {label_id_fluid}")
            
            # Debug: Print a sample of the annotations
            if len(video_annotations) > 0:
                sample_row = video_annotations.iloc[0]
                print(f"Sample annotation row keys: {sample_row.index.tolist()}")
                if 'data' in sample_row:
                    print(f"Sample data keys: {sample_row['data'].keys() if isinstance(sample_row['data'], dict) else 'Not a dict'}")
                if 'free_fluid_foreground' in sample_row:
                    polygons = sample_row['free_fluid_foreground']
                    print(f"Sample free_fluid_foreground type: {type(polygons)}")
                    print(f"Sample free_fluid_foreground value: {str(polygons)[:100]}...")
            
            # If this is exam 91, first look at all frames to find any with valid polygons
            if is_exam_91:
                print("\nSCAN FOR VALID POLYGONS IN EXAM 91:")
                valid_frames = []
                
                for idx, row in video_annotations.iterrows():
                    frame_num = row.get('frameNumber', 'unknown')
                    polygons = row.get('free_fluid_foreground')
                    label_id = row.get('labelId', 'unknown')
                    
                    print(f"Frame {frame_num}, Label: {label_id}")
                    print(f"  Polygons type: {type(polygons)}")
                    
                    if isinstance(polygons, list):
                        print(f"  Polygons length: {len(polygons)}")
                        
                        if len(polygons) > 0:
                            print(f"  ✓ Found valid polygon data!")
                            valid_frames.append((idx, frame_num, label_id, len(polygons)))
                
                if valid_frames:
                    print(f"\nFound {len(valid_frames)} frames with valid polygons:")
                    for v_idx, v_frame, v_label, v_count in valid_frames:
                        print(f"  Frame {v_frame}, Label {v_label}: {v_count} polygons")
                    
                    # Use the first valid frame
                    valid_frame, frame_number, _, _ = valid_frames[0]
                    row = video_annotations.loc[valid_frame]
                    free_fluid_polygons = row['free_fluid_foreground']
                    print(f"\nSelected frame {frame_number} for tracking with {len(free_fluid_polygons)} polygons")
            
            # Regular processing for non-exam-91
            if valid_frame is None:
                for idx, row in video_annotations.iterrows():
                    print(f"\nChecking frame {row.get('frameNumber', 'unknown')}:")
                    polygons = row.get('free_fluid_foreground')
                    print(f"  Polygons type: {type(polygons)}")
                    print(f"  Polygons value: {str(polygons)[:50]}...")
                    
                    if isinstance(polygons, list):
                        print(f"  Polygons length: {len(polygons)}")
                        if len(polygons) > 0:
                            valid_frame = idx
                            frame_number = int(row['frameNumber'])
                            free_fluid_polygons = polygons
                            print(f"  ✓ Found valid frame {frame_number} with {len(polygons)} polygons")
                            break
                        else:
                            print(f"  ✗ Empty polygon list")
                    else:
                        print(f"  ✗ Not a valid polygon list")
            
            if valid_frame is None:
                print(f"No valid polygons found in any frame for {study_uid}/{series_uid}, skipping")
                results['failed_videos'].append({
                    'video': video_path,
                    'study_uid': study_uid,
                    'series_uid': series_uid,
                    'exam_number': exam_number,
                    'error': 'No valid polygons found'
                })
                continue
            
            # Get the issue type
            issue_type = "unknown"
            if 'issue_type' in video_annotations.columns:
                issue_type = video_annotations.loc[valid_frame, 'issue_type']
                print(f"Issue type: {issue_type}")
            
            try:
                cap = cv2.VideoCapture(video_path)
                if not cap.isOpened():
                    print(f"Could not open video: {video_path}")
                    results['failed_videos'].append({
                        'video': video_path,
                        'study_uid': study_uid,
                        'series_uid': series_uid,
                        'exam_number': exam_number,
                        'error': 'Could not open video'
                    })
                    continue
                
                # Get video properties
                frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                
                print(f"Video dimensions: {frame_width}x{frame_height}")
                print(f"Total frames: {total_frames}")
                
                # Create initial mask from polygons
                from src.multi_frame_tracking.utils import polygons_to_mask
                initial_mask = polygons_to_mask(free_fluid_polygons, frame_height, frame_width)
                
                # Save the initial mask for verification
                mask_path = os.path.join(video_output_dir, f"initial_mask_frame_{frame_number}.png")
                cv2.imwrite(mask_path, initial_mask * 255)
                print(f"Saved initial mask to {mask_path} (pixel sum: {np.sum(initial_mask)})")
                
                # Process using the enhanced function
                try:
                    print(f"Fetching annotations for {study_uid}/{series_uid}")
                    from .consolidated_tracking import process_video_with_multi_frame_tracking_enhanced
                except ImportError:
                    print("Warning: Could not import from relative path, trying absolute import")
                    from src.consolidated_tracking import process_video_with_multi_frame_tracking_enhanced
                
                print("\n##############################################")
                print("# CALLING MULTI-FRAME TRACKING FUNCTION #")
                print("##############################################")
                print(f"Flow processor method: {flow_processor.method}")
                print(f"Study/Series: {study_uid}/{series_uid}")
                
                # Call the multi-frame processing function
                result = process_video_with_multi_frame_tracking_enhanced(
                    video_path=video_path,
                    annotations_df=video_annotations,
                    study_uid=study_uid,
                    series_uid=series_uid,
                    flow_processor=flow_processor,
                    output_dir=video_output_dir,
                    annotations_json=annotations_json,
                    mdai_client=mdai_client,
                    label_id_fluid=label_id_fluid,
                    label_id_machine=label_id_machine,
                    label_id_no_fluid=label_id_no_fluid,
                    project_id=project_id,
                    dataset_id=dataset_id,
                    upload_to_mdai=False,  # We'll upload manually with ground truth label
                    debug_mode=True
                )
                
                if not result or 'all_masks' not in result:
                    print("WARNING: No masks generated by the algorithm")
                    results['failed_videos'].append({
                        'video': video_path,
                        'study_uid': study_uid,
                        'series_uid': series_uid,
                        'exam_number': exam_number,
                        'error': 'No masks generated by algorithm'
                    })
                    continue
                
                algorithm_masks = result['all_masks']
                print(f"Generated {len(algorithm_masks)} masks")
                
                # Debug: Print mask types
                mask_types = {}
                for frame_idx, mask_info in algorithm_masks.items():
                    if isinstance(mask_info, dict) and 'type' in mask_info:
                        mask_type = mask_info['type']
                        mask_types[mask_type] = mask_types.get(mask_type, 0) + 1
                
                print("Mask types generated:")
                for mask_type, count in mask_types.items():
                    print(f"  {mask_type}: {count}")
                
                # Prepare masks for upload with the ground truth label
                annotations_to_upload = []
                
                for frame_idx, mask_info in algorithm_masks.items():
                    # Skip if this is a no-fluid frame
                    if isinstance(mask_info, dict) and mask_info.get('is_no_fluid', False):
                        print(f"Skipping frame {frame_idx} (marked as no-fluid)")
                        continue
                        
                    # Skip if this frame is in a no-fluid region
                    frame_row = video_annotations[video_annotations['frameNumber'] == frame_idx]
                    if not frame_row.empty and frame_row.iloc[0].get('is_no_fluid', False):
                        print(f"Skipping frame {frame_idx} (in no-fluid region)")
                        continue
                    
                    current_time = time.time()
                    elapsed_time = current_time - start_time
                    
                    # Check time limit
                    if elapsed_time > MAX_PROCESSING_TIME:
                        print(f"\nReached maximum processing time ({MAX_PROCESSING_TIME} seconds). Stopping processing.")
                        break
                        
                    # Check frame limit
                    if frames_processed >= MAX_FRAMES_TO_PROCESS:
                        print(f"\nReached maximum frame limit ({MAX_FRAMES_TO_PROCESS}). Stopping processing.")
                        break
                    
                    # Print progress every 30 seconds
                    if current_time - last_progress_time > 30:
                        print(f"\nProgress update:")
                        print(f"  Frames processed: {frames_processed}")
                        print(f"  Time elapsed: {elapsed_time:.1f} seconds")
                        print(f"  Current frame: {frame_idx}")
                        last_progress_time = current_time
                        
                    frames_processed += 1
                    
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
                    
                    # Create annotation based on type
                    try:
                        if is_clear:
                            print(f"\nPreparing no-fluid annotation for frame {frame_idx}")
                            annotation = {
                                'labelId': ground_truth_label_id,
                                'StudyInstanceUID': study_uid,
                                'SeriesInstanceUID': series_uid,
                                'frameNumber': int(frame_idx),
                                'groupId': label_id_machine,
                                'note': f'Ground truth: No fluid frame (predicted by algorithm) - Exam #{exam_number}'
                            }
                            annotations_to_upload.append(annotation)
                            print("✓ Added no-fluid annotation")
                        else:
                            binary_mask = (mask > 0.5).astype(np.uint8)
                            if np.sum(binary_mask) > 0:
                                print(f"\nPreparing fluid annotation for frame {frame_idx}")
                                import mdai
                                mask_data = mdai.common_utils.convert_mask_data(binary_mask)
                                if mask_data:
                                    annotation = {
                                        'labelId': ground_truth_label_id,
                                        'StudyInstanceUID': study_uid,
                                        'SeriesInstanceUID': series_uid,
                                        'frameNumber': int(frame_idx),
                                        'data': mask_data,
                                        'groupId': label_id_machine,
                                        'note': f'Ground truth: Fluid annotation (predicted by algorithm, type: {mask_type}) - Exam #{exam_number}'
                                    }
                                    annotations_to_upload.append(annotation)
                                    print("✓ Added fluid annotation")
                        
                    except Exception as e:
                        print(f"Error preparing annotation for frame {frame_idx}: {str(e)}")
                        continue
                
                # When preparing annotations for upload, adjust frame numbers back to MD.ai's 1-based system
                if annotations_to_upload:
                    print("\nAdjusting frame numbers back to MD.ai 1-based indexing for upload...")
                    for annotation in annotations_to_upload:
                        # Add 1 to convert back to MD.ai's 1-based frame numbering
                        system_frame = int(annotation['frameNumber'])
                        annotation['frameNumber'] = system_frame + 1
                        annotation['note'] = f"{annotation.get('note', '')} [System frame: {system_frame}]"
                    print(f"Adjusted {len(annotations_to_upload)} frame numbers for MD.ai upload")
                
                # Print upload debug info
                print(f"\n=== UPLOAD STATUS ===")
                print(f"Total annotations prepared: {len(annotations_to_upload)}")
                print(f"Should upload: {should_upload}")
                print(f"MD.ai client ready: {mdai_client is not None}")
                
                # Upload annotations if we have them and upload flag is set
                if annotations_to_upload and should_upload and mdai_client:
                    print(f"\nStarting upload of {len(annotations_to_upload)} annotations to MD.ai...")
                    print(f"First annotation sample:")
                    print(json.dumps(annotations_to_upload[0], indent=2))
                    
                    # Import here to avoid circular import
                    from src.multi_frame_tracking.utils import delete_existing_annotations
                    
                    # Delete existing ground truth annotations first
                    try:
                        deleted_count = delete_existing_annotations(
                            client=mdai_client,
                            study_uid=study_uid,
                            series_uid=series_uid,
                            label_id=ground_truth_label_id,
                            group_id=label_id_machine
                        )
                        print(f"✓ Deleted {deleted_count} existing ground truth annotations")
                    except Exception as e:
                        print(f"Error deleting existing annotations: {str(e)}")
                    
                    try:
                        print("\nUploading new annotations...")
                        failed_annotations = mdai_client.import_annotations(
                            annotations=annotations_to_upload,
                            project_id=project_id,
                            dataset_id=dataset_id
                        )
                        
                        if failed_annotations:
                            successful_count = len(annotations_to_upload) - len(failed_annotations)
                            print(f"✓ Uploaded {successful_count} out of {len(annotations_to_upload)} annotations")
                            print(f"✗ Failed to upload {len(failed_annotations)} annotations")
                            print("\nFirst failed annotation:")
                            print(json.dumps(failed_annotations[0], indent=2))
                            results['total_upload_success'] += successful_count
                            results['total_upload_failures'] += len(failed_annotations)
                        else:
                            print(f"✓ Successfully uploaded all {len(annotations_to_upload)} annotations")
                            results['total_upload_success'] += len(annotations_to_upload)
                        
                        results['total_annotations_created'] += len(annotations_to_upload)
                        
                    except Exception as e:
                        print(f"Error during upload: {str(e)}")
                        traceback.print_exc()
                        results['total_upload_failures'] += len(annotations_to_upload)
                        results['failed_videos'].append({
                            'video': video_path,
                            'study_uid': study_uid,
                            'series_uid': series_uid,
                            'exam_number': exam_number,
                            'error': f'Upload error: {str(e)}'
                        })
                elif not mdai_client:
                    print("\n✗ Cannot upload: MD.ai client not available")
                elif not should_upload:
                    print(f"\n✗ Skipping upload of {len(annotations_to_upload)} annotations (upload flag not set)")
                    print(f"Use --upload flag to enable MD.ai uploads")
                    results['total_annotations_created'] += len(annotations_to_upload)
                else:
                    print("\n✗ No annotations to upload")
                
                print("\n=== END UPLOAD STATUS ===")
                
                cap.release()
                
                # Record success
                results['processed_videos'].append({
                    'video': video_path,
                    'study_uid': study_uid,
                    'series_uid': series_uid,
                    'exam_number': exam_number,  
                    'annotations_created': len(annotations_to_upload),
                    'upload_successful': should_upload,
                    'issue_type': issue_type
                })
            
            except Exception as e:
                print(f"Error processing video: {str(e)}")
                traceback.print_exc()
                results['failed_videos'].append({
                    'video': video_path,
                    'study_uid': study_uid,
                    'series_uid': series_uid,
                    'exam_number': exam_number,  
                    'error': f'Processing error: {str(e)}'
                })
                continue
        
        except Exception as e:
            print(f"Error processing video {video_path}: {str(e)}")
            traceback.print_exc()
            results['failed_videos'].append({
                'video': video_path,
                'study_uid': study_uid,
                'series_uid': series_uid,
                'exam_number': exam_number,
                'error': str(e)
            })
        
        # Add completion message with timing
        total_time = time.time() - start_time
        print(f"\nProcessing complete for video {i+1}/{len(video_paths)}")
        print(f"Total frames processed: {frames_processed}")
        print(f"Total processing time: {total_time:.1f} seconds")
        print(f"Average time per frame: {total_time/frames_processed:.1f} seconds")
    
    # Generate summary
    results['summary'] = {
        'total_videos': len(video_paths),
        'successful_videos': len(results['processed_videos']),
        'failed_videos': len(results['failed_videos']),
        'total_annotations_created': results['total_annotations_created'],
        'total_upload_success': results['total_upload_success'],
        'total_upload_failures': results['total_upload_failures'],
        'exam_numbers_processed': sorted(list(set([v['exam_number'] for v in results['processed_videos'] + results['failed_videos']])))
    }
    
    # Save detailed results
    results_path = os.path.join(output_dir, 'ground_truth_creation_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nGround truth dataset creation complete!")
    print(f"Results saved to: {results_path}")
    print(f"Exam numbers processed: {', '.join(map(str, results['summary']['exam_numbers_processed']))}")
    
    return results
def create_feedback_loop_report(results, base_output_dir):
    """
    Create a comprehensive report comparing multiple feedback loop iterations
    
    Args:
        results: Dictionary with results from all iterations
        base_output_dir: Base output directory
    """
    report_path = os.path.join(base_output_dir, "feedback_loop_report.md")
    
    with open(report_path, 'w') as f:
        f.write("# Ground Truth Feedback Loop Report\n\n")
        
        # Create summary table
        f.write("## Summary of Iterations\n\n")
        f.write("| Iteration | Videos Processed | Annotations Created | Mean IoU | Mean Dice |\n")
        f.write("|-----------|-----------------|---------------------|----------|----------|\n")
        
        for iteration in results.get('iterations', []):
            iter_num = iteration.get('iteration_number', 0)
            gt_results = iteration.get('ground_truth_results', {})
            eval_results = iteration.get('evaluation_results', {})
            
            videos_processed = gt_results.get('summary', {}).get('total_videos', 0)
            annotations_created = gt_results.get('summary', {}).get('total_annotations_created', 0)
            
            mean_iou = eval_results.get('summary', {}).get('overall_mean_iou', 0)
            mean_dice = eval_results.get('summary', {}).get('overall_mean_dice', 0)
            
            f.write(f"| {iter_num} | {videos_processed} | {annotations_created} | {mean_iou:.4f} | {mean_dice:.4f} |\n")
        
        # Create detailed sections for each iteration
        for iteration in results.get('iterations', []):
            iter_num = iteration.get('iteration_number', 0)
            eval_results = iteration.get('evaluation_results', {})
            
            f.write(f"\n## Iteration {iter_num} Details\n\n")
            
            # Add overall metrics
            if 'summary' in eval_results:
                summary = eval_results['summary']
                f.write("### Overall Metrics\n\n")
                f.write(f"- Mean IoU: {summary.get('overall_mean_iou', 0):.4f}\n")
                f.write(f"- Mean Dice: {summary.get('overall_mean_dice', 0):.4f}\n")
                f.write(f"- Best Video: {summary.get('best_video', 'N/A')}\n")
                f.write(f"- Worst Video: {summary.get('worst_video', 'N/A')}\n\n")
            
            # List per-video metrics
            f.write("### Per-Video Metrics\n\n")
            f.write("| Video | Mean IoU | Mean Dice |\n")
            f.write("|-------|----------|----------|\n")
            
            for video_id, metrics in eval_results.items():
                if video_id != 'summary' and isinstance(metrics, dict) and 'metrics' in metrics:
                    video_metrics = metrics['metrics']
                    f.write(f"| {video_id} | {video_metrics.get('mean_iou', 0):.4f} | {video_metrics.get('mean_dice', 0):.4f} |\n")
    
    # Create visualization
    if results.get('iterations', []):
        # Plot IoU progression across iterations
        iterations = [i.get('iteration_number', 0) for i in results['iterations']]
        ious = [i.get('evaluation_results', {}).get('summary', {}).get('overall_mean_iou', 0) for i in results['iterations']]
        dices = [i.get('evaluation_results', {}).get('summary', {}).get('overall_mean_dice', 0) for i in results['iterations']]
        
        plt.figure(figsize=(10, 6))
        plt.plot(iterations, ious, 'b-', label='Mean IoU')
        plt.plot(iterations, dices, 'g-', label='Mean Dice')
        plt.xlabel('Iteration')
        plt.ylabel('Score')
        plt.title('Performance Metrics Across Feedback Loop Iterations')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        
        chart_path = os.path.join(base_output_dir, "feedback_loop_metrics.png")
        plt.savefig(chart_path)
    
    return report_path

def find_video_file(study_uid, series_uid):
    """
    Find a video file by looking in all data directories.
    
    Args:
        study_uid (str): Study instance UID
        series_uid (str): Series instance UID
        
    Returns:
        str: Path to the video file if found, None otherwise
    """
    # First check the standard path in the most recent download
    data_dirs = sorted([d for d in os.listdir('data') 
                     if os.path.isdir(os.path.join('data', d)) and 
                     (d.startswith('mdai_') or d == 'images')], 
                    key=lambda d: os.path.getmtime(os.path.join('data', d)), 
                    reverse=True)
    
    # Check all data directories
    for data_dir in data_dirs:
        video_path = os.path.join('data', data_dir, study_uid, f"{series_uid}.mp4")
        if os.path.exists(video_path):
            return video_path
    
    # If not found, do a more comprehensive search
    for root, dirs, files in os.walk('data'):
        for file in files:
            if file.endswith('.mp4') and file.startswith(f"{series_uid}"):
                return os.path.join(root, file)
    
    return None