import os
import time
from dotenv import load_dotenv
import mdai
import numpy as np
import cv2
import json
import pandas as pd
import matplotlib.pyplot as plt

# Forcing reload of .env file
load_dotenv(override=True)

# Setting up MD.ai configuration
ACCESS_TOKEN = os.getenv('MDAI_TOKEN')
DOMAIN = os.getenv('DOMAIN')
PROJECT_ID = os.getenv('PROJECT_ID')
DATASET_ID = os.getenv('DATASET_ID')
DATA_DIR = os.getenv('DATA_DIR')

# Label IDs
LABEL_ID_FLUID_OF = os.getenv('LABEL_ID_FLUID_OF')
LABEL_ID_MACHINE_GROUP = os.getenv('LABEL_ID_MACHINE_GROUP')

# Specific annotation file
SPECIFIC_ANNOTATIONS_FILE = os.path.join(DATA_DIR, "mdai_ucsf_project_x9N2LJBZ_annotations_dataset_D_V688LQ_labelgroup_G_7n3P09_2025-03-10-013102.json")

def verify_annotations_content(annotations_path):
    """Print key information about the annotations file to verify its content"""
    with open(annotations_path, 'r') as f:
        annotations_json = json.load(f)
    
    # Check file structure
    datasets = annotations_json.get('datasets', [])
    studies_count = 0
    annotations_count = 0
    
    for dataset in datasets:
        studies = dataset.get('studies', [])
        studies_count += len(studies)
        annotations_count += len(dataset.get('annotations', []))
    
    print(f"\nAnnotations file structure:")
    print(f"File path: {annotations_path}")
    print(f"Datasets: {len(datasets)}")
    print(f"Studies: {studies_count}")
    print(f"Annotations: {annotations_count}")
    
    # Print the first few studies and their exam numbers
    print("\nFirst 5 studies and their exam numbers:")
    count = 0
    for dataset in datasets:
        for study in dataset.get('studies', []):
            if count < 5:
                print(f"Exam #{study.get('number')}: {study.get('StudyInstanceUID')}")
                count += 1
            else:
                break
    
    return annotations_json

def find_exam_number(study_uid, annotations_json):
    """Find the exam number for a given StudyInstanceUID"""
    
    # First, check in the annotations data
    datasets = annotations_json.get('datasets', [])
    for dataset in datasets:
        for study in dataset.get('studies', []):
            if study.get('StudyInstanceUID') == study_uid:
                exam_number = study.get('number')
                if exam_number:
                    return exam_number
    
    # If not found, try other approaches
    try:
        for dataset in datasets:
            studies = dataset.get('studies', [])
            for study in studies:
                if study.get('StudyInstanceUID') == study_uid:
                    return study.get('number', 'Unknown')
    except Exception:
        pass
    
    return "Unknown"

def list_all_studies(annotations_json):
    """List all studies and their exam numbers"""
    studies = []
    for dataset in annotations_json.get('datasets', []):
        for study in dataset.get('studies', []):
            studies.append({
                'StudyUID': study.get('StudyInstanceUID'),
                'ExamNumber': study.get('number')
            })
    return studies

def save_mask_visualization(mask, filename):
    """
    Save a visualization of the mask for debugging purposes
    """
    plt.figure(figsize=(8, 8))
    plt.imshow(mask, cmap='gray')
    plt.title(f'Mask - Sum: {np.sum(mask)}, Max: {np.max(mask)}, Min: {np.min(mask)}')
    plt.colorbar()
    plt.savefig(filename)
    plt.close()
    print(f"Saved mask visualization to {filename}")

def create_label_dictionary(annotations_json):
    """
    Create a dictionary mapping label names to label IDs based on the annotations JSON file
    """
    # Check if the JSON file contains labels data
    if 'labels' in annotations_json:
        labels = annotations_json['labels']
        # Create a dictionary mapping label names to IDs
        label_dict = {}
        for label in labels:
            if 'name' in label and 'id' in label:
                label_dict[label['name']] = label['id']
        
        print(f"Created label dictionary with {len(label_dict)} labels:")
        for name, id in label_dict.items():
            print(f"  {name}: {id}")
        
        return label_dict
    else:
        print("Warning: No labels found in the annotations JSON.")
        return {}

def test_mask_upload_with_specific_file():
    """Test uploading a mask annotation using a specific annotations file"""
    print("Starting MD.ai mask upload test with specific annotations file...")
    
    try:
        # Connect to MD.ai
        print(f"\nConnecting to MD.ai ({DOMAIN})...")
        mdai_client = mdai.Client(domain=DOMAIN, access_token=ACCESS_TOKEN)
        project = mdai_client.project(project_id=PROJECT_ID)
        print("Successfully connected to MD.ai")
        
        # Check if the specific file exists
        if not os.path.exists(SPECIFIC_ANNOTATIONS_FILE):
            print(f"Error: Specified annotations file does not exist: {SPECIFIC_ANNOTATIONS_FILE}")
            return {
                'success': False,
                'study_uid': None,
                'series_uid': None,
                'frame_number': None,
                'annotations_json': None
            }
        
        print(f"Using specific annotations file: {SPECIFIC_ANNOTATIONS_FILE}")
        
        # Verify file content
        annotations_json = verify_annotations_content(SPECIFIC_ANNOTATIONS_FILE)
        
        # Create label dictionary
        label_dict = create_label_dictionary(annotations_json)
        
        # Use the provided label IDs if they exist, otherwise try to find them
        if LABEL_ID_FLUID_OF:
            label_id = LABEL_ID_FLUID_OF
            print(f"Using provided Fluid-OF label ID: {label_id}")
        else:
            # Try to find the label ID from the dictionary
            label_id = label_dict.get('Fluid-OF')
            if not label_id:
                print("Error: Could not find Fluid-OF label ID.")
                return {
                    'success': False,
                    'study_uid': None,
                    'series_uid': None,
                    'frame_number': None,
                    'annotations_json': None
                }
            print(f"Found Fluid-OF label ID from dictionary: {label_id}")
        
        # Process annotations
        all_annotations = []
        for dataset in annotations_json.get('datasets', []):
            if 'annotations' in dataset:
                all_annotations.extend(dataset['annotations'])
        
        # Convert to df
        annotations_df = pd.json_normalize(all_annotations, sep='_')
        
        # Find annotations with frameNumber
        valid_annotations = annotations_df[
            (annotations_df['frameNumber'].notna())
        ].copy()
        
        if len(valid_annotations) == 0:
            print("No valid annotations found with frameNumber")
            return {
                'success': False,
                'study_uid': None,
                'series_uid': None,
                'frame_number': None,
                'annotations_json': annotations_json
            }
            
        # Use the first valid annotation as reference
        reference = valid_annotations.iloc[0]
        study_uid = reference['StudyInstanceUID']
        series_uid = reference['SeriesInstanceUID']
        frame_number = int(reference['frameNumber'])  # Ensure frame number is an integer
        
        print(f"\nFound reference annotation:")
        print(f"  StudyUID: {study_uid}")
        print(f"  SeriesUID: {series_uid}")
        print(f"  Frame: {frame_number}")
        
        # Create a more visible mask - using a C-shaped mask that's more distinctive
        width, height = 512, 512  
        mask = np.zeros((height, width), dtype=np.uint8)
        
        # Create a C-shaped mask that will be more obvious in the UI
        cv2.ellipse(mask, (256, 256), (150, 100), 0, 45, 315, 1, thickness=-1)
        
        # Save a visualization of the mask for debugging
        debug_dir = os.path.join(os.getcwd(), "debug_output")
        os.makedirs(debug_dir, exist_ok=True)
        mask_viz_path = os.path.join(debug_dir, "test_mask.png")
        save_mask_visualization(mask, mask_viz_path)
        
        # Print mask statistics
        print(f"\nMask statistics:")
        print(f"  Shape: {mask.shape}")
        print(f"  Sum (number of 1's): {np.sum(mask)}")
        print(f"  Min: {np.min(mask)}")
        print(f"  Max: {np.max(mask)}")
        print(f"  Mask visualization saved to: {mask_viz_path}")
        
        # IMPORTANT: Use MD.ai's built-in function to convert mask to proper format
        mask_data = mdai.common_utils.convert_mask_data(mask)
        
        print(f"\nMask data from convert_mask_data:")
        print(f"  Type: {type(mask_data)}")
        
        # Create annotation dictionary - directly use the mask_data without modifying the format
        annotation = {
            'labelId': label_id,
            'StudyInstanceUID': study_uid,
            'SeriesInstanceUID': series_uid,
            'frameNumber': int(frame_number),
            'data': mask_data,
            'groupId': LABEL_ID_MACHINE_GROUP
        }
        
        # Print the exact annotation being sent
        print(f"\nSending annotation:")
        for key, value in annotation.items():
            if key == 'data':
                print(f"  {key}: [mask data object]")
            else:
                print(f"  {key}: {value}")
        
        # Try to upload using import_annotations
        print("\nAttempting upload with import_annotations method...")
        try:
            # Store annotations in a list
            annotations_list = [annotation]
            
            # Upload annotations
            failed_annotations = mdai_client.import_annotations(
                annotations=annotations_list,
                project_id=PROJECT_ID,
                dataset_id=DATASET_ID
            )
            
            # Check if there were any failures
            if failed_annotations:
                print(f"❌ Some annotations failed to upload:")
                for failed in failed_annotations:
                    print(f"  Index {failed.get('index', 'unknown')}: {failed.get('reason', 'unknown reason')}")
                success = False
            else:
                print(f"✅ Success! All annotations uploaded successfully")
                success = True
                
        except Exception as e:
            print(f"❌ Error with import_annotations: {str(e)}")
            success = False
        
        # Try with alternate method - direct approach
        if not success:
            print("\nTrying alternative approach with direct mask data...")
            
            # Create a binary mask
            binary_mask = (mask > 0).astype(np.uint8)
            
            # Create annotation dictionary with direct mask format
            alternative_annotation = {
                'labelId': label_id,
                'StudyInstanceUID': study_uid,
                'SeriesInstanceUID': series_uid,
                'frameNumber': int(frame_number),
                'data': binary_mask.tolist(),  # Convert numpy array to list
                'groupId': LABEL_ID_MACHINE_GROUP
            }
            
            try:
                # Upload annotations
                alternative_failed = mdai_client.import_annotations(
                    annotations=[alternative_annotation],
                    project_id=PROJECT_ID,
                    dataset_id=DATASET_ID
                )
                
                # Check if there were any failures
                if alternative_failed:
                    print(f"❌ Alternative approach also failed:")
                    for failed in alternative_failed:
                        print(f"  Index {failed.get('index', 'unknown')}: {failed.get('reason', 'unknown reason')}")
                    alt_success = False
                else:
                    print(f"✅ Success with alternative approach!")
                    alt_success = True
                    
            except Exception as e:
                print(f"❌ Error with alternative approach: {str(e)}")
                alt_success = False
                
            success = success or alt_success
        
        # Return results
        return {
            'success': success,
            'study_uid': study_uid,
            'series_uid': series_uid,
            'frame_number': frame_number,
            'annotations_json': annotations_json,
            'mask_viz_path': mask_viz_path
        }
            
    except Exception as e:
        print(f"❌ Overall error: {str(e)}")
        import traceback
        traceback.print_exc()
        return {
            'success': False,
            'study_uid': None,
            'series_uid': None,
            'frame_number': None,
            'annotations_json': None
        }

if __name__ == "__main__":
    result = test_mask_upload_with_specific_file()
    
    if result['success']:
        exam_number = find_exam_number(result['study_uid'], result['annotations_json'])
        print(f"\n✅ The mask was uploaded to Exam #{exam_number}")
        print(f"Study UID: {result['study_uid']}")
        print(f"Series UID: {result['series_uid']}")
        print(f"Frame Number: {result['frame_number']}")
        
        print(f"\nIMPORTANT: Check the MD.ai frontend to verify this mask appears on Exam #{exam_number}")
        print(f"Look for a C-shaped annotation in the center of the image at frame {result['frame_number']}")
        print(f"The mask visualization is saved to: {result.get('mask_viz_path')}")
        
        # List some studies for reference
        print("\nListing first 10 studies and exam numbers:")
        all_studies = list_all_studies(result['annotations_json'])
        for i, study in enumerate(all_studies[:10]):
            print(f"Exam #{study['ExamNumber']}: {study['StudyUID']}")
    else:
        print("\n❌ Test failed. No mask was uploaded.")