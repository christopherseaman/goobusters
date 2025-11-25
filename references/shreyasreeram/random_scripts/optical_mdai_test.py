import os
import time
from dotenv import load_dotenv
import mdai
import numpy as np
import cv2
import json
import pandas as pd

# Forcing reload of .env file
load_dotenv(override=True)

#  setting up MD.ai configuration
ACCESS_TOKEN = os.getenv('MDAI_TOKEN')
DOMAIN = os.getenv('DOMAIN')
PROJECT_ID = os.getenv('PROJECT_ID')
DATASET_ID = os.getenv('DATASET_ID')
DATA_DIR = os.getenv('DATA_DIR')

# Label IDs
LABEL_ID_FLUID_OF = os.getenv('LABEL_ID_FLUID_OF')
LABEL_ID_MACHINE_GROUP = os.getenv('LABEL_ID_MACHINE_GROUP')

#specific annotation file (from manual export out of md.ai) --> but this one does not have SOPID's 
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
        frame_number = int(reference['frameNumber'])
        
        print(f"\nFound reference annotation:")
        print(f"  StudyUID: {study_uid}")
        print(f"  SeriesUID: {series_uid}")
        print(f"  Frame: {frame_number}")
        
        # Create a simple binary mask
        width, height = 512, 512  
        mask = np.zeros((height, width), dtype=np.uint8)
        mask[150:350, 150:350] = 1  # Simple square mask
        
        # Converting mask to MD.ai format
        mask_data = mdai.common_utils.convert_mask_data(mask)
        
        # A nnotation format
        annotation1 = {
            'labelId': LABEL_ID_FLUID_OF,
            'StudyInstanceUID': study_uid,
            'SeriesInstanceUID': series_uid,
            'frameNumber': frame_number,
            'data': mask_data,
            #'groupId': LABEL_ID_MACHINE_GROUP
        }
        
        # Trying to upload with Study/Series/Frame
        print("\nAttempting upload with Study/Series/Frame...")
        try:
            response1 = mdai_client.import_annotations(
                annotations=[annotation1],
                project_id=PROJECT_ID,
                dataset_id=DATASET_ID
            )
            
            if not response1:
                print(f"✅ Success! Mask uploaded with Study/Series/Frame")
                success1 = True
            else:
                print(f"❌ Failed to upload with Study/Series/Frame. Error: {response1}")
                success1 = False
                
        except Exception as e:
            print(f"❌ Error with Study/Series/Frame: {str(e)}")
            success1 = False
        
        # If SOPInstanceUID exists, try that too
        success2 = False
        if 'SOPInstanceUID' in reference:
            sop_uid = reference['SOPInstanceUID']
            print(f"\nFound SOPInstanceUID: {sop_uid}")
            
            annotation2 = {
                'labelId': LABEL_ID_FLUID_OF,
                'SOPInstanceUID': sop_uid,
                'data': mask_data,
                'groupId': LABEL_ID_MACHINE_GROUP
            }
            
            print("\nAttempting upload with SOPInstanceUID...")
            try:
                response2 = mdai_client.import_annotations(
                    annotations=[annotation2],
                    project_id=PROJECT_ID,
                    dataset_id=DATASET_ID
                )
                
                if not response2:
                    print(f"✅ Success! Mask uploaded with SOPInstanceUID")
                    success2 = True
                else:
                    print(f"❌ Failed to upload with SOPInstanceUID. Error: {response2}")
                    success2 = False
                    
            except Exception as e:
                print(f"❌ Error with SOPInstanceUID: {str(e)}")
                success2 = False
        
        return {
            'success': success1 or success2,
            'study_uid': study_uid,
            'series_uid': series_uid,
            'frame_number': frame_number,
            'annotations_json': annotations_json
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
        print(f"\nCheck the MD.ai frontend to verify this mask appears on Exam #{exam_number}")
        
        # List some studies for reference
        print("\nListing first 10 studies and exam numbers:")
        all_studies = list_all_studies(result['annotations_json'])
        for i, study in enumerate(all_studies[:10]):
            print(f"Exam #{study['ExamNumber']}: {study['StudyUID']}")
    else:
        print("\n❌ Test failed. No mask was uploaded.")