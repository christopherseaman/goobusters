import os
from dotenv import load_dotenv
import mdai
import numpy as np
import cv2
import json

# Force reload of .env file by setting load_dotenv(override=True)
load_dotenv(override=True)

# MD.ai configuration
ACCESS_TOKEN = os.getenv('MDAI_TOKEN')
DOMAIN = os.getenv('DOMAIN')
PROJECT_ID = os.getenv('PROJECT_ID')
DATASET_ID = os.getenv('DATASET_ID')

# Clean the label IDs - use the new group ID
LABEL_ID_FLUID_OF = os.getenv('LABEL_ID_FLUID_OF').split('#')[0].strip().strip('"')
LABEL_ID_MACHINE_GROUP = os.getenv('LABEL_ID_MACHINE_GROUP').split('#')[0].strip().strip('"')

LABEL_ID_MACHINE_GROUP = "G_RJY6Qn"  

def test_mask_upload(client, study_uid, series_uid, frame_num):
    """Test uploading a mask annotation using MD.ai's built-in utilities"""
    
    # Create a simple binary mask
    mask = np.zeros((1000, 1000), dtype=np.uint8)
    mask[300:700, 300:700] = 1  # Create a 400x400 square in the center
    
    # Convert mask to MD.ai format 
    mask_data = mdai.common_utils.convert_mask_data(mask)
    
    # Create the annotation
    annotation = {
        'labelId': LABEL_ID_FLUID_OF,
        'StudyInstanceUID': study_uid,
        'SeriesInstanceUID': series_uid,
        'frameNumber': frame_num,
        'data': mask_data,
        'groupId': LABEL_ID_MACHINE_GROUP
    }
    
    print(f"\nTrying to upload mask annotation to frame {frame_num}...")
    
    try:
        # Attempt to upload
        response = client.import_annotations(
            annotations=[annotation],
            project_id=PROJECT_ID,
            dataset_id=DATASET_ID
        )
        
        if not response:
            print(f"✅ Success! Mask uploaded to frame {frame_num}")
            return True
        else:
            print(f"❌ Failed to upload mask. Error: {response}")
            return False
            
    except Exception as e:
        print(f"❌ Error during upload: {str(e)}")
        return False

def main():
    print("Starting MD.ai mask upload test with updated group ID...")
    
    try:
        print(f"\nConnecting to MD.ai ({DOMAIN})...")
        client = mdai.Client(domain=DOMAIN, access_token=ACCESS_TOKEN)
        print("Successfully connected to MD.ai")
        
      
        study_uid = '1.2.826.0.1.3680043.8.498.63008608727004617468991609053114798931'  # Study #116
       
        series_uid = '1.2.826.0.1.3680043.8.498.45476566330031023753664372781992762733'
        
        # Print the IDs we're using
        print(f"Study UID: {study_uid}")
        print(f"Series UID: {series_uid}")
        print(f"Label ID: {LABEL_ID_FLUID_OF}")
        print(f"Group ID: {LABEL_ID_MACHINE_GROUP}")
        
        success = test_mask_upload(
            client=client,
            study_uid=study_uid,
            series_uid=series_uid,
            frame_num=0  # We know frame 0 works
        )
        
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()