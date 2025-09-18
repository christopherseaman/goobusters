#!/usr/bin/env python
import os
from dotenv import load_dotenv
import mdai
import sys
import numpy as np

# Load environment variables
load_dotenv('.env')

# Get credentials
MDAI_TOKEN = os.getenv('MDAI_TOKEN')
DOMAIN = os.getenv('DOMAIN')
PROJECT_ID = os.getenv('PROJECT_ID')
DATASET_ID = os.getenv('DATASET_ID')
LABEL_ID_FLUID_OF = os.getenv('LABEL_ID_FLUID_OF')
LABEL_ID_MACHINE_GROUP = os.getenv('LABEL_ID_MACHINE_GROUP')

print("MD.ai Client Testing")
print("===========================")
print(f"MDAI_TOKEN: {MDAI_TOKEN}")
print(f"DOMAIN: {DOMAIN}")
print(f"PROJECT_ID: {PROJECT_ID}")
print(f"DATASET_ID: {DATASET_ID}")
print(f"LABEL_ID_FLUID_OF: {LABEL_ID_FLUID_OF}")
print(f"LABEL_ID_MACHINE_GROUP: {LABEL_ID_MACHINE_GROUP}")
print("===========================")

# Verify credentials
if not all([MDAI_TOKEN, DOMAIN, PROJECT_ID, DATASET_ID, LABEL_ID_FLUID_OF, LABEL_ID_MACHINE_GROUP]):
    print("ERROR: Missing one or more required credentials")
    sys.exit(1)

# Try to initialize client
try:
    print("Attempting to initialize MD.ai client...")
    client = mdai.Client(domain=DOMAIN, access_token=MDAI_TOKEN)
    print("MD.ai client initialized successfully")
    
    # Try to get project info
    print(f"Attempting to retrieve project info for {PROJECT_ID}...")
    project = client.project(PROJECT_ID)
    print("Project info retrieved successfully")
    
    # Create a test mask
    print("Creating test mask...")
    # Create a 100x100 mask with a square in the middle
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[25:75, 25:75] = 1
    
    # Convert mask to MD.ai format
    mask_data = mdai.common_utils.convert_mask_data(mask)
    print(f"Mask data type: {type(mask_data)}")
    
    # Test import annotations endpoint
    print(f"Testing import_annotations endpoint...")
    test_annotation = {
        'labelId': LABEL_ID_FLUID_OF,
        'StudyInstanceUID': '1.2.826.0.1.3680043.8.498.18050612380255098469086741540114763661',
        'SeriesInstanceUID': '1.2.826.0.1.3680043.8.498.72553010565308306328905938562604820392',
        'frameNumber': 1,
        'data': mask_data,  # Use the properly converted mask data
        'groupId': LABEL_ID_MACHINE_GROUP
    }
    
    print(f"Submitting test annotation with proper mask data format")
    result = client.import_annotations(
        annotations=[test_annotation],
        project_id=PROJECT_ID,
        dataset_id=DATASET_ID
    )
    
    print(f"Import annotations result: {result}")
    if not result:
        print("Test annotation uploaded successfully")
    else:
        print(f"Failed to upload test annotation: {result}")
    
except Exception as e:
    print(f"ERROR: {str(e)}")
    import traceback
    traceback.print_exc() 