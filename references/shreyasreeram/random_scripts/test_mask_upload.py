#!/usr/bin/env python3
import os
import numpy as np
import mdai
from dotenv import load_dotenv

# Load environment variables
load_dotenv('.env')

# Get credentials from environment
MDAI_TOKEN = os.getenv('MDAI_TOKEN')
DOMAIN = os.getenv('DOMAIN', 'annotate.md.ai')
PROJECT_ID = os.getenv('PROJECT_ID')
DATASET_ID = os.getenv('DATASET_ID')
LABEL_ID_FLUID_OF = os.getenv('LABEL_ID_FLUID_OF', 'L_A5BNpq')
LABEL_ID_MACHINE_GROUP = os.getenv('LABEL_ID_MACHINE_GROUP', 'G_7n3P09')

# Test parameters - replace with your actual values
VIDEO_PATH = 'data/mdai_ucsf_project_x9N2LJBZ_images_dataset_D_V688LQ_2025-05-19-171632/1.2.826.0.1.3680043.8.498.18050612380255098469086741540114763661/1.2.826.0.1.3680043.8.498.72553010565308306328905938562604820392.mp4'
STUDY_UID = '1.2.826.0.1.3680043.8.498.18050612380255098469086741540114763661'
SERIES_UID = '1.2.826.0.1.3680043.8.498.72553010565308306328905938562604820392'
TEST_FRAME = 10  # Frame number to test with

print("MD.ai Mask Upload Test")
print("======================")
print(f"MDAI_TOKEN: {MDAI_TOKEN[:5]}...")
print(f"PROJECT_ID: {PROJECT_ID}")
print(f"DATASET_ID: {DATASET_ID}")
print(f"LABEL_ID_FLUID_OF: {LABEL_ID_FLUID_OF}")
print(f"STUDY_UID: {STUDY_UID[:15]}...")
print(f"SERIES_UID: {SERIES_UID[:15]}...")

# Initialize MD.ai client
print("\nInitializing MD.ai client...")
client = mdai.Client(domain=DOMAIN, access_token=MDAI_TOKEN)

# Create a simple mask (100x100 with a circle in the middle)
def create_test_mask(height=100, width=100, radius=30):
    """Create a simple circular mask for testing"""
    y, x = np.ogrid[:height, :width]
    center_y, center_x = height // 2, width // 2
    
    # Create circle
    dist_from_center = np.sqrt((x - center_x)**2 + (y - center_y)**2)
    mask = (dist_from_center <= radius).astype(np.uint8)
    
    return mask

# Create test mask
mask = create_test_mask(height=480, width=640)
print(f"\nCreated test mask: shape={mask.shape}, sum={np.sum(mask)}")

# Convert mask to MD.ai format
mask_data = mdai.common_utils.convert_mask_data(mask)
print(f"Converted mask to MD.ai format: type={type(mask_data)}")

# Create annotation
annotation = {
    'labelId': LABEL_ID_FLUID_OF,
    'StudyInstanceUID': STUDY_UID,
    'SeriesInstanceUID': SERIES_UID,
    'frameNumber': TEST_FRAME,
    'data': mask_data,
    'groupId': LABEL_ID_MACHINE_GROUP
}

# Print annotation details
print("\nAnnotation structure:")
print(f"labelId: {annotation['labelId']}")
print(f"StudyInstanceUID: {annotation['StudyInstanceUID'][:15]}...")
print(f"SeriesInstanceUID: {annotation['SeriesInstanceUID'][:15]}...")
print(f"frameNumber: {annotation['frameNumber']}")
print(f"data type: {type(annotation['data'])}")
print(f"groupId: {annotation['groupId']}")

# Upload to MD.ai
print("\nUploading annotation to MD.ai...")
try:
    response = client.import_annotations(
        annotations=[annotation],
        project_id=PROJECT_ID,
        dataset_id=DATASET_ID
    )
    
    if response and len(response) > 0:
        print(f"Upload failed: {response}")
    else:
        print("Upload successful!")
except Exception as e:
    print(f"Error during upload: {e}")

print("\nTest completed") 