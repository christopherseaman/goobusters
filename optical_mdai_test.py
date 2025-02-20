import os
from dotenv import load_dotenv
import mdai
import json

# Load environment variables
load_dotenv('.env')

# MD.ai configuration
ACCESS_TOKEN = os.getenv('MDAI_TOKEN')
DOMAIN = os.getenv('DOMAIN')
PROJECT_ID = os.getenv('PROJECT_ID')
DATASET_ID = os.getenv('DATASET_ID')

def create_test_annotation():
    """Create a test bounding box annotation using known-good associated IDs"""
    annotation = {
        'labelId': 'L_JykNe7',
        'StudyInstanceUID': '1.3.6.1.4.1.30071.8.345051894651.5791584882627065',
        'SeriesInstanceUID': '1.2.840.114340.3.8251050064157.2.20180508.222218.160',
        'SOPInstanceUID': '1.2.840.114340.3.8251050064157.3.20180508.222608.582.6',
        'frameNumber': 0,  # Note: using frameNumber 0 as shown in the metadata
        'data': {
            'x': 200,
            'y': 200,
            'width': 200,
            'height': 400
        },
        'groupId': 'L_1A4xv7'
    }
    return annotation

def test_upload(client):
    """Test uploading a single bounding box annotation"""
    print("\n=== Testing MD.ai Annotation Upload ===")
    
    try:
        # Create test annotation
        annotation = create_test_annotation()
        
        print("\nPrepared annotation:")
        print(json.dumps(annotation, indent=2))
        
        print("\nAttempting upload...")
        response = client.import_annotations(
            annotations=[annotation],
            project_id=PROJECT_ID,
            dataset_id=DATASET_ID
        )
        
        print(f"Response: {response}")
        
        if not response: 
            print("✅ Upload successful!")
            return True
        else:
            print(f"❌ Upload failed. Response: {response}")
            return False
        
    except Exception as e:
        print(f"❌ Error during upload: {str(e)}")
        return False

def main():
    print("Starting MD.ai upload test...")
    
    try:
        print(f"\nConnecting to MD.ai ({DOMAIN})...")
        client = mdai.Client(domain=DOMAIN, access_token=ACCESS_TOKEN)
        print("Successfully connected to MD.ai")
    except Exception as e:
        print(f"Error connecting to MD.ai: {str(e)}")
        return
    
    success = test_upload(client)
    
    if success:
        print("\n✅ Test completed successfully!")
    else:
        print("\n❌ Test failed")

if __name__ == "__main__":
    main()