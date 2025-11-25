import os
import json
import mdai
import traceback

def get_mdai_client():
    """
    Initialize and return an MD.ai client using environment variables
    
    Returns:
        mdai.Client: Initialized MD.ai client
    """
    domain = os.getenv('MDAI_DOMAIN', 'ucsf.md.ai')
    access_token = os.getenv('MDAI_TOKEN')
    
    if not access_token:
        raise ValueError("MDAI_TOKEN environment variable not set")
    
    return mdai.Client(domain=domain, access_token=access_token)

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
        
        # Use the "data" directory in the project root
        DATA_DIR = os.path.join(os.getcwd(), "data")
        if not os.path.exists(DATA_DIR):
            print(f"ERROR: Data directory '{DATA_DIR}' does not exist")
            return []
        
        print(f"Using data directory: {DATA_DIR}")
        
        # Find the latest annotations file
        annotation_files = [f for f in os.listdir(DATA_DIR) 
                           if f.startswith(f'mdai_ucsf_project_{project_id}_annotations_dataset_{dataset_id}') 
                           and f.endswith('.json')]
        
        if not annotation_files:
            print(f"No annotation files found in {DATA_DIR} matching pattern 'mdai_ucsf_project_{project_id}_annotations_dataset_{dataset_id}*.json'")
            return []
        
        # Sort by modification time (newest first)
        annotation_files.sort(key=lambda f: os.path.getmtime(os.path.join(DATA_DIR, f)), reverse=True)
        annotations_file = os.path.join(DATA_DIR, annotation_files[0])
        print(f"Using annotations file: {annotations_file}")
        
        # Load the JSON
        with open(annotations_file, 'r') as f:
            annotations_json = json.load(f)
        
        # Extract all annotations
        all_annotations = []
        for dataset in annotations_json.get('datasets', []):
            if 'annotations' in dataset:
                all_annotations.extend(dataset['annotations'])
        
        print(f"Total annotations in file: {len(all_annotations)}")
        
        # Identify available study/series combinations
        study_series_pairs = set()
        for ann in all_annotations:
            study = ann.get('studyInstanceUid') or ann.get('StudyInstanceUID')
            series = ann.get('seriesInstanceUid') or ann.get('SeriesInstanceUID')
            if study and series:
                study_series_pairs.add((study, series))
        
        print(f"Available study/series combinations: {len(study_series_pairs)}")
        
        # Print a few study/series combinations
        if study_series_pairs:
            print("Some available study/series combinations:")
            for i, (study, series) in enumerate(list(study_series_pairs)[:3]):
                print(f"  {i+1}. Study: {study}")
                print(f"     Series: {series}")
        
        # Check if our target study/series is available
        target_pair = (study_uid, series_uid)
        if target_pair in study_series_pairs:
            print(f"Target study/series found in annotations file")
        else:
            print(f"WARNING: Target study/series NOT found in annotations file")
            
            # Find closest match
            for study, series in study_series_pairs:
                if study == study_uid:
                    print(f"Found matching study UID with different series: {series}")
                elif series == series_uid:
                    print(f"Found matching series UID with different study: {study}")
        
        # Filter for the specific study/series
        filtered_annotations = [
            ann for ann in all_annotations
            if (ann.get('studyInstanceUid') == study_uid or ann.get('StudyInstanceUID') == study_uid) and 
               (ann.get('seriesInstanceUid') == series_uid or ann.get('SeriesInstanceUID') == series_uid)
        ]
        
        print(f"Annotations for this study/series: {len(filtered_annotations)}")
        
        # Further filter by label_id if provided
        if label_id:
            filtered_annotations = [
                ann for ann in filtered_annotations
                if ann.get('labelId') == label_id
            ]
        
        print(f"Final filtered annotations: {len(filtered_annotations)}")
        
        # Print frame numbers for the first few annotations
        if filtered_annotations:
            sample_frames = [ann.get('frameNumber') for ann in filtered_annotations[:5]]
            print(f"Sample frame numbers: {sample_frames}")
        
        return filtered_annotations
        
    except Exception as e:
        print(f"Error fetching annotations: {str(e)}")
        traceback.print_exc()
        return [] 