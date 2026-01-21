"""
Minimal common_utils for iOS - json_to_dataframe without pandas.
"""

import json


def json_to_dataframe(json_path: str, datasets=None):
    """
    Load annotations JSON and return as dict with annotations and studies lists.
    This is a simplified version that doesn't require pandas.
    
    The JSON structure from MD.ai has annotations nested under datasets:
    {
        "datasets": [
            {
                "id": "...",
                "name": "...",
                "studies": [...],
                "annotations": [...]
            }
        ]
    }
    """
    if datasets is None:
        datasets = []
    
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    all_annotations = []
    all_studies = []
    
    # Extract from each dataset (matching real mdai behavior)
    for d in data.get("datasets", []):
        dataset_id = d.get("id", "")
        dataset_name = d.get("name", "")
        
        # Filter by dataset IDs if specified
        if datasets and dataset_id not in datasets:
            continue
        
        # Add studies with dataset info
        for study in d.get("studies", []):
            study_copy = dict(study)
            study_copy["dataset"] = dataset_name
            study_copy["datasetId"] = dataset_id
            all_studies.append(study_copy)
        
        # Add annotations with dataset info
        for ann in d.get("annotations", []):
            ann_copy = dict(ann)
            ann_copy["dataset"] = dataset_name
            all_annotations.append(ann_copy)
    
    return {
        "annotations": all_annotations,
        "studies": all_studies,
    }
