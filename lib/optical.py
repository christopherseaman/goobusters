"""
Utility functions for tracking output metadata.
"""

import os
import json
import yaml
from datetime import datetime


def create_identity_file(
    video_output_dir: str,
    study_uid: str,
    series_uid: str,
    video_annotations,
    studies_data,
) -> None:
    """Create identity YAML file with video metadata."""
    first_annotation = video_annotations.iloc[0]

    exam_number = "Unknown"
    if not studies_data.empty and "StudyInstanceUID" in studies_data.columns:
        study_match = studies_data[
            studies_data["StudyInstanceUID"] == study_uid
        ]
        if not study_match.empty and "number" in study_match.columns:
            exam_number = int(study_match.iloc[0]["number"])

    if "labelName" not in video_annotations.columns:
        raise ValueError(
            f"labelName column missing. Columns: {list(video_annotations.columns)}"
        )

    unique_labels = video_annotations[
        ["labelId", "labelName"]
    ].drop_duplicates()
    labels_info = unique_labels.to_dict("records")

    identity_data = {
        "study_instance_uid": study_uid,
        "series_instance_uid": series_uid,
        "exam_number": exam_number,
        "dataset_name": first_annotation.get("dataset", "Unknown"),
        "dataset_id": first_annotation.get("datasetId", "Unknown"),
        "created_at": datetime.now().isoformat(),
        "annotation_count": len(video_annotations),
        "labels": labels_info,
    }

    identity_file_path = os.path.join(video_output_dir, "identity.yaml")
    try:
        with open(identity_file_path, "w") as f:
            yaml.dump(
                identity_data, f, default_flow_style=False, sort_keys=False
            )
    except Exception:
        pass


def copy_annotations_to_output(
    video_output_dir: str, video_annotations, annotations_data
) -> None:
    """Save input annotations JSON (excluding track_id annotations)."""
    try:
        input_annotations = [
            ann
            for ann in annotations_data.get("annotations", [])
            if "track_id" not in ann
        ]

        input_data = {
            "annotations": input_annotations,
            "studies": annotations_data.get("studies", []),
            "labels": annotations_data.get("labels", []),
        }

        annotations_file = os.path.join(
            video_output_dir, "input_annotations.json"
        )
        with open(annotations_file, "w") as f:
            json.dump(input_data, f, indent=2, default=str)
    except Exception:
        pass
