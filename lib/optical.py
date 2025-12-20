#!/usr/bin/env python3

"""
Optical Flow Tracking Utilities

Provides utility functions for managing tracking output metadata.
"""

import os
import json
import yaml
from datetime import datetime

# Fix for pydicom deprecation warning from mdai 0.16.0
import sys
import importlib.util


def patch_pydicom_for_forward_compatibility():
    """
    Patches pydicom to handle the deprecated API usage in a forward-compatible way.
    This eliminates the deprecation warning while maintaining functionality.
    """
    try:
        # Import pydicom to check current state
        import pydicom

        # The warning comes from pydicom/pixel_data_handlers/util.py when it imports
        # pack_bits from the deprecated location. We need to intercept this.
        if hasattr(pydicom, "pixel_data_handlers") and hasattr(
            pydicom, "pixels"
        ):
            # Ensure the util module has pack_bits from the new location
            if hasattr(pydicom.pixel_data_handlers, "util"):
                util_module = pydicom.pixel_data_handlers.util

                # If pack_bits exists in the new location, make sure it's accessible
                # from the old location without triggering the warning
                if hasattr(pydicom.pixels, "pack_bits"):
                    # Replace the import in the util module to use the new API
                    util_module.pack_bits = pydicom.pixels.pack_bits

                    # Patch the module's __getattr__ to redirect deprecated calls
                    original_getattr = getattr(
                        pydicom.pixel_data_handlers, "__getattr__", None
                    )

                    def patched_getattr(name):
                        if name == "pack_bits":
                            return pydicom.pixels.pack_bits
                        elif original_getattr:
                            return original_getattr(name)
                        else:
                            raise AttributeError(
                                f"module has no attribute '{name}'"
                            )

                    # Apply the patch if we have a modern pydicom
                    if hasattr(pydicom.pixel_data_handlers, "__getattr__"):
                        pydicom.pixel_data_handlers.__getattr__ = (
                            patched_getattr
                        )

        return True
    except (ImportError, AttributeError):
        return False


# Apply the compatibility patch
patch_pydicom_for_forward_compatibility()

# Import mdai with the patched pydicom
import mdai


def create_identity_file(
    video_output_dir: str,
    study_uid: str,
    series_uid: str,
    video_annotations,
    studies_data,
) -> None:
    """
    Create an identity YAML file for the video output folder containing metadata.

    Args:
        video_output_dir: Path to the video output directory
        study_uid: Study Instance UID
        series_uid: Series Instance UID
        video_annotations: DataFrame containing annotation data for this video
        studies_data: DataFrame containing studies data with exam numbers
    """
    # Extract metadata from the first annotation (they should all have the same metadata)
    first_annotation = video_annotations.iloc[0]

    # Get exam number from studies data
    # Handle case where studies_data is empty or doesn't have expected columns
    exam_number = "Unknown"
    if not studies_data.empty and "StudyInstanceUID" in studies_data.columns:
        study_match = studies_data[
            studies_data["StudyInstanceUID"] == study_uid
        ]
        if not study_match.empty and "number" in study_match.columns:
            exam_number = int(study_match.iloc[0]["number"])

    # Get all unique labels present in this video's annotations
    # Ensure labelName column exists (required for identity file)
    if "labelName" not in video_annotations.columns:
        raise ValueError(
            f"labelName column missing in video_annotations. Columns: {list(video_annotations.columns)}"
        )

    unique_labels = video_annotations[
        ["labelId", "labelName"]
    ].drop_duplicates()
    labels_info = unique_labels.to_dict("records")

    # Create identity data
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

    # Write YAML file
    identity_file_path = os.path.join(video_output_dir, "identity.yaml")
    try:
        with open(identity_file_path, "w") as f:
            yaml.dump(
                identity_data, f, default_flow_style=False, sort_keys=False
            )
    except Exception as e:
        pass  # Silent failure for background task


def copy_annotations_to_output(
    video_output_dir: str, video_annotations, annotations_data
) -> None:
    """
    Copy input annotations JSON data to the video output directory.
    This contains all original annotations for the video EXCEPT any with track_id.

    Args:
        video_output_dir: Path to the video output directory
        video_annotations: DataFrame containing annotation data for this video
        annotations_data: Full annotations data structure
    """
    try:
        # Filter out any annotations that have track_id (these are generated, not input)
        input_annotations = []
        for annotation in annotations_data.get("annotations", []):
            # Only include annotations that don't have track_id (original annotations)
            if "track_id" not in annotation:
                input_annotations.append(annotation)

        # Create input annotations data structure
        input_data = {
            "annotations": input_annotations,
            "studies": annotations_data.get("studies", []),
            "labels": annotations_data.get("labels", []),
        }

        # Save the input annotations data for this video
        annotations_file = os.path.join(
            video_output_dir, "input_annotations.json"
        )
        with open(annotations_file, "w") as f:
            json.dump(input_data, f, indent=2, default=str)

    except Exception as e:
        pass  # Silent failure for background task
