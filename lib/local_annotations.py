#!/usr/bin/env python3
"""
Local Annotations Converter

Converts local annotations (modified_annotations.json + WebP masks) to MD.ai JSON format
for retracking. This enables the feedback loop where users edit annotations in the web UI
and re-run optical flow tracking with updated human-verified annotations.
"""

import os
import json
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, Dict, List, Tuple


def mask_to_polygon(mask: np.ndarray, simplify_tolerance: float = 2.0) -> Optional[List[List[float]]]:
    """
    Convert binary mask to polygon coordinates using contour detection.

    Args:
        mask: Binary mask array (0 or 255)
        simplify_tolerance: Epsilon for Douglas-Peucker polygon simplification (default: 2.0)

    Returns:
        List of [x, y] coordinate pairs representing the polygon, or None if no contours found

    Note:
        - Uses cv2.findContours() to extract mask boundary
        - If multiple contours exist, uses the largest one
        - Applies polygon simplification to reduce point count
    """
    # Ensure mask is binary
    if mask.max() > 1:
        _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return None

    # Use largest contour if multiple exist
    largest_contour = max(contours, key=cv2.contourArea)

    # Simplify polygon using Douglas-Peucker algorithm
    epsilon = simplify_tolerance
    simplified_contour = cv2.approxPolyDP(largest_contour, epsilon, True)

    # Convert to list of [x, y] coordinates
    polygon = [[float(point[0][0]), float(point[0][1])] for point in simplified_contour]

    return polygon


def convert_local_to_annotations_df(
    study_uid: str,
    series_uid: str,
    annotations_dir: str = "annotations"
) -> pd.DataFrame:
    """
    Convert local annotations to DataFrame format for retracking.

    This function reads saved annotations from masks.json and loads the mask
    images directly (no polygon conversion needed for local files).

    Args:
        study_uid: Study Instance UID
        series_uid: Series Instance UID
        annotations_dir: Base directory for annotations (default: "annotations")

    Returns:
        pandas DataFrame with columns: frameNumber, labelId, data, mask
        - frameNumber: int, 0-based frame number
        - labelId: str, either LABEL_ID (fluid) or EMPTY_ID (empty)
        - data: None (not used for local masks)
        - mask: numpy array of the mask image
        Empty DataFrame if no annotations exist

    Note:
        - ONLY processes frames marked as is_annotation=true
        - Reads from: annotations/{study_uid}_{series_uid}/masks.json
        - Loads masks directly from: annotations/{study_uid}_{series_uid}/masks/
    """
    # Construct paths
    annotation_path = Path(annotations_dir) / f"{study_uid}_{series_uid}"
    masks_json_file = annotation_path / "masks.json"
    masks_dir = annotation_path / "masks"

    # Check if masks.json exists
    if not masks_json_file.exists():
        return pd.DataFrame(columns=['frameNumber', 'labelId', 'data', 'mask'])

    # Load masks.json
    with open(masks_json_file) as f:
        masks_data = json.load(f)

    # Process each annotation (only those marked is_annotation=true)
    annotations_list = []

    for entry in masks_data:
        # Only include human-verified annotations
        if not entry.get('is_annotation', False):
            continue

        frame_num = entry.get('frameNumber')
        label_id = entry.get('labelId', entry.get('label_id', ''))
        is_empty = entry.get('type') == 'empty'

        # Skip if no label_id or frame_num
        if not label_id or frame_num is None:
            continue

        # Construct mask path
        mask_file = entry.get('mask_file', f"frame_{frame_num:06d}_mask.webp")
        mask_path = masks_dir / mask_file

        # Load mask directly (no polygon conversion)
        if not mask_path.exists():
            print(f"Warning: Mask file not found for frame {frame_num}: {mask_path}")
            continue

        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            print(f"Warning: Failed to load mask for frame {frame_num}: {mask_path}")
            continue

        # Create annotation entry with mask directly
        annotation = {
            'frameNumber': frame_num,
            'labelId': label_id,
            'data': None,  # Not used for local masks
            'mask': mask   # Pass mask directly
        }

        annotations_list.append(annotation)

    # Convert to DataFrame
    if not annotations_list:
        return pd.DataFrame(columns=['frameNumber', 'labelId', 'data', 'mask'])

    df = pd.DataFrame(annotations_list)

    # Sort by frame number for consistency
    df = df.sort_values('frameNumber').reset_index(drop=True)

    return df


def get_local_annotations_summary(
    study_uid: str,
    series_uid: str,
    annotations_dir: str = "annotations"
) -> Dict[str, int]:
    """
    Get summary statistics for local annotations.

    Args:
        study_uid: Study Instance UID
        series_uid: Series Instance UID
        annotations_dir: Base directory for annotations (default: "annotations")

    Returns:
        Dictionary with counts: total_annotations, fluid_annotations, empty_annotations
    """
    df = convert_local_to_annotations_df(study_uid, series_uid, annotations_dir)

    if df.empty:
        return {
            'total_annotations': 0,
            'fluid_annotations': 0,
            'empty_annotations': 0
        }

    # Count by annotation type
    empty_count = df['data'].isna().sum()
    fluid_count = len(df) - empty_count

    return {
        'total_annotations': len(df),
        'fluid_annotations': fluid_count,
        'empty_annotations': empty_count
    }
