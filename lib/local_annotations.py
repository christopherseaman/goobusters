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

    This function reads modified annotations from the local annotations directory
    and converts them to the format expected by process_video_with_multi_frame_tracking().

    Args:
        study_uid: Study Instance UID
        series_uid: Series Instance UID
        annotations_dir: Base directory for annotations (default: "annotations")

    Returns:
        pandas DataFrame with columns: frameNumber, labelId, data
        - frameNumber: int, 0-based frame number
        - labelId: str, either LABEL_ID (fluid) or EMPTY_ID (empty)
        - data: dict with foreground polygon or None for empty frames
        Empty DataFrame if no annotations exist

    Note:
        - ONLY processes frames with label_id or empty_id (human-verified annotations)
        - Reads from: annotations/{study_uid}_{series_uid}/modified_annotations.json
        - Loads masks from: annotations/{study_uid}_{series_uid}/masks/frame_XXXXXX_mask.webp
        - For fluid frames (is_empty=false): Converts mask to polygon using cv2.findContours()
        - For empty frames (is_empty=true): Uses "data": null (MD.ai format for verified empty)

    Structure of modified_annotations.json:
        {
            "frame_N": {
                "label_id": "L_xxxxx",  // LABEL_ID or EMPTY_ID
                "is_empty": bool,       // true for empty frames, false for fluid
                "modified_at": "ISO timestamp"
            }
        }
    """
    # Construct paths
    annotation_path = Path(annotations_dir) / f"{study_uid}_{series_uid}"
    annotations_file = annotation_path / "modified_annotations.json"
    masks_dir = annotation_path / "masks"

    # Check if annotations exist
    if not annotations_file.exists():
        return pd.DataFrame(columns=['frameNumber', 'labelId', 'data'])

    # Load modified annotations
    with open(annotations_file) as f:
        modified_annotations = json.load(f)

    # Process each annotation
    annotations_list = []

    for frame_key, frame_data in modified_annotations.items():
        # Parse frame number from "frame_N" format
        frame_num = int(frame_key.split('_')[1])

        # Get annotation metadata
        label_id = frame_data.get('label_id', '')
        is_empty = frame_data.get('is_empty', False)

        # Skip if no label_id
        if not label_id:
            continue

        # Construct mask path
        mask_path = masks_dir / f"frame_{frame_num:06d}_mask.webp"

        # Create annotation entry
        annotation = {
            'frameNumber': frame_num,
            'labelId': label_id
        }

        if is_empty:
            # Empty frame: use None data (MD.ai format for verified empty frames)
            annotation['data'] = None
        else:
            # Fluid frame: convert mask to polygon
            if not mask_path.exists():
                print(f"Warning: Mask file not found for frame {frame_num}: {mask_path}")
                continue

            # Load mask
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if mask is None:
                print(f"Warning: Failed to load mask for frame {frame_num}: {mask_path}")
                continue

            # Convert to polygon
            polygon = mask_to_polygon(mask)
            if polygon is None or len(polygon) < 3:
                print(f"Warning: No valid polygon found for frame {frame_num}")
                continue

            # Create data structure matching MD.ai format
            annotation['data'] = {
                'foreground': polygon
            }

        annotations_list.append(annotation)

    # Convert to DataFrame
    if not annotations_list:
        return pd.DataFrame(columns=['frameNumber', 'labelId', 'data'])

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
