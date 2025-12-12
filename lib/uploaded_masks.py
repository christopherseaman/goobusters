"""
Utilities for converting uploaded mask archives to annotations DataFrame.

Handles extraction of .tgz archives and conversion to the format expected
by the tracking pipeline.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import cv2
import pandas as pd

from lib.mask_archive import MASK_METADATA_FILENAME


def convert_uploaded_masks_to_annotations_df(
    archive_bytes: bytes | None, extract_dir: Path
) -> tuple[pd.DataFrame, dict]:
    """
    Extract uploaded mask archive and convert to annotations DataFrame.

    Args:
        archive_bytes: Binary .tgz archive data (None if already extracted)
        extract_dir: Directory containing extracted files (or to extract into)

    Returns:
        Tuple of (annotations_df, metadata_dict)
        - annotations_df: DataFrame with columns: frameNumber, labelId, data, mask
        - metadata_dict: Metadata from archive's metadata.json

    Raises:
        ValueError: If archive is invalid or contains no annotations
    """
    # Extract archive if bytes provided
    if archive_bytes is not None:
        from lib.mask_archive import extract_mask_archive

        extract_dir.mkdir(parents=True, exist_ok=True)
        extract_mask_archive(archive_bytes, extract_dir)

    # Load metadata.json
    metadata_path = extract_dir / MASK_METADATA_FILENAME
    if not metadata_path.exists():
        raise ValueError("Archive missing metadata.json")

    with metadata_path.open() as f:
        metadata = json.load(f)

    # Process annotations from metadata
    # Per spec: use 'frames' array, filter for is_annotation=true frames
    # These are the frames with label_id that retracking uses as input
    frames_data = metadata.get("frames", [])
    if not frames_data:
        raise ValueError(
            "Archive metadata.json missing 'frames' array. "
            "Uploaded masks must include frames with frame_number, label_id, and filename."
        )

    # Filter for annotation frames (is_annotation=true) - these are the input for retracking
    # Retracking needs the actual mask files (frames), not just metadata
    annotations_list = []
    for entry in frames_data:
        # Only process frames marked as annotations (these have label_id)
        if not entry.get("is_annotation", False):
            continue

        frame_num = entry.get("frame_number")
        label_id = entry.get("label_id")

        if frame_num is None or not label_id:
            continue

        # Check if this is an empty annotation (no mask file)
        filename = entry.get("filename")
        if filename is None:
            # Empty annotation - no mask
            annotation = {
                "frameNumber": frame_num,
                "labelId": label_id,
                "data": None,
                "mask": None,  # Explicitly None for empty frames
            }
            annotations_list.append(annotation)
            continue

        # Load mask file
        mask_path = extract_dir / filename
        if not mask_path.exists():
            continue

        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            continue

        annotation = {
            "frameNumber": frame_num,
            "labelId": label_id,
            "data": None,  # Not used for local masks
            "mask": mask,
        }
        annotations_list.append(annotation)

    if not annotations_list:
        raise ValueError("No valid annotations found in archive")

    df = pd.DataFrame(annotations_list)
    df = df.sort_values("frameNumber").reset_index(drop=True)

    return df, metadata
