"""
Shared logic for determining which series are trackable.

This module provides a single source of truth for the trackability check:
a series is trackable if it has at least one LABEL_ID or EMPTY_ID annotation
AND the corresponding video file exists on disk.
"""

from __future__ import annotations

from pathlib import Path
from typing import Set, Tuple

import mdai
import pandas as pd

from lib.config import ServerConfig, SharedConfig
from track import find_annotations_file, find_images_dir

# Cache for trackable series sets (keyed by config parameters)
_trackable_series_cache: dict[str, Set[Tuple[str, str]]] = {}


def get_trackable_series(
    config: SharedConfig,
    data_dir: Path | str | None = None,
    project_id: str | None = None,
    dataset_id: str | None = None,
) -> Set[Tuple[str, str]]:
    """
    Get set of trackable (study_uid, series_uid) tuples.
    
    A series is trackable if:
    1. It has at least one annotation with labelId == LABEL_ID or labelId == EMPTY_ID
    2. The corresponding video file exists on disk
    
    This is the same logic used by track.py.
    
    Args:
        config: Configuration with label_id, empty_id, data_dir, project_id, dataset_id
        data_dir: Override data directory (defaults to config.data_dir)
        project_id: Override project ID (defaults to config.project_id)
        dataset_id: Override dataset ID (defaults to config.dataset_id)
    
    Returns:
        Set of (study_uid, series_uid) tuples for trackable series
    """
    data_dir = Path(data_dir or config.data_dir)
    project_id = project_id or config.project_id
    dataset_id = dataset_id or config.dataset_id
    
    # Create cache key from config parameters
    cache_key = f"{data_dir}|{project_id}|{dataset_id}|{config.label_id}|{config.empty_id}"
    
    # Return cached result if available
    if cache_key in _trackable_series_cache:
        return _trackable_series_cache[cache_key]
    
    # Find annotations file and images directory
    annotations_path = find_annotations_file(
        str(data_dir),
        project_id,
        dataset_id,
    )
    images_dir = find_images_dir(
        str(data_dir),
        project_id,
        dataset_id,
    )
    
    # Load annotations
    annotations_blob = mdai.common_utils.json_to_dataframe(annotations_path)
    annotations_df = pd.DataFrame(annotations_blob["annotations"])
    
    # Filter annotations for free fluid label AND empty frames (same as track.py)
    label_id = config.label_id
    empty_id = config.empty_id
    free_fluid_annotations = annotations_df[
        (annotations_df["labelId"] == label_id)
        | (annotations_df["labelId"] == empty_id)
    ].copy()
    
    # Check if video files exist (same as track.py)
    images_dir_str = str(images_dir)
    
    def construct_video_path(base_dir: str, study_uid: str, series_uid: str) -> str:
        """Construct video path matching track.py logic."""
        return f"{base_dir}/{study_uid}/{series_uid}.mp4"
    
    free_fluid_annotations["video_path"] = free_fluid_annotations.apply(
        lambda row: construct_video_path(
            images_dir_str, row["StudyInstanceUID"], row["SeriesInstanceUID"]
        ),
        axis=1,
    )
    free_fluid_annotations["file_exists"] = free_fluid_annotations[
        "video_path"
    ].apply(lambda path: Path(path).exists())
    
    # Group by series - only series with existing videos and free fluid annotations
    matched_annotations = free_fluid_annotations[
        free_fluid_annotations["file_exists"]
    ]
    video_groups = matched_annotations.groupby([
        "StudyInstanceUID",
        "SeriesInstanceUID",
    ])
    
    # Create set of trackable series (same logic as track.py)
    trackable_series_set = set(
        (study_uid, series_uid) for (study_uid, series_uid), _ in video_groups
    )
    
    # Cache the result
    _trackable_series_cache[cache_key] = trackable_series_set
    
    return trackable_series_set


def clear_trackable_series_cache() -> None:
    """
    Clear the trackable series cache.
    
    Useful for testing or when annotations are updated.
    """
    _trackable_series_cache.clear()


def is_series_trackable(
    study_uid: str,
    series_uid: str,
    config: SharedConfig,
    data_dir: Path | str | None = None,
    project_id: str | None = None,
    dataset_id: str | None = None,
) -> bool:
    """
    Check if a specific series is trackable.
    
    Args:
        study_uid: Study Instance UID
        series_uid: Series Instance UID
        config: Configuration
        data_dir: Override data directory
        project_id: Override project ID
        dataset_id: Override dataset ID
    
    Returns:
        True if series is trackable, False otherwise
    """
    trackable = get_trackable_series(config, data_dir, project_id, dataset_id)
    return (study_uid, series_uid) in trackable


