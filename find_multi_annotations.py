#!/usr/bin/env python3
"""Find videos with multiple free fluid annotations."""

import json
import os
from pathlib import Path

# Load environment
LABEL_ID = "L_13yPql"  # Free fluid label from dot.env
data_dir = Path("data/mdai_ucsf_project_x9N2LJBZ_images_2025-09-18-050340")
annotations_file = "data/mdai_ucsf_project_x9N2LJBZ_annotations_2025-09-18-050340.json"

# Find all videos
videos = {}
for mp4 in data_dir.rglob("*.mp4"):
    study_uid = mp4.parent.name
    series_uid = mp4.stem
    videos[f"{study_uid}/{series_uid}"] = str(mp4)

print(f"Found {len(videos)} videos")

# Load annotations
with open(annotations_file) as f:
    data = json.load(f)

# Count annotations per video
annotation_counts = {}
for dataset in data['datasets']:
    for annotation in dataset.get('annotations', []):
        if annotation.get('labelId') == LABEL_ID:
            study = annotation.get('StudyInstanceUID')
            series = annotation.get('SeriesInstanceUID')
            key = f"{study}/{series}"
            if key in videos:
                if key not in annotation_counts:
                    annotation_counts[key] = []
                # Try to get frame info if available
                sop = annotation.get('SOPInstanceUID', 'unknown')
                annotation_counts[key].append(sop)

# Find videos with multiple annotations
multi_annotation_videos = {k: v for k, v in annotation_counts.items() if len(v) > 1}

print(f"\nVideos with multiple free fluid annotations:")
for key, annotations in sorted(multi_annotation_videos.items(), key=lambda x: len(x[1]), reverse=True)[:10]:
    study, series = key.split('/')
    unique_frames = len(set(annotations))
    print(f"\n{key}")
    print(f"  Path: {videos[key]}")
    print(f"  Total annotations: {len(annotations)}")
    print(f"  Unique frames/SOPs: {unique_frames}")
    if unique_frames > 1:
        print(f"  *** Good candidate - annotations on different frames ***")

if not multi_annotation_videos:
    print("No videos found with multiple annotations")