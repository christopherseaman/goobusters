"""
Shared fixtures for blank-series (zero-annotation) tests.

Builds a hermetic MD.ai-style dataset on disk: an annotations JSON parseable by
mdai.common_utils.json_to_dataframe plus an images directory of real mp4s, with
both annotated series and orphan (zero-annotation) videos.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np

from lib.config import ClientConfig, ServerConfig

_STAMP = "2026-01-01-000000"


def _label(label_id: str, name: str) -> dict:
    """An MD.ai label dict with all keys json_to_dataframe selects."""
    return {
        "id": label_id,
        "name": name,
        "annotationMode": "freeform",
        "color": "#000000",
        "description": "",
        "radlexTagIds": [],
        "scope": "INSTANCE",
        "parentId": None,
    }


def write_test_video(path: Path, n_frames: int = 6, size: int = 64) -> int:
    """Write a real mp4 with n_frames distinct frames. Returns frames written."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, 10.0, (size, size))
    if not writer.isOpened():
        raise RuntimeError(f"Could not open VideoWriter for {path}")
    for i in range(n_frames):
        frame = np.full((size, size, 3), (i * 30) % 256, dtype=np.uint8)
        writer.write(frame)
    writer.release()
    return n_frames


def build_dataset_fixture(
    data_dir: Path,
    project_id: str,
    dataset_id: str,
    *,
    annotated: list[tuple[str, str]],
    orphans: list[tuple[str, str]],
    label_id: str,
    empty_id: str,
    n_frames: int = 6,
) -> Path:
    """
    Create annotations JSON + images dir under data_dir.

    annotated/orphans are (study_uid, series_uid) tuples. Annotated series get
    an annotation row (labelId=label_id); orphans get only an mp4 on disk, no
    annotation and no study entry. Returns the images directory path.
    """
    data_dir.mkdir(parents=True, exist_ok=True)

    annotated_studies = []
    seen = set()
    for idx, (study, _series) in enumerate(annotated):
        if study not in seen:
            annotated_studies.append({"StudyInstanceUID": study, "number": idx + 1})
            seen.add(study)

    annotations = [
        {
            "StudyInstanceUID": study,
            "SeriesInstanceUID": series,
            "labelId": label_id,
            "frameNumber": 0,
            "createdAt": "2026-01-01T00:00:00.000Z",
            "updatedAt": "2026-01-01T00:00:00.000Z",
        }
        for (study, series) in annotated
    ]

    export = {
        "datasets": [
            {
                "id": dataset_id,
                "name": "TestDataset",
                "studies": annotated_studies,
                "annotations": annotations,
            }
        ],
        "labelGroups": [
            {
                "id": "lg1",
                "name": "group",
                "labels": [
                    _label(label_id, "Fluid"),
                    _label(empty_id, "Empty"),
                ],
            }
        ],
    }

    annotations_path = (
        data_dir
        / f"mdai_test_project_{project_id}_annotations_dataset_{dataset_id}_{_STAMP}.json"
    )
    annotations_path.write_text(json.dumps(export))

    images_dir = (
        data_dir
        / f"mdai_test_project_{project_id}_images_dataset_{dataset_id}_{_STAMP}"
    )
    images_dir.mkdir(parents=True, exist_ok=True)
    for study, series in [*annotated, *orphans]:
        write_test_video(images_dir / study / f"{series}.mp4", n_frames=n_frames)

    return images_dir


def make_config(
    data_dir: Path,
    output_dir: Path,
    state_dir: Path,
    *,
    project_id: str,
    dataset_id: str,
    label_id: str,
    empty_id: str,
) -> ServerConfig:
    """Build a ServerConfig pointed at the fixture paths."""
    return ServerConfig(
        mdai_token="test-token",
        data_dir=data_dir,
        domain="test",
        project_id=project_id,
        dataset_id=dataset_id,
        label_id=label_id,
        empty_id=empty_id,
        flow_method="dis",
        server_state_path=state_dir,
        mask_storage_path=output_dir,
    )


def make_client_config(
    video_cache_path: Path,
    *,
    project_id: str,
    dataset_id: str,
    label_id: str,
    empty_id: str,
) -> ClientConfig:
    """Build a ClientConfig whose dataset lives under video_cache_path."""
    return ClientConfig(
        mdai_token="test-token",
        data_dir=video_cache_path,
        domain="test",
        project_id=project_id,
        dataset_id=dataset_id,
        label_id=label_id,
        empty_id=empty_id,
        flow_method="dis",
        video_cache_path=video_cache_path,
        # Unreachable so the non-blocking server activity fetch fails fast.
        server_url="http://127.0.0.1:1",
    )
