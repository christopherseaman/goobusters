"""
MD.ai dataset management utilities for the distributed client.

Responsible for downloading datasets via the MD.ai SDK and exposing metadata
about locally available studies/series for the WebView frontend.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Any, Dict, Iterable, List, Tuple

"""
Lightweight fallbacks for mdai/pandas on platforms where the full dependency
stack (numpy/pandas/opencv) is unavailable (e.g., iOS). We inject minimal
stubs so that mdai imports succeed and the JSON parsing/iteration used by
this module continue to work without heavy native extensions.
"""

# Provide minimal pandas replacement when not installed (no numpy required).
# The DataFrame implementation supports only the small subset needed here:
# - construction from list/dict
# - iterrows()
# - groupby(keys) returning a dict-like object with items() iteration
# This avoids pulling in numpy/pandas wheels on iOS.
try:
    import pandas as pd  # type: ignore
except ImportError:  # pragma: no cover - exercised on iOS only
    import sys
    import json

    class _LiteGroupBy:
        def __init__(self, groups: Dict[Tuple[Any, ...], List[Dict[str, Any]]]):
            self._groups = groups

        def items(self):
            return self._groups.items()

        def __iter__(self):
            # Pandas groupby iteration yields (key, group) pairs
            return iter(self._groups.items())

    class _LiteDataFrame:
        def __init__(self, data: Any):
            # Accept list of dicts or dict with list values
            if isinstance(data, dict):
                # Convert columnar dict to list of rows
                keys = list(data.keys())
                rows = []
                # assume all columns same length
                if keys:
                    for idx in range(len(data[keys[0]])):
                        rows.append({k: data[k][idx] for k in keys})
                self._rows = rows
            elif isinstance(data, list):
                self._rows = list(data)
            else:
                self._rows = []

        def iterrows(self) -> Iterable[Tuple[int, Dict[str, Any]]]:
            for idx, row in enumerate(self._rows):
                yield idx, row

        def groupby(self, keys: List[str]) -> _LiteGroupBy:
            groups: Dict[Tuple[Any, ...], List[Dict[str, Any]]] = {}
            for row in self._rows:
                key = tuple(row.get(k) for k in keys)
                groups.setdefault(key, []).append(row)
            return _LiteGroupBy(groups)

    class _LitePandasModule:
        DataFrame = _LiteDataFrame

    pd = _LitePandasModule()  # type: ignore
    sys.modules["pandas"] = pd  # Ensure subsequent imports see the stub

# Parse annotations without pandas if available
import json
import glob
from types import SimpleNamespace

from lib.config import ClientConfig


def find_annotations_file(data_dir: str, project_id: str, dataset_id: str) -> str:
    """
    Find the most recent annotations JSON file for the given project and dataset.
    """
    import os
    pattern = os.path.join(
        data_dir, f"mdai_*_project_{project_id}_annotations_dataset_{dataset_id}_*.json"
    )
    matches = glob.glob(pattern)
    if not matches:
        raise FileNotFoundError(
            f"No annotations file found: {project_id}, dataset {dataset_id} in {data_dir}. "
            f"Expected pattern: mdai_*_project_{project_id}_annotations_dataset_{dataset_id}_*.json"
        )
    return sorted(matches)[-1]


def find_images_dir(data_dir: str, project_id: str, dataset_id: str) -> str:
    """
    Find the most recent images directory for the given project and dataset.
    """
    import os
    pattern = os.path.join(
        data_dir, f"mdai_*_project_{project_id}_images_dataset_{dataset_id}_*"
    )
    matches = [d for d in glob.glob(pattern) if os.path.isdir(d)]
    if not matches:
        raise FileNotFoundError(
            f"No images directory found: {project_id}, dataset {dataset_id} in {data_dir}. "
            f"Expected pattern: mdai_*_project_{project_id}_images_dataset_{dataset_id}_*"
        )
    return sorted(matches)[-1]

# mdai import - uses vendored minimal version on iOS
import mdai
from mdai import common_utils


class DatasetNotReady(RuntimeError):
    """Raised when the MD.ai dataset is not present locally."""


@dataclass
class SeriesInfo:
    study_uid: str
    series_uid: str
    exam_number: Optional[int]
    series_number: Optional[int]
    dataset_name: str
    video_path: Path


class MDaiDatasetManager:
    """Download, cache, and enumerate MD.ai datasets for the client."""

    def __init__(self, config: ClientConfig) -> None:
        self.config = config
        self.video_cache_path = Path(config.video_cache_path)
        self.video_cache_path.mkdir(parents=True, exist_ok=True)
        self._client_lock = threading.Lock()
        self._client: Optional[Any] = None  # mdai.Client when available
        self._token_override: Optional[str] = None
        self._images_dir: Optional[Path] = None
        self._annotations_df: Optional[pd.DataFrame] = None
        self._studies_lookup: Optional[dict[str, dict]] = None

    # ------------------------------------------------------------------ MD.ai SDK
    def _client_instance(self) -> mdai.Client:
        with self._client_lock:
            if self._client is None:
                self._client = mdai.Client(
                    domain=self.config.domain,
                    access_token=self._token_override or self.config.mdai_token,
                )
            return self._client

    def set_token(self, token: Optional[str]) -> None:
        """
        Update the MD.ai token for subsequent SDK calls.
        Resets the underlying client so the next use picks up the new token.
        """
        with self._client_lock:
            self._token_override = token.strip() if token else None
            self._client = None

    def sync_dataset(self) -> Path:
        """
        Download/refresh the MD.ai dataset. Returns the images directory.
        """
        client = self._client_instance()
        project = client.project(
            project_id=self.config.project_id,
            dataset_id=self.config.dataset_id,
            path=str(self.video_cache_path),
        )
        self._images_dir = Path(project.images_dir)
        self._annotations_df = None  # force reload
        self._studies_lookup = None
        return self._images_dir

    # ------------------------------------------------------------ Local discovery
    def _ensure_images_dir(self) -> Path:
        if self._images_dir and self._images_dir.exists():
            return self._images_dir
        try:
            # Look in video_cache_path where sync_dataset downloads to
            images_dir = Path(
                find_images_dir(
                    str(self.video_cache_path),
                    self.config.project_id,
                    self.config.dataset_id,
                )
            )
            self._images_dir = images_dir
            return images_dir
        except FileNotFoundError:
            raise DatasetNotReady(
                "MD.ai dataset not found locally. Run /api/dataset/sync first."
            )

    def _ensure_annotations(self) -> pd.DataFrame:
        if self._annotations_df is not None and self._studies_lookup is not None:
            return self._annotations_df
        try:
            # Look in video_cache_path where sync_dataset downloads to
            annotations_path = find_annotations_file(
                str(self.video_cache_path),
                self.config.project_id,
                self.config.dataset_id,
            )
        except FileNotFoundError:
            raise DatasetNotReady(
                "Annotations JSON missing. Download dataset via /api/dataset/sync."
            )

        blob = mdai.common_utils.json_to_dataframe(annotations_path)
        annotations_df = pd.DataFrame(blob["annotations"])
        studies_df = pd.DataFrame(blob["studies"])
        self._annotations_df = annotations_df
        self._studies_lookup = {
            row["StudyInstanceUID"]: row for _, row in studies_df.iterrows()
        }
        return annotations_df

    def list_local_series(self) -> list[SeriesInfo]:
        images_dir = self._ensure_images_dir()
        annotations_df = self._ensure_annotations()
        studies_lookup = self._studies_lookup or {}

        series: list[SeriesInfo] = []
        grouped = annotations_df.groupby(["StudyInstanceUID", "SeriesInstanceUID"])
        for (study_uid, series_uid), _ in grouped:
            # Skip entries with missing UIDs
            if not study_uid or not series_uid:
                continue
            video_path = images_dir / study_uid / f"{series_uid}.mp4"
            if not video_path.exists():
                continue

            study_info = studies_lookup.get(study_uid, {})
            exam_number = study_info.get("number")
            raw_series_number = study_info.get("SeriesNumber") or study_info.get(
                "seriesNumber"
            )
            try:
                parsed_series_number = int(raw_series_number)
            except (TypeError, ValueError):
                parsed_series_number = None

            series.append(
                SeriesInfo(
                    study_uid=study_uid,
                    series_uid=series_uid,
                    exam_number=int(exam_number)
                    if exam_number not in (None, "")
                    else None,
                    series_number=parsed_series_number,
                    dataset_name=str(study_info.get("dataset", "Unknown")),
                    video_path=video_path,
                )
            )

        series.sort(key=lambda item: (item.exam_number or 0, item.series_uid))
        return series

    def resolve_video(self, study_uid: str, series_uid: str) -> Path:
        images_dir = self._ensure_images_dir()
        video_path = images_dir / study_uid / f"{series_uid}.mp4"
        if not video_path.exists():
            raise FileNotFoundError(
                f"Video not found for {study_uid}/{series_uid}: {video_path}"
            )
        return video_path

    def dataset_ready(self) -> bool:
        try:
            self._ensure_images_dir()
            self._ensure_annotations()
            return True
        except DatasetNotReady:
            return False
