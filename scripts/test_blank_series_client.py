#!/usr/bin/env python3
"""
Test client-side discovery of orphan videos: MDaiDatasetManager.list_local_series
must surface a series that has an mp4 on disk but zero annotations, so it appears
in /api/videos and _find_series_info resolves it (unblocking /api/frames).

Run: uv run python scripts/test_blank_series_client.py
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import os
import sys

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)
if os.path.dirname(os.path.abspath(__file__)) not in sys.path:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from _blank_series_fixtures import build_dataset_fixture, make_client_config

from lib.client.mdai_client import MDaiDatasetManager

PROJECT_ID = "TESTPROJ"
DATASET_ID = "TESTDS"
LABEL_ID = "L_FLUID"
EMPTY_ID = "L_EMPTY"

ANNOTATED = ("1.study.annotated", "1.series.annotated")
ORPHAN = ("2.study.orphan", "2.series.orphan")


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        cache = Path(tmp) / "client_cache"
        build_dataset_fixture(
            cache, PROJECT_ID, DATASET_ID,
            annotated=[ANNOTATED], orphans=[ORPHAN],
            label_id=LABEL_ID, empty_id=EMPTY_ID,
        )
        config = make_client_config(
            cache, project_id=PROJECT_ID, dataset_id=DATASET_ID,
            label_id=LABEL_ID, empty_id=EMPTY_ID,
        )

        manager = MDaiDatasetManager(config)
        series = manager.list_local_series()
        keys = {(s.study_uid, s.series_uid) for s in series}

        assert ANNOTATED in keys, "annotated series missing from client list"
        assert ORPHAN in keys, "orphan video missing from client list"

        orphan_info = next(
            s for s in series if (s.study_uid, s.series_uid) == ORPHAN
        )
        assert orphan_info.video_path.exists(), "orphan video_path does not exist"
        assert orphan_info.exam_number is None, "orphan exam_number should be None"

        # resolve_video must succeed for the orphan (used by _find_series_info).
        resolved = manager.resolve_video(*ORPHAN)
        assert resolved.exists(), "resolve_video failed for orphan"

    print("PASS: client surfaces orphan video in list_local_series")
    return 0


if __name__ == "__main__":
    sys.exit(main())
