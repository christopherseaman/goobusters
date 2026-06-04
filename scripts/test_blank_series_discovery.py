#!/usr/bin/env python3
"""
Test that the server series index discovers orphan videos: a series with an
mp4 on disk but ZERO annotations must enter the index so get_series/list_series
succeed and select_next_series can offer it.

Hermetic: builds its own MD.ai-style dataset fixture (no real download needed).

Run: uv run python scripts/test_blank_series_discovery.py
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import os
import sys

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_lib_server = os.path.join(_project_root, "lib", "server")
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)
if _lib_server not in sys.path:
    sys.path.insert(0, _lib_server)
if os.path.dirname(os.path.abspath(__file__)) not in sys.path:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from _blank_series_fixtures import build_dataset_fixture, make_config

from lib.server.storage.series_manager import SeriesManager

PROJECT_ID = "TESTPROJ"
DATASET_ID = "TESTDS"
LABEL_ID = "L_FLUID"
EMPTY_ID = "L_EMPTY"

ANNOTATED = ("1.study.annotated", "1.series.annotated")
ORPHAN = ("2.study.orphan", "2.series.orphan")


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        data_dir = tmp_path / "data"
        output_dir = tmp_path / "output"
        state_dir = tmp_path / "state"

        build_dataset_fixture(
            data_dir, PROJECT_ID, DATASET_ID,
            annotated=[ANNOTATED], orphans=[ORPHAN],
            label_id=LABEL_ID, empty_id=EMPTY_ID,
        )
        config = make_config(
            data_dir, output_dir, state_dir,
            project_id=PROJECT_ID, dataset_id=DATASET_ID,
            label_id=LABEL_ID, empty_id=EMPTY_ID,
        )

        manager = SeriesManager(config)

        # Orphan video must be discoverable (currently raises FileNotFoundError).
        orphan_meta = manager.get_series(*ORPHAN)
        assert orphan_meta.study_uid == ORPHAN[0]
        assert orphan_meta.series_uid == ORPHAN[1]
        assert orphan_meta.video_path.endswith(f"{ORPHAN[0]}/{ORPHAN[1]}.mp4"), (
            f"orphan video_path wrong: {orphan_meta.video_path}"
        )
        # No study row for the orphan -> graceful fallback metadata.
        assert orphan_meta.exam_number is None, "orphan exam_number should be None"

        # Both annotated and orphan appear in the listing.
        listed = {(m.study_uid, m.series_uid) for m in manager.list_series()}
        assert ANNOTATED in listed, "annotated series missing from index"
        assert ORPHAN in listed, "orphan series missing from index"

        # Annotated must remain intact (regression guard).
        annotated_meta = manager.get_series(*ANNOTATED)
        assert annotated_meta.exam_number == 1, (
            f"annotated exam_number {annotated_meta.exam_number} != 1"
        )

        # With the annotated series completed, the orphan is auto-offered.
        manager.mark_complete(*ANNOTATED)
        selected = manager.select_next_series(user_email="tester@test")
        assert selected is not None, "select_next_series returned None"
        assert (selected.study_uid, selected.series_uid) == ORPHAN, (
            f"expected orphan to be selected, got {selected.study_uid}/{selected.series_uid}"
        )

    print("PASS: orphan video discovered, listed, and auto-offered")
    return 0


if __name__ == "__main__":
    sys.exit(main())
