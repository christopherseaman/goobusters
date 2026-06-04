#!/usr/bin/env python3
"""
Test the blank-series worker path: a "blank" job synthesizes openable
artifacts for an orphan video and marks itself completed.

Run: uv run python scripts/test_blank_series_worker.py
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

from _blank_series_fixtures import build_dataset_fixture, make_config

from lib.mask_archive import mask_series_dir
from lib.server.retrack_worker import process_blank_job
from lib.server.storage.retrack_queue import RetrackQueue
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

        queue = RetrackQueue(config.server_state_path / "retrack_queue.json")
        job = queue.enqueue_blank(*ORPHAN)
        assert job.job_type == "blank", f"job_type {job.job_type} != blank"

        dequeued = queue.dequeue()
        assert dequeued is not None and dequeued.job_type == "blank"

        process_blank_job(dequeued, config, manager)

        series_out = mask_series_dir(
            Path(config.mask_storage_path), config.flow_method, *ORPHAN
        )
        assert (series_out / "masks.tar").exists(), "masks.tar not synthesized"
        assert (series_out / "frames.tar").exists(), "frames.tar not synthesized"
        assert (series_out / "frametype.json").exists(), "frametype.json missing"

        # Tracking status is filesystem-derived; masks.tar => completed.
        assert manager.get_tracking_status(*ORPHAN) == "completed", (
            f"tracking_status {manager.get_tracking_status(*ORPHAN)} != completed"
        )

        # Job marked completed (not failed/processing).
        status = queue.get_job_status(*ORPHAN)
        assert status is not None and status.status == "completed", (
            f"job status {status.status if status else None} != completed"
        )

    print("PASS: blank job synthesized artifacts and completed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
