#!/usr/bin/env python3
"""
Test the post-startup orphan fallback: GET /api/masks on an orphan video that
has NOT yet been synthesized must return 202 TRACK_PENDING and enqueue a blank
synthesis job (not 500 TRACK_FAILED). Covers videos added after startup.

Run: uv run python scripts/test_blank_series_lazy_fallback.py
"""

from __future__ import annotations

import json
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

from lib.server.start import create_app
from lib.server.storage.retrack_queue import RetrackQueue

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

        # No synthesis performed: orphan has a video but no masks yet.
        app = create_app(config, skip_startup=True)
        client = app.test_client()

        resp = client.get(f"/api/masks/{ORPHAN[0]}/{ORPHAN[1]}")
        assert resp.status_code == 202, (
            f"expected 202 pending, got {resp.status_code}: {resp.get_data()[:300]}"
        )
        body = json.loads(resp.get_data())
        assert body.get("error_code") == "TRACK_PENDING", (
            f"error_code {body.get('error_code')} != TRACK_PENDING"
        )

        # A blank synthesis job must have been enqueued for the orphan.
        queue = RetrackQueue(config.server_state_path / "retrack_queue.json")
        job = queue.get_job_status(*ORPHAN)
        assert job is not None, "no job enqueued for orphan"
        assert job.job_type == "blank", f"job_type {job.job_type} != blank"

        # Simulate the worker failing the job (e.g. corrupt video): no masks.tar.
        queue.dequeue()  # -> processing
        queue.mark_failed(*ORPHAN, job.new_version_id, "corrupt video")

        # A subsequent GET must NOT re-enqueue forever: surface the failure.
        resp2 = client.get(f"/api/masks/{ORPHAN[0]}/{ORPHAN[1]}")
        assert resp2.status_code == 500, (
            f"after failure expected 500, got {resp2.status_code}: {resp2.get_data()[:200]}"
        )
        assert json.loads(resp2.get_data()).get("error_code") == "TRACK_FAILED"

        # No additional job was enqueued (still exactly one job for the orphan).
        all_jobs = [
            j for j in queue._load_queue()
            if (j.study_uid, j.series_uid) == ORPHAN
        ]
        assert len(all_jobs) == 1, (
            f"failed orphan re-enqueued: {len(all_jobs)} jobs (infinite-loop bug)"
        )

    print("PASS: un-synthesized orphan returns 202; failed synthesis does not re-enqueue")
    return 0


if __name__ == "__main__":
    sys.exit(main())
