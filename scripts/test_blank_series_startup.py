#!/usr/bin/env python3
"""
Test startup orphan discovery: find_orphan_series must select videos with zero
annotations and no synthesized masks, and exclude trackable series and
already-synthesized ones (so a restart doesn't re-enqueue completed work).

Run: uv run python scripts/test_blank_series_startup.py
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
from lib.server.storage.retrack_queue import RetrackQueue
from lib.server.storage.series_manager import SeriesManager
from lib.server.startup import enqueue_blank_jobs, find_orphan_series
from lib.trackable_series import clear_trackable_series_cache, get_trackable_series

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
        clear_trackable_series_cache()
        trackable = get_trackable_series(config)

        assert ANNOTATED in trackable, "annotated series should be trackable"
        assert ORPHAN not in trackable, "orphan must not be trackable"

        mask_root = Path(config.mask_storage_path)
        orphans = find_orphan_series(
            manager, trackable, mask_root, config.flow_method
        )
        keys = {(s.study_uid, s.series_uid) for s in orphans}
        assert ORPHAN in keys, "orphan not discovered for blank synthesis"
        assert ANNOTATED not in keys, "trackable series wrongly flagged as orphan"

        # Once masks.tar exists, the orphan must NOT be re-flagged (idempotent
        # restart).
        series_out = mask_series_dir(mask_root, config.flow_method, *ORPHAN)
        series_out.mkdir(parents=True, exist_ok=True)
        (series_out / "masks.tar").write_bytes(b"stub")
        orphans_after = find_orphan_series(
            manager, trackable, mask_root, config.flow_method
        )
        keys_after = {(s.study_uid, s.series_uid) for s in orphans_after}
        assert ORPHAN not in keys_after, "synthesized orphan re-flagged on restart"

        # enqueue_blank_jobs must not double-enqueue an orphan that already has
        # an active job (crash+restart leaves a pending job that find_orphan_series
        # still flags because masks.tar was never written).
        (series_out / "masks.tar").unlink()  # undo the stub: orphan again
        orphans2 = find_orphan_series(manager, trackable, mask_root, config.flow_method)
        queue = RetrackQueue(config.server_state_path / "retrack_queue.json")
        queue.enqueue_blank(*ORPHAN)  # leftover pending job from prior boot
        n = enqueue_blank_jobs(queue, orphans2)
        assert n == 0, f"double-enqueued an orphan with an active job (n={n})"
        dup = [j for j in queue._load_queue() if (j.study_uid, j.series_uid) == ORPHAN]
        assert len(dup) == 1, f"expected 1 job for orphan, found {len(dup)}"

    print("PASS: find_orphan_series + enqueue_blank_jobs dedupe correctly")
    return 0


if __name__ == "__main__":
    sys.exit(main())
