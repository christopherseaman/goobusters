# Goobusters Tasks

## To Do

- [ ] lib/optical.py cleanup - 1998 lines but only 2 exported functions used (create_identity_file, copy_annotations_to_output). Consider refactoring or removing dead code. 
- [ ] dot.env.example update for new/modified options
- [ ] `tracked_annotations` should include TRACK_ID, EMPTY_ID, and LABEL_ID
- [ ] lib/debug_visualization.py abandoned. If needed, will will have to fix debug viz and reincorporate
- [ ] Local annotaion feedback loop? See references/teef for simpler example app with single annotation type
    - [ ] Server for reviewing/modifying annotations (LABEL_ID, EMPTY_ID, and TRACK_ID)
    - [ ] Save human reviewed/modified free fluid (LABEL_ID), no fluid (EMPTY_ID) annotations outside data/ bc data reflects mdai truth (maybe local db? duckdb with file?); may be better to store as masks (grayscale images) for local iterations, update annotation from masks on tracking runs (think this happens already but may need fixing)
    - [ ] Re-run tracking with locally updated annotations (LABEL_ID & EMPTY_ID annotations, no TRACK_ID; if in local db then don't source mdai annotations json)

## Done

- [x] Update mdai integration to ONLY pull the PROJECT_ID and DATASET_ID from dot.env (not all data from the project) `mdai_client.project(project_id=PROJECT_ID, dataset_id=DATASET_ID, path=DATA_DIR)` from https://docs.md.ai/annotator/python/guides/general/
- [x] Clean code and remove unused or overly complex bits in the current fork (incl. remove deepflow as too slow to be practical)
    - [x] Removed deepflow
    - [x] Removed temporal smoothing (unproven complexity)
    - [x] Removed unused files: parallel_processor.py, video_capture_manager.py, diagnose.py
- [x] Empty frame annotations (allow annotations for frames with no fluid in bidirectional tracking with EMPTY_ID from dot.env)
- [x] Jerky tracking? Tracking seems to jump frame-to-frame (noted, no changes made)
- [x] Performance tuning? Anything to be done with cpu vs gpu accel or parallelization? (reviewed, already optimized)
