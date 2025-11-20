1. Clean code and remove unused or overly complex bits in the current fork (incl. remove deepflow as too slow to be practical)
2. Update mdai integration to ONLY pull the PROJECT_ID and DATASET_ID from dot.env (not all data from the project) `mdai_client.project(project_id=PROJECT_ID, dataset_id=DATASET_ID, path=DATA_DIR)` from https://docs.md.ai/annotator/python/guides/general/ 
3. Empty frame annotations (allow annotations for frames with no fluid in bidirectional tracking with EMPTY_ID from dot.env)
4. Jerky tracking? Tracking seems to jump frame-to-frame
5. Performance tuning? Anything to be done with cpu vs gpu accel or parallelization?