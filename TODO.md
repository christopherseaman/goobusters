Please review the codebase first. We need to do a few things:
1. output frames and video should always be created (currently only with debug enabled)
2. need to generate an identity file within each video output folder (yaml?) that has the studyinstanceUID, seriesInstanceUID, exam #, series #, dataset name and id
3. The output masks should use $TRACK_ID from dot.env and the video outputs shoudl color the tracked fluid in a different color than the original annotations
