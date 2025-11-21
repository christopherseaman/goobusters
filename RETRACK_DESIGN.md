# Local Annotation Feedback Loop - Design Plan

## Current State Analysis

**What exists:**
- [app.py](app.py) - Flask server with DuckDB storage for modified annotations
- `/api/save_mask` - Saves modified masks to DuckDB
- `/api/mark_empty` - Marks frames as EMPTY_ID in DuckDB
- [track.py](track.py) - Reads annotations from MD.ai JSON, generates TRACK_ID predictions

**What's missing:**
- Integration layer to merge local DB annotations with MD.ai data
- Re-tracking endpoint/mechanism
- UI button to trigger re-tracking

## Architecture Design

### 1. Data Flow

```
Initial Run:
MD.ai JSON → track.py → output/{method}/{study_uid}_{series_uid}/ → app.py displays

Iteration Loop:
app.py (user edits) → DuckDB (local storage)
                    ↓
              track.py reads:
              1. Check DuckDB for (study, series, frame)
              2. If found: use local LABEL_ID/EMPTY_ID
              3. If not: fall back to MD.ai JSON
                    ↓
              New TRACK_ID predictions → output/ → app.py displays
```

### 2. Required Components

**A. New lib module: `lib/annotation_db.py`**
- `get_local_annotations(method, study_uid, series_uid)` → DataFrame
- `merge_annotations(mdai_df, local_df)` → Combined DataFrame
  - Prioritize local modifications over MD.ai data
  - Maintain MD.ai structure (columns, data types)
  - Include both LABEL_ID and EMPTY_ID annotations

**B. Modified track.py flow:**
```python
# Current (line 142-152):
annotations_df = pd.DataFrame(annotations_data['annotations'])
free_fluid_annotations = annotations_df[annotations_df['labelId'] == LABEL_ID]

# New flow:
from lib.annotation_db import merge_annotations, get_local_annotations

annotations_df = pd.DataFrame(annotations_data['annotations'])
# For each video being tracked, merge with local DB
for (study_uid, series_uid), video_annotations in video_groups:
    local_annotations = get_local_annotations(method, study_uid, series_uid)
    merged = merge_annotations(video_annotations, local_annotations)
    # Use merged for tracking
```

**C. New app.py endpoint: `/api/retrack`**
```python
@app.route('/api/retrack/<method>/<study_uid>/<series_uid>', methods=['POST'])
def retrack_video(method, study_uid, series_uid):
    # Run track.py in subprocess for specific video
    # Set environment variables: TEST_STUDY_UID, TEST_SERIES_UID, FLOW_METHOD
    # Return status/progress
```

**D. UI Changes:**
- Add "Re-track" button (⚡) to top-right control buttons
- Show progress/status during re-tracking
- Reload video data after completion

### 3. Implementation Strategy

**Phase 1: Data Layer** (lib/annotation_db.py)
- Create functions to read from DuckDB
- Convert masks back to annotation format
- Merge logic with MD.ai annotations

**Phase 2: Track Integration**
- Modify track.py to check local DB before MD.ai
- Test with single video
- Validate output format matches

**Phase 3: UI/API Layer**
- Add re-track endpoint to app.py
- Add button to viewer
- Handle async/progress feedback

### 4. Key Considerations

**Annotation Format Consistency:**
- Local DB stores: base64 PNG masks + frame metadata
- MD.ai format: polygon coordinates in `data.foreground`
- Need conversion: mask → polygon or track.py handles masks directly?

**Solution:** Check if `process_video_with_multi_frame_tracking` can accept mask images directly rather than requiring polygon data. Looking at [lib/multi_frame_tracker.py:194-268](lib/multi_frame_tracker.py#L194-L268), it calls `_classify_annotations()` which likely expects polygon data.

**Alternative approach:** Store modified masks as PNG files in output directory, have tracking read from there instead of converting back to annotations.

**Performance:**
- Re-tracking single video should be fast (seconds to minutes)
- Run in background subprocess to avoid blocking Flask
- Return immediately with task ID, poll for status

**Data Integrity:**
- Local DB modifications should not affect MD.ai source
- Clear distinction between LABEL_ID (human), EMPTY_ID (human), TRACK_ID (machine)
- Version tracking of iterations

## Questions for User:

1. **Annotation format:** Should we convert stored masks back to MD.ai polygon format, or have tracking read masks directly?

2. **Re-tracking scope:** Re-track single video only, or batch re-track all videos with local modifications?

3. **Progress feedback:** Simple spinner, or detailed progress (frame N of M)?

4. **Method selection:** Re-track with same method, or allow choosing different method (e.g., switch from farneback to raft)?

## Implementation Phases

### Phase 1: Data Layer
- [ ] Create `lib/annotation_db.py` module
- [ ] Implement `get_local_annotations()` function
- [ ] Implement `merge_annotations()` function
- [ ] Test data merging with sample annotations

### Phase 2: Track Integration
- [ ] Modify track.py to use annotation_db
- [ ] Test single video re-tracking
- [ ] Validate output format

### Phase 3: UI/API Layer
- [ ] Add `/api/retrack` endpoint to app.py
- [ ] Add re-track button to viewer UI
- [ ] Implement progress feedback
- [ ] Test end-to-end flow
