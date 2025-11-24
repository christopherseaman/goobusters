# EMPTY_ID Annotation Format

## MD.ai Format for Empty/"No Fluid" Frames

Human-verified empty frames in MD.ai use the following format:

```json
{
  "labelId": "L_75K42J",  // EMPTY_ID from dot.env
  "frameNumber": 0,       // 0-based frame number (MD.ai uses 0-based indexing)
  "StudyInstanceUID": "...",
  "SeriesInstanceUID": "...",
  "data": null  // OR data: {"foreground": [], "background": []}
}
```

**Key Points:**
- `labelId == EMPTY_ID` is the **authoritative indicator** for empty frames
- Empty frames have **NO polygon data** - `data: null` or `data: {"foreground": []}`
- Empty frames are **human-verified** annotations, NOT just lack of annotations
- Frames without any annotation are **unreviewed**, NOT empty
- **Critical:** We use EMPTY_ID annotations (not lack of annotations) to identify blank/empty frames

**Note:** Some existing MD.ai data files may have EMPTY_ID annotations with polygon data, but this is incorrect. The correct format for empty frames is `data: null` or empty foreground array.

## Our System Format

### In `tracked_annotations.json` (output from track.py):

```json
{
  "labelId": "L_75K42J",  // EMPTY_ID
  "frameNumber": 0,
  "StudyInstanceUID": "...",
  "SeriesInstanceUID": "...",
  "data": null,  // MD.ai convention for empty frames in output
  "type": "empty",
  "is_annotation": true
}
```

### In `annotations/{method}/{study_uid}_{series_uid}/modified_annotations.json` (from app.py):

```json
{
  "frame_0": {
    "label_id": "L_75K42J",  // EMPTY_ID
    "is_empty": true,
    "modified_at": "2025-01-01T12:00:00"
  }
}
```

This format is converted to MD.ai format when retracking (see RETRACK_DESIGN.md).

**Note:** While MD.ai input may have polygon data with EMPTY_ID, our output uses `data: null` to clearly indicate an empty frame.

