# Distributed Architecture Specification

## Overview

Split the monolithic annotation editor into a client-server architecture:
- **Server**: Centralized mask generation and retrack coordination service
- **Client**: iPad app with embedded web UI for annotation

## Architecture Goals

1. **PHI Compliance**: Server NEVER serves videos or PHI - only masks (non-PHI)
2. **Multi-user coordination**: Track who is working on which videos to prevent conflicts
3. **Lightweight clients**: All clients download videos directly from MD.ai using MD.ai SDK
4. **Efficient data transfer**: Only transmit masks (WebP format in .tgz archives)
5. **Secure communication**: HTTPS via Caddy reverse proxy (not optional)

## Components

### Server Component

**Responsibilities:**
- Run optical flow tracking (DIS method from `track.py` logic)
- Store and serve tracked masks for all videos
- Coordinate which series are being edited and by whom
- Handle retrack requests when annotations are modified
- Maintain MD.ai dataset cache for tracking engine only
- Serve mask data via REST API

**Key Features:**
- Lazy tracking: Track series on-demand when first requested (not all upfront)
- Activity tracking: Track last activity timestamp per series via keep-alive pings (30s interval)
- Conflict warnings: Display recent activity to prevent concurrent edits
- Retrack queue: FIFO processing with parallel support (3-6 concurrent users)
- Dual version storage: Maintain production version + temporary version during retrack
- Error codes: Server returns codes, client looks up emoji/message

**Technology Stack:**
- Flask for REST API
- Filesystem-based storage (JSON files for metadata, NO SQLite)
- Existing tracking pipeline (`MultiFrameTracker`, `OpticalFlowProcessor`)
- Caddy for HTTPS reverse proxy with Let's Encrypt

**API Endpoints:**
```
# Authentication (all endpoints require valid MDAI_TOKEN in header)
POST /api/auth/validate                   # Validate token, get user info

# Series navigation
GET  /api/series                          # List all series with metadata
GET  /api/series/next                     # Get next incomplete series (server-selected)
GET  /api/series/{study_uid}/{series_uid} # Get series details + activity history

# Series status management
POST /api/series/{study_uid}/{series_uid}/complete # Mark series as done
POST /api/series/{study_uid}/{series_uid}/reopen   # Unmark as done

# Mask operations
GET  /api/masks/{study_uid}/{series_uid}  # Get current masks as .tgz + version ID
POST /api/masks/{study_uid}/{series_uid}  # Submit edits as .tgz, trigger retrack

# Retrack status
GET  /api/retrack/status/{study_uid}/{series_uid} # Check retrack progress

# Activity tracking (keep-alive)
POST /api/series/{study_uid}/{series_uid}/activity # Update last activity timestamp
```

### Client Component

**Responsibilities:**
- iPad app with embedded web UI (NOT just Safari - needs MD.ai SDK)
- Download videos from MD.ai using Python SDK
- Extract video frames locally
- Fetch masks from server for display
- Cache mask edits locally until save
- Submit annotation changes to server
- Handle reset/revert of local changes

**Key Features:**
- Native app wrapper with embedded WebView
- Python backend for MD.ai auth, dataset download, video frame extraction
- Mask overlay rendering (server-provided masks + local edits)
- Edit caching: Maintain unsaved changes in LocalStorage
- Reset functionality: Discard local cache, revert to server masks
- Touch-friendly UI (existing viewer.js already supports this)

**Technology Stack:**
- Native iPad app wrapper (Pythonista or bundled Python interpreter)
- Python backend: MD.ai SDK integration, frame extraction
- Existing viewer.js/viewer.css frontend (reuse as-is)
- LocalStorage for edit cache
- Fetch API for server communication
- **Client does NOT perform tracking** - only frame extraction and mask overlay

**Critical Constraint**: All clients (desktop and iPad) download videos using MD.ai SDK. Server NEVER serves PHI/videos, only masks.

## User Workflow

### Initial Load
1. Client starts, connects to server with MD.ai token
2. User clicks "Next" to get series assignment
3. Server selects series (smart selection, skips recently viewed)
4. Server records view timestamp, returns series metadata + activity history
5. Client checks for recent edits, shows warning if series active within last hour
6. Client downloads video from MD.ai directly (via SDK)
7. Client extracts frames locally (PNG format)
8. Client fetches masks from server (WebP .tgz + version ID)
9. Client renders video with mask overlay (green=annotation, orange=tracked)

**Error states:**
- ☕ "Server is preparing masks..." (pending/processing initial tracking)
- 💩🔥 "Tracking failed for this series. Contact admin." (failed initial tracking)

### Editing
1. User draws/modifies annotations in browser
2. Changes cached in LocalStorage (not yet saved)
3. Client shows "unsaved changes" indicator
4. User can continue editing or reset to discard

### Saving
1. User clicks "Save"
2. Client packages annotations as .tgz + previous_version_id
3. Client POSTs to server `/api/masks/{study}/{series}`
4. Server validates version ID (conflict check)
5. If valid: Server generates new version ID, queues retrack
6. Server returns new version ID + retrack status
7. Client updates local version ID, clears cache
8. Client shows ⏳ "Retracking..." (15-30s typical, 2-3min timeout)
9. Client polls retrack status every 2 seconds
10. Client fetches updated masks from server when complete

**Error states:**
- ⏳ "Retracking..." (in progress, show hourglass)
- 💩🔥 "Retrack failed. Contact admin." (if server returns error code)
- Timeout after 2-3 minutes: Offer "Reset" or "Keep Waiting"

### Reset
1. User clicks "Reset"
2. Client discards LocalStorage cache
3. Client refetches masks from server
4. Client renders with server masks (original state)

### Next Video
1. User clicks "Next" (or marks current as "done" first)
2. Server smart-selects next series (skips recently viewed, selects longest-unviewed)
3. Repeat from Initial Load step 4

### Conflict Scenario
1. User A and User B both load same series
2. User A saves first (version 1 → version 2)
3. User B tries to save (still has version 1)
4. Server rejects: previous_version_id doesn't match current
5. Client shows conflict warning
6. User B must reset to get latest, then re-apply edits

## Communication Protocol

### Mask Transfer Format

**GET masks (download .tgz):**
```http
GET /api/masks/{study_uid}/{series_uid}
Authorization: Bearer {MDAI_TOKEN}

Response 200:
Headers:
  Content-Type: application/x-tar+gzip
  X-Version-ID: abc123def456 (or null if never edited)
  X-Mask-Count: 145
  X-Flow-Method: dis
  X-Generated-At: 2025-01-15T12:34:56Z

Body: Binary .tgz archive containing:
  - metadata.json
  - frame_000001.webp (if mask exists)
  - frame_000002.webp
  - ...

Response 202 (tracking in progress):
{
  "status": "pending",
  "error_code": "TRACK_PENDING"
}

Response 500 (tracking failed):
{
  "status": "failed",
  "error_code": "TRACK_FAILED",
  "error_message": "Details here"
}
```

**metadata.json structure:**
```json
{
  "study_uid": "1.2.840...",
  "series_uid": "1.2.840...",
  "version_id": "abc123..." or null,
  "flow_method": "dis",
  "generated_at": "2025-01-15T12:34:56Z",
  "frame_count": 150,
  "mask_count": 145,
  "frames": [
    {
      "frame_number": 0,
      "has_mask": true,
      "is_annotation": true,
      "label_id": "L_13yPql",
      "filename": "frame_000001.webp"
    },
    {
      "frame_number": 1,
      "has_mask": true,
      "is_annotation": false,
      "filename": "frame_000002.webp"
    },
    {
      "frame_number": 5,
      "has_mask": false,
      "is_annotation": true,
      "label_id": "L_75K42J"
    }
  ]
}
```

### Annotation Submission Format

**POST masks (upload .tgz, trigger retrack):**
```http
POST /api/masks/{study_uid}/{series_uid}
Authorization: Bearer {MDAI_TOKEN}
Content-Type: application/x-tar+gzip
X-Previous-Version-ID: abc123def456 (or null)
X-Editor: user@example.com

Body: Binary .tgz archive containing:
  - metadata.json
  - frame_000001.webp (edited mask)
  - frame_000005.webp
  - ...

Response 200 (success):
{
  "success": true,
  "version_id": "def456ghi789",
  "retrack_queued": true,
  "queue_position": 2
}

Response 409 (version conflict):
{
  "error_code": "VERSION_MISMATCH",
  "current_version": "xyz789",
  "your_version": "abc123",
  "message": "Someone else edited this series. Please reset and re-apply changes."
}

Response 409 (retrack in progress):
{
  "error_code": "RETRACK_IN_PROGRESS",
  "message": "This series is currently being retracked. Wait or reload."
}
```

**Annotation metadata.json:**
```json
{
  "study_uid": "1.2.840...",
  "series_uid": "1.2.840...",
  "previous_version_id": "abc123..." or null,
  "editor": "user@example.com",
  "edited_at": "2025-01-15T12:45:00Z",
  "annotations": [
    {
      "frame_number": 0,
      "label_id": "L_13yPql",
      "filename": "frame_000001.webp"
    },
    {
      "frame_number": 4,
      "label_id": "L_13yPql",
      "filename": "frame_000005.webp"
    },
    {
      "frame_number": 10,
      "label_id": "L_75K42J",
      "filename": null
    }
  ]
}
```

### Version-Based Conflict Prevention

Uses optimistic locking via version IDs:
- Each mask set has unique version_id: `hash(server_received_at_timestamp + user_email)`
- Unedited series have `null` version ID
- Client includes previous_version_id when saving
- Server validates: previous_version_id must match current version
- On mismatch: server rejects save (HTTP 409), client shows conflict warning
- User must reset to latest version, then re-apply edits
- Collision handling: Treat as generic server error (astronomically unlikely), user retries

## Three Annotation Types

Server uses ONLY these annotation types from MD.ai:

1. **LABEL_ID** (`L_13yPql`): Human-verified fluid mask (green in UI)
2. **EMPTY_ID** (`L_75K42J`): Human-verified "no fluid" frame (no mask)
3. **TRACK_ID**: Server-generated tracked mask (orange in UI)

**Critical**: MD.ai may contain TRACK_ID or other annotation types in dataset, but server IGNORES these for initial tracking. Only LABEL_ID and EMPTY_ID are used as inputs to tracking algorithm.

**Client behavior:**
- Trust server masks as authoritative
- NEVER fall back to MD.ai annotations for tracking
- Use "Mark as Empty" button to submit EMPTY_ID (clears mask on frame)

## Data Storage

### Server Storage Structure

**Filesystem-based** (NO SQLite):

```
server_state/
├── series/
│   ├── {study_uid}_{series_uid}/
│   │   ├── metadata.json           # Series info, exam_id, completion status, tracking status
│   │   ├── version_current.json    # Current version ID, editor, timestamp
│   │   ├── version_temp.json       # Temporary version during retrack (deleted on success/failure)
│   │   ├── activity.json           # Last 2 users' activity timestamps
│   │   └── edit_history.json       # Edit log (optional, for audit)
├── activity_log.json               # Recent activity across all series (for "next" selection)
└── retrack_queue.json              # Pending retracks (FIFO)
```

**Mask storage** (same as current `output/` structure):
```
output/
├── dis/                            # Flow method
│   ├── {study_uid}_{series_uid}/
│   │   ├── identity.yaml           # Metadata
│   │   ├── masks/                  # Current production masks
│   │   │   ├── frame_000001.webp
│   │   │   └── ...
│   │   ├── masks_temp/             # Temporary masks during retrack
│   │   │   └── ...                 # (promoted to masks/ on success, deleted on failure)
│   │   └── masks.tgz               # Archive for serving (generated from masks/)
```

### Client Storage Structure

```
client_cache/
├── data/                           # MD.ai dataset (videos downloaded via SDK)
├── frames/                         # Extracted video frames (PNG)
│   ├── {study_uid}_{series_uid}/
│   │   ├── frame_000001.png
│   │   └── ...
└── masks/                          # Current masks from server (WebP)
    ├── {study_uid}_{series_uid}/
    │   └── ... (extracted from .tgz)
```

## Server Error Codes

**Client looks up codes to display appropriate UI:**

- `TRACK_PENDING`: ☕ "Server is preparing masks..."
- `TRACK_PROCESSING`: ☕ "Server is preparing masks..."
- `TRACK_FAILED`: 💩🔥 "Tracking failed for this series. Contact admin."
- `RETRACK_PENDING`: ⏳ "Retracking..." (show hourglass)
- `RETRACK_PROCESSING`: ⏳ "Retracking..." (show hourglass)
- `RETRACK_FAILED`: 💩🔥 "Retrack failed. Contact admin."
- `VERSION_MISMATCH`: "Someone else edited this series. Please reset and re-apply changes."
- `RETRACK_IN_PROGRESS`: "This series is currently being retracked. Wait or reload."
- `ALL_DONE`: 🎉 "All done! No more videos to annotate."
- `ALL_ACTIVE`: ⏳ "Almost done! All remaining videos are being edited by others."

## New Features

### A. Exam ID Selection

- **Source**: `exam_number` field from MD.ai annotations JSON
- **Hierarchy**: Projects → Datasets → Exams → Series (usually 1 series per exam)
- **Uniqueness**: Exam ID + Series Number guaranteed unique within dataset
- **Display**: "Exam 194, Series 1" (human-readable) instead of long series UID
- **Navigation**: List series by exam_id within dataset

**identity.yaml example:**
```yaml
study_instance_uid: 1.2.826.0.1.3680043.8.498.84378034445648120915087955063435521251
series_instance_uid: 1.2.826.0.1.3680043.8.498.13186914227253558193575435031594620951
exam_number: 194
dataset_name: PECARN Video
dataset_id: D_V688LQ
```

### B. Smart "Next" Selection

- **Server chooses next video** (not client-side)
- **Skip recently viewed**: Avoid series viewed by anyone in last N minutes (configurable)
- **Fallback**: If all recently viewed, select video with longest time since last view
- **Keep-alive**: 30-second pings update activity timestamp (proxy metric, not stateful)
- **Activity tracking**: Server tracks last 2 users' activity timestamps per series

**API response:**
```json
{
  "study_uid": "...",
  "series_uid": "...",
  "exam_id": 194,
  "series_number": 1,
  "last_active_by": "user@example.com",
  "last_active_at": "2025-01-15T13:00:00Z",
  "tracking_status": "completed"
}
```

**No available series:**
```json
{
  "no_available_series": true
}
```

### C. Completion Tracking

- **Mark as "done"**: Remove from active rotation
- Completed videos no longer served by "next" button
- **Later**: Review mode to browse completed videos
- Allows focused annotation workflow on incomplete videos

## iPad App Architecture

**Critical**: iPad CANNOT just use Safari because:
- MD.ai Python SDK required for video download (no JS client available)
- App handles: config, dataset download, video-to-frame parsing
- WebView displays server UI after setup

**App Structure:**
1. Native iPad app wrapper
2. Python backend (Pythonista or bundled interpreter)
3. MD.ai dataset download via Python SDK
4. Video frame extraction locally (PNG format)
5. Server connection for mask fetch/submit only
6. WebView serves existing viewer.js/viewer.css UI
7. Local frame serving to WebView (file:// or localhost)

**Deployment:**
- Python runs in background for MD.ai operations
- Most UI rendered in WebView (reuse existing viewer.js)
- Touch controls already supported by existing CSS
- Videos NEVER transmitted to server (downloaded directly from MD.ai)

## Configuration

**Shared (dot.env):**
- `MDAI_TOKEN`: MD.ai API access token
- `PROJECT_ID`, `DATASET_ID`: MD.ai project identifiers
- `LABEL_ID`: Free fluid label identifier (`L_13yPql`)
- `EMPTY_ID`: No-fluid frame label identifier (`L_75K42J`)
- `FLOW_METHOD`: Optical flow method (default: `dis`)

**Server-specific:**
- `SERVER_PORT`: Default 5000
- `SERVER_HOST`: Default 0.0.0.0
- `RECENT_VIEW_THRESHOLD_MINUTES`: Default 60 (for conflict warnings)
- `RETRACK_WORKERS`: Default 2 (parallel retrack processing)
- `MASK_STORAGE_PATH`: Where to store tracked masks

**Client-specific:**
- `CLIENT_PORT`: Default 8080
- `SERVER_URL`: e.g., https://goobusters.lab.example.com
- `USER_EMAIL`: Identifier for this client (used in edit history)
- `VIDEO_CACHE_PATH`: Local MD.ai dataset location

## Implementation Details

### Initial Mask Generation

- Server runs `track.py` logic with DIS tracking method
- Triggered **lazily** on first series request (not upfront for entire dataset)
- Uses ONLY LABEL_ID and EMPTY_ID annotations from MD.ai (ignores TRACK_ID)
- On failure: Server stores error state with error_code
- Client displays ☕ (startup), 💩🔥 (failure), or ⏳ (in-progress) based on error_code
- No masks available until tracking succeeds

### Retrack Behavior

- **UI indicator**: Large ⏳ emoji while waiting
- **Typical duration**: 15-30 seconds
- **Client timeout**: 2-3 minutes
- **On timeout**: Offer "Reset" or "Keep Waiting" options
- **Server failure**: Explicit failure state passed to client (no auto-timeouts)
- **Parallel support**: Multiple series can retrack simultaneously (up to RETRACK_WORKERS)

### Dual Version Storage During Retrack

- **Concurrent save protection**: Reject saves while temp version exists
- **Scenario**: User A saves → retrack starts → temp version created
- **User B tries to save**: Server detects temp version → HTTP 409 Conflict
- **Timeout**: Temp version deleted after 5 minutes if retrack stalls
- **Storage**: `masks_temp/` directory and `version_temp.json` during retrack
- **Success**: Promote `masks_temp/` to `masks/`, delete temp version
- **Failure**: Delete `masks_temp/`, keep production masks, delete temp version

### Server Startup Sequence

1. Initialize filesystem storage structure
2. Download MD.ai dataset (incremental update if already exists)
3. Populate series metadata from annotations
4. Start API server (don't wait for tracking)
5. Track series lazily on first request

**Status endpoint:**
```http
GET /api/status

Response:
{
  "ready": true,
  "series_pending": 10,
  "series_completed": 140,
  "series_failed": 2
}
```

## Success Criteria

**Core Functionality:**
- [ ] Server tracks and serves masks for all videos
- [ ] Client downloads videos via MD.ai SDK (not from server)
- [ ] Client fetches and displays server masks
- [ ] Client trusts server masks, never falls back to MD.ai annotations
- [ ] Annotation changes trigger retrack on server (async, FIFO)
- [ ] Reset functionality reverts to server state
- [ ] Parallel retrack supports 3-6 concurrent users

**Navigation & Selection:**
- [ ] Exam ID selection (human-readable instead of series UID)
- [ ] "Next" button uses smart server-side selection
- [ ] Skip recently viewed series (configurable threshold)
- [ ] Fallback to longest-since-viewed when all recent
- [ ] Mark series as "done" to remove from rotation

**Conflict Prevention:**
- [ ] Version ID validation on save
- [ ] Display last editor and timestamp before opening
- [ ] Strong warning if edited within last hour
- [ ] Reject save if previous_version_id doesn't match
- [ ] Flag conflict and require resend on mismatch

**Authentication & Security:**
- [ ] MD.ai token authentication on all API endpoints
- [ ] HTTPS deployment works with Caddy
- [ ] Token validation against MD.ai API

**Platform Support:**
- [ ] Works on desktop browsers (Chrome, Firefox, Safari)
- [ ] iPad app with MD.ai SDK integration
- [ ] iPad can download videos and extract frames locally
- [ ] Multiple users can work on different series simultaneously

**Performance:**
- [ ] Mask fetch < 2s
- [ ] Retrack completes < 60s per video
- [ ] Parallel retrack doesn't degrade performance

## Project Structure

```
goobusters/
├── server/                    # NEW: Server component
│   ├── api/
│   │   ├── __init__.py
│   │   ├── auth.py           # Authentication endpoints
│   │   ├── series.py         # Series navigation endpoints
│   │   ├── masks.py          # Mask serving/submission endpoints
│   │   └── retrack.py        # Retrack queue endpoints
│   ├── storage/              # Filesystem-based storage
│   │   ├── series_manager.py
│   │   ├── version_manager.py
│   │   └── retrack_queue.py
│   ├── server.py             # Main server entrypoint
│   ├── retrack_worker.py     # Background retrack processor
│   └── dot.env.server        # Server-specific config
│
├── client/                    # NEW: Client app
│   ├── static/               # Reuse existing
│   │   ├── css/viewer.css
│   │   └── js/viewer.js      # Enhanced with server API calls
│   ├── templates/            # Reuse existing
│   │   └── viewer.html
│   ├── client.py             # Minimal Flask server (iPad app backend)
│   ├── mdai_client.py        # MD.ai dataset download
│   └── dot.env.client        # Client-specific config
│
├── lib/                       # SHARED: Tracking logic
│   ├── __init__.py
│   ├── multi_frame_tracker.py
│   ├── opticalflowprocessor.py
│   ├── optical.py
│   ├── performance_config.py
│   └── video_capture_manager.py
│
├── data/                      # MD.ai dataset cache (both server & client)
├── output/                    # Current tracking output (becomes server-only)
├── track.py                   # LEGACY: Keep for testing
├── app.py                     # LEGACY: Keep for testing
├── DISTRIBUTED_ARCHITECTURE.md
└── dot.env                    # Shared config
```

## Summary

**Key Design Choices:**

1. **PHI Compliance**: All clients download videos from MD.ai using SDK, server NEVER serves PHI
2. **Data Transfer**: .tgz archives for masks (not individual API requests), avoids 100-1000x requests per series
3. **Version Control**: `hash(timestamp + email)` for version IDs, optimistic locking with collision detection
4. **Storage**: Filesystem-based (NO SQLite), uses existing patterns from current codebase
5. **Error Handling**: Server returns codes, client looks up emoji/message (☕, 💩🔥, ⏳, 🎉)
6. **Client Architecture**: iPad app with Python backend for MD.ai SDK, WebView for UI
7. **Smart Navigation**: Server-side selection, skip recently viewed, keep-alive pings
8. **Tracking**: Lazy on-demand (not all upfront), DIS method, FIFO retrack queue with parallel support
9. **Annotations**: Three types (LABEL_ID, EMPTY_ID, TRACK_ID), server ignores MD.ai TRACK_ID for initial tracking
10. **Security**: HTTPS via Caddy with Let's Encrypt (not optional)
