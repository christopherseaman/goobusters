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

## Core Server Operation: Mask Generation and Flow

**This section defines the fundamental operation of the server - how masks are generated, served, and retracked. This is core to the specification.**

### Mask Flow: Initial Tracking → Client → Retracking

**Complete Flow**:

1. **Server Startup (Before serving clients)**:
   - Server downloads MD.ai dataset using token + project_id + dataset_id
   - Server downloads videos and annotations to `data/` directory
   - Server builds series index from downloaded annotations
   - **Server generates masks for all series** (on startup):
     - For each series: Read MD.ai annotations (polygons for `LABEL_ID` and `EMPTY_ID`)
     - Convert polygons → mask images (binary masks)
     - Run optical flow tracking → generates mask images for all frames
     - Save all mask images as `.webp` files to `output/{flow_method}/{study_uid}_{series_uid}/masks/`
     - Update series metadata: `tracking_status = "completed"`
   - Server is now ready to serve masks immediately

2. **Client Downloads Masks**:
   - Client requests masks: `GET /api/masks/{study}/{series}`
   - Server reads `.webp` files from disk
   - Server builds metadata.json with `frames` array (all frames with `is_annotation`, `label_id`, `filename`)
   - Server packages masks + metadata into `.tgz` archive
   - Server sends archive to client

3. **Client Edits Masks**:
   - Client extracts archive (has `.webp` files + `metadata.json`)
   - Client modifies mask images (edits `.webp` files)
   - Client updates `metadata.json` (preserves `frames` array structure)

4. **Client Uploads Edited Masks**:
   - Client packages edited masks + updated metadata into `.tgz` archive
   - Client uploads: `POST /api/masks/{study}/{series}` with `X-Previous-Version-ID` header
   - Server extracts archive (has edited `.webp` files + updated `metadata.json`)
   - Server queues retrack job

5. **Server Retracks**:
   - Retrack worker loads edited mask images (`.webp` files) from uploaded archive
   - Worker filters `frames` array for `is_annotation=true` entries
   - Worker loads actual mask image data from `.webp` files
   - Worker uses those masks as input for optical flow tracking
   - Worker generates new tracked masks for all frames
   - Worker saves new masks, promotes to production

**Key Points**:
- **Server startup**: Downloads MD.ai dataset (videos + annotations) using token + project + dataset config
- **Server startup**: Generates masks for ALL series (MD.ai polygons → mask images → tracking) before serving clients
- **Client download**: Server reads pre-generated mask images from disk (masks already exist)
- **Retracking**: Client's edited mask images → server uses as input
- **Mask images**: ALWAYS generated locally by server (never downloaded from MD.ai)
- **Tracking timing**: Both dataset download AND mask generation happen on startup; server ready when all masks generated

## Implementation Roadmap

| Phase | Focus | Key Deliverables | Validation (real runs only) |
| --- | --- | --- | --- |
| 0. Baseline & Config Sync | Confirm shared `dot.env`, MD.ai credentials, disk layout | Fresh `dot.env` copies (`dot.env.server`, `dot.env.client`), documented secrets loading, hydrated `data/` cache | `uv run python3 track.py --help` (ensures deps), MD.ai SDK smoke test downloading single study |
| 1. Server Core | Flask API skeleton, storage scaffolding, MD.ai dataset download on startup, initial mask generation on startup | `/api/status`, `/api/series`, `/api/masks` (GET) read-only paths wired to filesystem, MD.ai dataset sync, tracking workers for all series | Server downloads real dataset on startup, generates masks for all series on startup, then serves masks immediately |
| 2. Client Foundation | iPad Python backend + WebView hooking, MD.ai download loop | `client/client.py` serving viewer assets, MD.ai SDK-based downloader, frame extractor feeding WebView | Run client on desktop simulator with MD.ai token, verify frames rendered with placeholder masks fetched from server |
| 3. Collaboration & Retrack | Versioning, conflict handling, retrack queue, activity pings | Version files, `/api/masks` POST path, `/api/retrack/status`, keep-alive endpoint, queue workers (>= `RETRACK_WORKERS`) | Multi-user test: two real clients editing same series, observe 409 conflict + retrack completion with true masks |
| 4. Hardening & Ops | HTTPS, observability, disaster recovery, completion UX | Caddy config, structured logging, backup scripts, completion state flows, alerting on worker failure | End-to-end test on staging: full annotation loop incl. mark-complete, TLS validation via `curl https://.../api/status` |

**Exit criteria per phase**: No phase is “done” until commands listed in Validation are executed against the real pipeline—no mocks, no synthetic payloads.

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
- Initial tracking: Generate masks for all series on startup (before serving clients)
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

#### Server Work Packages

| ID | Task | Description | Dependencies | Validation |
| --- | --- | --- | --- | --- |
| S1 | Storage bootstrap | Implement `server/storage/series_manager.py` to materialize `server_state/` tree, migrate legacy `output/` layout | Access to MD.ai metadata json, filesystem permissions | Run `python server/storage/series_manager.py --init` then `ls server_state/series | wc -l` to ensure entries created |
| S2 | Auth middleware | Build token validator hitting MD.ai `/me` endpoint once per token + cache result for 15 min | MD.ai SDK, shared `dot.env` secrets | `curl -H "Authorization: Bearer $MDAI_TOKEN" /api/auth/validate` returns real MD.ai profile json |
| S3 | Series navigation API | Wire `/api/series`, `/api/series/next`, `/api/series/{...}` to storage, include activity metadata | S1, S2 | Manual GETs via `httpie` showing real series data from dataset snapshot |
| S4 | Mask serving | Package current `output/dis/{series}` into .tgz via `tarfile`, stream with headers, fall back to pending/failure payloads | S1, tracking outputs | Download `.tgz`, untar, diff vs `output/` mask files byte-for-byte |
| S5 | Mask submission + retrack queue | Accept uploads, enforce version checks, enqueue retrack jobs, expose `/api/retrack/status` | S1-S4, queue implementation | Submit actual edited mask archive, observe new `version_temp.json`, retrack worker promoting `masks_temp` after running `track.py` |
| S6 | Activity & completion | Keep-alive handler, `complete` / `reopen` endpoints, smart-selection heuristics (`RECENT_VIEW_THRESHOLD_MINUTES`) | S3, S5 | Simulate two clients pinging every 30s, assert oldest inactive series served via `/api/series/next` |
| S7 | Observability & TLS | Structured JSON logs, health/status endpoints, Caddy config with HTTPS + MD.ai token passthrough | All prior | `curl -v https://server/api/status` shows valid cert, logs contain request IDs |

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
- Detect/handle newer dataset versions (client vs server) and coordinate refresh

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

#### Client Work Packages

| ID | Task | Description | Dependencies | Validation |
| --- | --- | --- | --- | --- |
| C1 | Runtime wrapper | Package Python runtime (Pyto or Pyodide-on-iOS) + native shell that launches backend and WebView | Apple dev account, shared config loader | Build IPA, install on test iPad, confirm Python process starts and serves local HTTP port |
| C2 | MD.ai sync | Implement `client/mdai_client.py` to auth, list series, download exams, and store under `client_cache/data` | C1, MD.ai SDK | Download real exam (>=500 MB) over Wi-Fi, hash compare with MD.ai CLI output |
| C3 | Frame extraction | Use ffmpeg/opencv to split MP4 to PNG frames, maintain manifest consumed by WebView; auto-extract after dataset sync | C2, ffmpeg binary bundled | Run extractor on downloaded video, ensure frame count matches metadata and PNGs accessible via `file://` path; verify post-sync auto-extraction |
| C4 | Viewer integration | Reuse `static/js/viewer.js` + CSS inside WebView, inject fetch adapters hitting server APIs | C1 | Manual interaction: load video, see masks overlay (initially from GET masks), draw annotations with touch |
| C5 | Local edit cache | Persist edits in LocalStorage + filesystem, support reset + unsaved indicator | C4 | Start edit, kill app, relaunch → edits still present until Save; Reset clears |
| C6 | Save + retrack UX | Package edits into .tgz, attach `X-Previous-Version-ID`, display retrack spinner, poll `/api/retrack/status` | C4, C5, server S5 | Real save cycle using staging server; verify new version ID returned and spinner hides when GET masks matches |
| C7 | Conflict + completion flows | Surface warnings (recent activity, version mismatch), mark complete UI, next navigation linking to server selection | C6, server S6 | Two devices editing same series; second sees warning + 409 toast, Next button fetches new series metadata |

### Shared Components & Testing Discipline

| ID | Task | Description | Owners | Validation |
| --- | --- | --- | --- | --- |
| SH1 | Config refactor | Derive `dot.env.server` / `dot.env.client` from shared template, enforce typed loader shared via `lib/config.py` | Server & Client | Run `python lib/config.py --dump` on both sides; diff outputs except scoped keys |
| SH2 | Tracking library contract | Expose `run_tracking(study_uid, series_uid, flow_method)` wrapper to be imported by server workers, ensuring no duplication of `track.py` options | Server workers | Execute wrapper directly against MD.ai series, compare outputs to legacy `track.py` run |
| SH3 | Artifact packaging utilities | Shared module to tar/un-tar mask archives, emit metadata.json, validate schema | Server & Client | Round-trip test: client creates archive → server validates → server re-archives → client diff ensures identical manifest |
| SH4 | Real test matrix | Document required manual/automated runs per phase (tracking smoke, retrack conflict, offline resilience) | All | Checklist stored in repo, each run referencing actual command + timestamp, no synthetic mocks |

All functionality must be verified with real MD.ai downloads and actual mask archives. If production data is unreachable, the corresponding task remains blocked rather than using fake payloads.

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
- 💩🔥 "Tracking failed for this series. Contact admin." (failed initial tracking on startup)
- Note: Masks should be available immediately after server startup completes

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

### Initial Mask Generation (On Startup)

**When it happens**: On server startup, before serving any clients. Server cannot serve masks without generating them first.

**Process**:
1. Server reads MD.ai annotations (polygons for `LABEL_ID` and `EMPTY_ID` only)
2. Server converts polygons to mask images (binary masks)
3. Server runs optical flow tracking to generate mask images for `TRACK_ID` on all frames
4. Server saves all mask images as `.webp` files to `output/{flow_method}/{study_uid}_{series_uid}/masks/`
5. Server updates series metadata: `tracking_status = "completed"`

**Key points**:
- Initial tracking uses MD.ai annotations (polygons) as input, NOT mask images
- Mask images are generated locally by the server
- Tracking happens on startup for ALL series (time-consuming but necessary)
- All frames get mask images (either from annotations or tracked)
- Server ready to serve masks immediately after startup completes

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
2. **Download MD.ai dataset** (incremental update if already exists):
   - Use MD.ai token + project_id + dataset_id from config
   - Connect to MD.ai using SDK: `mdai.Client(domain=DOMAIN, access_token=MDAI_TOKEN)`
   - Download/update project: `client.project(project_id=PROJECT_ID, dataset_id=DATASET_ID, path=DATA_DIR)`
   - This downloads videos and annotations to `data/` directory (same as `track.py`)
   - **Note**: Currently server expects dataset to exist; implementation needs to add this download step
3. **Populate series metadata** from downloaded annotations:
   - Build series index from annotations JSON (already implemented in `SeriesManager._build_index_from_annotations`)
   - Create series metadata files in `server_state/series/`
4. **Generate masks for all series** (on startup, before serving):
   - For each series in the dataset:
     - Read MD.ai annotations (polygons for `LABEL_ID` and `EMPTY_ID`)
     - Convert polygons → mask images (binary masks)
     - Run optical flow tracking → generates mask images for all frames
     - Save all mask images as `.webp` files to `output/{flow_method}/{study_uid}_{series_uid}/masks/`
     - Update series metadata: `tracking_status = "completed"`
   - **Note**: This is time-consuming (15-25 seconds per video), but necessary - server cannot serve masks without generating them first
   - Can run in parallel workers to speed up (similar to retrack queue)
5. **Start API server** (after masks are generated, or start API immediately and track in background)

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

## Validation Plan

- **Tracking pipeline smoke**: `uv run python3 track.py --study $TEST_STUDY_UID --series $TEST_SERIES_UID --flow dis` executes end-to-end after every change under `lib/` to guarantee parity with production outputs.
- **Server API contract**:
  - Start server via `uv run python3 server/server.py`.
  - Issue `curl` requests for `/api/status`, `/api/series/next`, `/api/masks/{study}/{series}` pulling real .tgz files and comparing against `output/`.
  - Upload edited archive with `http --form POST ... < archive.tgz` to ensure retrack queue writes `version_temp.json` and spawns worker invoking real tracking code.
- **Client-device loop**:
  - Launch iPad build, authenticate with MD.ai, download actual study, confirm PNG extraction count equals `metadata.json.frame_count`.
  - Save edits to staging server, observe retrack completion and updated masks.
- **Conflict + completion scenario**:
  - Run two physical devices, have both save; second must receive HTTP 409 from real server.
  - After completion, `GET /api/series/next` should skip the finished series.
- **TLS + observability**:
  - Deploy Caddy reverse proxy, run `curl -v https://server/api/status` from fresh machine to validate certificate chain.
  - Tail structured logs ensuring each request logs command name, MD.ai user email (hashed), and duration.

Testing must always target real MD.ai downloads and mask artifacts; placeholder payloads or mocked APIs are not accepted evidence of functionality.

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
8. **Tracking**: Initial masks generated on startup for all series, DIS method, FIFO retrack queue with parallel support
9. **Annotations**: Three types (LABEL_ID, EMPTY_ID, TRACK_ID), server ignores MD.ai TRACK_ID for initial tracking
10. **Security**: HTTPS via Caddy with Let's Encrypt (not optional)
