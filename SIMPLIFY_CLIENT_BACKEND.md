# Client Backend Simplification Plan

## Overview

Simplify the client backend architecture by eliminating the proxy layer. The frontend should communicate directly with the server for all non-PHI operations, while the client backend focuses solely on PHI-related local operations.

## Current Architecture

```
Frontend (Browser)
    ↓
Client Backend (localhost:8080)
    ├─→ Proxies to Server (hivemind:5000) for masks, series, metadata
    ├─→ Serves local video frames (PHI)
    ├─→ Downloads from MD.ai (PHI)
    └─→ Extracts frames locally
```

## Target Architecture

```
Frontend (Browser)
    ├─→ Server (hivemind:5000) DIRECTLY for masks, series, metadata
    └─→ Client Backend (localhost:8080) for:
        ├─→ Local video frames (PHI)
        ├─→ MD.ai dataset sync (PHI)
        └─→ Frame extraction (PHI)
```

## Client Backend Responsibilities (Post-Simplification)

### KEEP: PHI-Related Operations
1. **MD.ai Dataset Sync**
   - Download videos and annotations from MD.ai API
   - Store in local cache (`client_cache/data/`)

2. **Frame Extraction**
   - Convert MP4 videos to WebP frames locally
   - Store frames in `client_cache/frames/`

3. **Local Frame Serving**
   - Serve video frames via HTTP to frontend
   - Endpoint: `GET /api/frames/{study_uid}/{series_uid}/{frame_number}`

4. **Static File Serving**
   - Serve HTML/CSS/JS for viewer UI

5. **User Settings**
   - Store/retrieve user preferences locally

### REMOVE: Proxy Operations
All proxy routes should be removed:
- `GET /proxy/<path:path>` → Frontend calls server directly
- `GET /api/masks/*` → Frontend calls server directly
- `POST /api/masks/*` → Frontend calls server directly
- `GET /api/series` → Frontend calls server directly
- `POST /api/series/*/activity` → Frontend calls server directly
- `POST /api/series/*/complete` → Frontend calls server directly
- `GET /api/video/*` (metadata) → Server has this from MD.ai annotations

## Implementation Steps

### 1. Update Frontend to Use Direct Server URLs

**Files to modify:**
- `ios/Goobusters/Goobusters/python-app/static/js/viewer.js`

**Changes:**
- Add server URL configuration (read from config or window object)
- Update all mask-related fetch calls to use server URL directly
- Update all series-related fetch calls to use server URL directly
- Keep frame-serving calls pointing to client backend (localhost:8080)

**Example:**
```javascript
// Before
fetch('/api/masks/...')

// After
fetch(`${SERVER_URL}/api/masks/...`)

// Frame serving stays local
fetch('/api/frames/...')  // Still localhost:8080
```

### 2. Remove Proxy Routes from Client Backend

**Files to modify:**
- `ios/Goobusters/Goobusters/python-app/lib/client/start.py`

**Routes to remove:**
- `/proxy/<path:path>`
- `/api/masks/<study_uid>/<series_uid>` (GET and POST)
- `/api/series`
- `/api/series/<study_uid>/<series_uid>/activity`
- `/api/series/<study_uid>/<series_uid>/complete`
- `/api/video/<method>/<study_uid>/<series_uid>` (metadata)

**Routes to keep:**
- `/api/frames/<study_uid>/<series_uid>/<int:frame_number>`
- `/api/dataset/sync` (if still using POST for sync trigger)
- Static file serving routes
- User settings routes (if any)

### 3. Update Configuration

**Files to modify:**
- `ios/Goobusters/Goobusters/python-app/dot.env.defaults`
- `ios/Goobusters/Goobusters/python-app/ios_config.py`

**Add:**
- `SERVER_URL` configuration to be injected into frontend

**Example in template:**
```html
<script>
    const SERVER_URL = '{{ server_url }}';
    const CLIENT_URL = 'http://localhost:8080';
</script>
```

### 4. Simplify Client Backend Dependencies

**After removing proxy functionality:**
- Remove any server-specific HTTP client code
- Remove caching logic for proxied data
- Simplify error handling (no need to handle server connection errors)

### 5. Update CORS Configuration on Server

**Files to modify:**
- `lib/server/start.py`

**Changes:**
- Ensure CORS allows requests from `localhost:8080` (iPad client frontend)
- Add appropriate CORS headers for cross-origin requests

### 6. Testing Checklist

- [ ] Frontend can fetch masks from server directly
- [ ] Frontend can submit edited masks to server directly
- [ ] Frontend can fetch series list from server directly
- [ ] Frontend can mark series complete via server directly
- [ ] Frontend can fetch frames from client backend (localhost:8080)
- [ ] MD.ai dataset sync still works through client backend
- [ ] Frame extraction still works through client backend
- [ ] No broken proxy routes (404s in console)
- [ ] CORS headers work correctly for cross-origin requests

## Benefits

1. **Simpler architecture** - Clear separation: PHI stays local, everything else goes to server
2. **Eliminates caching issues** - No more dual-layer caching between client backend and server
3. **Reduces code complexity** - Fewer routes, less error handling, clearer responsibilities
4. **Easier debugging** - Direct server communication visible in browser network tab
5. **Better PHI compliance** - Clearer boundary: client backend = PHI zone only

## PHI Compliance Verification

After simplification, verify:
- ✅ Videos never leave iPad
- ✅ Frames never sent to server
- ✅ Server only receives/serves masks (non-PHI)
- ✅ Client backend is the only component touching PHI data

## Rollback Plan

If issues arise:
1. Revert frontend changes (restore proxy calls)
2. Keep proxy routes in client backend
3. Document specific blockers for future attempts
