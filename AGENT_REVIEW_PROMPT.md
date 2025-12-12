## Fresh Agent Review Prompt (Goobusters)

Copy/paste this entire prompt to a new coding agent.

### Prompt
You are a senior Python engineer joining mid-stream.

**Repo:** `goobusters`

**Primary spec:** `DISTRIBUTED_ARCHITECTURE.md` (authoritative, plus clarifications in `DECISION_REGISTER.md`).

**Hard requirements:**
- No mocks. No fake data. No placeholder tests.
- Prefer config-driven behavior over hardcoding.
- KISS/DRY.

#### What you need to do
1. Assess current implementation vs spec for the distributed server (startup tracking, mask serving, retracking).
2. Run the real-data end-to-end test runner and make it pass:
   - `uv venv --seed`
   - `uv pip install -r requirements.txt`
   - Ensure `dot.env` exists (copy from `dot.env.example` and fill real values)
   - `uv run python scripts/test_full_system.py`
3. Fix the failures with real behavior (no shortcuts).
4. When behavior is ambiguous, record the decision in `DECISION_REGISTER.md`. 

---

## What we’ve been working on (tasks + intent)

### 1) Core Server Operation (startup)
- Goal: On server startup (before serving clients), ensure the MD.ai dataset is present and generate initial masks for all trackable series.
- Key spec section: `DISTRIBUTED_ARCHITECTURE.md` → “Core Server Operation: Mask Generation and Flow”.
- Nuance: dataset downloads are large; startup should be idempotent and avoid re-downloading unless explicitly forced.

### 2) Logging
- Goal: eliminate print spam; log to a file by default under `server/log/` with sortable timestamp names (e.g. `YYMMDD-HHMMSS.log`).

### 3) Completion markers and serving precedence
- Goal: treat `masks.tgz` + `metadata.json` as completion artifacts.
- Serving precedence: if retrack output exists, serve it first (`retrack/masks.tgz` > `masks.tgz`).

### 4) API correctness (mask download + upload + retrack status)
- Goal endpoints:
  - `GET /api/masks/{study}/{series}`: serve prebuilt `.tgz` if present; otherwise return 202 and trigger tracking if needed.
  - `POST /api/masks/{study}/{series}`: accept edited `.tgz`, enforce optimistic lock with `X-Previous-Version-ID`, enqueue retrack.
  - `GET /api/retrack/status/{study}/{series}`: report queue/processing/completed/failed.

### 5) “Same logic as track.py” filtering
- Goal: server’s notion of trackable series must match `track.py`:
  - series has at least one `LABEL_ID` or `EMPTY_ID` annotation
  - and corresponding video exists on disk

### 6) One metadata interpretation
- Goal: `metadata.json` uses one canonical shape: a `frames` array.

### 7) Testing discipline
- Goal: `scripts/test_full_system.py` orchestrates server + worker + tests + shutdown, using real data.

---

## Known pain points observed
- Worktree friction: changes were often made inside Cursor worktrees (`.cursor/worktrees/...`) and were hard to apply to the main repo. Prefer working directly in `/Users/christopher/Documents/goobusters`.
- Missing config: server fails fast if required keys are missing (expects `dot.env` or env vars).
- Import/package issues: server often needs to run as a module (`python -m server.server`) and `server/` should be a Python package.
- Requirements drift: some variants of `requirements.txt` were missing `httpx`, breaking the test runner.

---

## Concrete checklist for review
1. Run `uv run python scripts/test_full_system.py` and capture the first failing stack trace.
2. Confirm these areas match the spec intent:
   - `server/startup.py` (dataset reuse vs re-download; trackable filtering)
   - `server/server.py` (logging + performs startup init before serving)
   - `server/api/routes.py` (GET masks, POST masks, retrack status)
   - `server/tracking_worker.py` + `server/retrack_worker.py` (archive built once on completion; retrack writes under `retrack/`)
   - `lib/mask_archive.py` (archive format + metadata schema)
   - `lib/config.py` (server isolation under `server/`, avoid duplicated `flow_method`)
3. Ensure `scripts/test_startup_verification.py` validates trackable series (track.py logic), not all series in the dataset index.
4. Ensure server does not re-download MD.ai exports if they already exist.

---

## Suggested next step after tests are green
Re-review `DISTRIBUTED_ARCHITECTURE.md` roadmap and implement remaining missing server pieces, expanding end-to-end tests accordingly.
