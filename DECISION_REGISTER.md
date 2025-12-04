# Decision Register

| ID | Date | Decision / Assumption | Rationale | Impact |
| --- | --- | --- | --- | --- |
| DR-001 | 2025-12-04 | iPad build ships with Pyto-style embedded Python runtime (or Pyodide fallback) to guarantee MD.ai SDK + ffmpeg execution. | Pythonista is aging and cannot reliably bundle native wheels; Pyto/Pyodide let us pin Python 3.11 and required deps. | Influences client packaging tasks (C1-C3) and ensures MD.ai + OpenCV run identically on iPad and desktop. |
| DR-002 | 2025-12-04 | Both client and server use the shared Python `tarfile` utility for `.tgz` mask archives instead of shelling out. | Keeps packaging logic DRY, avoids platform-specific `tar` flags, and simplifies schema validation hooks. | Drives shared task SH3 and enforces identical archive manifests across platforms for testing. |
| DR-003 | 2025-12-04 | Typed config loader (`lib/config.py`) becomes single source for `dot.env`, with scoped prefixes per role. | Prevents divergence between server/client env files and provides validation before processes start. | Required for SH1, unlocks deterministic deployments and consistent test credentials. |
| DR-004 | 2025-12-04 | Server bootstrapping requires a hydrated MD.ai dataset; we fail fast if `find_annotations_file` cannot locate exports. | Guarantees real-series metadata instead of placeholder manifests and enforces "no mocks" policy. | Ops run `uv run python3 track.py` (or MD.ai download) before launching server to generate `server_state`. |
