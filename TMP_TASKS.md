1. POST masks endpoint — accept edited masks, extract .tgz, validate version
2. Retrack queue + worker — port retrack logic from app.py, adapt to new storage, queue system
3. Lazy tracking trigger — background worker for initial tracking on first request
4. Frontend integration — update viewer.js to use new APIs (can happen in parallel)
5. iPad app foundation — start C1 (runtime wrapper) when ready