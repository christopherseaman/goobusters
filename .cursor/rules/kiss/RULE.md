---
alwaysApply: true
---

# Reminder to KISS/DRY

- reuse shared methods in lib/
- track.py works so lean on that implementation
- ALWAYS address the root cause of each issue
- ALWAYS test by (at minimum) running `uv run server/server.py -dk && uv run client/client.py -kd` and checking client/log/latest and server/log/latest