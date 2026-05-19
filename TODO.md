# Goobusters Tasks

## To Do

### Bugfixes

- [ ] Possible to have multitouch pinch/zoom for image/mask canvas without affecting toolbar buttons?
- [?] Frame image/mask canvas inside toolbars instead of having toolbars on top of image/mask

### UI/UX Improvements

- [ ] (lower priority) improved series navigation modal

### Onboarding & Auth

- [ ] Fetch active users from server to populate username dropdown dynamically
    - Add endpoint: GET /api/users/active (returns list of users from recent activity)
    - Use server list if available, fall back to fixed list on error
- [ ] Add "Other" option to username dropdown to allow custom name entry
    - Show text input when "Other" selected
    - Save custom name and add to dynamic list

### Data & Annotation Handling

- [ ] Handle series with no fluid annotations gracefully (serve blank masks/metadata so client can open; no tracking needed)
    - Empty frames (EMPTY_ID) already work; this is about series with zero annotations
    - Validate "No Fluid" frame annotation compatible with mdai json syntax

### Investigations

- [ ] Jumpy video in annotation editor app but not in tracked_video.mp4? Example: exam 19; 1.2.826.0.1.3680043.8.498.12762211632497404572246503032980657292_1.2.826.0.1.3680043.8.498.90262783102403545676047413537747709850

### Test Coverage

- [ ] Add test coverage — bugs keep reaching TestFlight because there's no automated layer between "code looks right locally" and "app runs in the iOS bundle / WKWebView / against the real server". Known examples this gap allowed through:
    - **iOS bundle artifact regressions** — `users.yaml` wasn't copied by `ios/scripts/bundle_python.sh` after the annotator-list refactor; `_load_annotators()` silently returned `[]` and the name dropdown rendered blank on fresh installs. A post-bundle smoke test that imports `lib.client.start` against the bundled paths and asserts critical files load would have caught it at build time.
    - **WKWebView API differences** — `alert()` is a silent no-op in WKWebView without a `WKUIDelegate`, so the `loadNextSeries() → no_available_series → alert()` path produced a blank app instead of a message. A WebView-context render test (or even a grep rule banning `alert()` in `static/js/`) would surface this class of issue.
    - **Server startup crash on stale mdai token** — `create_app(skip_startup=False)` runs `download_mdai_dataset()` before `app.run()`; an expired token raises during `_test_endpoint()`, Flask never binds, Cloudflare returns a fast 502, and the iOS app sits at "Connecting to server..." forever. A boot-time smoke test (server stands up against a fake mdai responder, or at minimum a `/healthz` ping in CI after restart) would distinguish "code change broke startup" from "operational state drifted".
    - **md.ai now 403s image export** (per `plan.md`) — vendored `mdai/client.py` `project()` always exports annotations + images together; image 403 crashes sync. No test pins the contract with the SDK, so policy changes upstream break us silently. Worth at least a contract test that calls `client.project(annotations_only=True)` against a known dataset.
    - **Shell pipeline exit-code masking** — `command | tee log` returns tee's exit code, so a failing `testflight_push.py` was reported as success once this session. Worth a lint/convention check (`set -o pipefail` in every script that pipes, or a CI rule that fails on missing `pipefail`).

## Blocked

- [ ] (BLOCKED: REVISIT IF ANOTHER EXAMPLE ARISES) No LABEL_ID but still being included? 1.2.826.0.1.3680043.8.498.90435151582213456262290795805216481896_1.2.826.0.1.3680043.8.498.37052967828633660121479146607377040574
- [ ] (BLOCKED: NEED TEAM TO WEIGH IN ON METHODS) Send test annotations back to mdai for the Pelvic-1 dataset (MUST BE PELVIC-1 FOR TESTING, CHECK dot.env VARS TWICE)
