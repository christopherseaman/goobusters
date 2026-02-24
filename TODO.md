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

## Blocked

- [ ] (BLOCKED: REVISIT IF ANOTHER EXAMPLE ARISES) No LABEL_ID but still being included? 1.2.826.0.1.3680043.8.498.90435151582213456262290795805216481896_1.2.826.0.1.3680043.8.498.37052967828633660121479146607377040574
- [ ] (BLOCKED: NEED TEAM TO WEIGH IN ON METHODS) Send test annotations back to mdai for the Pelvic-1 dataset (MUST BE PELVIC-1 FOR TESTING, CHECK dot.env VARS TWICE)
