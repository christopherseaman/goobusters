# Goobusters iPad App

Native iOS app wrapper for the Goobusters annotation viewer.

## Setup

### 1. Prerequisites
- Xcode 15.0 or later
- iOS 17.0+ SDK
- Apple Developer account (Team ID: `KTGSS9PB3A`)

### 2. Simulator Setup

Run the setup script to boot an iPad Air simulator:

```bash
./setup-simulator.sh
```

Or manually:
1. Open Xcode
2. Window → Devices and Simulators
3. Create/boot an iPad Air simulator
4. Set orientation to Landscape (if desired)

### 3. Build and Run

1. Open `ios/Goobusters/Goobusters.xcodeproj` in Xcode
2. Select the "Goobusters" scheme
3. Select iPad Air simulator as destination
4. Press Cmd+R to build and run

## Project Structure

- `GoobustersApp.swift` - Main app entry point
- `GoobustersView.swift` - SwiftUI view with WebView
- `BackendManager.swift` - Manages Python backend process (TODO: implement)
- `Info.plist` - App configuration (bundle ID: `org.badmath.goobusters`)

## Current Status

- ✅ Xcode project created
- ✅ Basic SwiftUI app structure
- ✅ WebView configured to load `http://localhost:8080`
- ⏳ BackendManager needs implementation (Python runtime integration)
- ⏳ Python backend launch logic (Pyto vs Pyodide decision)

## Next Steps

1. **C1: Runtime wrapper** - Choose and integrate Python runtime (Pyto or Pyodide-on-iOS)
2. **C2: MD.ai sync** - Implement dataset sync in `BackendManager`
3. **C3: Frame extraction** - Bundle ffmpeg and implement frame extraction

See `TODO.md` for full task list.

