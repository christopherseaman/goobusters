# Goobusters iOS App - Python Runtime Integration

## ✅ Status: SUCCESS - Python is Running!

Python 3.14 is successfully running in the iOS simulator (iPad Air 11-inch M3) and serving HTTP requests.

### Verification Results

```bash
$ curl http://127.0.0.1:8080/healthz
OK - Python is running!

$ curl http://127.0.0.1:8080/
<html>
  <h1>🎉 Python is working on iOS!</h1>
  <p><strong>Python Version:</strong> 3.14.0 (main, Jan 12 2026, 19:33:34) [Clang 17.0.0]</p>
  ...
</html>
```

---

## Architecture Overview

### Components

1. **Python.xcframework** (`ios/Goobusters/Goobusters/Python.xcframework/`)
   - iOS device slice: `ios-arm64`
   - iOS simulator slice: `ios-arm64_x86_64-simulator`
   - Python 3.14 standard library
   - Build utilities for runtime installation

2. **Objective-C Bridge** (`BackendPythonRunner.mm`)
   - Wraps Python C API (`Python.h`)
   - Initializes Python interpreter with `PyConfig`
   - Executes Python scripts via `PyRun_SimpleFileExFlags`
   - Runs on dedicated serial dispatch queue

3. **Swift Interface** (`PythonBackendRunner.swift`)
   - Swift-friendly async API
   - Environment configuration (`PYTHONHOME`, `PYTHONPATH`)
   - Error handling with typed errors

4. **Application Manager** (`BackendManager.swift`)
   - SwiftUI `@ObservableObject` for UI integration
   - Health check polling
   - Status message updates

5. **Python Application Bundle** (`python-app/`)
   - Entry point: `start_server_simple.py` (currently active)
   - Full Flask server: `start_server.py` (ready for deployment)
   - iOS configuration: `ios_config.py`
   - Client libraries: `lib/client/`

### Build Process

The Xcode build includes a "Prepare Python Runtime" script phase that:

1. Detects simulator vs device build
2. Copies Python standard library to app bundle (`python/lib/`)
3. Converts `.so` extension modules to `.framework` bundles
4. Code signs all frameworks
5. Copies `python-app/` directory to app bundle

---

## Current Configuration

### Active Server: Simple HTTP Server

**File:** `python-app/start_server_simple.py`

**Why:** Minimal dependencies (no Flask, numpy, etc. required). Uses only Python standard library.

**Endpoints:**
- `GET /` - HTML page showing Python version and system info
- `GET /healthz` - Returns `200 OK` with plain text

**Code:**
```python
from http.server import HTTPServer, BaseHTTPRequestHandler

class SimpleHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/healthz":
            self.send_response(200)
            self.send_header("Content-Type", "text/plain")
            self.end_headers()
            self.wfile.write(b"OK - Python is running!\n")
        # ... (see file for full implementation)

def main(port=8080):
    server = HTTPServer(('127.0.0.1', port), SimpleHandler)
    server.serve_forever()
```

---

## Next Steps

### Option 1: Continue with Simple Server

The simple HTTP server is perfect for testing Python integration. To extend it:

1. Add more endpoints to `start_server_simple.py`
2. Implement annotation viewing/editing logic
3. Keep dependencies minimal (stdlib only)

### Option 2: Switch to Full Flask Server

To use the full-featured Flask server with client sync capabilities:

1. **Install Python packages in the xcframework**
   
   The Python runtime needs Flask and dependencies. These must be:
   - Built for iOS ARM64 architecture
   - Placed in `Python.xcframework/.../lib/python3.14/site-packages/`
   - Or bundled in `python-app/lib/` as vendored packages

2. **Update entry script**
   
   In `BackendManager.swift`, change:
   ```swift
   private let entryScript = "python-app/start_server.py"
   ```

3. **Add required packages**
   
   See `python-app/requirements-ios.txt` for needed packages:
   - flask
   - python-dotenv
   - httpx
   - numpy
   - pillow

### Option 3: Hybrid Approach

- Keep simple server for annotation UI
- Use Python only for data sync and mask manipulation
- Heavy processing (optical flow tracking) stays on server

---

## Running the App

### Via Xcode (Recommended)

1. Open project:
   ```bash
   open ios/Goobusters/Goobusters.xcodeproj
   ```

2. Select target: **iPad Air 11-inch (M3)** simulator

3. Build and run: `Cmd+R`

4. Watch console for Python initialization logs:
   - `[BackendPythonRunner] Python initialized successfully`
   - `[iOS Python] Starting HTTP server on port 8080...`

### Via Command Line

```bash
cd ios/Goobusters

# Build
xcodebuild -scheme Goobusters \
  -sdk iphonesimulator \
  -destination 'platform=iOS Simulator,id=F8AC1653-3335-47D7-B850-7FA209AE90FF' \
  build

# Install
xcrun simctl install F8AC1653-3335-47D7-B850-7FA209AE90FF \
  ~/Library/Developer/Xcode/DerivedData/Goobusters-*/Build/Products/Debug-iphonesimulator/Goobusters.app

# Launch
xcrun simctl launch F8AC1653-3335-47D7-B850-7FA209AE90FF org.badmath.goobusters

# Test
curl http://127.0.0.1:8080/healthz
# Should output: OK - Python is running!
```

### Simulator Device ID

iPad Air 11-inch (M3): `F8AC1653-3335-47D7-B850-7FA209AE90FF`

To list all simulators:
```bash
xcrun simctl list devices
```

---

## File Structure

```
ios/Goobusters/
├── Goobusters.xcodeproj          # Xcode project
└── Goobusters/
    ├── GoobustersApp.swift        # App entry point
    ├── GoobustersView.swift       # SwiftUI main view with WebView
    ├── BackendManager.swift       # Python backend lifecycle manager
    ├── PythonBackendRunner.swift  # Swift Python wrapper
    ├── BackendPythonRunner.h      # ObjC header
    ├── BackendPythonRunner.mm     # ObjC Python bridge
    ├── Goobusters-Bridging-Header.h  # Swift<->ObjC bridge
    ├── Info.plist                 # App configuration
    ├── Assets.xcassets/           # App icons, colors
    ├── Python.xcframework/        # Python 3.14 runtime
    │   ├── ios-arm64/             # Device slice
    │   ├── ios-arm64_x86_64-simulator/  # Simulator slice
    │   └── build/utils.sh         # Build utilities
    └── python-app/                # Python application code
        ├── start_server_simple.py # ✅ Current: Simple HTTP server
        ├── start_server.py        # Full Flask server (needs packages)
        ├── test_minimal.py        # Test script
        ├── ios_config.py          # iOS path configuration
        ├── requirements-ios.txt   # Python dependencies
        ├── lib/                   # Python libraries
        │   ├── client/            # Client-specific code
        │   ├── config.py
        │   ├── mask_archive.py
        │   └── ...
        ├── static/                # Web UI assets
        └── templates/             # HTML templates
```

---

## Environment Setup

### Python Environment Variables

Set in `PythonBackendRunner.swift` before Python initialization:

```swift
let pythonHome = (resourcePath as NSString).appendingPathComponent("python")
setenv("PYTHONHOME", pythonHome, 1)

let pythonPath = [
    (resourcePath as NSString).appendingPathComponent("python/lib/python3.14"),
    (resourcePath as NSString).appendingPathComponent("python/lib/python3.14/lib-dynload"),
    (resourcePath as NSString).appendingPathComponent("python-app"),
    (resourcePath as NSString).appendingPathComponent("python-app/lib"),
    (resourcePath as NSString).appendingPathComponent("python-app/lib/client")
].joined(separator: ":")
setenv("PYTHONPATH", pythonPath, 1)
```

### iOS Sandbox Paths

Python scripts use iOS sandbox directories via `ios_config.py`:

```python
def get_ios_paths():
    documents_dir = os.path.expanduser("~/Documents")
    return {
        "BUNDLE_PATH": bundle_path,
        "DATA_DIR": os.path.join(documents_dir, "goobusters_data"),
        "CACHE_DIR": os.path.join(documents_dir, "goobusters_cache"),
        "CLIENT_CACHE_DIR": os.path.join(documents_dir, "goobusters_client_cache"),
    }
```

---

## Troubleshooting

### Python doesn't start

1. Check Xcode console for errors
2. Look for `[BackendPythonRunner]` log messages
3. Verify `python-app/start_server_simple.py` exists in bundle
4. Check that Python.xcframework is embedded

### Server not responding on port 8080

1. Wait 30 seconds after launch (initialization can be slow)
2. Check `BackendManager` status messages in UI
3. Verify no firewall blocking localhost:8080
4. Try killing and relaunching the app

### Build failures

1. Clean build folder: `Cmd+Shift+K` in Xcode
2. Delete DerivedData:
   ```bash
   rm -rf ~/Library/Developer/Xcode/DerivedData/Goobusters-*
   ```
3. Rebuild: `Cmd+B`

### Simulator won't boot

```bash
# Shutdown
xcrun simctl shutdown F8AC1653-3335-47D7-B850-7FA209AE90FF

# Boot
xcrun simctl boot F8AC1653-3335-47D7-B850-7FA209AE90FF

# Open Simulator app
open -a Simulator
```

---

## Technical Details

### Python Initialization

The Python interpreter is initialized with these settings:

```c
PyConfig config;
PyConfig_InitIsolatedConfig(&config);

config.install_signal_handlers = 0;  // iOS doesn't allow
config.write_bytecode = 0;            // .pyc files not needed
config.use_environment = 1;           // Read PYTHONHOME/PATH
config.buffered_stdio = 0;            // Immediate output
config.parse_argv = 0;                // We set argv manually
config.site_import = 1;               // Import site.py

Py_InitializeFromConfig(&config);
```

### Threading Model

- Python runs on a dedicated serial dispatch queue: `org.badmath.goobusters.python`
- HTTP server uses Python's built-in threading for request handling
- Swift UI updates via `@MainActor` and `@Published` properties

### App Lifecycle

1. **App Launch** → `GoobustersApp.swift`
2. **View Appears** → `GoobustersView.onAppear()`
3. **Start Backend** → `BackendManager.start()`
4. **Init Python** → `PythonBackendRunner.start()`
5. **Configure** → `BackendPythonRunner.configureInterpreter()`
6. **Run Script** → `BackendPythonRunner.runEntryScriptAtPath()`
7. **Poll Health** → `BackendManager.waitForServer()`
8. **Show WebView** → Once health check passes

---

## References

- **Python Apple Support**: `ios/Goobusters/python-apple-support/`
- **Python.org iOS Guide**: https://docs.python.org/3/using/ios.html
- **BeeWare Project**: https://beeware.org/ (Python on mobile platforms)

---

## Success Metrics ✅

- [x] Build completes without errors
- [x] App launches on iPad Air 11" simulator  
- [x] Python 3.14 initializes successfully
- [x] HTTP server starts and binds to port 8080
- [x] `/healthz` endpoint returns 200 OK
- [x] WebView loads and displays Python-served content

**Status: FULLY OPERATIONAL**

Last verified: 2026-01-16 14:43 PST
Python version: 3.14.0
iOS SDK: 26.2 (iOS 18.2 simulator)
Device: iPad Air 11-inch (M3) - Simulator
