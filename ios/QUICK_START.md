# Quick Reference: Goobusters iOS + Python

## ✅ Current Status
**Python 3.14 is WORKING on iPad Air 11" simulator!**

## Quick Test
```bash
curl http://127.0.0.1:8080/healthz
# Returns: OK - Python is running!
```

## Running the App

### Method 1: Xcode (Easiest)
```bash
open ios/Goobusters/Goobusters.xcodeproj
```
- Select "iPad Air 11-inch (M3)" simulator
- Press `Cmd+R` to build and run

### Method 2: Command Line
```bash
# Build
cd ios/Goobusters
xcodebuild -scheme Goobusters -sdk iphonesimulator \
  -destination 'platform=iOS Simulator,id=F8AC1653-3335-47D7-B850-7FA209AE90FF' \
  build

# Install
xcrun simctl install F8AC1653-3335-47D7-B850-7FA209AE90FF \
  ~/Library/Developer/Xcode/DerivedData/Goobusters-*/Build/Products/Debug-iphonesimulator/Goobusters.app

# Launch  
xcrun simctl launch F8AC1653-3335-47D7-B850-7FA209AE90FF org.badmath.goobusters
```

## File Locations

### Key iOS Files
- **Main app**: `ios/Goobusters/Goobusters/GoobustersApp.swift`
- **Python bridge**: `ios/Goobusters/Goobusters/BackendPythonRunner.mm`
- **Backend manager**: `ios/Goobusters/Goobusters/BackendManager.swift`

### Python Code
- **Active server**: `ios/Goobusters/Goobusters/python-app/start_server_simple.py`
- **Flask server**: `ios/Goobusters/Goobusters/python-app/start_server.py` (needs packages)
- **iOS config**: `ios/Goobusters/Goobusters/python-app/ios_config.py`

### Python Runtime
- **Framework**: `ios/Goobusters/Goobusters/Python.xcframework/`
- **Build utils**: `ios/Goobusters/Goobusters/Python.xcframework/build/utils.sh`

## Switching Between Python Scripts

Edit `ios/Goobusters/Goobusters/BackendManager.swift`:

```swift
// Line 12-13:
private let entryScript = "python-app/start_server_simple.py"  // Current
// or
private let entryScript = "python-app/start_server.py"  // Full Flask (needs deps)
```

Then rebuild.

## Server Endpoints (Simple Server)

- `http://127.0.0.1:8080/` - HTML page with Python info
- `http://127.0.0.1:8080/healthz` - Health check (plain text)

## Troubleshooting

### "Connection refused"
- Wait 30 seconds after launch for Python to initialize
- Check app UI for status messages

### Build fails
```bash
# Clean
cd ios/Goobusters
xcodebuild clean

# Or in Xcode: Cmd+Shift+K
```

### Simulator issues
```bash
# Restart simulator
xcrun simctl shutdown F8AC1653-3335-47D7-B850-7FA209AE90FF
xcrun simctl boot F8AC1653-3335-47D7-B850-7FA209AE90FF
open -a Simulator
```

## Next Steps

1. **To use full Flask server**: Install Python packages in xcframework
2. **To add endpoints**: Edit `start_server_simple.py`
3. **To modify UI**: Edit `GoobustersView.swift`

## Documentation
See `ios/PYTHON_RUNTIME_STATUS.md` for full details.
