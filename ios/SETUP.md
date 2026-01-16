# Goobusters iOS App - Python Integration Setup

## Overview

The iOS app embeds Python 3.13 to run the Flask backend locally on the iPad. This requires:

1. Python.xcframework (already downloaded to `Frameworks/`)
2. PythonKit Swift Package (for Swift ↔ Python interop)
3. Bundled Python standard library
4. Bundled Goobusters Python code

## Step 1: Add Python.xcframework to Xcode

1. Open `Goobusters.xcodeproj` in Xcode
2. Select the **Goobusters** target in the project navigator
3. Go to **General** → **Frameworks, Libraries, and Embedded Content**
4. Click **+** and then **Add Other...** → **Add Files...**
5. Navigate to `Frameworks/Python.xcframework` and add it
6. Ensure it says **Embed & Sign** in the Embed column

## Step 2: Add PythonKit Swift Package

1. In Xcode, go to **File** → **Add Package Dependencies...**
2. Enter the URL: `https://github.com/pvieito/PythonKit`
3. Select version **Up to Next Major Version** from `0.3.0`
4. Click **Add Package**
5. Select the **PythonKit** library and add it to **Goobusters** target

## Step 3: Add Python Standard Library to Bundle

The Python.xcframework needs its standard library bundled with the app.

1. In Finder, go to `Frameworks/Python.xcframework/ios-arm64/`
2. Copy the `python-stdlib` folder
3. Drag it into the Xcode project under the **Goobusters** group
4. In the dialog, select:
   - ✅ Copy items if needed
   - ✅ Create folder references (NOT groups)
   - Target: ✅ Goobusters

## Step 4: Bundle Python App Code

Run the bundle script to package the Goobusters Python code:

```bash
cd /Users/cseaman/Documents/goobusters
./ios/scripts/bundle_python.sh
```

Then add the generated `python-app` folder to Xcode:

1. Drag `ios/Goobusters/Goobusters/python-app` into the project
2. Select:
   - ✅ Copy items if needed
   - ✅ Create folder references
   - Target: ✅ Goobusters

## Step 5: Add PythonRunner.swift to Build

The `PythonRunner.swift` file was created but needs to be added to the Xcode project:

1. In Xcode, right-click the **Goobusters** folder in the navigator
2. Select **Add Files to "Goobusters"...**
3. Select `PythonRunner.swift`
4. Ensure **Goobusters** target is checked

## Step 6: Update Info.plist for Local Networking

Add App Transport Security exception for localhost:

1. Open `Info.plist`
2. Add the following (or edit in Xcode's property list editor):

```xml
<key>NSAppTransportSecurity</key>
<dict>
    <key>NSAllowsLocalNetworking</key>
    <true/>
</dict>
```

## Step 7: Build and Test

1. Select an iPad simulator or connected iPad
2. Build and run (⌘R)
3. The app should:
   - Show "Initializing Python..."
   - Show "Starting Flask server..."
   - Display the web UI once the server is ready

## Troubleshooting

### "Module not found" Python errors
- Ensure `python-stdlib` is added as a **folder reference** (blue folder icon), not a group
- Check that PYTHONPATH includes both stdlib and app paths

### Server never becomes ready
- Check Xcode console for Python errors
- Verify all dependencies are bundled (run `bundle_python.sh`)

### App crashes on launch
- Ensure Python.xcframework is set to **Embed & Sign**
- Check that the framework supports the target architecture (arm64 for device, x86_64 for Intel simulator)

## File Structure

```
ios/
├── Goobusters/
│   ├── Goobusters.xcodeproj
│   ├── Frameworks/
│   │   └── Python.xcframework/     # Python runtime
│   └── Goobusters/
│       ├── GoobustersApp.swift
│       ├── GoobustersView.swift
│       ├── BackendManager.swift
│       ├── PythonRunner.swift
│       ├── python-stdlib/          # Python standard library (added via Xcode)
│       └── python-app/             # Goobusters Python code (generated)
└── scripts/
    └── bundle_python.sh            # Script to bundle Python app code
```

## Dependencies to Bundle

The following Python packages need to be compiled for iOS and bundled:

**Required (core functionality):**
- flask
- python-dotenv
- numpy
- pillow

**Required (for tracking - may need iOS-specific builds):**
- opencv-contrib-python → Need iOS wheel
- scipy
- scikit-image

**Heavy/Optional (consider server-only):**
- torch, torchvision → Very large, may not have iOS wheels
- pandas → Large but has iOS support

## Alternative: Thin Client Mode

If bundling all dependencies proves difficult, the app can run in "thin client" mode:
- Connect to a remote server (goo.badmath.org) instead of local Python
- Only need to bundle minimal Python for local file serving
- Tracking happens server-side
