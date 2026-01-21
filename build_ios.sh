#!/bin/bash
# Build, install, and run Goobusters iOS app on simulator
#
# Prerequisites:
#   - Xcode installed
#   - dot.env configured with secrets (MDAI_TOKEN, etc.)
#
# Usage:
#   ./build_ios.sh              # Build and run on default simulator
#   ./build_ios.sh --skip-build # Just install and run (if already built)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Configuration
SIMULATOR_NAME="iPad Air 11-inch (M3)"
BUNDLE_ID="org.badmath.goobusters"
PROJECT="ios/Goobusters/Goobusters.xcodeproj"
SCHEME="Goobusters"

# Parse arguments
SKIP_BUILD=false
if [ "$1" = "--skip-build" ]; then
    SKIP_BUILD=true
fi

# Find or boot simulator
echo "=== Finding Simulator ==="
SIMULATOR_ID=$(xcrun simctl list devices available | grep "$SIMULATOR_NAME" | grep -oE '[A-F0-9-]{36}' | head -1)

if [ -z "$SIMULATOR_ID" ]; then
    echo "Error: Simulator '$SIMULATOR_NAME' not found"
    echo "Available simulators:"
    xcrun simctl list devices available | grep iPad
    exit 1
fi

echo "Using simulator: $SIMULATOR_NAME ($SIMULATOR_ID)"

# Check if simulator is booted
BOOT_STATUS=$(xcrun simctl list devices | grep "$SIMULATOR_ID" | grep -o "(Booted)" || true)
if [ -z "$BOOT_STATUS" ]; then
    echo "Booting simulator..."
    xcrun simctl boot "$SIMULATOR_ID" 2>/dev/null || true
    sleep 3
fi

if [ "$SKIP_BUILD" = false ]; then
    # Generate python-app bundle
    echo ""
    echo "=== Bundling Python Code ==="
    bash ios/scripts/bundle_python.sh

    # Build with Xcode
    echo ""
    echo "=== Building iOS App ==="
    xcodebuild -project "$PROJECT" \
        -scheme "$SCHEME" \
        -destination "id=$SIMULATOR_ID" \
        -configuration Debug \
        build | tail -20
fi

# Find built app
APP_PATH=$(find ~/Library/Developer/Xcode/DerivedData -name "Goobusters.app" -path "*/Debug-iphonesimulator/*" 2>/dev/null | head -1)

if [ -z "$APP_PATH" ]; then
    echo "Error: Built app not found. Run without --skip-build first."
    exit 1
fi

echo ""
echo "=== Installing App ==="
# Kill any running instance
pkill -9 -f Goobusters 2>/dev/null || true

# Uninstall previous version
xcrun simctl uninstall "$SIMULATOR_ID" "$BUNDLE_ID" 2>/dev/null || true

# Install new version
xcrun simctl install "$SIMULATOR_ID" "$APP_PATH"

echo ""
echo "=== Launching App ==="
xcrun simctl launch "$SIMULATOR_ID" "$BUNDLE_ID"

# Open Simulator app to bring it to foreground
open -a Simulator

echo ""
echo "=== Done ==="
echo "App running on $SIMULATOR_NAME"
echo "Flask server: http://127.0.0.1:8080"
