#!/bin/bash
# Build, install, and run Goobusters iOS app on simulator
#
# Prerequisites:
#   - Xcode installed
#   - dot.env configured (MDAI_TOKEN, SERVER_URL, etc.)
#
# Usage:
#   ./build_ios.sh              # Build and run (prompts for credentials on first run)
#   ./build_ios.sh --skip-build # Just install and run (if already built)
#   ./build_ios.sh --debug      # Build with MDAI_TOKEN + USER_EMAIL pre-bundled
#   ./build_ios.sh --archive    # Create archive for TestFlight (device build)
#
# Startup behavior (handled by client):
#   - If credentials present -> syncs automatically on startup
#   - If credentials missing -> shows setup page to enter token/name

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
DEBUG_BUILD=false
ARCHIVE_BUILD=false
for arg in "$@"; do
    case $arg in
        --skip-build) SKIP_BUILD=true ;;
        --debug) DEBUG_BUILD=true ;;
        --archive) ARCHIVE_BUILD=true ;;
    esac
done

if [ "$DEBUG_BUILD" = true ]; then
    echo "=== DEBUG BUILD MODE ==="
    echo "  - MDAI_TOKEN will be bundled from dot.env"
    echo "  - USER_EMAIL: debugger-$(hostname -s)"
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

if [ "$ARCHIVE_BUILD" = true ]; then
    # Archive build for TestFlight
    # Archive settings are configured in the Xcode scheme (Release config, reveal in Organizer)
    # Xcode automatically uses ~/Library/Developer/Xcode/Archives/YYYY-MM-DD/ by default
    echo ""
    echo "=== Bundling Python Code ==="
    bash ios/scripts/bundle_python.sh
    
    echo ""
    echo "=== Creating Archive ==="
    echo "Using Xcode scheme settings (Release configuration)"
    
    xcodebuild -project "$PROJECT" \
        -scheme "$SCHEME" \
        -sdk iphoneos \
        archive 2>&1 | tail -10
    
    if [ $? -eq 0 ]; then
        echo ""
        echo "✓ Archive created in default Xcode location"
        echo ""
        echo "To upload to TestFlight:"
        echo "1. Open Xcode → Window → Organizer"
        echo "2. Select the Goobusters archive from today"
        echo "3. Click 'Distribute App' → 'App Store Connect'"
    else
        echo "Archive failed - check logs above"
        exit 1
    fi
    exit 0
fi

if [ "$SKIP_BUILD" = false ]; then
    # Generate python-app bundle
    echo ""
    echo "=== Bundling Python Code ==="
    if [ "$DEBUG_BUILD" = true ]; then
        bash ios/scripts/bundle_python.sh --debug
    else
        bash ios/scripts/bundle_python.sh
    fi

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

# Uninstall previous version (clean install)
echo "Uninstalling previous version..."
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
