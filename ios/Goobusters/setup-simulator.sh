#!/bin/bash
# Setup iPad Air simulator for Goobusters development

set -e

echo "Setting up iPad Air simulator for Goobusters..."

# List available iPad simulators
echo "Available iPad simulators:"
xcrun simctl list devices available | grep -i "ipad" | grep -i "air" || echo "No iPad Air found, listing all iPads:"
xcrun simctl list devices available | grep -i "ipad" | head -5

# Find iPad Air (prefer latest iOS version)
IPAD_DEVICE=$(xcrun simctl list devices available | grep -i "ipad.*air" | head -1 | sed 's/.*(\([^)]*\)).*/\1/' | tr -d ' ')

if [ -z "$IPAD_DEVICE" ]; then
    echo "No iPad Air found. Creating one..."
    # Get latest iOS runtime
    RUNTIME=$(xcrun simctl list runtimes available | grep -i "ios" | tail -1 | sed 's/.*(\([^)]*\)).*/\1/' | tr -d ' ')
    if [ -z "$RUNTIME" ]; then
        echo "Error: No iOS runtime found. Please install one via Xcode."
        exit 1
    fi
    echo "Using runtime: $RUNTIME"
    IPAD_DEVICE=$(xcrun simctl create "iPad Air Goobusters" "iPad Air" "$RUNTIME" 2>/dev/null || echo "")
fi

if [ -z "$IPAD_DEVICE" ]; then
    echo "Error: Could not find or create iPad Air simulator"
    exit 1
fi

echo "Using iPad Air device: $IPAD_DEVICE"

# Boot the simulator
echo "Booting simulator..."
xcrun simctl boot "$IPAD_DEVICE" 2>/dev/null || echo "Simulator already booted"

# Set to landscape orientation
echo "Setting to landscape orientation..."
xcrun simctl setenv "$IPAD_DEVICE" SIMULATOR_ORIENTATION "Landscape Left" 2>/dev/null || true

# Open Simulator app
echo "Opening Simulator app..."
open -a Simulator

# Wait a moment for simulator to be ready
sleep 2

echo ""
echo "✅ iPad Air simulator is ready!"
echo "   Device ID: $IPAD_DEVICE"
echo ""
echo "Next steps:"
echo "1. Open ios/Goobusters/Goobusters.xcodeproj in Xcode"
echo "2. Select the 'Goobusters' scheme"
echo "3. Select the iPad Air simulator as the destination"
echo "4. Press Cmd+R to build and run"

