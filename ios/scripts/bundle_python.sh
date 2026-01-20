#!/bin/bash
# Bundle Goobusters Python code for iOS app
# This script packages the Python backend code that will be embedded in the iOS app

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
OUTPUT_DIR="$SCRIPT_DIR/../Goobusters/Goobusters/python-app"

echo "=== Bundling Goobusters Python Code ==="
echo "Source: $REPO_ROOT"
echo "Output: $OUTPUT_DIR"

# Clean previous bundle
rm -rf "$OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR"

# Copy lib directory (contains client code and utilities)
echo "Copying lib modules..."
mkdir -p "$OUTPUT_DIR/lib"
cp "$REPO_ROOT/lib/config.py" "$OUTPUT_DIR/lib/"
cp "$REPO_ROOT/lib/mask_archive.py" "$OUTPUT_DIR/lib/"
cp "$REPO_ROOT/lib/mask_utils.py" "$OUTPUT_DIR/lib/"
cp "$REPO_ROOT/lib/frame_extractor.py" "$OUTPUT_DIR/lib/"
touch "$OUTPUT_DIR/lib/__init__.py"

# Copy client module (lib/client/)
echo "Copying client module..."
mkdir -p "$OUTPUT_DIR/lib/client"
cp "$REPO_ROOT/lib/client/start.py" "$OUTPUT_DIR/lib/client/"
cp "$REPO_ROOT/lib/client/mdai_client.py" "$OUTPUT_DIR/lib/client/"
cp "$REPO_ROOT/lib/client/frame_extractor.py" "$OUTPUT_DIR/lib/client/" 2>/dev/null || true
touch "$OUTPUT_DIR/lib/client/__init__.py"

# Also create top-level client module reference for imports
mkdir -p "$OUTPUT_DIR/client"
cat > "$OUTPUT_DIR/client/__init__.py" << 'EOF'
# Re-export from lib.client for compatibility
from lib.client.mdai_client import *
EOF

# Copy static files (HTML, JS, CSS)
echo "Copying static files..."
cp -r "$REPO_ROOT/static" "$OUTPUT_DIR/"

# Copy templates
if [ -d "$REPO_ROOT/templates" ]; then
    echo "Copying templates..."
    cp -r "$REPO_ROOT/templates" "$OUTPUT_DIR/"
fi

# Copy default config (NOT .env with secrets!)
echo "Copying default config..."
cp "$REPO_ROOT/dot.env.defaults" "$OUTPUT_DIR/"

# Create iOS-specific config overrides
cat > "$OUTPUT_DIR/ios_config.py" << 'EOF'
"""iOS-specific configuration for Goobusters"""
import os

def get_ios_paths():
    """Get paths appropriate for iOS app sandbox"""
    # App bundle resource path
    bundle_path = os.path.dirname(os.path.abspath(__file__))

    # Documents directory for user data (writable)
    documents_dir = os.path.expanduser("~/Documents")

    return {
        "BUNDLE_PATH": bundle_path,
        "STATIC_DIR": os.path.join(bundle_path, "static"),
        "TEMPLATES_DIR": os.path.join(bundle_path, "templates"),
        "DATA_DIR": os.path.join(documents_dir, "goobusters_data"),
        "CACHE_DIR": os.path.join(documents_dir, "goobusters_cache"),
        "CLIENT_CACHE_DIR": os.path.join(documents_dir, "goobusters_client_cache"),
    }

def configure_environment():
    """Set up environment for iOS"""
    paths = get_ios_paths()

    # Ensure directories exist
    for key in ["DATA_DIR", "CACHE_DIR", "CLIENT_CACHE_DIR"]:
        os.makedirs(paths[key], exist_ok=True)

    # Set environment variables
    os.environ.setdefault("DATA_DIR", paths["DATA_DIR"])
    os.environ.setdefault("CLIENT_CACHE_DIR", paths["CLIENT_CACHE_DIR"])

    return paths
EOF

# Create a simple startup script
cat > "$OUTPUT_DIR/start_server.py" << 'EOF'
"""iOS app entry point - starts the Flask server"""
import os
import sys

# Add bundle path to Python path
bundle_path = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, bundle_path)

# Configure iOS environment
from ios_config import configure_environment, get_ios_paths
paths = configure_environment()

def main(port=8080):
    """Start the Goobusters client Flask server"""
    print(f"[iOS] Starting Flask server on port {port}...")
    print(f"[iOS] Bundle path: {bundle_path}")
    print(f"[iOS] Data dir: {paths['DATA_DIR']}")

    # Import and create the Flask app
    from lib.client.start import create_app
    app = create_app(skip_startup=False)

    # Run Flask (this blocks)
    app.run(host="127.0.0.1", port=port, threaded=True, use_reloader=False)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()
    main(port=args.port)
EOF

# Create healthz endpoint patch for the client app
cat > "$OUTPUT_DIR/healthz_patch.py" << 'EOF'
"""Adds /healthz endpoint to the Flask app for iOS startup detection"""

def add_healthz(app):
    @app.route("/healthz")
    def healthz():
        return "ok", 200
    return app
EOF

# Remove any .pyc files and __pycache__ directories
echo "Cleaning compiled Python files..."
find "$OUTPUT_DIR" -name "*.pyc" -delete
find "$OUTPUT_DIR" -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true

# Remove any .env files (secrets!)
echo "Removing secret files..."
find "$OUTPUT_DIR" -name ".env" -delete
find "$OUTPUT_DIR" -name "*.env" ! -name "dot.env.defaults" -delete

# Create requirements-ios.txt for reference
cat > "$OUTPUT_DIR/requirements-ios.txt" << 'EOF'
# Python packages needed for iOS client
# These must be compiled for iOS ARM64 or be pure Python

# Core (likely have iOS wheels)
flask
python-dotenv
httpx

# Data handling
numpy
pillow

# These are heavyweight and may not have iOS wheels:
# pandas - optional, for data analysis
# scipy - optional, for advanced processing
# opencv-contrib-python - REQUIRED for image processing, needs iOS build
# torch/torchvision - likely too large, tracking should be server-side

# Note: The iOS client primarily acts as a thin client that:
# 1. Syncs data from MD.ai
# 2. Proxies requests to the server
# 3. Displays the WebView UI
# Heavy processing (tracking) happens on the server
EOF

# List what was bundled
echo ""
echo "=== Bundle Contents ==="
find "$OUTPUT_DIR" -type f -name "*.py" | sort
echo ""
echo "=== Other Files ==="
find "$OUTPUT_DIR" -type f ! -name "*.py" | sort | head -20
echo ""
echo "Total files: $(find "$OUTPUT_DIR" -type f | wc -l)"
echo "Bundle size: $(du -sh "$OUTPUT_DIR" | cut -f1)"
echo ""
echo "=== Bundle Complete ==="
echo ""
echo "Next steps:"
echo "1. Add python-app folder to Xcode project as folder reference"
echo "2. Ensure Python.xcframework is embedded in the app"
echo "3. Add PythonKit Swift package"
echo "4. Build and test on simulator"