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

# Copy vendor directory if it exists in lib/client/
if [ -d "$REPO_ROOT/lib/client/vendor" ]; then
    echo "Copying vendor packages from lib/client/vendor..."
    rm -rf "$OUTPUT_DIR/vendor"
    cp -r "$REPO_ROOT/lib/client/vendor" "$OUTPUT_DIR/"
fi

# Also create top-level client module reference for imports
mkdir -p "$OUTPUT_DIR/client"
cat > "$OUTPUT_DIR/client/__init__.py" << 'EOF'
# Re-export from lib.client for compatibility
from lib.client.mdai_client import *
EOF

# Copy static files (HTML, JS, CSS)
echo "Copying static files..."
rm -rf "$OUTPUT_DIR/static"
cp -r "$REPO_ROOT/static" "$OUTPUT_DIR/"

# Copy templates
if [ -d "$REPO_ROOT/templates" ]; then
    echo "Copying templates..."
    rm -rf "$OUTPUT_DIR/templates"
    cp -r "$REPO_ROOT/templates" "$OUTPUT_DIR/"
    
    # Inject Cloudflare headers from dot.env into viewer.html if present
    if [ -f "$REPO_ROOT/dot.env" ]; then
        echo "Reading Cloudflare headers from dot.env..."
        CF_ACCESS_CLIENT_ID=$(grep "^CF_ACCESS_CLIENT_ID=" "$REPO_ROOT/dot.env" 2>/dev/null | cut -d'=' -f2- | tr -d '"' || echo "")
        CF_ACCESS_CLIENT_SECRET=$(grep "^CF_ACCESS_CLIENT_SECRET=" "$REPO_ROOT/dot.env" 2>/dev/null | cut -d'=' -f2- | tr -d '"' || echo "")
        
        # Replace placeholder values in viewer.html if CF vars exist
        if [ -n "$CF_ACCESS_CLIENT_ID" ] && [ -n "$CF_ACCESS_CLIENT_SECRET" ]; then
            echo "Injecting Cloudflare headers into viewer.html..."
            # Use | as delimiter - only need to escape | and & in replacement string
            CF_ID_ESC=$(printf '%s' "$CF_ACCESS_CLIENT_ID" | sed 's/[|&]/\\&/g')
            CF_SECRET_ESC=$(printf '%s' "$CF_ACCESS_CLIENT_SECRET" | sed 's/[|&]/\\&/g')
            sed -i '' "s|const CF_ACCESS_CLIENT_ID = \"\";|const CF_ACCESS_CLIENT_ID = \"$CF_ID_ESC\";|g" "$OUTPUT_DIR/templates/viewer.html"
            sed -i '' "s|const CF_ACCESS_CLIENT_SECRET = \"\";|const CF_ACCESS_CLIENT_SECRET = \"$CF_SECRET_ESC\";|g" "$OUTPUT_DIR/templates/viewer.html"
        fi
    fi
fi

# Copy default config and merge non-secret values from dot.env
echo "Copying default config..."
cp "$REPO_ROOT/dot.env.defaults" "$OUTPUT_DIR/"

# Merge non-secret config values from dot.env if it exists
if [ -f "$REPO_ROOT/dot.env" ]; then
    echo "Merging non-secret config from dot.env..."
    # List of non-secret keys to copy (NOT MDAI_TOKEN!)
    NON_SECRET_KEYS="DOMAIN PROJECT_ID DATASET_ID LABEL_ID EMPTY_ID FLOW_METHOD SERVER_URL"
    for key in $NON_SECRET_KEYS; do
        # Get value from dot.env (remove quotes)
        value=$(grep "^$key=" "$REPO_ROOT/dot.env" 2>/dev/null | cut -d'=' -f2- | tr -d '"' || echo "")
        if [ -n "$value" ] && [ "$value" != "REPLACE_ME" ] && [ "$value" != "REPLACE_ME_LABEL_ID" ] && [ "$value" != "REPLACE_ME_EMPTY_ID" ]; then
            echo "  Setting $key"
            # Update or append in dot.env.defaults
            if grep -q "^$key=" "$OUTPUT_DIR/dot.env.defaults"; then
                sed -i '' "s|^$key=.*|$key=$value|" "$OUTPUT_DIR/dot.env.defaults"
            else
                echo "$key=$value" >> "$OUTPUT_DIR/dot.env.defaults"
            fi
        fi
    done
fi

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

def _load_dotenv_defaults(bundle_path):
    """Load config values from dot.env.defaults and set as environment variables"""
    defaults_file = os.path.join(bundle_path, "dot.env.defaults")
    if not os.path.exists(defaults_file):
        return
    
    with open(defaults_file, "r") as f:
        for line in f:
            line = line.strip()
            # Skip comments and empty lines
            if not line or line.startswith("#"):
                continue
            # Parse KEY=VALUE (handle # comments at end of line)
            if "=" in line:
                key, _, value = line.partition("=")
                key = key.strip()
                value = value.split("#")[0].strip()  # Remove inline comments
                # Remove quotes if present
                if value.startswith('"') and value.endswith('"'):
                    value = value[1:-1]
                elif value.startswith("'") and value.endswith("'"):
                    value = value[1:-1]
                if key and value:
                    os.environ.setdefault(key, value)

def configure_environment():
    """Set up environment for iOS"""
    paths = get_ios_paths()

    # Load config from dot.env.defaults in bundle
    _load_dotenv_defaults(paths["BUNDLE_PATH"])

    # Ensure directories exist
    for key in ["DATA_DIR", "CACHE_DIR", "CLIENT_CACHE_DIR"]:
        os.makedirs(paths[key], exist_ok=True)

    # Set/override path environment variables
    os.environ["DATA_DIR"] = paths["DATA_DIR"]
    os.environ["CLIENT_CACHE_DIR"] = paths["CLIENT_CACHE_DIR"]
    os.environ["VIDEO_CACHE_PATH"] = os.path.join(paths["DATA_DIR"], "video_cache")
    os.environ["FRAMES_CACHE_PATH"] = os.path.join(paths["DATA_DIR"], "frames_cache")

    return paths
EOF

# Create a simple startup script
cat > "$OUTPUT_DIR/start_server.py" << 'EOF'
"""iOS app entry point - starts the Flask server"""
import os
import sys
import traceback
import io

# Add bundle path to Python path
bundle_path = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, bundle_path)

# Add vendor directory for vendored packages (Flask, Jinja2, etc)
vendor_path = os.path.join(bundle_path, "vendor")
if os.path.isdir(vendor_path):
    sys.path.insert(0, vendor_path)

# Create log file in Documents for debugging
log_dir = os.path.expanduser("~/Documents")
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, "goobusters_startup.log")

def log(msg):
    """Write to both stderr and log file"""
    print(msg, file=sys.stderr)
    with open(log_file, "a") as f:
        f.write(msg + "\n")

# Clear log file
with open(log_file, "w") as f:
    f.write("=== Goobusters Startup Log ===\n")

log(f"[iOS] Bundle path: {bundle_path}")
log(f"[iOS] Vendor path: {vendor_path}")
log(f"[iOS] sys.path: {sys.path}")

try:
    # Configure iOS environment
    from ios_config import configure_environment, get_ios_paths
    paths = configure_environment()
    log(f"[iOS] Data dir: {paths['DATA_DIR']}")
except Exception as e:
    log(f"[iOS] FAILED to configure environment: {e}")
    with open(log_file, "a") as f:
        traceback.print_exc(file=f)
    traceback.print_exc()
    sys.exit(1)

def main(port=8080):
    """Start the Goobusters client Flask server"""
    log(f"[iOS] Starting Flask server on port {port}...")

    try:
        # Import and create the Flask app
        log("[iOS] Importing lib.client.start...")
        from lib.client.start import create_app
        log("[iOS] Creating Flask app (skip_startup=True for iOS)...")
        # Skip startup sync on iOS - dataset may not exist yet
        app = create_app(skip_startup=True)
        log("[iOS] Starting Flask server...")

        # Run Flask (this blocks)
        app.run(host="127.0.0.1", port=port, threaded=True, use_reloader=False)
    except Exception as e:
        log(f"[iOS] FAILED to start Flask: {e}")
        with open(log_file, "a") as f:
            traceback.print_exc(file=f)
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    try:
        import argparse
        # Clean up sys.argv - iOS Python runner may pass script path as argument
        # Keep only actual arguments (those starting with --)
        clean_argv = [sys.argv[0]] + [arg for arg in sys.argv[1:] if arg.startswith("--")]
        parser = argparse.ArgumentParser(prog="GoobustersBackend")
        parser.add_argument("--port", type=int, default=8080)
        args = parser.parse_args(clean_argv[1:])
        main(port=args.port)
    except Exception as e:
        log(f"[iOS] FATAL: {e}")
        with open(log_file, "a") as f:
            traceback.print_exc(file=f)
        traceback.print_exc()
        sys.exit(1)
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

# Remove macOS-specific compiled extensions (.so files) - iOS needs pure Python fallbacks
echo "Removing macOS compiled extensions..."
find "$OUTPUT_DIR" -name "*.so" -type f -delete

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