#!/bin/bash
# Bundle Goobusters Python code for iOS app
# This script packages the Python backend code that will be embedded in the iOS app
#
# Usage:
#   ./bundle_python.sh           # Normal build (strips mdai.token)
#   ./bundle_python.sh --debug   # Debug build with mdai.token + user_email bundled
#
# Note: The client handles sync automatically based on credentials.
# - If credentials present -> syncs on startup
# - If credentials missing -> shows setup page

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
OUTPUT_DIR="$SCRIPT_DIR/../Goobusters/Goobusters/python-app"

# Parse arguments
DEBUG_BUILD=false
if [ "$1" = "--debug" ]; then
    DEBUG_BUILD=true
    echo "=== DEBUG BUILD MODE ==="
fi

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

# Copy templates and inject Cloudflare headers
if [ -d "$REPO_ROOT/templates" ]; then
    echo "Copying templates..."
    rm -rf "$OUTPUT_DIR/templates"
    cp -r "$REPO_ROOT/templates" "$OUTPUT_DIR/"
    
    # Inject Cloudflare headers from dot.yaml into viewer.html if present
    if [ -f "$REPO_ROOT/dot.yaml" ]; then
        echo "Reading Cloudflare headers from dot.yaml..."
        # Extract values using grep/sed (works without yq)
        CF_ACCESS_CLIENT_ID=$(grep -A 2 "cloudflare_headers:" "$REPO_ROOT/dot.yaml" 2>/dev/null | grep "CF-Access-Client-Id:" | sed 's/.*CF-Access-Client-Id: *//' | tr -d ' ')
        CF_ACCESS_CLIENT_SECRET=$(grep -A 3 "cloudflare_headers:" "$REPO_ROOT/dot.yaml" 2>/dev/null | grep "CF-Access-Client-Secret:" | sed 's/.*CF-Access-Client-Secret: *//' | tr -d ' ')
        
        # Replace placeholder values in viewer.html if CF vars exist
        if [ -n "$CF_ACCESS_CLIENT_ID" ] && [ -n "$CF_ACCESS_CLIENT_SECRET" ]; then
            echo "Injecting Cloudflare headers into viewer.html..."
            # Escape special chars for sed
            CF_ID_ESC=$(printf '%s' "$CF_ACCESS_CLIENT_ID" | sed 's/[|&]/\\&/g')
            CF_SECRET_ESC=$(printf '%s' "$CF_ACCESS_CLIENT_SECRET" | sed 's/[|&]/\\&/g')
            sed -i '' "s|const CF_ACCESS_CLIENT_ID = \"\";|const CF_ACCESS_CLIENT_ID = \"$CF_ID_ESC\";|g" "$OUTPUT_DIR/templates/viewer.html"
            sed -i '' "s|const CF_ACCESS_CLIENT_SECRET = \"\";|const CF_ACCESS_CLIENT_SECRET = \"$CF_SECRET_ESC\";|g" "$OUTPUT_DIR/templates/viewer.html"
        fi
    fi
fi

# Copy dot.yaml config (stripping mdai.token unless --debug)
echo "Copying dot.yaml config..."
if [ -f "$REPO_ROOT/dot.yaml" ]; then
    if [ "$DEBUG_BUILD" = true ]; then
        echo "  [DEBUG] Including mdai.token"
        cp "$REPO_ROOT/dot.yaml" "$OUTPUT_DIR/dot.yaml"
        
        # In debug mode, set default username to debugger-$hostname
        DEBUG_USER="debugger-$(hostname -s)"
        echo "  Setting user_email=$DEBUG_USER"
        # Add client.user_email if not present
        if ! grep -q "^client:" "$OUTPUT_DIR/dot.yaml"; then
            echo "" >> "$OUTPUT_DIR/dot.yaml"
            echo "client:" >> "$OUTPUT_DIR/dot.yaml"
            echo "    user_email: $DEBUG_USER" >> "$OUTPUT_DIR/dot.yaml"
        elif ! grep -q "user_email:" "$OUTPUT_DIR/dot.yaml"; then
            sed -i '' "/^client:/a\\
    user_email: $DEBUG_USER
" "$OUTPUT_DIR/dot.yaml"
        fi
    else
        # Strip mdai.token for non-debug builds
        echo "  Stripping mdai.token for release build"
        # Remove the token line from mdai section
        sed '/^[[:space:]]*token:/d' "$REPO_ROOT/dot.yaml" > "$OUTPUT_DIR/dot.yaml"
    fi
else
    echo "WARNING: dot.yaml not found at $REPO_ROOT/dot.yaml"
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

def _load_yaml_config(bundle_path):
    """Load config values from dot.yaml and set as environment variables"""
    import yaml
    
    yaml_file = os.path.join(bundle_path, "dot.yaml")
    if not os.path.exists(yaml_file):
        return {}
    
    with open(yaml_file) as f:
        config = yaml.safe_load(f) or {}
    
    # Flatten yaml to env vars for compatibility
    env_map = {
        "DOMAIN": ("mdai", "domain"),
        "PROJECT_ID": ("mdai", "project_id"),
        "DATASET_ID": ("mdai", "dataset"),
        "MDAI_TOKEN": ("mdai", "token"),
        "LABEL_ID": ("mdai", "label_id"),
        "EMPTY_ID": ("mdai", "empty_id"),
        "TRACK_ID": ("mdai", "track_id"),
        "SERVER_URL": ("server", "url"),
        "FLOW_METHOD": ("optical_flow", "method"),
        "USER_EMAIL": ("client", "user_email"),
    }
    
    for env_key, yaml_path in env_map.items():
        val = config
        for part in yaml_path:
            if isinstance(val, dict) and part in val:
                val = val[part]
            else:
                val = None
                break
        if val is not None:
            os.environ.setdefault(env_key, str(val))
    
    return config

def configure_environment():
    """Set up environment for iOS"""
    paths = get_ios_paths()

    # Load config from dot.yaml in bundle
    _load_yaml_config(paths["BUNDLE_PATH"])

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
"""iOS app entry point - starts the Flask server

Startup flow (handled by create_app):
1. If token + name present -> always sync
2. If credentials missing -> skip sync, show setup page
3. If sync fails -> log error, show setup page for retry
"""
import os
import sys
import traceback

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
    print(msg, file=sys.stderr, flush=True)
    with open(log_file, "a") as f:
        f.write(msg + "\n")
        f.flush()

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
        from lib.config import load_config
        
        log("[iOS] Loading config...")
        # Change to bundle directory so load_config finds dot.yaml
        original_cwd = os.getcwd()
        os.chdir(bundle_path)
        try:
            config = load_config("client")
        finally:
            os.chdir(original_cwd)
        log(f"[iOS] Config: server_url={config.server_url}, user={config.user_email}")
        log(f"[iOS] Token present: {bool(config.mdai_token and config.mdai_token != 'not_configured_yet')}")
        
        log("[iOS] Creating Flask app...")
        # Debug: check credentials path
        data_dir = os.environ.get("DATA_DIR", "NOT SET")
        creds_path = os.path.join(data_dir, "credentials.json") if data_dir != "NOT SET" else "UNKNOWN"
        creds_exists = os.path.exists(creds_path) if data_dir != "NOT SET" else False
        log(f"[iOS] Credentials path: {creds_path}, exists: {creds_exists}")
        if creds_exists:
            with open(creds_path) as f:
                log(f"[iOS] Credentials content: {f.read()}")
        
        # Frontend will handle sync after connecting to backend
        app = create_app(config=config)
        
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
find "$OUTPUT_DIR" -name "*.env" -delete
find "$OUTPUT_DIR" -name "dot.env*" -delete

# Create requirements-ios.txt for reference
cat > "$OUTPUT_DIR/requirements-ios.txt" << 'EOF'
# Python packages needed for iOS client
# These must be compiled for iOS ARM64 or be pure Python

# Core (likely have iOS wheels)
flask
pyyaml
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
echo "=== Config Files ==="
find "$OUTPUT_DIR" -type f -name "*.yaml" | sort
echo ""
echo "=== Other Files ==="
find "$OUTPUT_DIR" -type f ! -name "*.py" ! -name "*.yaml" | sort | head -20
echo ""
echo "Total files: $(find "$OUTPUT_DIR" -type f | wc -l)"
echo "Bundle size: $(du -sh "$OUTPUT_DIR" | cut -f1)"
echo ""
echo "=== Bundle Complete ==="
