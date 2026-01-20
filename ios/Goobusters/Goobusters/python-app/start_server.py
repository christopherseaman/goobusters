"""iOS app entry point - starts the Flask server"""
import os
import sys

# Add bundle path to Python path
bundle_path = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, bundle_path)

# Configure iOS environment
from ios_config import configure_environment, get_ios_paths
paths = configure_environment()

# Log to file in cache dir for debugging
try:
    log_dir = os.path.join(paths["CACHE_DIR"], "log")
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, "ios_python.log")
    log_file = open(log_path, "w", buffering=1)
    sys.stdout = log_file
    sys.stderr = log_file
except Exception:
    pass

def main(port=8080):
    """Start the Goobusters client Flask server"""
    print(f"[iOS] Starting Flask server on port {port}...")
    print(f"[iOS] Bundle path: {bundle_path}")
    print(f"[iOS] Data dir: {paths['DATA_DIR']}")

    # Import and create the Flask app
    from lib.client.start import create_app
    # Skip startup dataset sync; frontend handles blocking sync modal
    app = create_app(skip_startup=True)

    # Run Flask (this blocks)
    app.run(host="127.0.0.1", port=port, threaded=True, use_reloader=False)

if __name__ == "__main__":
    # Call main directly without argparse (argparse hangs on iOS)
    main(port=8080)
