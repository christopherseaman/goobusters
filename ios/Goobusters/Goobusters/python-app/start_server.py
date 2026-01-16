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
