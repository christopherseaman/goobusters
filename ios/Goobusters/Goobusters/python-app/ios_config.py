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
