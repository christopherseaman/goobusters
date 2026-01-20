"""iOS-specific configuration for Goobusters"""

import os
import json
import sys
import types
from pathlib import Path

# Stub heavy dependencies BEFORE any mdai imports
# This allows the stock MD.ai SDK to work on iOS without heavy deps

# Find vendor directory (should be in python-app/vendor)
_current_file = os.path.abspath(__file__)
_python_app_dir = os.path.dirname(_current_file)
_vendor_dir = os.path.join(_python_app_dir, "vendor")

if os.path.exists(_vendor_dir) and _vendor_dir not in sys.path:
    sys.path.insert(0, _vendor_dir)

# Stub pydicom and all submodules that MD.ai SDK might import
if "pydicom" not in sys.modules:
    _pydicom_stub = types.ModuleType("pydicom")
    _pydicom_stub.__version__ = "0.0.0"
    
    # Stub pydicom.filereader
    _pydicom_filereader_stub = types.ModuleType("pydicom.filereader")
    _pydicom_filereader_stub.dcmread = lambda *args, **kwargs: None  # No-op function
    _pydicom_stub.filereader = _pydicom_filereader_stub
    
    # Stub pydicom.dataset
    _pydicom_dataset_stub = types.ModuleType("pydicom.dataset")
    _pydicom_dataset_stub.Dataset = type("Dataset", (), {})  # Empty class
    _pydicom_dataset_stub.FileMetaDataset = type("FileMetaDataset", (), {})  # Empty class
    _pydicom_stub.dataset = _pydicom_dataset_stub
    
    # Stub pydicom.pixel_data_handlers.util
    _pydicom_pixel_handlers_stub = types.ModuleType("pydicom.pixel_data_handlers")
    _pydicom_pixel_handlers_util_stub = types.ModuleType("pydicom.pixel_data_handlers.util")
    _pydicom_pixel_handlers_util_stub.pack_bits = lambda *args, **kwargs: None  # No-op function
    _pydicom_pixel_handlers_stub.util = _pydicom_pixel_handlers_util_stub
    _pydicom_stub.pixel_data_handlers = _pydicom_pixel_handlers_stub
    
    # Stub pydicom.sequence
    _pydicom_sequence_stub = types.ModuleType("pydicom.sequence")
    _pydicom_sequence_stub.Sequence = type("Sequence", (), {})  # Empty class
    _pydicom_stub.sequence = _pydicom_sequence_stub
    
    sys.modules["pydicom"] = _pydicom_stub
    sys.modules["pydicom.filereader"] = _pydicom_filereader_stub
    sys.modules["pydicom.dataset"] = _pydicom_dataset_stub
    sys.modules["pydicom.pixel_data_handlers"] = _pydicom_pixel_handlers_stub
    sys.modules["pydicom.pixel_data_handlers.util"] = _pydicom_pixel_handlers_util_stub
    sys.modules["pydicom.sequence"] = _pydicom_sequence_stub

# Stub numpy for MD.ai SDK (SDK imports it but we don't use it)
if "numpy" not in sys.modules:
    _numpy_stub = types.ModuleType("numpy")
    _numpy_stub.__version__ = "0.0.0"
    # Stub numpy._core._multiarray_umath to prevent import errors
    _numpy_core_stub = types.ModuleType("numpy._core")
    _numpy_core_multiarray_stub = types.ModuleType("numpy._core._multiarray_umath")
    _numpy_core_stub._multiarray_umath = _numpy_core_multiarray_stub
    _numpy_stub._core = _numpy_core_stub
    sys.modules["numpy"] = _numpy_stub
    sys.modules["numpy._core"] = _numpy_core_stub
    sys.modules["numpy._core._multiarray_umath"] = _numpy_core_multiarray_stub

# Stub pandas for MD.ai SDK (SDK imports it in dicom_utils.py but we use our own json_to_dataframe)
# We must stub it BEFORE the SDK imports it, so it's here at module import time
if "pandas" not in sys.modules:
    _pandas_stub = types.ModuleType("pandas")
    _pandas_stub.__version__ = "0.0.0"
    # Stub DataFrame class so SDK can import it (even though it won't work)
    _pandas_stub.DataFrame = type("DataFrame", (), {})
    sys.modules["pandas"] = _pandas_stub

# Stub other optional modules (all used in optional features we don't use)
for mod_name in ["nibabel", "dicom2nifti", "cv2", "yaml"]:
    if mod_name not in sys.modules:
        stub = types.ModuleType(mod_name)
        stub.__version__ = "0.0.0"
        if mod_name == "cv2":
            stub.imread = lambda *args: None
        elif mod_name == "yaml":
            stub.safe_load = lambda *args: {}
            stub.dump = lambda *args, **kwargs: ""
        sys.modules[mod_name] = stub

# Stub the dicom_utils module BEFORE vendor.mdai.client imports it
_dicom_utils_stub = types.ModuleType("vendor.mdai.utils.dicom_utils")
_dicom_utils_stub.iterate_content_seq = lambda *args: None  # No-op function
sys.modules["vendor.mdai.utils.dicom_utils"] = _dicom_utils_stub


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
        "SETTINGS_FILE": os.path.join(documents_dir, "goobusters_settings.json"),
    }


def load_persisted_settings(settings_file: str) -> dict:
    """Load persisted settings from disk (token, user_email, etc.)"""
    if os.path.exists(settings_file):
        try:
            with open(settings_file, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            print(f"[ios_config] Warning: Could not load settings from {settings_file}: {e}")
    return {}


def save_persisted_settings(settings_file: str, settings: dict) -> None:
    """Save settings to disk"""
    try:
        os.makedirs(os.path.dirname(settings_file), exist_ok=True)
        with open(settings_file, "w") as f:
            json.dump(settings, f)
    except IOError as e:
        print(f"[ios_config] Warning: Could not save settings to {settings_file}: {e}")


def configure_environment():
    """Set up environment for iOS"""
    paths = get_ios_paths()

    # Ensure directories exist
    for key in ["DATA_DIR", "CACHE_DIR", "CLIENT_CACHE_DIR"]:
        os.makedirs(paths[key], exist_ok=True)

    # Set environment variables for paths (override any existing values)
    os.environ["DATA_DIR"] = paths["DATA_DIR"]
    os.environ["CLIENT_CACHE_DIR"] = paths["CLIENT_CACHE_DIR"]

    # Load persisted settings (token, user_email)
    persisted = load_persisted_settings(paths["SETTINGS_FILE"])
    persisted_token = persisted.get("mdai_token", "not_configured_yet")
    persisted_email = persisted.get("user_email")

    # Set MD.ai config values for iOS
    # Use persisted token if available, otherwise placeholder
    os.environ["MDAI_TOKEN"] = persisted_token
    os.environ["DOMAIN"] = "ucsf.md.ai"  # Match dot.env.defaults
    os.environ["PROJECT_ID"] = "x9N2LJBZ"  # Match dot.env.defaults
    os.environ["DATASET_ID"] = "D_V688LQ"  # Match dot.env.defaults
    os.environ["LABEL_ID"] = "L_13yPql"  # Match dot.env.defaults
    os.environ["EMPTY_ID"] = "L_75K42J"  # Match dot.env.defaults
    os.environ["FLOW_METHOD"] = "dis"
    
    # Set user email if persisted (for ClientConfig)
    if persisted_email:
        # Note: USER_EMAIL is read by load_config() for ClientConfig.user_email
        os.environ["USER_EMAIL"] = persisted_email
    
    # SERVER_URL is loaded from dot.env.defaults (or overridden by os.environ if set)
    # Don't override it here - let it come from config files
    # Only set fallback if truly not set (config loading happens after this, so check if it will be set)
    # Note: dot.env.defaults has SERVER_URL=http://10.1.0.10:5000 for simulator
    # We should NOT override with localhost:5000 as that won't work from simulator

    # Store settings file path for later use
    paths["SETTINGS_FILE"] = paths["SETTINGS_FILE"]

    return paths
