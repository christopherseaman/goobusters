#!/usr/bin/env python3
"""Debug script to identify the exact source and type of the pydicom warning."""

import warnings
import sys

# Capture all warnings
warnings.simplefilter('always')

# Custom warning handler to see details
def warning_handler(message, category, filename, lineno, file=None, line=None):
    print(f"WARNING TYPE: {category.__name__}")
    print(f"MESSAGE: {message}")
    print(f"FILE: {filename}")
    print(f"LINE: {lineno}")
    print("=" * 50)

# Set custom warning handler
old_showwarning = warnings.showwarning
warnings.showwarning = warning_handler

print("Testing mdai import and basic usage...")
try:
    import mdai
    print(f"mdai version: {mdai.__version__ if hasattr(mdai, '__version__') else 'unknown'}")
    
    # Try basic mdai operations that might trigger pydicom
    print("Testing mdai operations...")
    # This might trigger the warning if mdai uses pydicom internally
    
except Exception as e:
    print(f"Error during test: {e}")

print("Testing direct pydicom import...")
try:
    import pydicom
    print(f"pydicom version: {pydicom.__version__}")
    
    # Try accessing the deprecated module directly
    try:
        from pydicom import pixel_data_handlers
        print("pixel_data_handlers module still accessible")
    except ImportError as e:
        print(f"pixel_data_handlers import failed: {e}")
        
    # Try the new recommended import
    try:
        from pydicom.pixels import pack_bits
        print("New pydicom.pixels.pack_bits import successful")
    except ImportError as e:
        print(f"New pydicom.pixels import failed: {e}")
        
except Exception as e:
    print(f"Error testing pydicom: {e}")

# Restore original warning handler
warnings.showwarning = old_showwarning