#!/usr/bin/env python3
"""
Simple test script to verify Python is working in the iOS app.
This creates a minimal test that prints to console.
"""

import sys
import os


def main():
    print("=" * 60)
    print("PYTHON TEST - iOS App")
    print("=" * 60)
    print(f"Python version: {sys.version}")
    print(f"Python executable: {sys.executable}")
    print(f"Current working directory: {os.getcwd()}")
    print(f"sys.path:")
    for p in sys.path:
        print(f"  - {p}")
    print("=" * 60)
    print("Python is working correctly!")
    print("=" * 60)

    # Keep the script running for a bit so we can see output
    import time

    print("Waiting 10 seconds before exit...")
    time.sleep(10)
    print("Test complete - exiting")


if __name__ == "__main__":
    main()
