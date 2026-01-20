#!/usr/bin/env python3
"""Minimal test to verify Python execution"""

import sys
import os

# Write to both stdout and stderr
sys.stdout.write("PYTHON TEST: stdout working\n")
sys.stdout.flush()
sys.stderr.write("PYTHON TEST: stderr working\n")
sys.stderr.flush()

print("PYTHON TEST: print() working")
print(f"Python version: {sys.version}")
print(f"Working directory: {os.getcwd()}")
print(f"Script path: {__file__}")

# Create a marker file to prove we ran
marker_file = os.path.join(os.path.expanduser("~"), "Documents", "python_test_ran.txt")
os.makedirs(os.path.dirname(marker_file), exist_ok=True)
with open(marker_file, "w") as f:
    f.write(f"Python test ran successfully!\n")
    f.write(f"Python version: {sys.version}\n")
    f.write(f"Working directory: {os.getcwd()}\n")
print(f"Created marker file: {marker_file}")

print("PYTHON TEST COMPLETE - exiting in 5 seconds")
import time

time.sleep(5)
print("PYTHON TEST EXITING NOW")
