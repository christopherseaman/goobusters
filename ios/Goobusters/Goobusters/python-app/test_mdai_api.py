#!/usr/bin/env python3
"""Test that iOS client can actually talk to MD.ai API"""

import sys
import os

# Add vendor to path
sys.path.insert(0, 'vendor')

from lib.client.ios_mdai_client import MDaiClient
from lib.config import load_config

config = load_config("client")

print("=" * 60)
print("Testing MD.ai API connection with iOS client")
print("=" * 60)

# Create client
try:
    client = MDaiClient(
        domain=config.domain,
        access_token=config.mdai_token
    )
    print(f"✓ Client created: {config.domain}")
except Exception as e:
    print(f"✗ Failed to create client: {e}")
    sys.exit(1)

# Try a simple API call (just test connectivity, don't download)
print("\nTesting API connectivity...")
try:
    # Make a simple request to check if token works
    response = client._request("GET", "/api/projects")
    print(f"✓ API responded with status: {response.status_code}")
    print(f"✓ Authentication successful")
    
    projects = response.json()
    print(f"✓ Found {len(projects)} projects")
    
except Exception as e:
    print(f"✗ API call failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 60)
print("✓ iOS client successfully communicates with MD.ai API!")
print("=" * 60)
