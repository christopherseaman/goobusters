#!/usr/bin/env python3
"""Test iOS client with real MD.ai API"""

import sys
import os

# Use iOS client code
sys.path.insert(0, 'ios/Goobusters/Goobusters/python-app')
sys.path.insert(0, 'ios/Goobusters/Goobusters/python-app/vendor')

from ios.Goobusters.Goobusters.python_app.lib.client.ios_mdai_client import MDaiClient
from lib.config import load_config

config = load_config("shared")

print("=" * 60)
print("Testing MD.ai API with iOS httpx-based client")
print("=" * 60)

try:
    client = MDaiClient(
        domain=config.domain,
        access_token=config.mdai_token
    )
    print(f"✓ Client created for domain: {config.domain}")
except Exception as e:
    print(f"✗ Failed to create client: {e}")
    sys.exit(1)

# Test API connectivity
print("\nTesting API request...")
try:
    response = client._request("GET", "/api/projects")
    print(f"✓ API responded: HTTP {response.status_code}")
    
    projects = response.json()
    print(f"✓ Received {len(projects)} projects")
    
    if projects:
        print(f"  First project: {projects[0].get('name', 'N/A')}")
    
except Exception as e:
    print(f"✗ API call failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 60)
print("✓ SUCCESS: iOS client works with real MD.ai API!")
print("=" * 60)
