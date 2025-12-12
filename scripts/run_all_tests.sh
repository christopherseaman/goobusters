#!/bin/bash
# Run all test scripts for the distributed architecture

set -e

echo "============================================================"
echo "Running Distributed Architecture Test Suite"
echo "============================================================"
echo ""

# Check if server is running
echo "Checking if server is running..."
if ! curl -s http://localhost:5000/api/status > /dev/null 2>&1; then
    echo "⚠️  Server not running. Start it with:"
    echo "   uv run python server/server.py"
    echo ""
    echo "Continuing with queue tests only..."
    echo ""
    uv run python scripts/test_retrack_worker.py
else
    echo "✓ Server is running"
    echo ""
    
    # Run queue tests
    echo "Running retrack queue tests..."
    uv run python scripts/test_retrack_worker.py
    echo ""
    
    # Run API tests
    echo "Running server API tests..."
    uv run python scripts/test_server_api.py
fi

echo ""
echo "============================================================"
echo "Test suite completed"
echo "============================================================"
