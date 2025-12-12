#!/usr/bin/env python3
"""
Comprehensive test script that starts server, worker, runs all tests, and shuts down.

This script:
1. Starts the Flask server in background
2. Starts the retrack worker in background
3. Waits for services to be ready
4. Runs all test suites
5. Collects results
6. Shuts down all services
7. Reports final status

Usage: uv run python scripts/test_full_system.py
"""

from __future__ import annotations

import atexit
import signal
import subprocess
import sys
import time
from pathlib import Path

import httpx

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


class ServiceManager:
    """Manages background services (server and worker)."""

    def __init__(self):
        self.server_process = None
        self.worker_process = None
        self.base_url = "http://localhost:5000"

    def start_server(self) -> bool:
        """Start the Flask server."""
        print("[1/4] Starting server...")
        try:
            # Use server/server.py directly since run_server.py may not exist
            self.server_process = subprocess.Popen(
                [sys.executable, "server/server.py"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=Path(__file__).parent.parent,
            )
            print(
                "  ✓ Server process started (PID: {})".format(
                    self.server_process.pid
                )
            )

            # Wait for server to be ready
            max_wait = 10
            for _ in range(max_wait):
                try:
                    response = httpx.get(f"{self.base_url}/healthz", timeout=1)
                    if response.status_code == 200:
                        print("  ✓ Server is ready")
                        return True
                except Exception:
                    time.sleep(1)

            print("  ✗ Server failed to start (timeout)")
            return False
        except Exception as exc:
            print(f"  ✗ Failed to start server: {exc}")
            return False

    def start_worker(self) -> bool:
        """Start the retrack worker."""
        print("\n[2/4] Starting retrack worker...")
        try:
            # Use server/retrack_worker.py directly
            self.worker_process = subprocess.Popen(
                [sys.executable, "server/retrack_worker.py"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=Path(__file__).parent.parent,
            )
            print(
                "  ✓ Worker process started (PID: {})".format(
                    self.worker_process.pid
                )
            )
            time.sleep(2)  # Give worker a moment to initialize
            return True
        except Exception as exc:
            print(f"  ✗ Failed to start worker: {exc}")
            return False

    def stop_all(self) -> None:
        """Stop all background services."""
        print("\n[Cleanup] Stopping services...")
        if self.server_process:
            try:
                self.server_process.terminate()
                self.server_process.wait(timeout=5)
                print("  ✓ Server stopped")
            except Exception:
                self.server_process.kill()
                print("  ✓ Server killed")

        if self.worker_process:
            try:
                self.worker_process.terminate()
                self.worker_process.wait(timeout=5)
                print("  ✓ Worker stopped")
            except Exception:
                self.worker_process.kill()
                print("  ✓ Worker killed")

    def check_health(self) -> bool:
        """Check if services are healthy."""
        try:
            response = httpx.get(f"{self.base_url}/healthz", timeout=2)
            return response.status_code == 200
        except Exception:
            return False


def run_test_script(script_name: str) -> tuple[bool, str]:
    """Run a test script and return (success, output)."""
    script_path = Path(__file__).parent / script_name
    if not script_path.exists():
        return False, f"Test script not found: {script_name}"

    try:
        # Run test with generous timeout - tests have internal timeouts (30 minutes)
        # but we allow extra time for setup/teardown
        result = subprocess.run(
            [sys.executable, str(script_path)],
            capture_output=True,
            text=True,
            timeout=2400,  # 40 minute absolute max (tests have 30min internal timeout + buffer)
            cwd=Path(__file__).parent.parent,
        )
        # Return success only if exit code is 0
        # Exit code 0 = test passed, non-zero = test failed
        success = result.returncode == 0
        output_text = result.stdout + result.stderr
        return success, output_text
    except subprocess.TimeoutExpired:
        return False, f"Test timed out after 5 minutes"
    except Exception as exc:
        return False, f"Error running test: {exc}"


def main() -> None:
    """Run full system test suite."""
    print("=" * 70)
    print("Full System Test Suite")
    print("=" * 70)
    print("\nThis will:")
    print("  1. Start server and retrack worker")
    print("  2. Run all test suites")
    print("  3. Shut down services")
    print("  4. Report results")
    print()

    manager = ServiceManager()

    # Register cleanup on exit
    def cleanup():
        manager.stop_all()

    atexit.register(cleanup)
    signal.signal(signal.SIGINT, lambda s, f: (cleanup(), sys.exit(1)))
    signal.signal(signal.SIGTERM, lambda s, f: (cleanup(), sys.exit(1)))

    # Start services
    if not manager.start_server():
        print("\n❌ Failed to start server. Aborting.")
        sys.exit(1)

    if not manager.start_worker():
        print(
            "\n⚠️  Failed to start worker. Continuing with server-only tests..."
        )

    # Wait a moment for everything to settle
    time.sleep(2)

    # Run tests
    print("\n[3/4] Running test suites...")
    print("=" * 70)

    test_results = []

    # Test 0: Core Server Operation (startup verification)
    print("\n📋 Test: Core Server Operation (Startup Verification)")
    print("-" * 70)
    print(
        "  ⏳ Verifying server startup: dataset download and mask generation..."
    )
    success, output = run_test_script("test_startup_verification.py")
    test_results.append(("Core Server Operation (Startup)", success, output))
    if success:
        print("  ✓ PASSED - Server startup completed successfully")
        if "All series have masks" in output:
            print("     ✓ Dataset verified")
            print("     ✓ Masks generated for all series")
    else:
        print("  ✗ FAILED - Server startup did not complete correctly")
        print(output[-800:])
        print("\n  ⚠️  Other tests may fail if startup did not complete")

    # Test 1: Retrack queue operations (doesn't need server)
    print("\n📋 Test: Retrack Queue Operations")
    print("-" * 70)
    success, output = run_test_script("test_retrack_worker.py")
    test_results.append(("Retrack Queue", success, output))
    if success:
        print("  ✓ PASSED")
    else:
        print("  ✗ FAILED")
        print(output[-500:])  # Last 500 chars of output

    # Test 2: Lazy tracking (needs server) - takes 15-25 seconds
    print("\n📋 Test: Lazy Tracking (Initial Tracking)")
    print("-" * 70)
    print("  ⏳ This test triggers tracking and waits for completion...")
    success, output = run_test_script("test_lazy_tracking.py")
    test_results.append(("Lazy Tracking (Initial)", success, output))
    if success:
        # Verify test actually completed tracking successfully
        if "Tracking completed" in output and "Verified:" in output:
            # Extract elapsed time if present
            import re

            time_match = re.search(r"completed.*?in (\d+)s", output.lower())
            if time_match:
                elapsed = int(time_match.group(1))
                print(f"  ✓ PASSED (tracking completed in {elapsed}s)")
            else:
                print("  ✓ PASSED (tracking completed)")
        elif "Tracking failed" in output or "✗" in output:
            print("  ✗ FAILED - tracking did not complete successfully")
            print(
                "     This means no series with completed tracking is available for retracking test"
            )
        else:
            print("  ⚠ PASSED but tracking completion not verified")
        print(output[-800:])  # Show output to verify waiting occurred
    else:
        print("  ✗ FAILED")
        print(
            "     Tracking did not complete - this will prevent retracking test from running"
        )
        print(output[-800:])

    # Test 3: Full API flow including RETRACKING (needs server and worker)
    print("\n📋 Test: Server API (includes retracking)")
    print("-" * 70)
    print("  ⏳ This test includes retracking which takes 15-25 seconds...")
    success, output = run_test_script("test_server_api.py")
    test_results.append(("Server API (Retracking)", success, output))
    if success:
        # Verify retrack actually completed and was verified
        if "retrack completed in" in output.lower() and "Verified:" in output:
            # Extract elapsed time if present
            import re

            time_match = re.search(
                r"retrack completed in (\d+)s", output.lower()
            )
            if time_match:
                elapsed = int(time_match.group(1))
                print(f"  ✓ PASSED (retrack completed in {elapsed}s)")
            else:
                print("  ✓ PASSED (retrack completed)")
        else:
            print("  ⚠ PASSED but retrack may not have been verified")
        print(output[-1000:])  # Show output to verify waiting occurred
    else:
        print("  ✗ FAILED")
        # Show more output for retracking failures
        print(output[-1000:])

    # Shutdown
    print("\n[4/4] Shutting down services...")
    manager.stop_all()

    # Report results
    print("\n" + "=" * 70)
    print("Test Results Summary")
    print("=" * 70)

    passed = sum(1 for _, success, _ in test_results if success)
    total = len(test_results)

    for test_name, success, _ in test_results:
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"  {status} - {test_name}")

    print("\n" + "=" * 70)
    print(f"Total: {passed}/{total} tests passed")
    print("=" * 70)

    if passed == total:
        print("\n✅ All tests passed!")
        sys.exit(0)
    else:
        print(f"\n❌ {total - passed} test(s) failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
