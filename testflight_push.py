#!/usr/bin/env python3
"""Build and upload Goobusters to TestFlight.

Prerequisites:
    - App Store Connect API key in ~/.appstoreconnect/private_keys/AuthKey_KEYID.p8
    - dot.yaml configured with app_store.key_id and app_store.issuer_id

Usage:
    uv run python3 testflight_push.py           # Full build + upload
    uv run python3 testflight_push.py --upload  # Upload existing archive (skip build)
"""

import os
import plistlib
import shutil
import subprocess
import sys
import time
from pathlib import Path

import jwt
import requests
import yaml

SCRIPT_DIR = Path(__file__).parent
PROJECT = SCRIPT_DIR / "ios/Goobusters/Goobusters.xcodeproj"
INFO_PLIST = SCRIPT_DIR / "ios/Goobusters/Goobusters/Info.plist"
SCHEME = "Goobusters"
BUILD_DIR = SCRIPT_DIR / "build"
DERIVED_DATA = BUILD_DIR / "DerivedData"
ARCHIVE_PATH = BUILD_DIR / "Goobusters.xcarchive"
EXPORT_PATH = BUILD_DIR / "export"


def run(
    cmd: list[str], check: bool = True, **kwargs
) -> subprocess.CompletedProcess:
    print(f"$ {' '.join(cmd)}")
    return subprocess.run(cmd, check=check, **kwargs)


def load_config() -> dict:
    with open(SCRIPT_DIR / "dot.yaml") as f:
        return yaml.safe_load(f)


def set_build_number() -> str:
    """Set CFBundleVersion to Unix timestamp and return it."""
    build_number = str(int(time.time()))

    with open(INFO_PLIST, "rb") as f:
        plist = plistlib.load(f)

    plist["CFBundleVersion"] = build_number

    with open(INFO_PLIST, "wb") as f:
        plistlib.dump(plist, f)

    return build_number


def get_build_number() -> str:
    """Get current CFBundleVersion from Info.plist."""
    with open(INFO_PLIST, "rb") as f:
        plist = plistlib.load(f)
    return plist.get("CFBundleVersion", "1")


def get_last_commit_message() -> str:
    """Get the last git commit message."""
    result = subprocess.run(
        ["git", "log", "-1", "--pretty=%B"],
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()


def set_test_notes(
    key_id: str,
    issuer_id: str,
    key_path: Path,
    bundle_id: str,
    build_number: str,
):
    """Wait for build to process and set 'What to Test' from last commit message."""
    # Generate JWT token
    private_key = key_path.read_text()
    token = jwt.encode(
        {
            "iss": issuer_id,
            "exp": int(time.time()) + 1200,
            "aud": "appstoreconnect-v1",
        },
        private_key,
        algorithm="ES256",
        headers={"kid": key_id},
    )
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }
    api = "https://api.appstoreconnect.apple.com/v1"

    # Get app ID
    resp = requests.get(
        f"{api}/apps?filter[bundleId]={bundle_id}", headers=headers
    )
    resp.raise_for_status()
    app_id = resp.json()["data"][0]["id"]

    # Wait for build to be processed
    print(
        f"Waiting for build {build_number} to be processed", end="", flush=True
    )
    build_id = None
    for _ in range(30):  # 10 min max (30 * 20s)
        resp = requests.get(
            f"{api}/apps/{app_id}/builds?limit=10", headers=headers
        )
        resp.raise_for_status()
        for build in resp.json().get("data", []):
            if build["attributes"].get("version") == build_number:
                if build["attributes"].get("processingState") == "VALID":
                    build_id = build["id"]
                    break
        if build_id:
            break
        print(".", end="", flush=True)
        time.sleep(20)
    print()

    if not build_id:
        raise TimeoutError(f"Build {build_number} not ready after 10 minutes")
    print(f"Build {build_number} ready")

    # Get commit message and set "What to Test"
    commit_msg = get_last_commit_message()
    print(
        f"Setting test notes: {commit_msg[:60]}{'...' if len(commit_msg) > 60 else ''}"
    )

    # Get existing localization or create new
    resp = requests.get(
        f"{api}/builds/{build_id}/betaBuildLocalizations", headers=headers
    )
    resp.raise_for_status()
    loc_id = None
    for loc in resp.json().get("data", []):
        if loc["attributes"].get("locale") == "en-US":
            loc_id = loc["id"]
            break

    if loc_id:
        payload = {
            "data": {
                "type": "betaBuildLocalizations",
                "id": loc_id,
                "attributes": {"whatsNew": commit_msg},
            }
        }
        resp = requests.patch(
            f"{api}/betaBuildLocalizations/{loc_id}",
            headers=headers,
            json=payload,
        )
    else:
        payload = {
            "data": {
                "type": "betaBuildLocalizations",
                "attributes": {"locale": "en-US", "whatsNew": commit_msg},
                "relationships": {
                    "build": {"data": {"type": "builds", "id": build_id}}
                },
            }
        }
        resp = requests.post(
            f"{api}/betaBuildLocalizations", headers=headers, json=payload
        )
    resp.raise_for_status()
    print("Test notes set successfully")


def main():
    skip_build = "--upload" in sys.argv

    # Load config
    config = load_config()
    app_store = config.get("app_store", {})
    key_id = app_store.get("key_id")
    issuer_id = app_store.get("issuer_id")
    bundle_id = app_store.get("bundle_id", "org.badmath.goobusters")

    if not key_id or not issuer_id:
        print(
            "Error: app_store.key_id and app_store.issuer_id must be set in dot.yaml"
        )
        sys.exit(1)

    # Verify API key exists
    key_path = (
        Path.home() / f".appstoreconnect/private_keys/AuthKey_{key_id}.p8"
    )
    if not key_path.exists():
        print(f"Error: API key not found at {key_path}")
        print("Download from App Store Connect → Users and Access → Keys")
        sys.exit(1)

    print("=== TestFlight Push ===")
    print(f"  Key ID: {key_id}")
    print(f"  Issuer: {issuer_id}")
    print(f"  Bundle: {bundle_id}")
    print()

    if not skip_build:
        # Clean build
        print("=== Cleaning Build ===")
        if BUILD_DIR.exists():
            shutil.rmtree(BUILD_DIR)
        # Also clean global DerivedData for Goobusters
        derived_data = Path.home() / "Library/Developer/Xcode/DerivedData"
        for p in derived_data.glob("Goobusters-*"):
            shutil.rmtree(p)
        BUILD_DIR.mkdir(parents=True)

        # Increment build number
        print()
        print("=== Setting Build Number ===")
        build_number = set_build_number()
        print(f"Build number: {build_number}")

        # Bundle Python code
        print()
        print("=== Bundling Python Code ===")
        run(["bash", "ios/scripts/bundle_python.sh"], cwd=SCRIPT_DIR)

        # Create archive
        print()
        print("=== Creating Archive ===")
        run([
            "xcodebuild",
            "-project",
            str(PROJECT),
            "-scheme",
            SCHEME,
            "-sdk",
            "iphoneos",
            "-configuration",
            "Release",
            "-derivedDataPath",
            str(DERIVED_DATA),
            "-archivePath",
            str(ARCHIVE_PATH),
            "archive",
        ])

        if not ARCHIVE_PATH.exists():
            print("Error: Archive failed")
            sys.exit(1)
        print(f"Archive created: {ARCHIVE_PATH}")
    else:
        build_number = get_build_number()
        print(f"Using existing build: {build_number}")

    # Create ExportOptions.plist
    print()
    print("=== Exporting IPA ===")
    export_options = BUILD_DIR / "ExportOptions.plist"
    export_options.write_text("""\
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>method</key>
    <string>app-store-connect</string>
    <key>destination</key>
    <string>upload</string>
    <key>signingStyle</key>
    <string>automatic</string>
    <key>uploadSymbols</key>
    <false/>
</dict>
</plist>
""")

    # Export and upload directly to App Store Connect
    # (destination: upload in ExportOptions.plist handles the upload)
    print()
    print("=== Uploading to TestFlight ===")

    if EXPORT_PATH.exists():
        shutil.rmtree(EXPORT_PATH)

    # Use system rsync to avoid Homebrew rsync incompatibility with Xcode
    env = os.environ.copy()
    env["PATH"] = "/usr/bin:" + env.get("PATH", "")

    run(
        [
            "xcodebuild",
            "-exportArchive",
            "-archivePath",
            str(ARCHIVE_PATH),
            "-exportPath",
            str(EXPORT_PATH),
            "-exportOptionsPlist",
            str(export_options),
            "-allowProvisioningUpdates",
        ],
        env=env,
    )

    print()
    print("Build uploaded to App Store Connect")

    # Set test notes after build is processed
    print()
    print("=== Setting Test Notes ===")
    try:
        set_test_notes(key_id, issuer_id, key_path, bundle_id, build_number)
    except Exception as e:
        print(f"Warning: Failed to set test notes: {e}")
        print(
            "Build uploaded but test notes not set. Set manually in App Store Connect."
        )

    print()
    print("=== Done ===")


if __name__ == "__main__":
    main()
