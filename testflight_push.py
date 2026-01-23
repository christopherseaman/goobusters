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
from pathlib import Path

import yaml

SCRIPT_DIR = Path(__file__).parent
PROJECT = SCRIPT_DIR / "ios/Goobusters/Goobusters.xcodeproj"
INFO_PLIST = SCRIPT_DIR / "ios/Goobusters/Goobusters/Info.plist"
SCHEME = "Goobusters"
BUILD_DIR = SCRIPT_DIR / "build"
DERIVED_DATA = BUILD_DIR / "DerivedData"
ARCHIVE_PATH = BUILD_DIR / "Goobusters.xcarchive"
EXPORT_PATH = BUILD_DIR / "export"


def run(cmd: list[str], check: bool = True, **kwargs) -> subprocess.CompletedProcess:
    print(f"$ {' '.join(cmd)}")
    return subprocess.run(cmd, check=check, **kwargs)


def load_config() -> dict:
    with open(SCRIPT_DIR / "dot.yaml") as f:
        return yaml.safe_load(f)


def increment_build_number() -> str:
    """Increment CFBundleVersion in Info.plist and return new version."""
    with open(INFO_PLIST, "rb") as f:
        plist = plistlib.load(f)
    
    current = int(plist.get("CFBundleVersion", "1"))
    new_version = str(current + 1)
    plist["CFBundleVersion"] = new_version
    
    with open(INFO_PLIST, "wb") as f:
        plistlib.dump(plist, f)
    
    return new_version


def main():
    skip_build = "--upload" in sys.argv

    # Load config
    config = load_config()
    app_store = config.get("app_store", {})
    key_id = app_store.get("key_id")
    issuer_id = app_store.get("issuer_id")
    bundle_id = app_store.get("bundle_id", "org.badmath.goobusters")

    if not key_id or not issuer_id:
        print("Error: app_store.key_id and app_store.issuer_id must be set in dot.yaml")
        sys.exit(1)

    # Verify API key exists
    key_path = Path.home() / f".appstoreconnect/private_keys/AuthKey_{key_id}.p8"
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
        print("=== Incrementing Build Number ===")
        new_build = increment_build_number()
        print(f"Build number: {new_build}")
        
        # Bundle Python code
        print()
        print("=== Bundling Python Code ===")
        run(["bash", "ios/scripts/bundle_python.sh"], cwd=SCRIPT_DIR)

        # Create archive
        print()
        print("=== Creating Archive ===")
        run([
            "xcodebuild",
            "-project", str(PROJECT),
            "-scheme", SCHEME,
            "-sdk", "iphoneos",
            "-configuration", "Release",
            "-derivedDataPath", str(DERIVED_DATA),
            "-archivePath", str(ARCHIVE_PATH),
            "archive"
        ])

        if not ARCHIVE_PATH.exists():
            print("Error: Archive failed")
            sys.exit(1)
        print(f"Archive created: {ARCHIVE_PATH}")

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
    
    run([
        "xcodebuild",
        "-exportArchive",
        "-archivePath", str(ARCHIVE_PATH),
        "-exportPath", str(EXPORT_PATH),
        "-exportOptionsPlist", str(export_options),
        "-allowProvisioningUpdates"
    ], env=env)

    print()
    print("=== Done ===")
    print("Build uploaded to App Store Connect")
    print("Check TestFlight in App Store Connect for processing status")


if __name__ == "__main__":
    main()
