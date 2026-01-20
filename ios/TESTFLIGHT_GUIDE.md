# TestFlight Distribution Guide

This guide walks you through sharing your Goobusters iOS app via TestFlight for beta testing.

## Prerequisites

1. **Apple Developer Account** ($99/year)
   - Enroll at https://developer.apple.com/programs/
   - Ensure your account is active and in good standing

2. **App Store Connect Setup**
   - Go to https://appstoreconnect.apple.com
   - Create a new app (if not already created)
   - Fill in app information:
     - Name: Goobusters
     - Primary Language: English
     - Bundle ID: `org.badmath.goobusters` (must match Xcode project)
     - SKU: Any unique identifier (e.g., `goobusters-001`)

3. **Xcode Signing Configuration**
   - Open `ios/Goobusters/Goobusters.xcodeproj` in Xcode
   - Select the **Goobusters** target
   - Go to **Signing & Capabilities** tab
   - Check **Automatically manage signing**
   - Select your **Team** (your Apple Developer account)
   - Xcode will automatically create/update provisioning profiles

## Step 1: Update Version and Build Numbers

Before archiving, update your app version:

1. In Xcode, select the **Goobusters** target
2. Go to **General** tab
3. Update:
   - **Version**: e.g., `1.0` (user-facing version)
   - **Build**: e.g., `1` (increment for each upload)

Or edit `Info.plist` directly:
- `CFBundleShortVersionString`: Version (e.g., "1.0")
- `CFBundleVersion`: Build number (e.g., "1")

## Step 2: Configure Archive Build Settings

1. In Xcode, select the **Goobusters** scheme
2. Click the scheme dropdown → **Edit Scheme...**
3. Select **Archive** in the left sidebar
4. Set **Build Configuration** to **Release**
5. Click **Close**

## Step 3: Archive the App

### Method 1: Using Xcode (Recommended)

1. In Xcode, select **Any iOS Device** or **Generic iOS Device** from the device dropdown (not a simulator)
2. Go to **Product** → **Archive**
3. Wait for the archive to complete (this may take several minutes)
4. The **Organizer** window will open automatically when done

### Method 2: Using Command Line

```bash
cd /Users/christopher/Documents/goobusters/ios/Goobusters

# Clean build folder
xcodebuild clean -project Goobusters.xcodeproj -scheme Goobusters

# Create archive
xcodebuild archive \
  -project Goobusters.xcodeproj \
  -scheme Goobusters \
  -configuration Release \
  -archivePath ~/Desktop/Goobusters.xcarchive \
  -destination 'generic/platform=iOS' \
  CODE_SIGN_IDENTITY="Apple Development" \
  DEVELOPMENT_TEAM="YOUR_TEAM_ID"
```

Replace `YOUR_TEAM_ID` with your Apple Developer Team ID (found in Apple Developer account settings).

## Step 4: Upload to App Store Connect

### Method 1: Using Xcode Organizer

1. In the **Organizer** window (Window → Organizer if not open)
2. Select your archive
3. Click **Distribute App**
4. Select **App Store Connect**
5. Click **Next**
6. Select **Upload**
7. Click **Next**
8. Review signing options (usually "Automatically manage signing")
9. Click **Next**
10. Review summary and click **Upload**
11. Wait for upload to complete (progress shown in Organizer)

### Method 2: Using Command Line (xcrun altool)

```bash
# Validate the archive first
xcrun altool --validate-app \
  -f ~/Desktop/Goobusters.ipa \
  -t ios \
  --apiKey YOUR_API_KEY \
  --apiIssuer YOUR_ISSUER_ID

# Upload
xcrun altool --upload-app \
  -f ~/Desktop/Goobusters.ipa \
  -t ios \
  --apiKey YOUR_API_KEY \
  --apiIssuer YOUR_ISSUER_ID
```

**Note**: You'll need to export an IPA first (see below) and create an API key in App Store Connect.

### Method 3: Using Transporter App

1. Export the archive as an IPA:
   - In Organizer, select archive → **Distribute App**
   - Choose **App Store Connect** → **Export**
   - Save the `.ipa` file
2. Open **Transporter** app (from Mac App Store or Xcode)
3. Drag the `.ipa` file into Transporter
4. Click **Deliver**

## Step 5: Configure TestFlight in App Store Connect

1. Go to https://appstoreconnect.apple.com
2. Select your **Goobusters** app
3. Click **TestFlight** tab
4. Wait for processing (can take 5-30 minutes)
5. Once processing completes, you'll see the build listed

## Step 6: Add TestFlight Testers

### Internal Testing (Up to 100 testers)

1. In TestFlight, go to **Internal Testing**
2. Click **+** to create a new group (e.g., "Development Team")
3. Add testers:
   - Click **Add Testers**
   - Enter email addresses of team members
   - They must accept the invitation email
4. Select your build
5. Click **Start Testing**

### External Testing (Up to 10,000 testers, requires App Review)

1. In TestFlight, go to **External Testing**
2. Click **+** to create a new group
3. Add testers (same as above)
4. **Important**: You must provide:
   - App description
   - What to Test notes
   - Screenshots (at least one)
   - Privacy policy URL (if app collects data)
5. Submit for Beta App Review (can take 24-48 hours)
6. Once approved, testers can install

## Step 7: Testers Install the App

Testers will:

1. Receive an email invitation (or you can share a public link for external testing)
2. Install **TestFlight** app from App Store (if not already installed)
3. Open the invitation link or open TestFlight app
4. Tap **Accept** and **Install**
5. The app will appear on their home screen with an orange dot

## Troubleshooting

### Archive Fails with Signing Errors

- Ensure your Apple Developer account is properly configured in Xcode
- Check that Bundle ID matches App Store Connect
- Verify certificates are valid (Xcode → Preferences → Accounts → Download Manual Profiles)

### Upload Fails

- Check internet connection
- Verify App Store Connect API access (if using command line)
- Try using Xcode Organizer instead of command line
- Check that the build number is unique (increment if needed)

### Build Processing Takes Too Long

- Normal processing time: 5-30 minutes
- If stuck > 1 hour, check App Store Connect status page
- Try uploading a new build with incremented build number

### Testers Can't Install

- Verify they accepted the invitation email
- Check that their device is registered (for internal testing)
- Ensure they have TestFlight app installed
- Verify the build passed processing

### "Invalid Bundle" Errors

- Ensure all required app icons are included
- Check that Info.plist has all required keys
- Verify Python.xcframework is properly embedded and signed
- Check that minimum iOS version is set appropriately

## Incremental Builds

For subsequent TestFlight uploads:

1. **Increment Build Number** (not version, unless it's a new release)
   - In Xcode: General tab → Build number
   - Or edit `Info.plist`: `CFBundleVersion`
2. **Archive** again (Product → Archive)
3. **Upload** the new archive
4. **Wait for processing**
5. **Add to TestFlight group** (or it will auto-update if using same group)

## Best Practices

1. **Version Numbering**: Use semantic versioning (e.g., 1.0.0, 1.0.1, 1.1.0)
2. **Build Numbers**: Always increment, even for small fixes
3. **Test Notes**: Always provide "What to Test" notes for external testing
4. **Feedback**: Encourage testers to use TestFlight's built-in feedback feature
5. **Expiration**: TestFlight builds expire after 90 days (90 days from upload, not first install)

## Quick Reference Commands

```bash
# Check current version/build
plutil -p ios/Goobusters/Goobusters/Info.plist | grep -E "(CFBundleShortVersionString|CFBundleVersion)"

# Increment build number (macOS)
cd ios/Goobusters/Goobusters
/usr/libexec/PlistBuddy -c "Set :CFBundleVersion $(($(/usr/libexec/PlistBuddy -c "Print :CFBundleVersion" Info.plist) + 1))" Info.plist

# List available archives
ls -la ~/Library/Developer/Xcode/Archives/
```

## Additional Resources

- [App Store Connect Help](https://help.apple.com/app-store-connect/)
- [TestFlight Documentation](https://developer.apple.com/testflight/)
- [Xcode Archive Guide](https://developer.apple.com/documentation/xcode/distributing-your-app-for-beta-testing-and-releases)
