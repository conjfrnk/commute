#!/bin/bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"
# Override BUNDLE_ID in the environment to keep an existing install's identity (location grants, login item).
BUNDLE_ID="${BUNDLE_ID:-local.commute-bar}"

# The app runs its Python side through venv/bin/python; bootstrap it on first build.
if [ ! -x venv/bin/python ]; then
    # PYTHON overrides the search; otherwise take the newest interpreter on PATH that is new enough.
    PYTHON="${PYTHON:-}"
    if [ -z "$PYTHON" ]; then
        for candidate in python3.15 python3.14 python3.13 python3.12 python3; do
            if command -v "$candidate" >/dev/null 2>&1 && \
               "$candidate" -c 'import sys; sys.exit(sys.version_info[:2] < (3, 12))' 2>/dev/null; then
                PYTHON="$candidate"; break
            fi
        done
    fi
    if [ -z "$PYTHON" ] || ! "$PYTHON" -c 'import sys; sys.exit(sys.version_info[:2] < (3, 12))' 2>/dev/null; then
        echo "Commute Bar needs Python 3.12 or newer (for example: brew install python)" >&2
        exit 1
    fi
    "$PYTHON" -m venv venv
    venv/bin/pip install -r requirements.txt
    venv/bin/playwright install chromium
fi

swift build -c release
APP_DIR="$ROOT_DIR/build/Commute Bar.app"
mkdir -p "$APP_DIR/Contents/MacOS" "$APP_DIR/Contents/Resources"
ICONSET_DIR="$ROOT_DIR/.build/CommuteBar.iconset"
mkdir -p "$ICONSET_DIR"
for ICON_SIZE in 16 32 128 256 512; do
    sips -z "$ICON_SIZE" "$ICON_SIZE" "$ROOT_DIR/Assets/icon.png" --out "$ICONSET_DIR/icon_${ICON_SIZE}x${ICON_SIZE}.png" >/dev/null
    RETINA_SIZE=$((ICON_SIZE * 2))
    sips -z "$RETINA_SIZE" "$RETINA_SIZE" "$ROOT_DIR/Assets/icon.png" --out "$ICONSET_DIR/icon_${ICON_SIZE}x${ICON_SIZE}@2x.png" >/dev/null
done
iconutil -c icns "$ICONSET_DIR" -o "$APP_DIR/Contents/Resources/CommuteBar.icns"
cp .build/release/CommuteBar "$APP_DIR/Contents/MacOS/CommuteBar"
python3 - "$APP_DIR/Contents/Info.plist" "$ROOT_DIR" "$BUNDLE_ID" <<'PY'
import plistlib, sys
with open(sys.argv[1], 'wb') as f:
    plistlib.dump(dict(CFBundleName='Commute Bar', CFBundleDisplayName='Commute Bar',
        CFBundleIdentifier=sys.argv[3], CFBundleVersion='1',
        CFBundleShortVersionString='1.0', CFBundleExecutable='CommuteBar', CFBundlePackageType='APPL',
        CFBundleIconFile='CommuteBar',
        LSUIElement=True, LSMinimumSystemVersion='13.0', NSHighResolutionCapable=True,
        NSLocationUsageDescription='Choose your commute direction when you are at home or work.',
        NSLocationWhenInUseUsageDescription='Choose your commute direction when you are at home or work.',
        CommuteProjectPath=sys.argv[2]), f)
PY
# Ad-hoc signing defaults to a requirement of this build's cdhash, so every rebuild looks like a new app and macOS
# asks for location permission again. Pin the requirement to the bundle identifier so the grant survives rebuilds.
codesign --force --sign - --requirements "=designated => identifier \"$BUNDLE_ID\"" "$APP_DIR"
codesign --verify --deep --strict "$APP_DIR"
plutil -lint "$APP_DIR/Contents/Info.plist"
printf '%s\n' "$APP_DIR"
