#!/bin/bash
# Peresmeshnik.app and its .dmg on macOS (arm64), ad-hoc signed. From the repo root, after
# `uv sync --locked --group build`:  packaging/macos/build.sh
#   -> dist/Peresmeshnik.app, dist/Peresmeshnik-<version>-macos-arm64.dmg
# Models are not in the bundle: the setup wizard downloads them.
set -euo pipefail
cd "$(dirname "$0")/../.."
PY=.venv/bin/python
VERSION=$($PY -c 'import peresmeshnik; print(peresmeshnik.__version__)')
APP=dist/Peresmeshnik.app
DMG=dist/Peresmeshnik-$VERSION-macos-arm64.dmg

# the Piper runner is its own program (GPL-3.0): built from piper-runner/, never imported by the app.
# Its packages are hash-pinned; --no-deps: onnxruntime and numpy stay as uv.lock has them
uv pip install --python "$PY" --no-deps --require-hashes -r packaging/piper-requirements.txt
$PY -m PyInstaller --noconfirm --clean --distpath dist --workpath build/pyinstaller packaging/macos/Peresmeshnik.spec

if compgen -G 'native/macos/tap/*.swift' > /dev/null; then  # the Core Audio process tap of the call app (S1)
  # minos 12: without it the helper gets the runner's 15.0 and would not start on macOS 14 at all
  swiftc -O -target arm64-apple-macos12 native/macos/tap/*.swift -o "$APP/Contents/MacOS/peresmeshnik-tap"
elif [ "${REQUIRE_TAP:-}" = 1 ]; then  # a release without the call's sound would be no release
  echo 'no native/macos/tap/*.swift: the call sound helper is missing' >&2
  exit 1
fi
if [ -f native/macos/audio/main.swift ]; then  # the Core Audio helper of me -> them (S2): BlackHole 2ch, the microphone
  native/macos/audio/build.sh
  cp native/macos/audio/build/peresmeshnik-audio "$APP/Contents/MacOS/"
  test -x "$APP/Contents/MacOS/peresmeshnik-audio"
elif [ "${REQUIRE_TAP:-}" = 1 ]; then  # the release guard: without it the call hears nothing of the translation
  echo 'no native/macos/audio/main.swift: the me -> them helper is missing' >&2
  exit 1
fi
# the GPL check (the app's own Analysis must not import piper; the spec copies its warnings before piper's) and the licenses
$PY packaging/licenses.py "$APP/Contents/Resources/licenses" build/pyinstaller/Peresmeshnik/warn-app.txt Contents/MacOS/piper-runner

codesign --force --deep --sign - "$APP"  # ad-hoc: no Apple ID; macOS asks to "Open Anyway" once (README)
codesign --verify --deep --strict --verbose=1 "$APP"

STAGE=dist/dmg  # the .dmg's contents; kept, and brought up to date by the next build
mkdir -p "$STAGE"
rsync -a --delete "$APP/" "$STAGE/Peresmeshnik.app/"
ln -sfn /Applications "$STAGE/Applications"
hdiutil create -volname "Peresmeshnik $VERSION" -srcfolder "$STAGE" -ov -format UDZO "$DMG"
du -sh "$APP" "$DMG"
