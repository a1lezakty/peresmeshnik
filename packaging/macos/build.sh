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
uv pip install --python "$PY" --no-deps --require-hashes -r packaging/macos/piper-requirements.txt
$PY -m PyInstaller --noconfirm --clean --distpath dist --workpath build/pyinstaller packaging/macos/Peresmeshnik.spec
WARN=build/pyinstaller/Peresmeshnik/warn-app.txt  # the app's own Analysis (the spec copies it before piper's)
test -s "$WARN" || { echo "no $WARN: the GPL check cannot run" >&2; exit 1; }
if grep -qE 'excluded module named piper( |\.)' "$WARN"; then  # piper and piper.voice, not piper_x
  echo 'the app imports piper (GPL): it must only talk to piper-runner over the pipe' >&2
  exit 1
fi

if compgen -G 'native/macos/tap/*.swift' > /dev/null; then  # the Core Audio process tap of the call app (S1)
  # minos 12: without it the helper gets the runner's 15.0 and would not start on macOS 14 at all
  swiftc -O -target arm64-apple-macos12 native/macos/tap/*.swift -o "$APP/Contents/MacOS/peresmeshnik-tap"
elif [ "${REQUIRE_TAP:-}" = 1 ]; then  # a release without the call's sound would be no release
  echo 'no native/macos/tap/*.swift: the call sound helper is missing' >&2
  exit 1
fi
L="$APP/Contents/Resources/licenses"
$PY packaging/licenses.py "$L"
cp LICENSE "$L/Peresmeshnik-LICENSE.txt"
cp piper-runner/LICENSE "$L/piper-runner-LICENSE.txt"
cat > "$L/piper-runner-SOURCE.txt" <<EOF
Contents/MacOS/piper-runner is a separate program under GPL-3.0-or-later, built with PyInstaller from:
- piper-runner/piper_runner.py of https://github.com/a1lezakty/peresmeshnik (version $VERSION, tag v$VERSION);
- piper-tts 1.8.0: https://github.com/OHF-Voice/piper1-gpl, tag v1.8.0 (its PyPI sdist has no C sources);
- espeak-ng, built statically into piper/espeakbridge.so: https://github.com/espeak-ng/espeak-ng, commit 724808c
  (the GIT_TAG in piper1-gpl v1.8.0's CMakeLists.txt);
- onnxruntime and numpy, whose licenses are in this folder.
Peresmeshnik itself never imports it; it talks to it over stdin/stdout.
EOF

codesign --force --deep --sign - "$APP"  # ad-hoc: no Apple ID; macOS asks to "Open Anyway" once (README)
codesign --verify --deep --strict --verbose=1 "$APP"

STAGE=dist/dmg  # the .dmg's contents; kept, and brought up to date by the next build
mkdir -p "$STAGE"
rsync -a --delete "$APP/" "$STAGE/Peresmeshnik.app/"
ln -sfn /Applications "$STAGE/Applications"
hdiutil create -volname "Peresmeshnik $VERSION" -srcfolder "$STAGE" -ov -format UDZO "$DMG"
du -sh "$APP" "$DMG"
