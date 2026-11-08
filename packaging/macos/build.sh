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

# the Piper runner is its own program (GPL-3.0): built from piper-runner/, never imported by the app;
# the locked versions stay (onnxruntime, numpy): piper-tts only adds itself and pathvalidate
mkdir -p build
uv export --frozen --no-hashes --no-emit-project --format requirements.txt > build/constraints.txt
uv pip install --python "$PY" piper-tts==1.8.0 --constraint build/constraints.txt
$PY -m PyInstaller --noconfirm --clean --distpath dist --workpath build/pyinstaller packaging/macos/Peresmeshnik.spec
if grep -q 'excluded module named piper ' build/pyinstaller/Peresmeshnik/warn-Peresmeshnik.txt; then
  echo 'the app imports piper (GPL): it must only talk to piper-runner over the pipe' >&2
  exit 1
fi

if compgen -G 'native/macos/tap/*.swift' > /dev/null; then  # the Core Audio process tap of the call app (S1)
  # minos 12: without it the helper gets the runner's 15.0 and would not start on macOS 14 at all
  swiftc -O -target arm64-apple-macos12 native/macos/tap/*.swift -o "$APP/Contents/MacOS/peresmeshnik-tap"
fi
L="$APP/Contents/Resources/licenses"
$PY packaging/licenses.py "$L"
cp LICENSE "$L/Peresmeshnik-LICENSE.txt"
cp piper-runner/LICENSE "$L/piper-runner-LICENSE.txt"
cat > "$L/piper-runner-SOURCE.txt" <<EOF
Contents/MacOS/piper-runner is a separate program under GPL-3.0-or-later, built with PyInstaller from:
- piper-runner/piper_runner.py of https://github.com/a1lezakty/peresmeshnik (version $VERSION, tag v$VERSION);
- piper-tts 1.8.0 (with espeak-ng built in): https://pypi.org/project/piper-tts/1.8.0/#files (sdist),
  https://github.com/OHF-Voice/piper1-gpl (tag v1.8.0);
- onnxruntime and numpy, whose licenses are in this folder.
Peresmeshnik itself never imports it; it talks to it over stdin/stdout.
EOF

codesign --force --deep --sign - "$APP"  # ad-hoc: no Apple ID; macOS asks to "Open Anyway" once (README)
codesign --verify --deep --strict --verbose=1 "$APP"

STAGE=$(mktemp -d)
cp -R "$APP" "$STAGE/"
ln -s /Applications "$STAGE/Applications"
hdiutil create -volname "Peresmeshnik $VERSION" -srcfolder "$STAGE" -ov -format UDZO "$DMG"
du -sh "$APP" "$DMG"
