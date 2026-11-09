#!/bin/bash
# Peresmeshnik for Windows (x64): the one-folder app, its .zip and the Inno Setup installer. From the repo root in Git
# Bash, after `uv sync --locked --group build`, with Inno Setup 6 installed:  packaging/windows/build.sh
#   -> dist/Peresmeshnik/, dist/Peresmeshnik-<version>-windows-x64.zip, dist/Peresmeshnik-<version>-windows-x64-setup.exe
# Models are not in the build: the setup wizard downloads them. Unsigned: SmartScreen asks once (README).
set -euo pipefail
cd "$(dirname "$0")/../.."
PY=.venv/Scripts/python.exe
VERSION=$($PY -c 'import peresmeshnik; print(peresmeshnik.__version__)')
APP=dist/Peresmeshnik
NAME=Peresmeshnik-$VERSION-windows-x64
ISCC=${ISCC:-"/c/Program Files (x86)/Inno Setup 6/ISCC.exe"}

# the Piper runner is its own program (GPL-3.0): built from piper-runner/, never imported by the app.
# Its packages are hash-pinned; --no-deps: onnxruntime and numpy stay as uv.lock has them
uv pip install --python "$PY" --no-deps --require-hashes -r packaging/piper-requirements.txt
$PY -m PyInstaller --noconfirm --clean --distpath dist --workpath build/pyinstaller packaging/windows/Peresmeshnik.spec
if [ -f native/windows/loopback/build.cmd ]; then  # the call app's sound: the WASAPI process loopback helper (S1, MSVC), given its full path
  MSYS_NO_PATHCONV=1 cmd /c 'native\windows\loopback\build.cmd' "$(cygpath -wa "$APP/peresmeshnik-loopback.exe")"
  test -f "$APP/peresmeshnik-loopback.exe"
elif [ "${REQUIRE_LOOPBACK:-}" = 1 ]; then  # a release without the call's sound would be no release
  echo 'no native/windows/loopback/build.cmd: the call sound helper is missing' >&2
  exit 1
fi
# the GPL check (the app's own Analysis must not import piper; the spec copies its warnings before piper's) and the licenses
$PY packaging/licenses.py "$APP/licenses" build/pyinstaller/Peresmeshnik/warn-app.txt piper-runner.exe

$PY -m zipfile -c "dist/$NAME.zip" "$APP"  # unpacks to a Peresmeshnik folder
MSYS_NO_PATHCONV=1 "$ISCC" /Q "/DVersion=$VERSION" packaging/windows/Peresmeshnik.iss  # Git Bash would rewrite /D... as a path
du -sh "$APP" "dist/$NAME.zip" "dist/$NAME-setup.exe"
