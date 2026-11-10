#!/bin/bash
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0 OR GPL-3.0-or-later
# (GPL too: it is part of the GPL program piper-runner's Corresponding Source)
# The sources a release carries next to its builds (release.yml, job sources), as one Peresmeshnik-<version>-sources.tar:
# - the GPL-3.0 piper-runner (piper-runner-SOURCE.txt): this repository's runner and build scripts at this commit,
#   piper1-gpl v1.8.0 and espeak-ng 724808c (the GIT_TAG in its CMakeLists.txt; built into piper's espeakbridge);
# - the LGPL-2.1 mpg123 in libsndfile of the Windows build (LGPL-SOURCE.txt): mpg123 1.32.9 and vcpkg's port of it at
#   b7601ed (portfile and patches), the version packaging/licenses.py finds there (MPG123): change both together.
# Each download is checked by the SHA-256 pinned here (mpg123's is the same file as the SHA-512 in vcpkg's portfile).
# Usage: packaging/sources.sh <version> <out dir>
set -euo pipefail
v=$1 out=$2
root=$(cd "$(dirname "$0")/.." && pwd)
tmp=$(mktemp -d)
src=$tmp/Peresmeshnik-$v-sources
mkdir -p "$src/piper-runner" "$src/mpg123/vcpkg-port-b7601ed" "$out"

get() {  # <url> <file> <sha256>
  curl -sfL --retry 3 -o "$2.part" "$1" || { echo "sources: cannot download $1" >&2; exit 1; }
  echo "$3  $2.part" | sha256sum -c --quiet - || { echo "sources: $1 is not the file pinned here (SHA-256)" >&2; exit 1; }
  mv "$2.part" "$2"
}

get https://github.com/OHF-Voice/piper1-gpl/archive/refs/tags/v1.8.0.tar.gz "$src/piper-runner/piper1-gpl-1.8.0.tar.gz" \
  0a22987a6157f14e2fb7cf4aa702b547467fd0448bea9f233abd503c3ff4a0e1
esp=724808c5a83f9ef95fdd0db886ba7ba537ff224a
get https://github.com/espeak-ng/espeak-ng/archive/$esp.tar.gz "$src/piper-runner/espeak-ng-$esp.tar.gz" \
  7d8ebd78201923a443245362a928999977d1721c3eeba596eafc6ea8db7e243c
git -C "$root" archive --format=tar.gz --prefix="peresmeshnik-$v/" -o "$src/piper-runner/peresmeshnik-$v-runner.tar.gz" \
  HEAD piper-runner packaging LICENSE pyproject.toml uv.lock  # the runner, its build scripts, the pinned build tools

get https://downloads.sourceforge.net/project/mpg123/mpg123/1.32.9/mpg123-1.32.9.tar.bz2 "$src/mpg123/mpg123-1.32.9.tar.bz2" \
  03b61e4004e960bacf2acdada03ed94d376e6aab27a601447bd4908d8407b291
port=https://raw.githubusercontent.com/microsoft/vcpkg/b7601ed37e1e1ce11a4f61b4b94a23e39d465f1d/ports/mpg123
while read -r f sum; do
  get "$port/$f" "$src/mpg123/vcpkg-port-b7601ed/$f" "$sum"
done <<'EOF'
portfile.cmake 1387d7eefdc648e983346dcded90c3e10794f5aef2ed490c460946edd63539a7
vcpkg.json a6d42bf47e06c405057af98b66ffa80b43fcce09a8cf4d9b0440a36789e20f3e
fix-dllexport.diff 7ef49d04ac4d19c946b7e1f9d65006ca88bee107340218984e86c2b1af776b57
have-fpu.diff 9eb6c504ccb0577854a7af6b64efbb9575631bbd589afad5a4aa6f014634212b
pkgconfig.diff f615aa6de8c86fc9741e39fe5346a2d541d80daf45bcb75d31b3b4f81baf29f2
EOF

cat > "$src/README.txt" <<EOF
Sources for Peresmeshnik $v (https://github.com/a1lezakty/peresmeshnik, tag v$v), next to its builds:

piper-runner/ - the Corresponding Source of the GPL-3.0 program piper-runner in the macOS and Windows builds
  (licenses/piper-runner-SOURCE.txt there): peresmeshnik-$v-runner.tar.gz (piper-runner/ and the build scripts in
  packaging/), piper1-gpl-1.8.0.tar.gz (https://github.com/OHF-Voice/piper1-gpl, tag v1.8.0) and
  espeak-ng-$esp.tar.gz (https://github.com/espeak-ng/espeak-ng, the commit piper1-gpl builds into espeakbridge).
mpg123/ - mpg123 1.32.9 (LGPL-2.1), built into libsndfile of the Windows build (licenses/LGPL-SOURCE.txt there):
  mpg123-1.32.9.tar.bz2 as released, and vcpkg-port-b7601ed/, the portfile and patches it was built with
  (https://github.com/microsoft/vcpkg/tree/b7601ed37e1e1ce11a4f61b4b94a23e39d465f1d/ports/mpg123).

SHA-256:
$(cd "$src" && find . -type f ! -name README.txt | sort | xargs sha256sum)
EOF
tar -cf "$out/Peresmeshnik-$v-sources.tar" -C "$tmp" "Peresmeshnik-$v-sources"
echo "sources: $out/Peresmeshnik-$v-sources.tar"
