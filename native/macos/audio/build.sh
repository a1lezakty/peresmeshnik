#!/bin/sh
# Builds the helper: native/macos/audio/build/peresmeshnik-audio (Xcode command line tools; ARCH=x86_64 for Intel).
set -e
cd "$(dirname "$0")"
mkdir -p build
swiftc -O -target "${ARCH:-arm64}-apple-macos12" main.swift -o build/peresmeshnik-audio \
  -Xlinker -sectcreate -Xlinker __TEXT -Xlinker __info_plist -Xlinker Info.plist
