"""macOS sound devices for "me -> them" through the native helper `peresmeshnik-audio` (native/macos/audio): Core Audio
does the work in one small process per stream (record, play, loop), as parec / pacat / module-loopback do on Linux.

Devices are named by their Core Audio UID (stable across restarts and replugging); a name works too. The virtual
microphone is BlackHole 2ch, installed by the user: the app plays the translation into it, the call app records it.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

HELPER = 'peresmeshnik-audio'
BLACKHOLE = 'BlackHole 2ch'
BLACKHOLE_UID = 'BlackHole2ch_UID'
INSTALL = ('BlackHole 2ch is not installed. It is the virtual microphone that carries your voice to the call app. '
           'Install it with "brew install blackhole-2ch" in the Terminal, or with the installer from '
           'https://existential.audio/blackhole/ — then check again (still not found: restart the Mac).')
NO_PERMISSION = ('Peresmeshnik may not use the microphone: System Settings → Privacy & Security → Microphone → '
                 'turn Peresmeshnik on')


class BlackHoleMissing(RuntimeError):
    def __init__(self):
        super().__init__(INSTALL)


def helper() -> str:
    """$PERESMESHNIK_AUDIO, else next to the executable (Peresmeshnik.app/Contents/MacOS), else this checkout's build."""
    here = Path(__file__).resolve().parents[2] / 'native' / 'macos' / 'audio' / 'build' / HELPER
    for p in (os.environ.get('PERESMESHNIK_AUDIO'), Path(sys.executable).parent / HELPER, here, shutil.which(HELPER)):
        if p and Path(p).is_file():
            return str(p)
    raise RuntimeError(f'{HELPER} is not built: run native/macos/audio/build.sh (Xcode command line tools)')


def devices() -> list[dict]:
    """[{uid, name, inputs, outputs, rate, transport, default_input, default_output, id}] from Core Audio, now."""
    return json.loads(subprocess.run([helper(), 'list'], check=True, capture_output=True, text=True, timeout=10).stdout)


def permission() -> str:
    """The microphone permission: authorized | denied | restricted | undetermined."""
    return subprocess.run([helper(), 'permission'], capture_output=True, text=True, timeout=10).stdout.strip()


def is_blackhole(key: str) -> bool:
    return key in (BLACKHOLE, BLACKHOLE_UID)


def blackhole(devs: list[dict] | None = None) -> dict:
    """The BlackHole 2ch device, else BlackHoleMissing with how to install it."""
    for d in devices() if devs is None else devs:
        if d['outputs'] and (is_blackhole(d['uid']) or is_blackhole(d['name'])):
            return d
    raise BlackHoleMissing


def not_blackhole(key: str, what: str = 'microphone') -> str:
    """BlackHole 2ch carries our own output: as the microphone (or the headphones) the app would hear itself."""
    if key and is_blackhole(key):
        raise ValueError(f'{BLACKHOLE} is the virtual microphone for the call app, not your {what}: choose your real '
                         f'{what} in the settings')
    return key


def spawn(*args: str, stdout=subprocess.PIPE, stdin=subprocess.PIPE) -> subprocess.Popen:
    """The helper with its stdin kept open: record and loop end when it closes (when the app is gone)."""
    return subprocess.Popen([helper(), *args], stdin=stdin, stdout=stdout, stderr=subprocess.PIPE)
