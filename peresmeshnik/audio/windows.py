"""Windows sound devices for "me -> them" through the native helper `peresmeshnik-audio.exe` (native/windows/audio):
WASAPI does the work in one small process per stream (record, play, loop), as on macOS (audio/macos.py).

Devices are named by their endpoint id (stable across restarts and replugging); a name works too. The virtual
microphone is VB-Audio Virtual Cable, installed by the user: the app plays the translation into "CABLE Input", the call
app records "CABLE Output".
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import threading
from pathlib import Path

from peresmeshnik.audio import processed

HELPER = 'peresmeshnik-audio.exe'
CABLE = 'VB-Audio Virtual Cable'  # in both endpoints' names, also when the user renames them: "CABLE Input (VB-Audio ...)"
CABLE_IN, CABLE_OUT = 'CABLE Input', 'CABLE Output'
URL = 'https://vb-audio.com/Cable/'
INSTALL = (f'VB-Audio Virtual Cable is not installed. It is the virtual microphone that carries your voice to the call '
           f'app. Download it from {URL} (VB-CABLE Driver Pack), unpack the zip, right-click VBCABLE_Setup_x64.exe → '
           'Run as administrator → Install Driver, restart Windows — then check again.')
NO_PERMISSION = ('Peresmeshnik may not use the microphone: Windows Settings → Privacy & security → Microphone → turn on '
                 '"Microphone access" and "Let desktop apps access your microphone"')
NO_WINDOW = getattr(subprocess, 'CREATE_NO_WINDOW', 0)  # a console app started from the windowed .exe: no console flash


class CableMissing(RuntimeError):
    def __init__(self):
        super().__init__(INSTALL)


def helper() -> str:
    """$PERESMESHNIK_AUDIO, else next to the executable (the install folder), else this checkout's build."""
    here = Path(__file__).resolve().parents[2] / 'native' / 'windows' / 'audio' / 'build' / HELPER
    for p in (os.environ.get('PERESMESHNIK_AUDIO'), Path(sys.executable).parent / HELPER, here, shutil.which(HELPER)):
        if p and Path(p).is_file():
            return str(p)
    raise RuntimeError(f'{HELPER} is not built: run native\\windows\\audio\\build.cmd (Visual Studio C++ build tools)')


def devices() -> list[dict]:
    """[{uid, name, inputs, outputs, rate, transport, default_input, default_output}] of the active endpoints, now: a
    capture endpoint has inputs, a render one outputs; transport 'virt' = a software device (VB-Cable, Voicemeeter)."""
    r = subprocess.run([helper(), 'list'], capture_output=True, timeout=10, creationflags=NO_WINDOW)
    if r.returncode:  # the helper's own words ("is the Windows Audio service running?"), not only its exit code
        lines = r.stderr.decode(errors='replace').strip().splitlines()
        raise RuntimeError((lines[-1] if lines else f'{HELPER} list exited with {r.returncode}').removeprefix('peresmeshnik-audio: '))
    return json.loads(r.stdout.decode('utf-8'))


def is_ours(d: dict) -> bool:
    """VB-Cable's endpoints: they carry our own output (the virtual microphone), never the user's sound."""
    return CABLE in d['name']


def cable(devs: list[dict] | None = None) -> tuple[dict, dict]:
    """(CABLE Input, CABLE Output): where the app plays and what the call app records; else CableMissing."""
    devs = devices() if devs is None else devs
    plays = [d for d in devs if d['outputs'] and is_ours(d)]  # also "CABLE In 16ch" (pack 46): "CABLE Input" first
    play = next((d for d in plays if d['name'].startswith(CABLE_IN)), plays[0] if plays else None)
    recs = [d for d in devs if d['inputs'] and is_ours(d)]
    rec = next((d for d in recs if d['name'].startswith(CABLE_OUT)), recs[0] if recs else None)
    if not (play and rec):
        raise CableMissing
    return play, rec


def stand_in_mic() -> str:
    """$PERESMESHNIK_TEST_MIC: a virtual input the tests take as the microphone (a second cable on the CI runner); read
    only under pytest, so it never lets a virtual device through in the app."""
    return os.environ.get('PERESMESHNIK_TEST_MIC', '') if os.environ.get('PYTEST_CURRENT_TEST') else ''


def not_virtual(key: str, what: str = 'microphone') -> str:
    """CABLE Output carries our own translation: as the microphone the app would hear itself. Any other virtual input
    (another cable, Voicemeeter) is not the user's voice either; NVIDIA Broadcast and Krisp are (audio.processed)."""
    if not key or what != 'microphone' or key == stand_in_mic():
        return key
    d = next((d for d in devices() if d['inputs'] and key in (d['uid'], d['name'])), None)
    if d and is_ours(d):
        raise ValueError(f'{CABLE_OUT} is the virtual microphone for the call app, not your microphone: choose your real '
                         'microphone in the settings')
    if d and (d['transport'] == 'virt' or 'VB-Audio' in d['name']) and not processed(d['name']):
        raise ValueError(f'{d["name"]} is a virtual device, not your microphone: choose your real microphone in the settings')
    return key


def permission() -> str:
    """'denied' when Windows keeps the microphone from desktop apps (Privacy & security: the helper exits with 6 as it
    opens one), else 'granted' (it opened one, or there is none to open); found by starting a recording for a moment."""
    p = spawn('record', '--rate', '16000', stdout=subprocess.DEVNULL)
    first: list[bytes] = []
    t = threading.Thread(target=lambda: first.append(p.stderr.readline()), daemon=True)
    t.start()
    t.join(5)  # "using ..." or "none" once it is set up, else the line it dies with
    try:
        if first and first[0].startswith((b'using', b'none')):
            return 'granted'
        return 'denied' if p.wait(3) == 6 else 'granted'
    except subprocess.TimeoutExpired:
        return 'granted'
    finally:
        if p.poll() is None:
            p.stdin.close()
            p.wait(5)


def spawn(*args: str, stdout=subprocess.PIPE, stdin=subprocess.PIPE) -> subprocess.Popen:
    """The helper with its stdin kept open: record and loop end when it closes (when the app is gone)."""
    return subprocess.Popen([helper(), *args], stdin=stdin, stdout=stdout, stderr=subprocess.PIPE, creationflags=NO_WINDOW)
