"""Entry point of the frozen app (PyInstaller): the helper programs beside the executable, then the usual main.

Beside the executable (Contents/MacOS on macOS, the install folder on Windows): piper-runner (the Piper TTS runner,
its own GPL-3.0 program: the app talks to it over a pipe, ARCHITECTURE §10) and the call app's sound helper:
peresmeshnik-tap (macOS, the Core Audio process tap), peresmeshnik-loopback.exe (Windows, WASAPI process loopback), and
the me -> them sound helper peresmeshnik-audio.exe (Windows; on macOS audio/macos.py finds it beside the executable).
The voice pack (the CUDA clone) is its own process with its own Python: nothing of it is imported here.
"""
import os
import shlex
import sys
from pathlib import Path

here = Path(sys.executable).parent
if (here / 'piper-runner.exe').exists():  # Windows: the bare name, found first in the folder of the app's own .exe
    # (CreateProcess's search order); a full path with spaces would not survive the command's Windows-style split
    os.environ.setdefault('PERESMESHNIK_PIPER_CMD', 'piper-runner.exe')
elif (p := here / 'piper-runner').exists():
    os.environ.setdefault('PERESMESHNIK_PIPER_CMD', shlex.quote(str(p)))
if (p := here / 'peresmeshnik-tap').exists():
    os.environ.setdefault('PERESMESHNIK_TAP', str(p))
if (p := here / 'peresmeshnik-loopback.exe').exists():
    os.environ.setdefault('PERESMESHNIK_LOOPBACK', str(p))
if (p := here / 'peresmeshnik-audio.exe').exists():
    os.environ.setdefault('PERESMESHNIK_AUDIO', str(p))
sys.argv = [a for a in sys.argv if not a.startswith('-psn_')]  # an old Finder adds its process serial number
piped = sys.stderr is not None  # Windows: a console, a pipe or a file; a windowed .exe from Explorer has none
for name in ('stdout', 'stderr'):  # no stdio (None): print() and logging write nowhere instead of failing
    if getattr(sys, name) is None:
        setattr(sys, name, open(os.devnull, 'w', encoding='utf-8'))

from peresmeshnik.__main__ import main  # noqa: E402

try:
    sys.exit(main())
except Exception:  # an error of a command (e.g. the cloud refusing a key): its traceback and exit code 1. On Windows the
    if not piped:  # windowed bootloader would show it in a dialog and wait for a click; that only when nothing else sees it
        raise
    import traceback
    traceback.print_exc()
    sys.exit(1)
