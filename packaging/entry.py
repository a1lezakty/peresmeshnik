"""Entry point of the frozen app (PyInstaller): the helper programs beside the executable, then the usual main.

Contents/MacOS holds Peresmeshnik, piper-runner (the Piper TTS runner, its own GPL-3.0 program: the app talks to
it over a pipe, ARCHITECTURE §10) and peresmeshnik-tap (the Core Audio process tap of the call app's sound).
"""
import os
import shlex
import sys
from pathlib import Path

here = Path(sys.executable).parent
if (p := here / 'piper-runner').exists():
    os.environ.setdefault('PERESMESHNIK_PIPER_CMD', shlex.quote(str(p)))
if (p := here / 'peresmeshnik-tap').exists():
    os.environ.setdefault('PERESMESHNIK_TAP', str(p))
sys.argv = [a for a in sys.argv if not a.startswith('-psn_')]  # an old Finder adds its process serial number

from peresmeshnik.__main__ import main  # noqa: E402

sys.exit(main())
