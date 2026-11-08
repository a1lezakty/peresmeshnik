# SPDX-License-Identifier: GPL-3.0-or-later
"""Entry of the frozen piper-runner.exe on Windows: a part of that separate GPL-3.0 program (piper-runner/), not of
Peresmeshnik. The .exe is windowed, so that no console opens when the app starts it; an error ends it with its
traceback on stderr and exit code 1, instead of the windowed bootloader's dialog, which would wait for a click while
the app waits for the runner."""
import sys
import traceback

from piper_runner import main

try:
    main()
except Exception:
    traceback.print_exc()
    sys.exit(1)
