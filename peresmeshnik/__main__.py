"""Entry point. Stage 0 has no engines yet: the app starts in its idle state and waits.

tests/test_idle_cpu.py watches this process, so whatever the app grows into must keep the
idle state (translation off) at ~0 CPU: no timers, no polling loops, no redraws without changes.
"""
import argparse
import threading

from peresmeshnik import __version__


def main(argv=None):
    ap = argparse.ArgumentParser(prog='peresmeshnik')
    ap.add_argument('--version', action='version', version=__version__)
    ap.parse_args(argv)
    print(f'Peresmeshnik {__version__}: idle, translation off. Ctrl+C to quit.', flush=True)
    try:
        threading.Event().wait()  # blocks without waking up; the GUI event loop replaces this later
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    main()
