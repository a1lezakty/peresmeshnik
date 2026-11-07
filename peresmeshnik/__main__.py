"""Entry point. Without a command the app starts in its idle state and waits.

tests/test_idle_cpu.py watches this process, so whatever the app grows into must keep the
idle state (translation off) at ~0 CPU: no timers, no polling loops, no redraws without changes.
Commands live in their own modules (`cli(argv)`), imported only when used.
"""
import argparse
import importlib
import sys
import threading

from peresmeshnik import __version__

COMMANDS = {
    'models': ('peresmeshnik.models', 'list and download models (license screen, SHA-256)'),
    'cloud': ('peresmeshnik.cloud', 'Gemini Live direction on a file or a raw PCM pipe'),
    'run': ('peresmeshnik.pipeline', 'local direction on a file'),
}


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if argv[:1] and argv[0] in COMMANDS:
        return importlib.import_module(COMMANDS[argv[0]][0]).cli(argv[1:])
    ap = argparse.ArgumentParser(prog='peresmeshnik', epilog='commands: ' + '; '.join(f'{k} - {v[1]}' for k, v in COMMANDS.items()))
    ap.add_argument('--version', action='version', version=__version__)
    ap.parse_args(argv)
    print(f'Peresmeshnik {__version__}: idle, translation off. Ctrl+C to quit.', flush=True)
    try:
        threading.Event().wait()  # blocks without waking up; the GUI event loop replaces this later
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    sys.exit(main())
