"""Crash reports without the user's data (ARCHITECTURE §5, "Crash reports"). Nothing is sent: the tray shows a report,
copies it and opens a GitHub issue page with it, and the user sends it.

`install()` (the app's start): the log of the run goes, already cleaned, to <data dir>/logs/peresmeshnik.log (two files
of at most 512 KB); an unhandled exception (any thread, a Qt slot) writes <data dir>/reports/crash-<time>.txt at once;
a hard crash (a native library's segfault) is written by faulthandler into reports/fault-<pid>.txt and made a report at
the next start; `report(what, exc)` is the app's own (a direction that died, a helper that ended: the side restarts).

A report has the version, OS, CPU, RAM, GPU, the engines and models of the settings (downloaded or not), the stack and
the last 200 lines of the log. It never has what was said (subtitles, translations), audio, device names, the call
app, the voice profile's name, keys or the user's name in paths (`clean`, `scrub`):
- a log line keeps its format string and only its numbers ("microphone <…> is not there: using <…>"); the latency
  logger's JSON keeps only its numbers. Every log call of the app passes data as arguments (none builds its message);
- what is left as text (exception messages, the stack) loses the private values known: the settings' devices, call app
  and voice, the voice profiles, the devices seen this run (`private()`) and listed now, the key and anything shaped like
  one, the home folder and the user's name, and the last subtitles (`said()`, kept in memory only) whole and word by
  word (words of 4 letters and more: shorter ones are mostly "the", "and", "это").
"""
from __future__ import annotations

import collections
import faulthandler
import json
import logging
import logging.handlers
import os
import platform
import re
import shutil
import sys
import threading
import time
import traceback
from pathlib import Path

from peresmeshnik import __version__, config

LINES = 200  # of the log in a report
KEEP = 10  # reports kept
KEY = re.compile(r'AIza[0-9A-Za-z_\-]{20,}')
# a fault file's crash: a signal, or a named Windows exception ("access violation"); faulthandler logs every other SEH
# code too ("code 0x8001010d", COM's, handled), and a run killed after one would leave a false report
FATAL = re.compile(r'Fatal Python error|Windows fatal exception: (?!code 0x)')
PRIVATE_PROPS = ('.description', '.alias', '.nick', '.product.name', '.serial', '.address', '.user', '.host',
                 '.hostname', '.machine_id')  # PipeWire properties that can name the user's device, the user or the host
NOISY = ('websockets', 'google_genai', 'httpx', 'httpcore', 'urllib3', 'asyncio')
_private: set[str] = set()
_said: collections.deque = collections.deque(maxlen=40)  # the last subtitles' texts, in memory only
_partial: tuple = ()  # the clause in progress: its latest partial subtitle
_words = None  # their words as one pattern, made when needed
_lock = threading.Lock()
_fault = None
on_report = []  # callbacks(path, quiet): the tray says it once (quiet: the app's own message says it already)
log = logging.getLogger('peresmeshnik.crash')


def reports_dir() -> Path:
    return config.data_dir() / 'reports'


def log_file() -> Path:
    return config.data_dir() / 'logs' / 'peresmeshnik.log'


def private(*values: str) -> None:
    """Names the app saw that must not reach a report (a device, the call app)."""
    with _lock:
        _private.update(v for v in values if v and len(v) >= 3)


def said(*texts: str, final: bool = True) -> None:
    """Subtitles shown (what was said and its translation): scrubbed from any text that reaches a report. The finals
    are kept (the last 20 clauses); of the partials only the clause's latest (each grows into the next)."""
    global _words, _partial
    with _lock:
        if not final:
            _partial, _words = tuple(t for t in texts if t), None
            return
        _partial = ()
        for t in texts:
            if t and (not _said or _said[-1] != t):
                _said.append(t)
        _words = None


def scrub(text: str, words: bool = True) -> str:
    """Private values out of a text; `words`: the last subtitles' words too (exception texts: a log line's own format
    string is code, its text arguments are gone already)."""
    global _words
    with _lock:
        texts = (*_said, *_partial)
        values = sorted(_private | set(texts), key=len, reverse=True)
        if words and _words is None and texts:
            words = sorted({w for t in texts for w in re.findall(r'\w{4,}', t)}, key=len, reverse=True)
            _words = re.compile(r'\b(?:' + '|'.join(map(re.escape, words)) + r')\b', re.IGNORECASE) if words else False
        words = words and _words
    for v in values:
        text = text.replace(v, '<…>')
    if words:
        text = words.sub('<…>', text)
    text = KEY.sub('<key>', text)
    home = str(Path.home())
    for h in {home, home.replace('\\', '/'), home.replace('\\', '\\\\')}:
        text = re.sub(re.escape(h), '~', text, flags=re.IGNORECASE if sys.platform == 'win32' else 0)
    try:
        import getpass
        if len(user := getpass.getuser()) >= 3:
            text = re.sub(rf'\b{re.escape(user)}\b', '<user>', text)
    except Exception:
        pass
    return text


def clean(record: logging.LogRecord) -> str:
    """The record's message without its data: numbers stay, text arguments do not."""
    msg = record.msg if isinstance(record.msg, str) else '<…>'
    if record.name == 'peresmeshnik.latency' and msg.startswith('{'):
        try:
            msg = json.dumps({k: v for k, v in json.loads(msg).items() if isinstance(v, int | float | bool)})
        except ValueError:
            msg = '<…>'
    elif record.args:
        args = record.args if isinstance(record.args, tuple) else (record.args,)
        try:
            msg = msg % tuple(a if isinstance(a, int | float | bool) else '<…>' for a in args)
        except (TypeError, ValueError):
            pass
    line = scrub(f'{time.strftime("%H:%M:%S", time.localtime(record.created))} {record.levelname[0]} {record.name} {msg}',
                 words=False)
    if record.exc_info and record.exc_info[1] is not None:
        line += '\n' + scrub(_stack(record.exc_info[1]))
    return line


def _stack(e: BaseException) -> str:
    """The stack with the exception's own text cut short (it is the part that may carry data)."""
    lines = traceback.format_exception(type(e), e, e.__traceback__)
    return ''.join(lines[:-1]) + f'{type(e).__module__}.{type(e).__qualname__}: {str(e)[:300]}'


class _Handler(logging.handlers.RotatingFileHandler):
    def format(self, record: logging.LogRecord) -> str:
        return clean(record)


def _known_private(settings) -> None:
    if settings is not None:
        private(settings.ui.mic, settings.ui.headphones, settings.ui.call_app, settings.local.voice)
        try:
            from peresmeshnik.cloud import api_key
            private(api_key(settings.cloud))
        except BaseException:  # no key: SystemExit
            pass
    try:
        from peresmeshnik.voice import voices_dir
        private(*(p.name.removesuffix('.new').removesuffix('.old') for p in voices_dir().iterdir()))
    except OSError:
        pass


def _devices_now() -> None:
    """The devices there are now, for scrubbing: only through what the process has already loaded."""
    try:
        if sd := sys.modules.get('sounddevice'):
            private(*(d['name'] for d in sd.query_devices()))
        if ca := sys.modules.get('peresmeshnik.audio.coreaudio'):
            private(*(d['name'] for d in ca.devices()))
        if sys.platform.startswith('linux') and shutil.which('pactl'):
            import subprocess
            for kind in ('sinks', 'sources'):
                out = subprocess.run(['pactl', '-f', 'json', 'list', kind], capture_output=True, text=True, timeout=3).stdout
                for d in json.loads(out or '[]'):  # names and the user's or host's, not every property: "alsa", "100"
                    props = d.get('properties') or {}  # would eat paths and line numbers in the stack
                    private(d.get('name', ''), d.get('description', ''),
                            *(str(v) for k, v in props.items() if k.endswith(PRIVATE_PROPS)))
    except Exception:
        pass


def about(settings=None) -> str:
    from peresmeshnik import models
    lines = [f'Peresmeshnik {__version__} ({"installed" if getattr(sys, "frozen", False) else "from source"}), '
             f'Python {platform.python_version()}']
    if qt := sys.modules.get('PySide6.QtCore'):
        lines[0] += f', Qt {qt.qVersion()}'
    try:
        import psutil
        ram = f'{psutil.virtual_memory().total / 2 ** 30:.0f} GB RAM'
    except Exception:
        ram = 'RAM unknown'
    lines += [f'OS: {platform.platform()}', f'CPU: {platform.machine()}, {os.cpu_count()} threads, {ram}, '
              f'NVIDIA GPU: {"yes" if shutil.which("nvidia-smi") else "no"}']
    if settings is not None:
        s = settings

        def have(model_id: str) -> str:
            try:
                models.path(model_id)
                return model_id
            except Exception:
                return f'{model_id} (not downloaded)'
        for side, engine, src, tgt in (('them -> me', s.them_to_me, s.their_lang, s.my_lang),
                                       ('me -> them', s.me_to_them, s.my_lang, s.their_lang)):
            local = [have(s.local.asr.get(src, '?')), have(s.local.mt.get(f'{src}-{tgt}', '?'))]
            local.append(have(s.local.tts.get(tgt, '?')) if side.startswith('them') else
                         f'{have(s.local.clone)}, voice {"enrolled" if s.local.voice else "none"}')
            cloud = f'cloud {s.cloud.translate_model}, subtitles {"cloud" if s.cloud.subtitles else "local"}; ' \
                if engine == 'cloud' else ''
            lines.append(f'{side}: {engine} {src}->{tgt}: {cloud}local {", ".join(local)}')
    return '\n'.join(lines)


def _tail() -> list[str]:
    for h in logging.getLogger().handlers:
        if isinstance(h, _Handler):
            h.flush()
    f = log_file()
    lines: collections.deque = collections.deque(maxlen=LINES)
    for p in (f.with_name(f.name + '.1'), f):
        try:
            with open(p, encoding='utf-8', errors='replace') as fh:
                lines.extend(line.rstrip('\n') for line in fh)
        except OSError:
            pass
    return list(lines)


def write(what: str, stack: str, settings=None, log_lines: list[str] | None = None, quiet: bool = False) -> Path | None:
    """A report file; never raises (it runs while something is already wrong)."""
    try:
        _known_private(settings)
        _devices_now()
        text = '\n\n'.join([  # the header is made from the settings' engine ids only: no scrub (it would eat "local")
            scrub(f'What happened: {what}'), about(settings), scrub('Stack:\n' + stack.rstrip()),
            scrub('The last lines of the log (without what was said, device names or keys):\n' + '\n'.join(
                _tail() if log_lines is None else log_lines))]) + '\n'
        d = reports_dir()
        d.mkdir(parents=True, exist_ok=True)
        now = time.time()
        p = d / f'crash-{time.strftime("%Y%m%d-%H%M%S", time.localtime(now))}.{int(now * 1000) % 1000:03d}-{os.getpid()}.txt'
        p.write_text(text, 'utf-8')
        for old in sorted(d.glob('crash-*.txt'))[:-KEEP]:
            old.unlink(missing_ok=True)
        for cb in on_report:
            cb(p, quiet)
        return p
    except Exception:
        return None


def report(what: str, e: BaseException | None = None, settings=None, quiet: bool = False) -> Path | None:
    """The app's own report: a direction that died, a helper that ended (`quiet`: the app's message names the report)."""
    stack = _stack(e) if e is not None else ''.join(traceback.format_stack()[:-1])
    if settings is None:
        try:
            settings = config.load()
        except Exception:
            pass
    return write(what, stack, settings, quiet=quiet)


class Respawn:
    """A TTS whose helper process may die (the Piper runner): made again, and the clause said again if none of it was,
    instead of the pipeline's speak worker dying with it (the side would make all its engines again, BENCH). An error
    with the helper still alive is the pipeline's as before, and so is a helper dying more than RESTARTS times a minute
    (the side restarts, then stays off). `on_respawn(e)`: one message and a report."""
    RESTARTS = 3

    def __init__(self, make, on_respawn=lambda e: None, tts=None):
        self.make, self.on_respawn, self._times = make, on_respawn, []
        self.tts = tts or make()  # `tts`: one made already (by make_local, after the models it needs were found)
        self.sample_rate = self.tts.sample_rate

    @property
    def _p(self):  # the helper process (pipeline.cpu_s counts its CPU)
        return getattr(self.tts, '_p', None)

    def synth(self, text, lang, voice=None, speed=1.0):
        kw = {'speed': speed} if speed != 1.0 else {}  # the pipeline passes it only to a TTS that takes it
        inner, said = self.tts.synth(text, lang, voice, **kw), False
        try:
            for chunk in inner:
                said = True
                yield chunk
            return
        except GeneratorExit:  # the caller stopped early (barge-in): the TTS reads away the rest
            inner.close()
            raise
        except Exception as e:
            p, now = self._p, time.monotonic()
            self._times = [x for x in self._times if now - x < 60]
            if p is None or p.poll() is None or len(self._times) >= self.RESTARTS:
                raise
            self._times.append(now)
            try:
                self.tts.close()
            except Exception:
                pass
            self.tts = self.make()
            self.on_respawn(e)
        if not said:
            yield from self.tts.synth(text, lang, voice, **kw)

    def close(self) -> None:
        self.tts.close()


def _excepthook(prev):
    def hook(*args):
        e = args[0].exc_value if len(args) == 1 else args[1]  # threading.excepthook gets one argument
        if e is not None and not isinstance(e, KeyboardInterrupt | SystemExit):
            where = f' in thread {args[0].thread.name}' if len(args) == 1 and args[0].thread else ''
            report(f'unhandled {type(e).__name__}{where}', e)
        prev(*args)
    return hook


def install(settings=None, verbose: bool = False) -> list[Path]:
    """The app's start: earlier hard crashes become reports (returned: the tray says so), then the cleaned log file,
    faulthandler and the exception hooks."""
    global _fault
    _known_private(settings)
    found = []
    d = reports_dir()
    for f in sorted(d.glob('fault-*.txt')) if d.is_dir() else ():
        pid = f.stem.removeprefix('fault-')
        try:
            import psutil
            if not pid.isdigit() or psutil.pid_exists(int(pid)):
                continue  # its run is still on (this one, or a second start that the lock file stops)
            text = f.read_text('utf-8', 'replace').strip()
            f.unlink()
        except Exception:
            continue
        if FATAL.search(text) and (p := write('the app crashed (a fatal error in native code; its last run)', text, settings)):
            found.append(p)
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    for h in root.handlers:  # the terminal: as before
        if not isinstance(h, _Handler):
            h.setLevel(logging.INFO if verbose else logging.WARNING)
    for name in NOISY:
        logging.getLogger(name).setLevel(logging.WARNING)
    try:
        log_file().parent.mkdir(parents=True, exist_ok=True)
        root.addHandler(_Handler(log_file(), maxBytes=512 * 1024, backupCount=1, encoding='utf-8', delay=True))
        d.mkdir(parents=True, exist_ok=True)
        _fault = open(d / f'fault-{os.getpid()}.txt', 'w')  # noqa: SIM115 (open for the process's whole life)
        faulthandler.enable(_fault, all_threads=True)
    except OSError as e:
        log.warning('crash reports: %s', e)
    sys.excepthook = _excepthook(sys.excepthook)
    threading.excepthook = _excepthook(threading.excepthook)
    log.info('Peresmeshnik %s started', __version__)
    return found


def uninstall() -> None:
    """A clean exit: no fault file is left for the next start."""
    global _fault
    if _fault is not None:
        faulthandler.disable()
        _fault.close()
        Path(_fault.name).unlink(missing_ok=True)
        _fault = None


def issue_url(text: str, limit: int = 7000) -> str:
    """A new GitHub issue with the report (cut to fit a URL; the tray copies the whole of it too)."""
    from urllib.parse import quote
    first = text.splitlines()[0].removeprefix('What happened: ') if text else 'crash'
    body = text if len(quote(text)) <= limit else None
    if body is None:
        body, cut = text, len(text)
        while len(quote(body)) > limit:
            cut = cut * 3 // 4
            body = text[:cut]
        body += '\n\n(cut here: the whole report is in your clipboard, paste it below)'
    fence = '```'
    return (f'https://github.com/a1lezakty/peresmeshnik/issues/new?title={quote("Crash: " + first[:80])}'
            f'&body={quote(fence + chr(10) + body + chr(10) + fence)}')
