"""Crash reports without the user's data: what was said, device names, the call app, the voice's name and the key never
reach a report or the log file, whatever path they take (log arguments, the latency JSON, an exception's text, a
thread's unhandled exception); a hard crash (a segfault) becomes a report at the next start; the GitHub link fits."""
import json
import logging
import os
import subprocess
import sys
import threading
import time
from urllib.parse import unquote

import pytest

from peresmeshnik import config, crash

KEY = 'AIzaSyD3' + 'x' * 31  # shaped like a Gemini key
SAID = 'Мы договорились встретиться в среду'  # what someone said
TRANSLATION = 'We agreed to meet on Wednesday'
DEVICES = ('Jabra Evolve2 65', 'bluez_output.00_1B_66_AA_BB_CC.1', 'Telegram')  # mic, headphones, call app
VOICE = 'Ivan Petrov'


@pytest.fixture
def app(tmp_path, monkeypatch):
    monkeypatch.setattr(config, 'data_dir', lambda: tmp_path / 'data')
    monkeypatch.setenv('GEMINI_API_KEY', KEY)
    monkeypatch.setattr(crash, '_private', set())
    monkeypatch.setattr(crash, '_said', crash.collections.deque(maxlen=40))
    monkeypatch.setattr(crash, '_seen', set())
    monkeypatch.setattr(crash, '_partial', ())
    monkeypatch.setattr(crash, 'on_report', [])
    root = logging.getLogger()
    handlers, level = list(root.handlers), root.level
    hooks = sys.excepthook, threading.excepthook
    s = config.Settings()
    s.ui.mic, s.ui.headphones, s.local.voice = DEVICES[0], DEVICES[1], VOICE
    (tmp_path / 'data' / 'voices' / VOICE).mkdir(parents=True)
    crash.install(s)
    yield s
    crash.uninstall()
    for h in root.handlers:
        if h not in handlers:
            root.removeHandler(h)
            h.close()
    root.setLevel(level)
    sys.excepthook, threading.excepthook = hooks


LISTED = 'Galaxy Buds2 Pro of Ivan'  # a device the app never chose: only listed at the report (crash._devices_now)


def leaks(text: str) -> list[str]:
    low = text.lower()
    return [x for x in (KEY, SAID, TRANSLATION, *DEVICES, VOICE, 'среду', 'wednesday') if x.lower() in low]


def test_reports_and_the_log_hold_no_user_data(app, monkeypatch):
    import types
    monkeypatch.setitem(sys.modules, 'sounddevice', types.SimpleNamespace(query_devices=lambda: [{'name': LISTED}]))
    crash.private(DEVICES[2])  # the call app, as the app registers it when it routes the call
    for i in range(1, 30):  # partials of a long clause, then its final: one entry each, not one per partial
        crash.said(f'partial {i}', final=False)
    assert crash._partial == ('partial 29',) and len(crash._said) == 0
    crash.said(SAID, TRANSLATION)  # the subtitles shown, as the app tells it
    assert list(crash._said) == [SAID, TRANSLATION] and crash._partial == ()
    crash.said('Мы ещё', final=False)  # the next clause in progress: scrubbed too
    logging.getLogger('peresmeshnik.ui').info('overlay %s: %s | %s', 'them', SAID, TRANSLATION)
    logging.getLogger('peresmeshnik.latency').info(json.dumps({'clause': 3, 'text': SAID, 'translation': TRANSLATION,
                                                               'caption_lag_s': 0.42}, ensure_ascii=False))
    logging.getLogger('peresmeshnik.app').warning('microphone %s is not there: using %s', DEVICES[0], DEVICES[1])
    logging.getLogger('peresmeshnik.cloud.gemini').warning('connect failed (%s), retry', f'key={KEY} rejected')
    try:
        raise RuntimeError(f'the headphones are {DEVICES[1]}, the call app {DEVICES[2]}: {SAID} (key {KEY})')
    except RuntimeError:
        logging.getLogger('peresmeshnik.app').exception('control')

    def worker():
        raise ValueError(f'voice {VOICE} at {crash.Path.home() / "voices" / VOICE}: {meet} ({LISTED})')
    meet = 'meet on ' + 'wednesday'  # a variable: a frame's source line is code, scrubbed of private values only
    t = threading.Thread(target=worker, name='speak')
    t.start()
    t.join()
    reports = sorted(crash.reports_dir().glob('crash-*.txt'))
    assert len(reports) == 1
    text = reports[0].read_text('utf-8')
    logged = crash.log_file().read_text('utf-8')
    print(ascii(text[:2000]))  # a Windows console is cp1252
    assert not leaks(text) and not leaks(logged), (leaks(text), leaks(logged))  # 'wednesday': only word by word
    assert LISTED not in text and 'Buds2' not in text and ': <…> on <…> (<…>)' in text, text[-1500:]
    assert 'unhandled ValueError in thread speak' in text and 'in worker' in text  # what and where
    assert 'Peresmeshnik' in text and 'OS: ' in text and 'them -> me: local' in text and 'parakeet' in text
    assert 'microphone <…> is not there: using <…>' in text and '"caption_lag_s": 0.42' in text and '"clause": 3' in text
    assert str(crash.Path.home()) not in text


def test_a_segfault_becomes_a_report_at_the_next_start(app, tmp_path):
    code = ('import ctypes, sys; from peresmeshnik import config, crash; '
            f'config.data_dir = lambda: __import__("pathlib").Path({str(tmp_path / "data")!r}); '
            'crash.install(); ctypes.string_at(0)')  # a native crash
    p = subprocess.Popen([sys.executable, '-c', code], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    assert p.wait(60) != 0
    # by its python's pid: on Windows a venv's python.exe is a launcher that runs the real one as its child
    faults = [f for f in crash.reports_dir().glob('fault-*.txt') if f.name != f'fault-{os.getpid()}.txt']
    assert len(faults) == 1 and 'string_at' in faults[0].read_text(), faults  # faulthandler's stack of the crashed thread
    fault = faults[0]
    found = crash.install(app)  # the next start
    assert len(found) == 1 and not fault.exists()
    text = found[0].read_text('utf-8')
    assert 'the app crashed' in text and ('Segmentation fault' in text or 'access violation' in text.lower()), text
    assert not leaks(text)


def test_a_handled_windows_exception_in_a_fault_file_is_no_crash(app):
    """faulthandler on Windows logs every SEH code, handled ones too (COM's 0x8001010d): a run killed after one leaves
    its fault file, which is no crash report at the next start."""
    p = subprocess.Popen([sys.executable, '-c', 'pass'])
    p.wait()  # a pid no run has now
    f = crash.reports_dir() / f'fault-{p.pid}.txt'
    f.write_text('Windows fatal exception: code 0x8001010d\n\nThread 0x00001a2c (most recent call first):\n'
                 '  File "C:\\app\\peresmeshnik\\ui\\tray.py", line 120 in exec\n')
    assert crash.install(app) == [] and not f.exists()


def test_a_clean_exit_leaves_no_fault_file(app):
    crash.uninstall()
    assert not list(crash.reports_dir().glob('fault-*.txt'))


def test_issue_link_fits_and_says_it_was_cut():
    text = 'What happened: unhandled ValueError\n' + 'x' * 20_000
    url = crash.issue_url(text)
    assert len(url) < 8000 and url.startswith('https://github.com/a1lezakty/peresmeshnik/issues/new?title=Crash')
    assert 'the whole report is in your clipboard' in unquote(url)
    short = crash.issue_url('What happened: x\nshort')
    assert 'clipboard' not in unquote(short) and 'short' in unquote(short)


def test_report_hooks_tell_the_tray(app):
    seen = []
    crash.on_report.append(lambda p, quiet: seen.append((p, quiet)))
    p = crash.report('them -> me: the direction died', RuntimeError('worker'))
    q = crash.report('them -> me: restarted', RuntimeError('worker'), quiet=True)  # the app's message names it
    time.sleep(0.1)
    assert seen == [(p, False), (q, True)] and p.exists() and q.exists()


def test_a_dead_voice_helper_is_made_again_and_the_clause_said():
    """crash.Respawn: the TTS's helper process died -> a new TTS says the clause; an error with the helper alive, or a
    4th death within a minute, is the pipeline's (the side restarts)."""
    class Proc:
        code, exits_in = None, 0.0  # killed: the exit code; exits_in: it is still exiting this long after its pipe closed

        def wait(self, timeout=None):
            time.sleep(min(self.exits_in, timeout or 0))
            if self.code is None or self.exits_in > (timeout or 0):
                raise subprocess.TimeoutExpired('tts', timeout)
            return self.code

    made = []

    class Tts:
        sample_rate = 22050

        def __init__(self):
            self._p, self.fail = Proc(), None
            made.append(self)

        def synth(self, text, lang, voice=None, speed=1.0):
            if self._p.code is not None:
                raise BrokenPipeError(32, 'Broken pipe')
            if self.fail:
                raise self.fail
            yield text

        def close(self):
            pass
    seen = []
    r = crash.Respawn(Tts, seen.append)
    assert list(r.synth('a', 'ru')) == ['a'] and r._p is made[0]._p
    for i in range(3):
        made[-1]._p.code, made[-1]._p.exits_in = -9, 0.2 * i  # killed; still exiting for a moment after its pipe closed
        assert list(r.synth('b', 'ru')) == ['b']  # by a new one
    assert len(made) == 4 and len(seen) == 3 and all(isinstance(e, BrokenPipeError) for e in seen)
    made[-1]._p.code = -9
    with pytest.raises(BrokenPipeError):
        list(r.synth('c', 'ru'))  # the 4th within a minute
    r = crash.Respawn(Tts)
    made[-1].fail = RuntimeError('a bad request')  # the helper is alive: not a death
    with pytest.raises(RuntimeError, match='a bad request'):
        list(r.synth('d', 'ru'))


class FakeOverlay:
    def side(self, name):
        return type('S', (), {'show': lambda self, c: None, 'hide': lambda self: None, 'latency': lambda self, x: None})()


class FakeOut:
    def __init__(self):
        self.played = []

    def play(self, pcm, sr):
        self.played.append((time.monotonic(), len(pcm)))

    def flush(self):
        pass

    def stop(self):
        pass


def controller(monkeypatch, s, out):
    from pathlib import Path

    from peresmeshnik.app import Controller
    monkeypatch.setattr(Controller, '_output', lambda self, name, s: out)
    states = []
    ctl = Controller(FakeOverlay(), states.append, them_file=str(Path(__file__).parent / 'data' / 'call-en.ogg'), settings=s)
    return ctl, states


def wait_for(cond, timeout):
    end = time.monotonic() + timeout
    while not cond() and time.monotonic() < end:
        time.sleep(0.1)
    return cond()


def test_a_dying_direction_restarts_its_side_then_stays_off(app, monkeypatch):
    """A side whose translation dies (a worker, a helper process) starts again with one message and a report; after
    4 failures within a minute it stays off and says why; the app itself goes on."""
    from peresmeshnik.app import Side
    from peresmeshnik.engines import Caption

    class Dying:
        def feed(self, pcm):
            pass

        def events(self):
            yield Caption(SAID, TRANSLATION, True, 0)
            time.sleep(0.2)
            raise RuntimeError(f'pipeline worker died: piper runner exited (-11) after "{SAID}"')

        def close(self):
            pass
    monkeypatch.setattr(Side, '_local', lambda self, s: Dying())
    s = config.Settings()
    s.them_to_me = 'local'
    ctl, states = controller(monkeypatch, s, FakeOut())
    ctl.set_translation(True).result(30)
    assert wait_for(lambda: any('stopped' in x for x in states), 30), states
    restarted = [x for x in states if 'restarted after an error' in x]
    stopped = [x for x in states if 'stopped' in x]
    assert len(restarted) == 3 and len(stopped) == 1 and 'failed 4 times within a minute' in stopped[0]
    reports = list(crash.reports_dir().glob('crash-*.txt'))
    assert len(reports) == 4 and not any(leaks(p.read_text('utf-8')) for p in reports)
    assert 'them → me: the translation failed (RuntimeError' in reports[0].read_text('utf-8')
    assert not ctl.on and not ctl.sides
    ctl.quit()


class Loopback:  # capture_windows.ProcessLoopback as Controller._input makes it (windows_side)
    def __init__(self, on_end):
        self.on_end, self.n, self.stopped = on_end, len(self.made) + 1, False
        self.made.append(self)

    def start(self, cb):
        self.seen.append(('start', self.n))

    def retarget(self, app):
        assert not self.stopped, f'loopback {self.n} retargeted after its stop'
        self.seen.append(('retarget', self.n, app))

    def stop(self):
        self.stopped = True
        self.seen.append(('stop', self.n))


class LastingDirection:  # its events last until it is closed, as a real one's
    def __init__(self):
        self.closed = threading.Event()

    def feed(self, pcm):
        pass

    def events(self):
        self.closed.wait()
        yield from ()

    def close(self):
        self.closed.set()


def windows_side(s, monkeypatch):
    """Windows' branch of app.py on any OS: a controller translating them -> me from a fake loopback of the call app
    (muted), as _on leaves it. Returns (controller, states, seen: what the loopbacks and the muting did)."""
    import types

    from peresmeshnik import app as appmod
    from peresmeshnik.audio import capture_windows
    seen = []
    monkeypatch.setattr(Loopback, 'made', [], raising=False)
    monkeypatch.setattr(Loopback, 'seen', seen, raising=False)
    monkeypatch.setattr(appmod, 'sys', types.SimpleNamespace(**{**vars(sys), 'platform': 'win32'}))
    monkeypatch.setattr(capture_windows, 'check', lambda: (0, ''))
    monkeypatch.setattr(capture_windows, 'ProcessLoopback', Loopback)
    monkeypatch.setattr(capture_windows, 'mute_call', lambda app: seen.append(('mute', app)))
    monkeypatch.setattr(capture_windows, 'restore_call', lambda: seen.append(('restore',)))
    monkeypatch.setattr(appmod.Side, '_local', lambda self, s: LastingDirection())
    monkeypatch.setattr(appmod.Controller, '_output', lambda self, name, s: FakeOut())
    s.them_to_me, s.ui.call_app = 'local', DEVICES[2]
    states = []
    ctl = appmod.Controller(FakeOverlay(), states.append, settings=s)
    ctl.on, ctl.sides = True, [ctl._side('them', s)]
    ctl.sides[0].start()
    ctl._route(s)
    assert seen == [('start', 1), ('retarget', 1, DEVICES[2]), ('mute', DEVICES[2])], seen
    seen.clear()
    return ctl, states, seen


def test_a_restarted_call_side_on_windows_hears_the_call_meanwhile_and_is_routed_again(app, monkeypatch):
    """The call sound loopback's helper dies under them -> me: the call app is un-muted before the old loopback stops,
    the new side's loopback gets the call app and mutes it again; one message. A late end of the old loopback does
    not restart the new side."""
    ctl, states, seen = windows_side(app, monkeypatch)
    try:
        Loopback.made[0].on_end('the loopback helper ended (1): a Windows audio error')
        assert wait_for(lambda: any('restarted' in x for x in states), 10), states
        assert seen == [('retarget', 1, ''), ('restore',), ('stop', 1), ('start', 2), ('retarget', 2, DEVICES[2]),
                        ('mute', DEVICES[2])], seen
        assert len(states) == 1 and 'them → me restarted after an error (the call sound capture stopped' in states[0]
        Loopback.made[0].on_end('the loopback helper ended (1): late')  # the old one's, after the restart
        ctl.submit(lambda: None).result(5)
        assert len(Loopback.made) == 2 and len(states) == 1, states
        assert len(list(crash.reports_dir().glob('crash-*.txt'))) == 1
    finally:
        ctl.quit()


def test_a_call_app_that_is_not_running_is_no_restart_on_windows(app, monkeypatch):
    """Translation on before the call app runs (or it closed): the loopback ends with "<app> is not running". No helper
    died: no restart (no models made again, no crash report), the call app is un-muted and the tray says why."""
    ctl, states, seen = windows_side(app, monkeypatch)
    try:
        Loopback.made[0].on_end(f'{DEVICES[2]} is not running')
        assert wait_for(lambda: states, 10), states
        ctl.submit(lambda: None).result(5)
        assert seen == [('retarget', 1, ''), ('restore',)], seen  # un-muted; the side stays, nothing restarted
        assert states == [f'error: the call sound capture stopped: {DEVICES[2]} is not running'], states
        assert len(Loopback.made) == 1 and not list(crash.reports_dir().glob('crash-*.txt'))
    finally:
        ctl.quit()


WORK_MODELS = ('silero-vad', 'parakeet-tdt-0.6b-v3', 'quickmt-en-ru', 'piper-ru_RU-dmitri-medium')


def test_a_killed_piper_runner_restarts_the_side(app, monkeypatch):
    """The real helper: the Piper runner process is killed while them -> me translates a recording; only the runner
    starts again (crash.Respawn: the ASR and MT models stay loaded) and the side speaks again; one message, one report
    without what was said."""
    import os

    import psutil

    from peresmeshnik import models
    try:
        for m in WORK_MODELS:
            models.path(m)
    except models.ModelMissing as e:
        if os.environ.get('PERESMESHNIK_WORK_GATE') == '1':
            raise
        pytest.skip(f'needs the work-gate models: {e}')
    s = config.Settings()
    s.them_to_me, s.their_lang, s.my_lang = 'local', 'en', 'ru'
    out = FakeOut()
    ctl, states = controller(monkeypatch, s, out)
    try:
        ctl.set_translation(True).result(120)
        assert wait_for(lambda: out.played, 60), states
        runners = [p for p in psutil.Process().children(recursive=True) if 'piper' in ' '.join(p.cmdline())]
        assert runners, [p.cmdline() for p in psutil.Process().children(recursive=True)]
        t_kill = time.monotonic()
        for p in runners:
            p.kill()
        assert wait_for(lambda: any('the voice helper stopped' in x for x in states), 30), states
        assert wait_for(lambda: any(t > t_kill + 1 for t, _ in out.played), 60), 'no voice after the restart'
        back = min(t for t, _ in out.played if t > t_kill + 1) - t_kill
        print(f'piper runner killed -> the side speaks again in {back:.1f} s (only the runner starts again)')
        assert ctl.on and len([x for x in states if 'voice helper' in x]) == 1, states
        assert not any('restarted after an error' in x for x in states), states  # the side itself did not restart
        reports = list(crash.reports_dir().glob('crash-*.txt'))
        assert len(reports) == 1 and 'the voice helper died' in reports[0].read_text('utf-8')  # BrokenPipe or its exit
        assert not leaks(reports[0].read_text('utf-8'))
    finally:
        ctl.quit()


def test_the_loopholes_of_the_scrub_are_closed(app, monkeypatch):
    """L-42: what F1 got past the scrub: short words and names, a part of a Chinese or Japanese phrase (no spaces), the
    words of a clause said long ago, a third-party logger's ready message, the cause of a chained exception kept whole,
    the home folder URL-encoded and the host's name."""
    import platform
    import urllib.parse
    crash.said('Мой PIN 4821, собака Rex. Поднимите', 'My PIN is 4821, the dog is Rex. Raise it')
    crash.said('我们明天在新西伯利亚机场见面', '明日ノボシビルスク空港で会いましょう')
    for i in range(40):  # 40 clauses later
        crash.said(f'clause {i}', f'фраза {i}')
    host = platform.node()
    why = 'the cause: ' + 'Novosibirsk airport ' * 30 + 'END' + '-OF-CAUSE'  # not in the stack's source lines
    what = f'PIN dog Rex 新西伯利亚 ノボシビルスク at {urllib.parse.quote(str(crash.Path.home()), safe="")} on {host}'
    try:
        try:
            raise OSError(why)
        except OSError as cause:
            raise RuntimeError(what) from cause
    except RuntimeError:
        logging.getLogger('peresmeshnik.app').exception('control')
    logging.getLogger('onnxruntime.something').warning('a built message: собака Rex, 明天在新西伯利亚')
    logged = crash.log_file().read_text('utf-8')
    print(ascii(logged[-1500:]))
    for x in ('PIN', 'Rex', 'собака', '新西伯利亚', 'ノボシビルスク', '明天', urllib.parse.quote(str(crash.Path.home()), safe=''),
              'END-OF-CAUSE'):
        assert x not in logged, x
    assert host not in logged or len(host) < 3
    assert 'RuntimeError' in logged and 'OSError' in logged and 'onnxruntime.something' in logged  # what, still
    assert 'raise RuntimeError(what) from cause' in logged  # a frame is code: a word said ("raise") stays there


def test_the_apps_own_words_stay_in_what_happened(app):
    """CI: "the" said in a call does not eat the app's own text from "What happened"; the exception's text in it goes."""
    crash.said('the dog Rex barked', 'собака Рекс лаяла')
    e = ValueError('Rex barked at 3')
    p = crash.report(f'them → me: the translation failed (ValueError: {e})', e)
    first = p.read_text('utf-8').splitlines()[0]
    assert first.startswith('What happened: them → me: the translation failed (ValueError: ') and 'Rex' not in first, first


def test_a_generic_host_name_is_not_scrubbed(app, monkeypatch):
    """L-42 (review of PR #22): a host named like the system ("linux", "localhost") is no user's name: the stack keeps
    "x86_64-linux-gnu"; a host of the user's own is scrubbed."""
    import platform
    for host, kept in (('linux', True), ('localhost', True), ('ivans-laptop', False)):
        monkeypatch.setattr(crash, '_private', set())
        monkeypatch.setattr(platform, 'node', lambda host=host: host)
        crash.install(app)
        assert (crash.scrub(f'lib/x86_64-{host}-gnu/libc.so') == f'lib/x86_64-{host}-gnu/libc.so') is kept, host


def test_the_report_names_the_clone_engine_chosen_here(monkeypatch, tmp_path):
    """FIXES 6: the report's me -> them voice is languages.direction's (Pocket TTS on a CPU), not the settings' xtts-v2."""
    from peresmeshnik import languages, models
    monkeypatch.setattr(languages, 'gpu', lambda engine='': False)
    monkeypatch.setattr(models, 'hf_token', lambda: 'hf_test')
    monkeypatch.setenv('PERESMESHNIK_MODELS', str(tmp_path))
    s = config.Settings(me_to_them='local')
    line = next(x for x in crash.about(s).splitlines() if x.startswith('me -> them'))
    assert 'pocket-tts-en (not downloaded), voice none' in line and 'xtts' not in line
