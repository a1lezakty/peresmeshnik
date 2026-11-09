"""The me -> them side in a call (stage 5, S2): what the virtual mic gets, and the speakers' echo in my microphone.

- **My voice at the live level.** My turns go in real time through the app's path with the stock voice (Piper; the
  clone needs an NVIDIA GPU, its numbers are in BENCH stage 5): LocalDirection -> LiveSwitch -> a device in memory.
  The translation comes at my live voice's level (before stage 5 Piper and the clone spoke ~9 dB louder, peaks at full
  scale), so the hotkey's crossfade does not jump in loudness.
- **The speakers' echo in my microphone** (no headphones): them -> me plays its translation, the microphone hears it
  back through the room as loud as my own voice (bench/mic_echo.py `room`: laptop speakers, RT60 0.4-0.8 s, 80 ms).
  The 'me' side's gate (S1's audio/echo.py, reference: what the headphones play) zeroes it: it is not translated back
  to them and its partial subtitle does not cut what I hear (barge-in), my own words pass, also over its end.

Played: the Russian recordings of tests/data (as them -> me's translation would sound), mine: the English ones (other
people). The parts with models run with PERESMESHNIK_WORK_GATE=1 (CI job work-gate, every OS); without them, the
gate's part runs everywhere."""
import json
import os
import sys
import threading
import time
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'bench'))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from call import SR, NoTts, Scenario, clauses, score  # noqa: E402
from mic_echo import barge_ins, mic_gate, room  # noqa: E402
from test_echo import utterances  # noqa: E402

GATE = os.environ.get('PERESMESHNIK_WORK_GATE') == '1'
models = pytest.mark.skipif(not GATE and os.environ.get('GITHUB_JOB') != 'work-gate', reason='needs models: PERESMESHNIK_WORK_GATE=1')
DATA = Path(__file__).parent / 'data'
PLAYED = list(zip(utterances('call-ru.ogg'), (DATA / 'call-ru.txt').read_text(encoding='utf-8').splitlines()))
MINE = list(zip(utterances('call-en.ogg'), (DATA / 'call-en.txt').read_text(encoding='utf-8').splitlines()))


def speakers(gain_db=0.0, rt60=0.4, delay=0.08, echo_path=True):
    """3 rounds: the speakers say a translation, I answer 0.8 s after it, the last time 1 s before its end."""
    rng = np.random.default_rng(0)
    sc, t = Scenario(80), 0.5
    for k, ((x, xt), (y, yt)) in enumerate(zip(PLAYED, MINE)):
        sc.add('ours', x, t, xt, room(x, delay, gain_db, rt60, rng) if echo_path else None, delay)
        t += len(x) / SR + delay + (-1.0 if k == 2 else 0.8)
        sc.add('them', y, t, yt)  # 'them' in Scenario's terms: not our echo (my speech here)
        t += len(y) / SR + 0.8
    return sc.trim()


@pytest.mark.parametrize('gain_db,rt60,floor', [(0, 0.4, 0.85), (6, 0.4, 0.85), (-6, 0.4, 0.85), (0, 0.8, 0.85)])
def test_speakers_echo_is_zeroed_my_speech_is_not(gain_db, rt60, floor):
    """With the app's threshold for this side ([local] mic_echo_rho_on 0.4): a reverberant room (RT60 0.8 s) smears the
    echo's spectrum, at audio/echo.py's 0.5 only 81 % of its blocks were zeroed (89 % now; BENCH stage 5)."""
    _, s = mic_gate(speakers(gain_db, rt60))
    print(gain_db, rt60, s)
    assert s['echo_blocks_zeroed'] >= floor, s
    assert s['their_blocks_zeroed'] <= 0.02, s  # mine


def test_headphones_my_speech_is_not_zeroed():
    """With headphones nothing comes back, while the reference plays, also under my speech."""
    _, s = mic_gate(speakers(echo_path=False))
    assert s['their_blocks_zeroed'] == 0, s


def test_the_microphones_gate_takes_its_threshold(monkeypatch):
    """[local] mic_echo_rho_on for the 'me' side's gate (a reverberant room, BENCH stage 5); the call side's stays."""
    from test_echo import _controller

    from peresmeshnik import config
    s = config.Settings()
    assert s.local.mic_echo_rho_on == 0.4
    s.local.mic_echo_rho_on = 0.35
    ctl, _ = _controller(monkeypatch)
    assert ctl._side('me', s).echo.rho_on == 0.35 and ctl._side('them', s).echo.rho_on == 0.5


@pytest.mark.parametrize('rho,used', [(0.05, 0.2), (1.0, 0.8), (0.6, 0.6)])
def test_the_microphones_threshold_is_kept_in_range(tmp_path, capsys, rho, used):
    from peresmeshnik import config
    p = tmp_path / 'settings.toml'
    p.write_text(f'[local]\nmic_echo_rho_on = {rho}\n')
    assert config.load(p).local.mic_echo_rho_on == used
    assert ('outside 0.2..0.8' in capsys.readouterr().err) == (rho != used)


@pytest.fixture(scope='module')
def base():
    assert GATE, 'job work-gate without PERESMESHNIK_WORK_GATE=1: the scenarios would silently skip'
    from peresmeshnik.pipeline import make_local
    d = make_local('en', 'ru', threads=2)  # me -> them, my language English (the models the work-gate has)
    d.warm_up()
    yield d
    d.close()


@models
def test_speakers_echo_is_not_translated_back(base):
    from peresmeshnik.pipeline import LocalDirection
    d = LocalDirection(base.vad, base.asr, base.mt, NoTts(), 'en', 'ru')
    sc = speakers()
    rows = clauses(d, sc.capture)
    before = score(sc, rows) | {'barge_ins': barge_ins(sc, rows)}
    pcm, g = mic_gate(sc)
    rows = clauses(d, pcm)
    after = score(sc, rows) | g | {'barge_ins': barge_ins(sc, rows)}
    print('\nspeakers echo:', json.dumps({'before': before, 'after': after}))
    assert before['ours_out'] >= 2, before  # the scenario has an echo to find
    assert after['ours_out'] == 0 and after['barge_ins'] == 0, after
    assert after['them_recall'] >= before['them_recall'] - 0.05, (before, after)


def speech_db(x: np.ndarray, sr: int) -> float:
    """Level of speech: power mean of its 20 ms frames above -45 dBFS (pauses do not count)."""
    f = sr // 50
    p = np.mean(x[:len(x) // f * f].reshape(-1, f) ** 2, axis=1)
    return float(10 * np.log10(np.mean(p[p > 10 ** -4.5])))


@models
def test_my_voice_at_the_live_level(base):
    from me_voice import Sink, Traced, play, trace

    from peresmeshnik.audio.mic import PEAK, LiveSwitch
    from peresmeshnik.pipeline import LocalDirection, read_audio, run
    raw = []

    class Keep(Traced):
        def synth(self, text, lang, voice=None, **kw):
            for c in super().synth(text, lang, voice, **kw):
                raw.append(c)
                yield c
    tts = Keep(base.tts)
    d = LocalDirection(base.vad, base.asr, base.mt, tts, 'en', 'ru')
    d.realtime = True
    sw = trace(LiveSwitch(d))  # as the app's 'me' side: 24 kHz out
    mine = read_audio(str(DATA / 'call-en.ogg'))
    sink, events = Sink(sw.sample_rate, len(mine) / 16_000 + 60), []
    th = threading.Thread(target=play, args=(sw, sink, events))
    th.start()
    run(sw, np.concatenate([mine, np.zeros(8 * 16_000, np.float32)]), realtime=True)  # the microphone goes on
    d.finish()
    th.join()
    time.sleep(max(0.0, sink.end - time.perf_counter()))
    x = sink.x[:int((sink.end - sink.t0) * sink.sr) + 1]
    me, out, before = speech_db(mine, 16_000), speech_db(x, sink.sr), speech_db(np.concatenate(raw), tts.sample_rate)
    print(f'\nmy voice {me:.1f} dB; the stock voice {before:.1f} dB as it comes -> {out:.1f} dB out, '
          f'peak {np.abs(x).max():.2f} ({np.abs(np.concatenate(raw)).max():.2f}); {len(tts.said)} clauses')
    assert len(tts.said) >= 3
    assert abs(out - me) <= 2.0, (me, out, before)
    assert np.abs(x).max() <= PEAK + 1e-6
