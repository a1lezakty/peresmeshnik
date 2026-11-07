"""LocalDirection on fake engines: clause segmentation, event order, latency log, silence skips ASR. No models."""
import json
import logging

import numpy as np
import pytest

from peresmeshnik.engines import SAMPLE_RATE, Caption, Speech
from peresmeshnik.pipeline import LocalDirection, run

FRAME = 512


class EnergyVad:
    """Silero's event contract on an energy threshold."""
    min_silence_frames = 10  # 320 ms

    def __init__(self):
        self.pos, self.busy_s, self.speaking, self.silent = 0, 0.0, False, 0

    def feed(self, pcm):
        for i in range(0, len(pcm), FRAME):
            f = pcm[i:i + FRAME]
            self.pos += len(f)
            loud = float(np.abs(f).max()) > 0.1
            if not self.speaking:
                if loud:
                    self.speaking, self.silent = True, 0
                    yield 'start', f
                continue
            self.silent = 0 if loud else self.silent + 1
            if self.silent >= self.min_silence_frames:
                self.speaking = False
                yield 'end', f
            else:
                yield 'speech', f


class WordAsr:
    """One word per 0.5 s of audio; no punctuation, so long clauses are cut by length only."""
    def __init__(self):
        self.streams = 0

    def stream(self, lang):
        self.streams += 1
        return WordStream()


class WordStream:
    def __init__(self):
        self.pcm, self.offset, self.words = np.zeros(0, np.float32), 0, 0

    @property
    def duration(self):
        return len(self.pcm) / SAMPLE_RATE

    def text(self, n):
        return ' '.join(f'w{self.words + i}' for i in range(n))

    def feed(self, pcm, partial=True):
        self.pcm = np.concatenate([self.pcm, pcm])
        return self.text(int(self.duration * 2)) if len(self.pcm) % (10 * FRAME) == 0 else None

    def cut(self, force=False):
        if not force:
            return None
        n = int(self.duration * 2) - 1  # keep the last word
        at = n * SAMPLE_RATE // 2
        text, self.words = self.text(n), self.words + n
        self.pcm, self.offset = self.pcm[at:], self.offset + at
        return text

    def finish(self, settled_s=0.0):
        text = self.text(max(1, int(self.duration * 2)))
        self.offset += len(self.pcm)
        self.pcm = np.zeros(0, np.float32)
        return text


class UpperMt:
    def translate(self, text, src, tgt, context=()):
        return text.upper()


class ToneTts:
    sample_rate = 8000

    def synth(self, text, lang, voice=None):
        for _ in text.split():
            yield np.zeros(800, np.float32)


def tone(s):
    return (0.5 * np.sin(np.arange(int(s * SAMPLE_RATE)) * 0.1)).astype(np.float32)


def silence(s):
    return np.zeros(int(s * SAMPLE_RATE), np.float32)


def translate(pcm, caplog=None):
    asr = WordAsr()
    d = LocalDirection(EnergyVad(), asr, UpperMt(), ToneTts(), 'en', 'ru')
    d.realtime = False
    run(d, pcm)
    d.finish()
    return asr, list(d.events()), d.summary()


def test_clauses_events_and_log(caplog):
    caplog.set_level(logging.INFO, 'peresmeshnik.latency')
    asr, events, summary = translate(np.concatenate([silence(0.5), tone(1.5), silence(0.6), tone(1.0), silence(0.6)]))
    finals = [e for e in events if isinstance(e, Caption) and e.final]
    assert [c.translation for c in finals] == [c.original.upper() for c in finals]
    assert len(finals) == 2 and asr.streams == 2
    # one chunk per word; speech starts after the first caption, captions do not wait for speech
    kinds = ''.join('C' if isinstance(e, Caption) and e.final else 'S' for e in events if isinstance(e, Speech) or e.final)
    assert kinds.count('S') == sum(len(c.original.split()) for c in finals) and kinds.index('S') > kinds.index('C')
    rows = [json.loads(r.getMessage()) for r in caplog.records]
    assert [r['reason'] for r in rows] == ['pause', 'pause']
    assert {'asr_final_ms', 'mt_ms', 'tts_first_ms', 'tts_ms'} <= rows[0].keys()
    assert 'voice_lag_s' not in rows[0]  # not real time, so no lag numbers
    assert 2.0 < rows[0]['end_s'] < 2.2  # speech ended at 2.0 s; the trailing silence is not counted
    assert summary['clauses'] == 2 and summary['audio_s'] > 4


def test_silence_runs_no_asr():
    asr, events, _ = translate(silence(3.0))
    assert asr.streams == 0 and events == []


def test_long_speech_is_cut():
    caplog_rows = []
    handler = logging.Handler()
    handler.emit = lambda r: caplog_rows.append(json.loads(r.getMessage()))
    log = logging.getLogger('peresmeshnik.latency')
    log.addHandler(handler)
    log.setLevel(logging.INFO)
    try:
        _, events, _ = translate(np.concatenate([tone(12.0), silence(0.6)]))
    finally:
        log.removeHandler(handler)
    finals = [e.original for e in events if isinstance(e, Caption) and e.final]
    words = ' '.join(finals).split()
    assert words == [f'w{i}' for i in range(len(words))]  # nothing lost or repeated across cuts
    assert [r['reason'] for r in caplog_rows][:-1] == ['max'] * (len(finals) - 1)
    assert len(finals) >= 3


@pytest.mark.filterwarnings('ignore::pytest.PytestUnhandledThreadExceptionWarning')
def test_dead_worker_does_not_hang():
    class BrokenMt:
        def translate(self, *a, **k):
            raise RuntimeError('boom')

    d = LocalDirection(EnergyVad(), WordAsr(), BrokenMt(), ToneTts(), 'en', 'ru')
    run(d, np.concatenate([tone(1.0), silence(0.6), tone(1.0), silence(0.6)]))
    d.finish()  # returns although the translate worker died
    assert not [e for e in d.events() if isinstance(e, Caption) and e.final]
