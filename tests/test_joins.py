"""Clause joins (audio/joins.py): silence at a clause's edges cut, a clause that goes on starts as loud as the last ended.
No models: tones and room noise."""
import numpy as np

from peresmeshnik.audio import joins
from peresmeshnik.audio.joins import Joins, goes_on, speech_db

SR = 24_000


def noise(s, db=-60.0, seed=0):
    return (np.random.default_rng(seed).standard_normal(int(s * SR)) * 10 ** (db / 20)).astype(np.float32)


def tone(s, db=-20.0, fall_db=0.0):
    """A 200 Hz tone at `db` (rms), falling linearly by `fall_db` over its length (a clause's loud start, quiet end)."""
    t = np.arange(int(s * SR))
    g = 10 ** ((db - fall_db * t / len(t)) / 20) * np.sqrt(2)
    return (g * np.sin(2 * np.pi * 200 * t / SR)).astype(np.float32)


def chunks(x, n=1920):  # Pocket's 80 ms
    return [x[i:i + n] for i in range(0, len(x), n)]


def say(j, x, cut=False):
    return np.concatenate(list(j(chunks(x), SR, cut)))


def test_edges_cut_pauses_kept_nothing_lost():
    j = Joins()
    pause = noise(0.4, seed=1)
    x = np.concatenate([noise(0.3), tone(1.0), pause, tone(0.5), noise(0.4, seed=2)])
    y = say(j, x, cut=True)
    assert abs(len(y) - (2 * joins.EDGE_S + 1.0 + 0.4 + 0.5) * SR) <= 2 * SR * joins.FRAME_S  # the pause inside stays
    assert np.array_equal(y[int(joins.EDGE_S * SR):][:SR], x[int(0.3 * SR):][:SR])  # the speech as it was
    assert y[0] == 0 and y[-1] == 0 and np.abs(y[:5]).max() < np.abs(y[100:200]).max()  # cut noise fades: no click
    y2 = say(j, np.concatenate([noise(0.3), tone(0.5), noise(0.6)]))  # a sentence's end: PAUSE_S of it after
    assert abs(len(y2) - (joins.EDGE_S + 0.5 + joins.PAUSE_S) * SR) <= 2 * SR * joins.FRAME_S


def test_digital_silence_and_a_soft_end():
    x = np.concatenate([np.zeros(int(0.3 * SR), np.float32), tone(0.5), tone(0.2, -45.0), np.zeros(int(0.3 * SR), np.float32)])
    y = say(Joins(), x, cut=True)
    assert abs(len(y) - (2 * joins.EDGE_S + 0.7) * SR) <= 2 * SR * joins.FRAME_S  # -45 dBFS after -20 is still sound


def test_sound_goes_out_as_it_came_its_end_faded():
    x = tone(0.8)
    out = list(Joins()(chunks(x), SR))
    end = round(joins.FADE_S * SR)
    assert len(out) == len(chunks(x)) + 1 and all(len(c) for c in out)  # its last 5 ms held back, to fade them out
    y = np.concatenate(out)
    assert np.array_equal(y[:-end], x[:-end]) and y[-1] == 0  # an engine that stops mid-sound does not click


def test_no_sound_at_all_goes_out_as_it_came():
    x = np.zeros(SR, np.float32)
    out = list(Joins()(chunks(x), SR))
    assert len(out) == len(chunks(x)) and sum(map(len, out)) == len(x)


def test_a_clause_that_goes_on_starts_as_loud_as_the_last_ended():
    j = Joins()
    clause = np.concatenate([noise(0.2), tone(2.0, -20.0, fall_db=9.0), noise(0.3)])  # -20 dB down to -29 dB
    say(j, clause, cut=True)
    assert -29 < j.end_db < -27  # its last 0.4 s of speech
    y = say(j, clause)
    raw = speech_db(clause, SR, True, int(0.4 * SR))
    start = speech_db(y, SR, True, int(0.4 * SR))
    assert raw - start > 3.5  # was 7 dB above where the last one ended: turned down
    lead, b = int(joins.EDGE_S * SR), clause[int(0.2 * SR):]
    late = slice(int((joins.RAMP_S + 0.1) * SR), int((joins.RAMP_S + 0.5) * SR))  # past RAMP_S: its own loudness again
    assert np.allclose(y[lead:][late], b[late], atol=1e-6)
    c = say(j, clause)  # after a sentence's end: as it is
    assert np.allclose(c[lead:lead + SR // 2], b[:SR // 2], atol=1e-6)


def test_it_is_never_turned_up_and_a_broken_off_clause_leaves_no_level():
    j = Joins()
    say(j, tone(1.0, -10.0), cut=True)
    y = say(j, tone(1.0, -30.0))
    assert np.allclose(y[:SR // 2], tone(1.0, -30.0)[:SR // 2], atol=1e-6)  # no silence before it: nothing cut
    g = j(chunks(tone(2.0)), SR, cut=True)
    next(g)
    g.close()
    assert j.end_db is None


def test_closing_closes_the_engine():
    closed = []

    def synth():
        try:
            yield from chunks(tone(2.0))
        finally:
            closed.append(True)
    g = Joins()(synth(), SR)
    next(g)
    g.close()
    assert closed == [True]


def test_goes_on_text():
    assert goes_on('Мы пошли в магазин.') == 'Мы пошли в магазин,'
    assert goes_on('and then we went to the') == 'and then we went to the,'
    assert goes_on('Is it?') == 'Is it?' and goes_on('') == ''
