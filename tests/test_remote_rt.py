"""bench/remote_rt.py: the client-side player that turns arrival times into voice lags, jitter holds and underruns."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'bench'))
from remote_rt import FRAME, SAMPLE_RATE, Trace, clause_ends, play  # noqa: E402


def trace(chunks, ends_s):
    t = Trace()
    for k, end in enumerate(ends_s):
        t.text(0.0, {'type': 'clause', 'clause': k, 'end_s': end})
    t.chunks.update(chunks)
    return t


def test_clause_end_is_the_send_time_of_its_last_sample():
    dt = FRAME / SAMPLE_RATE
    sent = [10 + i * dt + 0.005 for i in range(100)]  # every frame 5 ms late; frame i holds the 32 ms before it is sent
    ends = clause_ends(trace({}, [1.0]), sent)
    assert abs(ends[0] - (10 - dt + 1.0 + 0.005)) < 1e-6  # as the pipeline's own clock (LocalDirection._wall)


def test_back_to_back_with_a_jitter_hold_and_underruns():
    ends = {0: 0.0, 1: 1.0}
    t = trace({0: [(0.5, 1.0), (1.6, 0.5)],  # the second chunk comes 0.1 s after the first ran dry
               1: [(1.2, 0.5)]}, [0.0, 1.0])  # clause 1 waits behind clause 0
    lags, under, under_s = play(t, ends, 0.0)
    assert lags == {0: 0.5, 1: 1.1} and under == 1 and abs(under_s - 0.1) < 1e-9
    lags, under, _ = play(t, ends, 0.2)  # the hold covers the late chunk
    assert abs(lags[0] - 0.7) < 1e-9 and under == 0 and abs(lags[1] - 1.2) < 1e-9


def test_dropped_clauses_are_left_out():
    t = trace({}, [1.0])
    t.clauses[0]['dropped'] = 4.2
    assert clause_ends(t, [0.0] * 100) == {}
