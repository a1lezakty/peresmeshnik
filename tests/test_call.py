"""Stage 5 call scenarios through the local pipeline with the real models (Silero, Parakeet, OPUS-MT, Piper), on the
recordings of tests/data: theirs = call-en.ogg, ours = ours-en.ogg (other people). bench/call.py builds the
scenarios and scores them; docs/BENCH.md "Stage 5" has the numbers on FLEURS.

- echo: our speech comes back from a far end on a loudspeaker (150 ms, -12 dB): without the gate it is translated
  back to us, with it not, and their words are not lost;
- long phrases: a monologue without pauses; the translated subtitle of a word comes earlier with the translated
  partials (stage 5) than with clause-end captions only (before), the words are as right;
- two people: their turns overlap; nobody is lost where the other one only overlaps the end.

Needs the models, so it runs with PERESMESHNIK_WORK_GATE=1 (CI job work-gate, every OS)."""
import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest

GATE = os.environ.get('PERESMESHNIK_WORK_GATE') == '1'
pytestmark = pytest.mark.skipif(not GATE and os.environ.get('GITHUB_JOB') != 'work-gate', reason='needs models: PERESMESHNIK_WORK_GATE=1')
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'bench'))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from call import SR, NoTts, Scenario, clauses, content, far_end, gate, norm, score, squeeze  # noqa: E402

from peresmeshnik.pipeline import read_audio  # noqa: E402

DATA = Path(__file__).parent / 'data'


def sentences(name):
    from test_echo import utterances
    return list(zip(utterances(f'{name}.ogg'), (DATA / f'{name}.txt').read_text(encoding='utf-8').splitlines()))


@pytest.fixture(scope='module')
def base():
    assert GATE, 'job work-gate without PERESMESHNIK_WORK_GATE=1: the scenarios would silently skip'
    from peresmeshnik.pipeline import make_local
    d = make_local('en', 'ru', threads=2)
    d.warm_up()
    yield d
    d.close()


def quiet(base):
    from peresmeshnik.pipeline import LocalDirection
    return LocalDirection(base.vad, base.asr, base.mt, NoTts(), 'en', 'ru')


def call_with_echo():
    rng = np.random.default_rng(0)
    sc, t = Scenario(90), 0.5
    for k, ((x, xt), (y, yt)) in enumerate(zip(sentences('ours-en'), sentences('call-en'))):
        sc.add('ours', x, t, xt, far_end(x, 0.15, -12, rng), 0.15)
        t += len(x) / SR + 0.15 + (-1.0 if k == 2 else 0.8)  # the last one: they talk over our echo
        sc.add('them', y, t, yt)
        t += len(y) / SR + 0.8
    return sc.trim()


def test_echo_is_not_translated(base):
    sc = call_with_echo()
    d = quiet(base)
    before = score(sc, clauses(d, sc.capture))
    pcm, g = gate(sc)
    after = score(sc, clauses(d, pcm)) | g
    print('\necho:', json.dumps({'before': before, 'after': after}))
    assert before['ours_out'] >= 2, before  # the scenario has an echo to find
    assert after['ours_out'] == 0, after
    assert after['them_recall'] >= before['them_recall'] - 0.05, (before, after)


def wer(ref: str, hyp: str) -> float:
    r, h = ref.split(), hyp.split()
    d = list(range(len(h) + 1))
    for i, a in enumerate(r, 1):
        prev, d[0] = d[0], i
        for j, b in enumerate(h, 1):
            prev, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, prev + (a != b))
    return d[-1] / max(len(r), 1)


def test_long_phrases_get_earlier_subtitles(base):
    sents = sentences('call-en')[:2] + sentences('ours-en')[:2]
    sc, t = Scenario(60), 0.5
    for x, text in sents:
        x = squeeze(x)
        sc.add('them', x, t, text)
        t += len(x) / SR + 0.1
    sc.trim()
    ref = norm(' '.join(text for _, text in sents))
    out = {}
    for name, partial in (('before', False), ('after', True)):
        base.partial_mt = partial
        rows = clauses(base, sc.capture, realtime=True)
        nw = lambda x: max(len(x['text'].split()), 1)  # noqa: E731
        spans = []
        for x in rows:
            for i in range(nw(x)):
                t_w = x['start_s'] + (x['end_s'] - x['start_s']) * (i + 0.5) / nw(x)
                first = min([x['end_s'] + x['caption_lag_s']] + [e + lag for k, e, lag in x.get('partials', ()) if k > i])
                spans.append(first - t_w)
        out[name] = {'subtitle_s_p50': round(float(np.median(spans)), 2), 'clauses': len(rows),
                     'wer': round(wer(ref, norm(' '.join(x['text'] for x in rows))), 3)}
    print('\nlong phrases:', json.dumps(out))
    assert out['after']['subtitle_s_p50'] <= out['before']['subtitle_s_p50'] - 0.5, out
    assert out['after']['wer'] <= out['before']['wer'] + 0.05, out


def test_two_people_overlapping_ends(base):
    """They take turns, each starting 0.5-1 s before the other one ends: every utterance is mostly out."""
    a, b = sentences('call-en'), sentences('ours-en')
    sc, t = Scenario(80), 0.5
    for k, (x, text) in enumerate([a[0], b[0], a[1], b[1], a[2], b[2]]):
        sc.add('them', x, t, text)
        t += len(x) / SR - (0.5 + 0.1 * k)
    sc.trim()
    rows = clauses(quiet(base), sc.capture)
    got = [{w: 0 for w in content(s['text'])} for s in sc.segs]
    for r in rows:
        for w in content(r['text']):
            i = next((i for i, s in enumerate(sc.segs) if w in got[i] and not got[i][w]
                      and s['start'] - 1.0 <= r['end_s'] and r['start_s'] <= s['end'] + 1.0), None)
            if i is not None:
                got[i][w] = 1
    recall = [round(sum(g.values()) / max(len(g), 1), 2) for g in got]
    print('\ntwo people:', recall, [r['text'] for r in rows])
    assert min(recall) >= 0.5, recall


def test_latency_per_link(base):
    """The call through the gate in real time with the voice and the app's queue policy: seconds from the end of a
    clause's speech to each link (BENCH stage 5 table). Bound loosely: a runner is slow and shared."""
    from call import LINKS, pct

    from peresmeshnik.pipeline import MAX_LAG_S
    pcm, _ = gate(call_with_echo())
    base.max_lag_s = MAX_LAG_S
    rows = clauses(base, pcm, realtime=True)
    out = {name: {'p50': pct([r[k] for r in rows if k in r], 50), 'p90': pct([r[k] for r in rows if k in r], 90)}
           for name, k in LINKS}
    print('\nlinks:', json.dumps(out | {'clauses': len(rows), 'dropped': sum('dropped' in r for r in rows)}))
    assert out['caption']['p50'] is not None and out['caption']['p50'] < 2.0, out
    assert out['voice']['p50'] is not None and out['voice']['p50'] < 4.0, out


def test_data_is_there():
    assert len(read_audio(str(DATA / 'ours-en.ogg'))) > 20 * SR
