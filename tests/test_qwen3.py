"""Qwen3-TTS in the voice pack (peresmeshnik/local/qwen3.py) on stand-ins of torch, faster_qwen3_tts and chatterbox
(tests/qwen3_stub.py): the length guard, Chatterbox taking over a voice Qwen3 ran away on, the speaker vector's cache,
and the runner's choice of the model code by --kind. No GPU, no weights."""
import sys
from pathlib import Path

import numpy as np
import pytest
import qwen3_stub
import soundfile as sf
from qwen3_stub import CHATTERBOX, QWEN

from peresmeshnik.engines import Voice
from peresmeshnik.local import qwen3, xtts
from peresmeshnik.local.clone import CloneUnavailable, VoicePackClone

STUB = [sys.executable, str(Path(__file__).with_name('qwen3_stub.py'))]


@pytest.fixture(autouse=True)
def fakes(monkeypatch):
    for name, m in qwen3_stub.fakes().items():
        monkeypatch.setitem(sys.modules, name, m)


def profile(root: Path, name: str, seconds: float = 15, level: float = 0.2) -> Voice:
    (root / name).mkdir(parents=True)
    sf.write(root / name / 'reference.wav', np.full(int(seconds * 16_000), level, np.float32), 16_000)
    return Voice(name, 'qwen3', root / name)


@pytest.fixture
def eng(tmp_path):
    (tmp_path / 'chatterbox').mkdir()
    return qwen3.Engine(tmp_path / 'qwen3', device='cuda', fallback_dir=tmp_path / 'chatterbox')


def whose(x: np.ndarray) -> tuple[int, int]:
    """Samples of Qwen3's audio and of Chatterbox's."""
    return int(np.sum(x == np.float32(QWEN))), int(np.sum(x == np.float32(CHATTERBOX)))


CHATTERBOX_ONLY = (0, 24_000 - xtts.FADE_IN + 1)  # a clause of its own: faded in


def say(e, text, v, lang='ru'):
    return np.concatenate(list(e.synth(text, lang, v)))


def test_a_clause_streams_from_qwen3_and_chatterbox_is_not_loaded(eng, tmp_path):
    me = profile(tmp_path, 'me')
    q, c = whose(say(eng, 'привет всем', me))
    assert c == 0 and 12 * qwen3_stub.CHUNK - xtts.FADE_IN <= q <= 12 * qwen3_stub.CHUNK  # FADE_IN: faded in
    assert eng.model.calls[0][1] == 'Russian' and eng.fallback is None and eng.info() == {
        'runaways': 0, 'fallback_voices': [], 'fallback_loaded': False}
    x = say(eng, 'Привет.', me)  # a sentence: its pause after it
    assert np.all(x[-int(xtts.SENTENCE_PAUSE_S * qwen3.SR):] == 0)
    with pytest.raises(ValueError, match="does not speak 'tr'"):
        say(eng, 'merhaba', me, 'tr')


def test_a_runaway_is_cut_and_chatterbox_speaks_that_voice_from_then_on(eng, tmp_path):
    me, you = profile(tmp_path, 'me'), profile(tmp_path, 'you')
    budget = xtts.budget_samples('loop loop loop')
    q, c = whose(x := say(eng, 'loop loop loop', me))
    assert budget - xtts.FADE_IN - xtts.FADE <= q < budget and c == 24_000 and x[-1] == np.float32(CHATTERBOX)
    assert eng.model.closed == 1 and eng.model.chunks <= budget // qwen3_stub.CHUNK + 1  # its generation stopped there
    assert eng.fallback.said == [('loop loop loop', 'ru', str(me.path / 'reference.wav'), qwen3.CFG)]
    calls, cb = len(eng.model.calls), eng.fallback
    assert whose(say(eng, 'привет', me)) == CHATTERBOX_ONLY and len(eng.model.calls) == calls  # Chatterbox only now
    assert eng.fallback is cb  # loaded once
    assert whose(say(eng, 'привет', you))[1] == 0  # another voice: still Qwen3
    assert eng.info() == {'runaways': 1, 'fallback_voices': [str(me.path)], 'fallback_loaded': True}
    eng.forget(me)  # a new recording: still Chatterbox (for this engine's life), conditioned again
    assert whose(say(eng, 'привет', me)) == CHATTERBOX_ONLY and len(eng.model.calls) == calls + 1


def test_without_a_fallback_a_runaway_is_only_cut(tmp_path):
    eng, me = qwen3.Engine(tmp_path / 'qwen3'), profile(tmp_path, 'me')
    budget = xtts.budget_samples('loop')
    for _ in range(2):
        q, c = whose(say(eng, 'loop', me))
        assert q < budget and c == 0
    assert eng.info() == {'runaways': 2, 'fallback_voices': [], 'fallback_loaded': False}


def test_barge_in_stops_qwen3_and_is_not_a_runaway(eng, tmp_path):
    me = profile(tmp_path, 'me')
    gen = eng.synth('loop', 'en', me)
    next(gen), next(gen)
    gen.close()
    assert eng.model.closed == 1 and eng.runaways == 0 and eng.fallback is None


def test_the_speaker_vector_is_cached_next_to_the_recording(eng, tmp_path):
    me = profile(tmp_path, 'me', seconds=15)
    say(eng, 'привет', me)
    assert eng.model.model.vectors == [10.0]  # the first 10 s of the recording
    cache = (me.path / 'qwen3.pt').read_bytes()
    other = qwen3.Engine(tmp_path / 'qwen3')
    say(other, 'привет', me)
    assert other.model.model.vectors == [] and (me.path / 'qwen3.pt').read_bytes() == cache  # read, not computed
    profile(tmp_path / 'new', 'me', seconds=3, level=0.5)
    (me.path / 'reference.wav').write_bytes((tmp_path / 'new' / 'me' / 'reference.wav').read_bytes())  # re-recorded
    other.forget(me)
    say(other, 'привет', me)
    assert other.model.model.vectors == [3.0] and other.model.calls[-1][2].x[0] == pytest.approx(0.5, abs=1e-3)
    assert other.similarity(me, np.full(16_000, 0.5, np.float32), 16_000) == pytest.approx(1.0)


def test_the_runner_picks_the_model_code_by_kind(tmp_path):
    me = profile(tmp_path, 'me')
    (tmp_path / 'chatterbox').mkdir()
    log = tmp_path / 'runner.log'
    p = VoicePackClone(tmp_path / 'qwen3', cmd=[*STUB, '--kind', 'qwen3', '--fallback-dir', str(tmp_path / 'chatterbox'),
                                                '--device', 'cpu'], log=log)
    try:
        assert p.sample_rate == 24_000 and p.info['torch'] == 'stub'
        assert whose(np.concatenate(list(p.synth('привет', 'ru', me))))[1] == 0
        q, c = whose(np.concatenate(list(p.synth('loop', 'ru', me))))
        assert 0 < q and c == 24_000
        assert {k: v for k, v in p.stats().items() if k != 'torch' and k != 'cuda'} == {
            'runaways': 1, 'fallback_voices': [str(me.path)], 'fallback_loaded': True}
        assert p.latents(me) and (me.path / 'qwen3.pt').exists()
    finally:
        p.close()
    assert 'qwen3: runaway in ' in log.read_text()
    with pytest.raises(CloneUnavailable, match="No module named 'TTS'"):  # the default kind: xtts.py, no coqui-tts here
        VoicePackClone(tmp_path / 'xtts', cmd=[*STUB, '--device', 'cpu'], log=log)


def test_the_xtts_cache_follows_the_recording(tmp_path):
    """COMMON 19:53 / FIXES 14: xtts.pt carries the recording's SHA-256 (voice.json's ref_sha256): a new take, or a
    0.3.0 cache without it, is conditioned again, never spoken from the old recording; the same take is read back."""
    import types
    v = profile(tmp_path, 'me')
    made = []
    eng = types.SimpleNamespace(_latents={}, device='cuda', _condition=lambda pcm, sr: made.append(len(pcm)) or {
        'gpt': qwen3_stub.FakeTensor([len(made)]), 'spk': qwen3_stub.FakeTensor([0])})
    gpt, _ = xtts.Engine.latents(eng, v)
    assert made and gpt.x[0] == 1 and (v.path / 'xtts.pt').exists()
    eng._latents.clear()
    xtts.Engine.latents(eng, v)
    assert len(made) == 1  # the same recording: from xtts.pt
    sf.write(v.path / 'reference.wav', np.full(16_000 * 15, 0.3, np.float32), 16_000)  # a new take
    eng._latents.clear()
    gpt, _ = xtts.Engine.latents(eng, v)
    assert len(made) == 2 and gpt.x[0] == 2


def test_a_voice_from_a_sample_and_chatterbox_from_it(eng):
    """voices/live.py: condition(pcm, sr) -> a state; Qwen3 speaks from its vector (the first REF_S s); a runaway gives
    that voice to Chatterbox conditioned from the sample in memory (no file), a new state of it is Qwen3's again."""
    import types
    st = eng.condition(np.full(15 * 16_000, 0.4, np.float32), 16_000)
    them = types.SimpleNamespace(path=None, state=st)
    assert whose(say(eng, 'привет', them))[1] == 0 and eng.model.model.vectors == [10.0]
    assert eng.model.calls[-1][2].x[0] == pytest.approx(0.4)
    assert eng.similarity(them, np.full(16_000, 0.4, np.float32), 16_000) == pytest.approx(1.0)
    q, c = whose(say(eng, 'loop', them))
    assert 0 < q and c == 24_000 and eng.fallback.said[-1][2] == ('sample', 10.0, 0.4)
    assert whose(say(eng, 'привет', them)) == CHATTERBOX_ONLY and eng.info()['fallback_voices'] == [st.key]
    again = types.SimpleNamespace(path=None, state=eng.condition(np.full(16_000, 0.2, np.float32), 16_000))
    assert whose(say(eng, 'привет', again))[1] == 0


def test_the_runner_speaks_a_state_of_qwen3(tmp_path):
    p = VoicePackClone(tmp_path / 'qwen3', cmd=[*STUB, '--kind', 'qwen3', '--device', 'cpu'], log=tmp_path / 'runner.log')
    try:
        import types
        sid = p.condition(np.full(16_000, 0.3, np.float32), 16_000)
        assert whose(np.concatenate(list(p.synth('привет', 'ru', types.SimpleNamespace(path=None, state=sid)))))[0] > 0
    finally:
        p.close()
