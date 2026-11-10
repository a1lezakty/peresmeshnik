"""Pocket TTS on onnxruntime (local/pocket.py) without its weights or the network: the safetensors reader, the graphs
fed with random weights of the right shapes, the voice-state cache and its reset, a stopped generator. With real weights
($PERESMESHNIK_POCKET_DIR: model.safetensors + tokenizer.model) one more test speaks."""
import json
import os
import re
import struct
from pathlib import Path

import numpy as np
import pytest

from peresmeshnik import voice
from peresmeshnik.local import pocket

POCKET_DIR = os.environ.get('PERESMESHNIK_POCKET_DIR')


def write_safetensors(path: Path, tensors: dict, dtypes: dict | None = None) -> None:
    """The file format by hand (a JSON header, then the raw tensors); `dtypes`: name -> 'BF16' etc."""
    header, blobs, at = {}, [], 0
    for name, x in tensors.items():
        kind = (dtypes or {}).get(name, {'float32': 'F32', 'float16': 'F16', 'int64': 'I64'}[str(x.dtype)])
        bf16 = (np.asarray(x, np.float32).view(np.uint32) >> 16).astype('<u2')
        raw = bf16.tobytes() if kind == 'BF16' else x.tobytes()
        header[name] = {'dtype': kind, 'shape': list(x.shape), 'data_offsets': [at, at + len(raw)]}
        blobs.append(raw)
        at += len(raw)
    h = json.dumps(header | {'__metadata__': {'format': 'pt'}}).encode()
    h += b' ' * (-len(h) % 8)
    path.write_bytes(struct.pack('<Q', len(h)) + h + b''.join(blobs))


def test_safetensors_reader(tmp_path):
    t = {'a': np.arange(6, dtype=np.float32).reshape(2, 3) - 2.5,
         'b': np.array([1.5, -2.0, 3.140625, 1e-3], np.float32),
         'c': np.array([0.5, 2.0], np.float16), 'i': np.array([[7, -1]], np.int64)}
    write_safetensors(tmp_path / 'm.safetensors', t, {'b': 'BF16'})
    w = pocket.read_safetensors(tmp_path / 'm.safetensors')
    assert np.array_equal(w['a'], t['a']) and w['a'].dtype == np.float32
    assert w['b'].dtype == np.float32 and np.array_equal(w['b'][:3], t['b'][:3])  # exact in bf16
    assert w['b'][3] == pytest.approx(1e-3, rel=1e-2)  # bf16 keeps 8 bits of mantissa: truncated
    assert w['c'].dtype == np.float32 and list(w['c']) == [0.5, 2.0]
    assert w['i'].dtype == np.int64 and w['i'].tolist() == [[7, -1]]


def test_registry_folders(tmp_path):
    """Kyutai rows hold model.safetensors; Shtorm-ru model_fast.safetensors and a config.yaml with its settings."""
    (tmp_path / 'model_fast.safetensors').touch()
    assert pocket.weights_file(tmp_path).name == 'model_fast.safetensors'
    (tmp_path / 'model.safetensors').touch()
    assert pocket.weights_file(tmp_path).name == 'model.safetensors'
    assert pocket.settings(tmp_path) == (pocket.TEMP, pocket.EOS_LOGIT)
    (tmp_path / 'config.yaml').write_text('# x\nweights_path: hf://a/b\ndefault_temperature: 0.5  # tuned\n'
                                          'flow_lm:\n  default_eos_threshold: 9\n', encoding='utf-8')
    assert pocket.settings(tmp_path) == (0.5, pocket.EOS_LOGIT)  # only top-level keys
    with pytest.raises(FileNotFoundError):
        pocket.weights_file(tmp_path / 'x')


def random_weights(eos_bias: float = 10.0) -> dict[str, np.ndarray]:
    """Every weight the graphs and the engine read, at its shape (graphs.json), random. EOS fires at once."""
    meta = json.loads((pocket.GRAPHS / 'graphs.json').read_text(encoding='utf-8'))
    rng, w = np.random.default_rng(0), {}
    for shapes in meta['graphs'].values():
        for name, s in shapes.items():  # <name>.T, .int8 (+ .int8_scale): made by the engine from the weight [out, in]
            if not name.endswith('.int8_scale'):
                base, flip = re.subn(r'\.(T|int8)$', '', name)
                w[base] = rng.standard_normal(s[::-1] if flip else s, np.float32) * 0.02
    d, ld = meta['dim'], meta['ldim']
    w |= {'flow_lm.conditioner.embed.weight': rng.standard_normal((4001, d), np.float32) * 0.02,
          'flow_lm.input_linear.weight': rng.standard_normal((d, ld), np.float32) * 0.02,
          'flow_lm.bos_emb': rng.standard_normal(ld, np.float32),
          'flow_lm.bos_before_voice': rng.standard_normal((1, 1, d), np.float32),
          'flow_lm.emb_std': np.ones(ld, np.float32), 'flow_lm.emb_mean': np.zeros(ld, np.float32)}
    w['flow_lm.out_eos.bias'] = np.full(1, eos_bias, np.float32)
    return w


@pytest.fixture(scope='module')
def model(tmp_path_factory):
    """A model folder with a tiny SentencePiece tokenizer; its weights come from random_weights(), not from a file."""
    import sentencepiece as spm
    d = tmp_path_factory.mktemp('pocket')
    lines = ['Hello there, this is a test.', 'Привет, это проверка голоса.', 'One two three four five six seven.'] * 20
    with open(d / 'tokenizer.model', 'wb') as f:
        spm.SentencePieceTrainer.train(sentence_iterator=iter(lines), model_writer=f, vocab_size=60,
                                       hard_vocab_limit=False, minloglevel=2)
    (d / 'model.safetensors').touch()
    w = random_weights()
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(pocket, 'read_safetensors', lambda path: w)
        eng = pocket.PocketTts(d, threads=1, model_id='pocket-tts-test')
    yield eng, w
    eng.close()


def test_the_weights_feed_the_graphs(model):
    """The arrays handed over are the graphs' weights: other values, other audio; missing or misshaped: refused."""
    eng, w = model
    lat = np.random.default_rng(1).standard_normal((1, 2, 32)).astype(np.float32)
    a, _ = eng.graphs.decode(lat, eng.graphs.decoder_state())
    b, _ = eng.graphs.decode(lat, eng.graphs.decoder_state())
    assert a.shape == (2 * 1920,) and np.isfinite(a).all() and np.array_equal(a, b)
    name = next(n for n in eng.graphs.meta['graphs']['mimi_decoder'] if n.endswith('weight'))
    w2 = dict(w) | {name: w[name] * 3}
    c, _ = pocket.Graphs(w2, threads=1).decode(lat, (0, eng.graphs.decoder_state()[1]))
    assert not np.allclose(a, c)
    with pytest.raises(ValueError, match='not a Pocket TTS model'):
        pocket.Graphs({k: v for k, v in w.items() if k != name}, threads=1)
    with pytest.raises(ValueError, match='the graph wants'):
        pocket.Graphs(dict(w) | {name: w[name][:1]}, threads=1)


def test_int8(model):
    """quantize: per output channel, symmetric; the int8 FlowLM graph follows the float one closely."""
    w = np.random.default_rng(6).standard_normal((8, 5)).astype(np.float32) * np.arange(1, 9, dtype=np.float32)[:, None]
    q, scale = pocket.quantize(w)
    assert q.dtype == np.int8 and q.shape == (5, 8) and np.abs(q).max() == 127
    assert np.abs(q.T * scale[:, None] - w).max() <= scale.max() / 2 + 1e-6
    eng, weights = model
    g8 = pocket.Graphs(weights, threads=1, int8=True)
    rng = np.random.default_rng(7)
    x, noise = rng.standard_normal((1, 5, 1024), np.float32), rng.standard_normal((1, 32), np.float32)
    a, _, _ = eng.graphs.flow(x, eng.graphs.empty_cache(), noise)
    b, _, _ = g8.flow(x, g8.empty_cache(), noise)
    assert not np.array_equal(a, b) and np.abs(a - b).max() < 0.05 * np.abs(a).max()


def test_the_decoder_streams(model):
    """Two latents one by one give the audio of both at once: the conv and attention states carry over."""
    g = model[0].graphs
    lat = np.random.default_rng(2).standard_normal((1, 2, 32)).astype(np.float32)
    whole, _ = g.decode(lat, g.decoder_state())
    first, st = g.decode(lat[:, :1], g.decoder_state())
    second, _ = g.decode(lat[:, 1:], st)
    assert np.allclose(np.concatenate([first, second]), whole, atol=1e-5)


def test_the_sample_is_cut_to_10_s(model, monkeypatch):
    eng = model[0]
    got = []
    real = eng.graphs.encode
    monkeypatch.setattr(eng.graphs, 'encode', lambda x: got.append(x.shape[-1]) or real(x))
    loud = np.random.default_rng(3).standard_normal(16_000 * 30).astype(np.float32) * 0.1  # 30 s, no pause at 10 s
    state = eng.condition(loud, 16_000)
    frames = int(np.ceil((10 * 24_000 + 0.08 * 24_000) / 1920))  # 10 s + the pause end_on_pause appends, whole frames
    assert got == [frames * 1920]
    assert state.shape == (6, 2, 16, 1 + frames, 64)  # BOS before the voice + one position per frame


def test_resample():
    """16 / 48 / 44.1 kHz to 24 kHz: a 1 kHz tone stays a 1 kHz tone of the same level, the length scales."""
    for sr in (16_000, 48_000, 44_100, 24_000):
        x = np.sin(2 * np.pi * 1000 * np.arange(sr) / sr).astype(np.float32)
        y = pocket.resample(x, sr)
        ref = np.sin(2 * np.pi * 1000 * np.arange(24_000) / 24_000)
        assert len(y) == 24_000 and y.dtype == np.float32 and np.abs(y[1000:-1000] - ref[1000:-1000]).max() < 2e-3


def test_end_on_pause():
    x = np.concatenate([np.full(24_000, 0.5, np.float32), np.zeros(24_000, np.float32)])  # 1 s loud, 1 s silence
    y = pocket.end_on_pause(x)
    assert len(y) == 24_000 + int(0.08 * 24_000) and y[0] == 0.5 and not y[24_000 - 1] and not y[-1]


def test_the_voice_state_is_cached_with_its_recording(model, tmp_path, monkeypatch):
    eng = model[0]
    rng = np.random.default_rng(4)
    v = voice.save(rng.standard_normal(16_000 * 3).astype(np.float32) * 0.1, 16_000, 'me', 'ru', root=tmp_path)
    calls = []
    real = eng.condition
    monkeypatch.setattr(eng, 'condition', lambda pcm, sr: calls.append(sr) or real(pcm, sr))
    s1 = eng.state(v)
    assert (v.path / 'pocket-pocket-tts-test.npz').exists() and len(calls) == 1
    assert eng.state(v) is s1 and len(calls) == 1  # in memory
    eng.forget(v)
    assert np.array_equal(eng.state(v), s1) and len(calls) == 1  # from the file
    v2 = voice.save(rng.standard_normal(16_000 * 3).astype(np.float32) * 0.1, 16_000, 'me', 'ru', root=tmp_path)
    s2 = eng.state(v2)  # a new recording of the same profile: the file is not its state any more
    assert len(calls) == 2 and not np.array_equal(s1, s2)
    v3 = eng.enroll(rng.standard_normal(16_000 * 3).astype(np.float32) * 0.1, 16_000, name='me', root=tmp_path)
    assert v3.engine == 'pocket' and len(calls) == 3


def test_synth_streams_and_stops(model, tmp_path):
    eng = model[0]
    v = voice.save(np.random.default_rng(5).standard_normal(16_000 * 2).astype(np.float32) * 0.1, 16_000, 'me', 'en',
                   root=tmp_path)
    with pytest.raises(ValueError, match='enrolled voice'):
        next(eng.synth('Hello.', 'en'))
    chunks = list(eng.synth('Hello there.', 'en', v))
    # EOS from the first frame: MIN_FRAMES frames, then 2 + 3 more for a short text
    assert len(chunks) == pocket.MIN_FRAMES + 5 and all(c.dtype == np.float32 and c.shape == (1920,) for c in chunks)
    assert chunks[0][0] == 0  # faded in
    gen = eng.synth('Hello there.', 'en', v)
    next(gen)
    gen.close()  # barge-in: nothing left running
    assert len(list(eng.synth('Hello there.', 'en', v))) == len(chunks)


def test_prompt_and_split(model):
    assert pocket.prompt('hello there') == ('Hello there.', 5)
    said = '“Hi?”, she said, and went on with the story'
    assert pocket.prompt(said + ',') == (said + '.', 3)
    assert pocket.prompt('It’s fine!')[0] == "It's fine!"
    assert pocket.prompt('« Oui ; non » (bien)', 'fr')[0] == 'Oui , non bien.'
    assert pocket.prompt('Это «так» — да', 'ru')[0] == 'Это «так» — да.'  # Shtorm: no rewrites
    eng = model[0]
    long = ' '.join(['One two three four five six seven.'] * 12)
    parts = eng.split(long)
    assert len(parts) > 1 and all(len(eng._sp.encode(p)) <= pocket.MAX_TOKENS for p in parts)
    assert ' '.join(parts).split() == long.split()
    assert list(eng.synth(' ... ', 'en', 'unused')) == []  # nothing to say: nothing said, no voice needed


@pytest.mark.skipif(not POCKET_DIR, reason='$PERESMESHNIK_POCKET_DIR: a Pocket TTS model folder')
def test_real_weights_speak(tmp_path):
    t = np.arange(24_000 * 4) / 24_000
    pcm = (0.1 * np.sin(2 * np.pi * 140 * t) * (np.sin(2 * np.pi * 2 * t) > 0)).astype(np.float32)
    eng = pocket.PocketTts(POCKET_DIR, threads=2, model_id='test')
    v = voice.save(pcm, 24_000, 'me', 'en', root=tmp_path)
    wav = np.concatenate(list(eng.synth('Hello there, this is a test.', 'en', v)))
    assert np.isfinite(wav).all() and 0.5 < len(wav) / 24_000 < 6
    eng.close()


def test_the_voice_state_follows_the_weights_and_survives_a_broken_file(model, tmp_path, monkeypatch):
    """The cache file names the weights' sha256 from the registry: other weights, the state made again; a broken or cut
    file is made again too, never raised."""
    assert pocket.registry_sha256('pocket-tts-en', 'model.safetensors').startswith('bc96b824')
    assert pocket.registry_sha256('', 'model.safetensors') == pocket.registry_sha256('x', 'model.safetensors') == ''
    eng = model[0]
    v = voice.save(np.random.default_rng(8).standard_normal(16_000 * 3).astype(np.float32) * 0.1, 16_000, 'me', 'ru',
                   root=tmp_path)
    calls = []
    real = eng.condition
    monkeypatch.setattr(eng, 'condition', lambda pcm, sr: calls.append(sr) or real(pcm, sr))
    cache = v.path / 'pocket-pocket-tts-test.npz'
    eng.state(v)
    good = cache.read_bytes()
    for broken in (b'not a zip', good[:len(good) // 2], b''):
        eng.forget(v)
        cache.write_bytes(broken)
        eng.state(v)
    assert len(calls) == 4 and cache.read_bytes() != b''
    eng.forget(v)
    eng.state(v)
    assert len(calls) == 4  # the file made again is read
    monkeypatch.setattr(eng, '_weights_sha', 'f' * 64)  # new weights of the same model in the registry
    eng.forget(v)
    eng.state(v)
    assert len(calls) == 5
    eng.forget(v)
    eng.state(v)
    assert len(calls) == 5


def test_a_voice_with_its_own_state_is_spoken_from_it():
    """COMMON 20:28 (S2): the other person's voice, conditioned live, carries its state and has no profile folder: state()
    returns it as it is, reads no file, computes nothing."""
    import types

    from peresmeshnik.local.pocket import PocketTts
    own = np.zeros((2, 3), np.float32)
    assert PocketTts.state(types.SimpleNamespace(), types.SimpleNamespace(state=own, path='/no/such/folder')) is own

