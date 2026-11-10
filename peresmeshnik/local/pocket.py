"""Kyutai Pocket TTS (100M, a voice cloned from a sample) on onnxruntime, behind `Tts` and `VoiceCloner`: no torch.

The graphs are our own export of Kyutai's MIT `pocket_tts` code (packaging/pocket/export.py) and hold no weights
(pocket_onnx/, graphs.json lists what each one needs); the weights are the model's own safetensors (Kyutai's
per language, or a community one with the same shapes such as Shtorm-PocketTTS-RU), read here with numpy and handed to
onnxruntime when the sessions are made. `model_dir` is a registry folder as downloaded: the weights (weights_file),
`tokenizer.model` (SentencePiece), maybe a `config.yaml` (settings).

A voice is a profile folder (voice.py): the first REF_S seconds of its recording pass the model once and the result,
the transformer's KV cache after the voice prompt, is cached next to the recording as `pocket-<model id>.npz`: it
depends on the model's weights, and on the recording (its SHA-256 is in the file: a new recording conditions again).

    python -m peresmeshnik.local.pocket --voice ~/.local/share/Peresmeshnik/voices/me --model-dir DIR --lang en "Hello"
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import struct
import zipfile
from collections.abc import Iterator
from pathlib import Path

import numpy as np

from peresmeshnik import voice as profiles
from peresmeshnik.engines import Voice
from peresmeshnik.local.pocket_text import prepare

GRAPHS = Path(__file__).with_name('pocket_onnx')
SR = 24_000
STATE_V = 2  # how the state is made (2: polyphase resampling): an older cache file is made again
REF_S = 10.0  # longer samples do not sound more alike, and at ~40 s the model breaks down (RESEARCH-clone 3.3)
TEMP, EOS_LOGIT, MIN_FRAMES, MAX_TOKENS = 0.3, -4.0, 6, 50  # pocket_tts defaults; EOS is not trusted before MIN_FRAMES
FADE_IN = SR // 200  # a fresh decoder state steps at its first samples (a click): 5 ms fade, as pocket_tts
# Characters absent from a language's training text, as its pocket_tts config rewrites them (the Kyutai rows have no
# config of their own); ';' reads as ',' in SEMICOLONS. A config.yaml next to the weights (community models) can set
# TEMP and EOS_LOGIT: settings()
_QUOTES = {c: '' for c in '"“”„«»()[]'} | {'’': "'", '‘': "'", '–': ','}
REPLACE = {'en': {'’': "'", '‘': "'"}, 'fr': _QUOTES | {':': ','}, 'de': _QUOTES | {'—': ','},
           'es': _QUOTES | {'¡': '', '¿': ''}, 'pt': _QUOTES, 'it': _QUOTES}
SEMICOLONS = {'fr', 'de'}
DTYPES = {'F32': '<f4', 'F16': '<f2', 'BF16': '<u2', 'F64': '<f8', 'I64': '<i8', 'I32': '<i4'}


def read_safetensors(path: Path | str) -> dict[str, np.ndarray]:
    """A safetensors file as numpy arrays (float ones as float32: bf16 widened by a shift), without the package:
    8 bytes of header length, the JSON header, the raw little-endian tensors. Each array owns its memory."""
    out = {}
    with open(path, 'rb') as f:
        n = struct.unpack('<Q', f.read(8))[0]
        header = json.loads(f.read(n))
        header.pop('__metadata__', None)
        for name, h in sorted(header.items(), key=lambda kv: kv[1]['data_offsets'][0]):
            if h['dtype'] not in DTYPES:
                raise ValueError(f'{path}: {name} is {h["dtype"]}, not a dtype this reader knows')
            b, e = h['data_offsets']
            dt = np.dtype(DTYPES[h['dtype']])
            f.seek(8 + n + b)
            x = np.fromfile(f, dt, (e - b) // dt.itemsize).reshape(h['shape'])
            if h['dtype'] == 'BF16':  # in place in the float32 result: no temporaries
                y = np.empty(x.shape, np.float32)
                y.view(np.uint32)[...] = x
                y.view(np.uint32)[...] <<= 16
                x = y
            elif dt.kind == 'f' and dt != np.float32:
                x = x.astype(np.float32)
            out[name] = x
    return out


def quantize(w: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """A Linear weight [out, in] as int8 [in, out] and a scale per output channel, symmetric: w ~ q.T * scale."""
    scale = np.abs(w).max(axis=1) / 127
    scale[scale == 0] = 1
    q = np.clip(np.rint(w / scale[:, None]), -127, 127).astype(np.int8)
    return np.ascontiguousarray(q.T), scale.astype(np.float32)


class Graphs:
    """The onnxruntime sessions of one model: FlowLM step (int8: its transformer's Linear layers in int8, as
    pocket_tts' quantize=True) and Mimi decoder; the Mimi encoder only while a voice is conditioned."""

    def __init__(self, weights: dict[str, np.ndarray], threads: int = 2, graphs: Path = GRAPHS,
                 int8: bool = False):
        import onnxruntime as ort
        self._ort, self._threads, self._dir, self._w = ort, threads, Path(graphs), weights
        self._q: dict = {}
        self.meta = json.loads((self._dir / 'graphs.json').read_text(encoding='utf-8'))
        self._flow, self._dec = self._session('flow_lm_int8' if int8 else 'flow_lm'), self._session('mimi_decoder')
        self._w = {n.removesuffix('.T'): weights[n.removesuffix('.T')] for n in self.meta['graphs']['mimi_encoder']}

    def _array(self, name: str) -> np.ndarray:
        """A graph's weight: `<name>.T` transposed (a Linear on 3-D input), `<name>.int8` / `.int8_scale` quantized
        (packaging/pocket/export.py)."""
        if name.endswith('.T') and name not in self._w:
            return np.ascontiguousarray(self._w[name[:-2]].T)
        if name.endswith(('.int8', '.int8_scale')):
            base, kind = name.rsplit('.', 1)
            if base not in self._q:  # both halves of one quantize(), done once
                self._q = {base: quantize(self._w[base])}
            return self._q[base][kind == 'int8_scale']
        return np.ascontiguousarray(self._w[name])

    def _session(self, name: str):
        ort, shapes, arrays = self._ort, self.meta['graphs'][name], []
        for n, s in shapes.items():
            try:
                arrays.append(self._array(n))
            except KeyError:
                raise ValueError(f'not a Pocket TTS model: no weight {n.removesuffix(".T")}') from None
            if list(arrays[-1].shape) != s:
                raise ValueError(f'not a Pocket TTS model of these shapes: {n} is {list(arrays[-1].shape)}, '
                                 f'the graph wants {s}')
        so = ort.SessionOptions()
        so.intra_op_num_threads, so.inter_op_num_threads = self._threads, 1
        so.enable_cpu_mem_arena = False  # the arena keeps the peak of every call (the voice prompt: +0.6 GB); as fast
        so.add_session_config_entry('session.intra_op.allow_spinning', '0')
        so.add_session_config_entry('session.inter_op.allow_spinning', '0')
        # onnxruntime copies them into the session (its C API: "TensorProto can't refer to the user provided buffers"),
        # so they are dropped here: 0.53 GB instead of 0.87 after loading (C, 1.31.1; audio identical with them freed)
        so.add_external_initializers(list(shapes), [ort.OrtValue.ortvalue_from_numpy(a) for a in arrays])
        return ort.InferenceSession(str(self._dir / f'{name}.onnx'), so, providers=['CPUExecutionProvider'])

    def _ov(self, x):
        ort = self._ort
        return x if isinstance(x, ort.OrtValue) else ort.OrtValue.ortvalue_from_numpy(np.ascontiguousarray(x))

    def empty_cache(self) -> np.ndarray:
        m = self.meta
        return np.zeros((m['layers'], 2, m['heads'], 0, m['head_dim']), np.float32)

    def flow(self, x: np.ndarray, cache, noise: np.ndarray):
        """FlowLM over input embeddings x [1, T, dim] after `cache` (the KV cache: numpy or OrtValue): the next
        latent [1, ldim] for this noise, the EOS logit, the new cache (an OrtValue: it stays in onnxruntime)."""
        cache = self._ov(cache)
        feeds = {'x': self._ov(x.astype(np.float32, copy=False)), 'cache': cache,
                 'offset': self._ov(np.array(cache.shape()[3], np.int64)),
                 'noise': self._ov(noise.astype(np.float32, copy=False))}
        latent, eos, out = self._flow.run_with_ort_values(['latent', 'eos', 'cache_out'], feeds)
        return latent.numpy(), float(eos.numpy().reshape(-1)[0]), out

    def decoder_state(self) -> tuple[int, list[np.ndarray]]:
        return 0, [np.zeros(s, np.float32) for s in self.meta['decoder_states']]

    def decode(self, latent: np.ndarray, state: tuple[int, list]) -> tuple[np.ndarray, tuple[int, list]]:
        """Mimi latents [1, n, ldim] (denormalized) to n * frame samples, streaming: `state` from decoder_state()."""
        offset, states = state
        feeds = {'latent': self._ov(latent.astype(np.float32, copy=False)),
                 'offset': self._ov(np.array(offset, np.int64)),
                 **{f'state{i}': self._ov(s) for i, s in enumerate(states)}}
        audio, *states = self._dec.run_with_ort_values(None, feeds)
        return audio.numpy().reshape(-1), (offset + latent.shape[1] * self.meta['decoder_steps_per_latent'], states)

    def encode(self, audio: np.ndarray) -> np.ndarray:
        """24 kHz audio [1, 1, k * frame] to the voice conditioning [1, k, dim] (a session for this call only)."""
        return self._session('mimi_encoder').run(None, {'audio': audio.astype(np.float32, copy=False)})[0]


def resample(x: np.ndarray, sr: int, to: int = SR) -> np.ndarray:
    """scipy.signal.resample_poly(x, up, down) in numpy (Kaiser window, beta 5): the voice sample as pocket_tts reads
    it. Linear interpolation instead cost the clone ~0.05 of speaker similarity (its state off by 8 %)."""
    g = math.gcd(sr, to)
    up, down = to // g, sr // g
    if up == down:
        return np.asarray(x, np.float32)
    rate = max(up, down)
    half = 10 * rate
    m = np.arange(-half, half + 1)
    h = np.sinc(m / rate) * np.kaiser(2 * half + 1, 5.0)
    pre = down - half % down
    h = np.concatenate([np.zeros(pre), h / h.sum() * up])
    taps = np.pad(h, (0, -len(h) % up)).reshape(-1, up).T  # polyphase: taps[p, r] = h[p + r * up]
    xs = np.append(np.asarray(x, np.float64), 0.0)  # index len(x): the zeros around the signal
    n_out, skip, out = -(-len(x) * up // down), (half + pre) // down, []
    for i in range(0, n_out, 32768):
        t = (np.arange(i, min(i + 32768, n_out)) + skip) * down
        idx = t[:, None] // up - np.arange(taps.shape[1])
        idx[(idx < 0) | (idx >= len(x))] = len(x)
        out.append((taps[t % up] * xs[idx]).sum(axis=1))
    return np.concatenate(out).astype(np.float32)


def end_on_pause(x: np.ndarray, sr: int = SR, pause_s: float = 0.08, fade_s: float = 0.02,
                 floor_db: float = 35.0) -> np.ndarray:
    """A voice prompt ending on exactly `pause_s` of silence (pocket_tts: a prompt that stops on speech makes the model
    go on with it): trailing 20 ms frames `floor_db` below the loudest are cut, the end faded out, zeros appended."""
    frame = int(0.02 * sr)
    n = len(x) // frame
    if not n:
        return x
    db = 20 * np.log10(np.sqrt(np.mean(np.square(x[:n * frame].reshape(n, frame)), axis=1)) + 1e-12)
    end = (int(np.nonzero(db > db.max() - floor_db)[0][-1]) + 1) * frame
    x = x[:end].copy()
    fade = min(int(fade_s * sr), end)
    x[end - fade:] *= np.linspace(1, 0, fade, dtype=np.float32)
    return np.concatenate([x, np.zeros(int(pause_s * sr), np.float32)])


def prompt(text: str, lang: str = 'en') -> tuple[str, int]:
    """The text as the model was trained on it (pocket_tts' prepare_text_prompt) and the frames to say after EOS."""
    text = ' '.join(text.split())
    if lang in REPLACE:
        text = re.sub(r'([.!?…])\s*[,;:]', r'\1', ' '.join(text.translate(str.maketrans(REPLACE[lang])).split()))
    if not text:
        raise ValueError('nothing to say')
    if lang in SEMICOLONS:
        text = text.replace(';', ',')
    after = (3 if len(text.split()) <= 4 else 1) + 2
    text = text[0].upper() + text[1:]
    core = text.rstrip('"\'”’)]» ')
    if core and core[-1] not in '.!?…':
        text = core.rstrip(',;:-–— ') + '.' + text[len(core):].strip() if core[-1] in ',;:-–—' else text + '.'
    return text, after


def weights_file(model_dir: Path) -> Path:
    """`model.safetensors` (Kyutai's rows), else `model_fast.safetensors` (Shtorm-ru), else the folder's only one."""
    for name in ('model.safetensors', 'model_fast.safetensors'):
        if (model_dir / name).exists():
            return model_dir / name
    found = sorted(model_dir.glob('*.safetensors'))
    if len(found) != 1:
        raise FileNotFoundError(f'{model_dir}: no Pocket TTS weights (*.safetensors: {len(found)})')
    return found[0]


def registry_sha256(model_id: str, name: str) -> str:
    """The sha256 of the weights file `name` in the registry row `model_id` (checked on download, not re-hashed here);
    '' when the registry has no such row or file (a folder given by hand, tests)."""
    from peresmeshnik import models
    try:
        return models.get(model_id).files[name]['sha256']
    except (models.ModelMissing, KeyError):
        return ''


def settings(model_dir: Path) -> tuple[float, float]:
    """(temperature, EOS logit threshold): pocket_tts' defaults, or a config.yaml's top-level default_temperature /
    default_eos_threshold. ponytail: those two keys by a regex, no YAML parser (not an app dependency)."""
    found = {}
    if (cfg := model_dir / 'config.yaml').exists():
        found = dict(re.findall(r'(?m)^(default_temperature|default_eos_threshold):\s*(-?[\d.]+)\s*(?:#.*)?$',
                                cfg.read_text(encoding='utf-8')))
    return float(found.get('default_temperature', TEMP)), float(found.get('default_eos_threshold', EOS_LOGIT))


class PocketTts:
    """Pocket TTS as the clone engine: speaks only in an enrolled voice."""
    sample_rate = SR

    def __init__(self, model_dir: Path | str, threads: int = 2, model_id: str = '', int8: bool = False):
        import sentencepiece
        model_dir = Path(model_dir)
        wfile = weights_file(model_dir)
        w = read_safetensors(wfile)
        self._weights_sha = registry_sha256(model_id, wfile.name)  # new weights of the same model: a new voice state
        self.temp, self.eos_logit = settings(model_dir)
        self.model_id = (model_id or model_dir.name) + ('-int8' if int8 else '')  # the voice state differs a little
        self.graphs = Graphs(w, threads, int8=int8)
        p = 'flow_lm.'
        self._embed, self._w_in = w[p + 'conditioner.embed.weight'], w[p + 'input_linear.weight']
        self._bos, self._bos_voice = w[p + 'bos_emb'], w[p + 'bos_before_voice'].reshape(1, 1, -1)
        self._std, self._mean = w[p + 'emb_std'], w[p + 'emb_mean']
        self._sp = sentencepiece.SentencePieceProcessor(model_file=str(model_dir / 'tokenizer.model'))
        self._ends = self._sp.encode('.!...?')[1:]  # sentence-end tokens (the first is the word boundary)
        self._commas = self._sp.encode(',;:')[1:]
        self._states: dict[tuple, np.ndarray] = {}
        self._rng = np.random.default_rng()

    # ---- voices
    def state(self, voice) -> np.ndarray:
        """The voice's state (the KV cache after its prompt): in memory, else `pocket-<model id>.npz` in the profile if
        it was made from this recording, else computed now and written there."""
        folder = Path(getattr(voice, 'path', voice))
        ref = folder / 'reference.wav'
        st = ref.stat()
        key = (str(folder), st.st_ino, st.st_mtime_ns, st.st_size)
        if key not in self._states:
            digest = f'{STATE_V}:{hashlib.sha256(ref.read_bytes()).hexdigest()}'
            if self._weights_sha:
                digest += f':{self._weights_sha}'
            cache = folder / f'pocket-{self.model_id}.npz'
            state = None
            try:
                with np.load(cache) as z:
                    state = z['state'] if str(z['ref']) == digest else None
            except (OSError, ValueError, KeyError, EOFError, zipfile.BadZipFile):  # a broken or cut file: made again
                pass
            if state is None:
                state = self.condition(*profiles.read_audio(ref))
                tmp = cache.with_name(cache.name + '.tmp')
                with open(tmp, 'wb') as f:
                    np.savez(f, state=state, ref=np.array(digest))
                os.replace(tmp, cache)
            self.forget(folder)
            self._states[key] = state
        return self._states[key]

    def forget(self, voice) -> None:
        """Drop a profile's state from memory (a new recording of it: the file is checked against the recording)."""
        folder = str(Path(getattr(voice, 'path', voice)))
        for k in [k for k in self._states if k[0] == folder]:
            del self._states[k]

    def condition(self, pcm: np.ndarray, sr: int) -> np.ndarray:
        """The model's state after the voice prompt: the first REF_S s of `pcm` at 24 kHz, ending on a pause."""
        x = np.asarray(pcm, np.float32)
        x = (x.mean(axis=1) if x.ndim > 1 else x)[:int(REF_S * sr)]
        x = end_on_pause(resample(x, sr))
        frame = self.graphs.meta['frame']
        x = np.pad(x, (0, -len(x) % frame))
        cond = np.concatenate([self._bos_voice, self.graphs.encode(x[None, None])], axis=1)
        _, _, cache = self.graphs.flow(cond, self.graphs.empty_cache(), np.zeros((1, len(self._bos)), np.float32))
        return cache.numpy()

    def enroll(self, pcm: np.ndarray, sample_rate: int, transcript: str | None = None, name: str = 'me',
               lang: str = 'ru', root: Path | None = None) -> Voice:
        v = profiles.save(pcm, sample_rate, name, lang, transcript, root)
        self.forget(v)
        self.state(v)
        return Voice(name=v.name, engine='pocket', path=v.path, meta=v.meta)

    # ---- speech
    def split(self, text: str, lang: str = 'en') -> list[str]:
        """Pieces of at most MAX_TOKENS tokens at sentence ends, a long sentence at commas (pocket_tts' split)."""
        piece = self._sp.id_to_piece

        def decimal(tokens: list[int], i: int) -> bool:  # 3.5: not a sentence end
            return (i > 1 and piece(tokens[i - 1]) == '.' and piece(tokens[i - 2])[-1:].isdigit()
                    and piece(tokens[i])[:1].isdigit())

        def cut(tokens: list[int], marks: list[int]) -> list[list[int]]:
            parts, prev = [[]], False
            for i, t in enumerate(tokens):
                if prev and t not in marks and not (marks is self._ends and decimal(tokens, i)):
                    parts.append([])
                parts[-1].append(t)
                prev = t in marks
            return [p for p in parts if p]
        pieces = []
        for s in cut(self._sp.encode(prompt(text, lang)[0]), self._ends):
            subs = cut(self._sp.encode(self._sp.decode(s).strip()), self._commas) if len(s) > MAX_TOKENS else [s]
            pieces += [(len(p), self._sp.decode(p)) for p in (subs if len(subs) > 1 else [s])]
        chunks: list[list] = []
        for n, t in pieces:
            if chunks and chunks[-1][0] + n <= MAX_TOKENS:
                chunks[-1] = [chunks[-1][0] + n, chunks[-1][1] + ' ' + t]
            else:
                chunks.append([n, t])
        return [t.strip() for _, t in chunks]

    def synth(self, text: str, lang: str, voice: Voice | None = None) -> Iterator[np.ndarray]:
        if voice is None:
            raise ValueError('the clone engine speaks only in an enrolled voice')
        text = prepare(text, lang)
        if not re.search(r'\w', text):
            return
        state = self.graphs._ov(self.state(voice))
        for part in self.split(text, lang):
            yield from self._say(part, state, lang)

    def _say(self, text: str, state, lang: str = 'en') -> Iterator[np.ndarray]:
        """One piece: the text after the voice, then a latent per 80 ms frame until EOS (+ a few frames), each decoded
        as soon as it is made."""
        text, after = prompt(text, lang)
        tokens = self._sp.encode(text)
        g, std = self.graphs, math.sqrt(self.temp)
        _, _, cache = g.flow(self._embed[tokens][None], state, np.zeros((1, len(self._bos)), np.float32))
        dec, latent, eos_at = g.decoder_state(), self._bos, None
        for i in range(math.ceil((len(tokens) / 3 + 2) * 12.5)):  # pocket_tts' length cap: 3 tokens a second + 2 s
            noise = self._rng.standard_normal((1, len(latent)), np.float32) * std
            out, eos, cache = g.flow((latent @ self._w_in.T)[None, None], cache, noise)
            if eos_at is None and eos > self.eos_logit and i >= MIN_FRAMES:
                eos_at = i
            if eos_at is not None and i >= eos_at + after:
                break
            latent = out[0]
            audio, dec = g.decode((latent * self._std + self._mean)[None, None], dec)
            if not i:
                audio[:FADE_IN] *= np.linspace(0, 1, FADE_IN, dtype=np.float32)
            yield audio

    def close(self) -> None:
        self.graphs = None
        self._states.clear()


def main(argv=None) -> None:
    import argparse
    import time

    import soundfile as sf
    ap = argparse.ArgumentParser(prog='python -m peresmeshnik.local.pocket',
                                 description='Say a text in an enrolled voice.')
    ap.add_argument('text')
    ap.add_argument('--voice', type=Path, required=True, help='voice profile folder')
    ap.add_argument('--lang', default='en')
    ap.add_argument('--out', type=Path, default=Path('pocket.wav'))
    ap.add_argument('--model-dir', type=Path, required=True, help='weights (*.safetensors) + tokenizer.model')
    ap.add_argument('--threads', type=int, default=2)
    ap.add_argument('--int8', action='store_true', help="the FlowLM transformer in int8: faster, smaller")
    a = ap.parse_args(argv)
    eng = PocketTts(a.model_dir, a.threads, int8=a.int8)
    v = profiles.load(a.voice.name, a.voice.parent)
    eng.state(v)
    t0, chunks = time.perf_counter(), []
    for c in eng.synth(a.text, a.lang, v):
        if not chunks:
            print(f'first audio {time.perf_counter() - t0:.2f} s', flush=True)
        chunks.append(c)
    wav = np.concatenate(chunks)
    dt = time.perf_counter() - t0
    sf.write(a.out, wav, SR)
    print(f'{len(wav) / SR:.2f} s of audio in {dt:.2f} s (RTF {dt * SR / len(wav):.2f}) -> {a.out}')


if __name__ == '__main__':
    main()
