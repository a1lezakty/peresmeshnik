"""Qwen3-TTS 0.6B Base for the voice pack's runner: streaming clone from a voice vector, with Chatterbox Multilingual
taking over a voice Qwen3 ran away on.

Qwen3 (faster-qwen3-tts, CUDA graphs) streams the first sound in ~0.1 s and conditions on the speaker vector only
(x-vector of the first <= 10 s of the recording, cached as `qwen3.pt` next to it). On some foreign voices it does not
stop: up to 163 s of repetition instead of 5-6 s (RESEARCH-clone.md §10.4, §11.4). The length guard is XTTS's
(`xtts.budget_samples`): past it the stream is cut, Chatterbox Multilingual (chatterbox-tts, MIT, no streaming,
+2-3 s) says that clause again, and so it does every later clause of that voice for the rest of this engine's life.
Chatterbox loads only at the first runaway (VRAM: Qwen3 ~2.4 GB, Chatterbox +3.2 GB).

A voice is a profile folder, or anything with `state`: condition()'s Sample (the other person in a call, voices/live.py),
which Chatterbox takes over as it does a folder.

Imports numpy and xtts.py (its guards); torch, faster_qwen3_tts and chatterbox only inside. Same interface as
xtts.py (voice-runner/voice_runner.py --kind qwen3).
"""
from __future__ import annotations

import contextlib
import hashlib
import io
import itertools
import re
import sys
from collections.abc import Iterator
from pathlib import Path

import numpy as np

if __package__:  # peresmeshnik.local.qwen3
    from . import xtts
else:  # the runner loads this file on its own: xtts.py is beside it
    import importlib.util
    _spec = importlib.util.spec_from_file_location('xtts', Path(__file__).with_name('xtts.py'))
    xtts = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(xtts)

SR = xtts.SR  # Qwen3's codec and Chatterbox's S3Gen: 24 kHz both
REF_S = 10  # the speaker vector: from the first 10 s of the recording
CHUNK = 2  # codec steps (1/12 s each) per streamed chunk: the first sound in ~0.1 s (RESEARCH-clone.md §3.2)
CFG = 0.5  # Chatterbox cfg_weight: 0 (upstream's advice for a foreign voice) said worse Russian (§11.4)
LANGS = {'en': 'English', 'ru': 'Russian', 'de': 'German', 'fr': 'French', 'es': 'Spanish', 'it': 'Italian',
         'pt': 'Portuguese', 'zh': 'Chinese', 'ja': 'Japanese', 'ko': 'Korean'}
cuda_info, cap_vram, folder = xtts.cuda_info, xtts.cap_vram, xtts.folder


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class Sample:
    """condition()'s state: Qwen3's speaker vector of a sample, and the sample for Chatterbox should it take over (its
    conditionals then too). `key`: its name in `fallen` and the stderr log."""
    _n = itertools.count(1)

    def __init__(self, spk, pcm: np.ndarray, sr: int):
        self.spk, self.pcm, self.sr, self.conds = spk, pcm, sr, None
        self.key = f'sample {next(self._n)}'


def voice_key(voice) -> str:
    return st.key if (st := getattr(voice, 'state', None)) is not None else str(folder(voice))


class Engine:
    sample_rate = SR

    def __init__(self, model_dir: Path | str, device: str | None = None, threads: int = 2,
                 fallback_dir: Path | str | None = None):
        import torch
        torch.set_num_threads(threads)
        with contextlib.suppress(RuntimeError):  # allowed once per process
            torch.set_num_interop_threads(1)
        from faster_qwen3_tts import FasterQwen3TTS
        self.device = device or 'cuda'  # CUDA graphs: no CPU path
        self.model = FasterQwen3TTS.from_pretrained(str(model_dir), device=self.device)
        self.model.warmup()  # captures the CUDA graphs now, not in the first clause
        self.fallback_dir, self.fallback = fallback_dir, None  # Chatterbox: loaded at the first runaway
        self.fallen: set[str] = set()  # voices Chatterbox speaks from now on
        self.runaways = 0
        self._xvec: dict[str, object] = {}
        self._conds: dict[str, object] = {}  # Chatterbox's conditionals per voice

    def _vector(self, pcm: np.ndarray, sr: int):
        items = self.model.model.create_voice_clone_prompt(ref_audio=(np.asarray(pcm, np.float32), sr), ref_text='',
                                                           x_vector_only_mode=True)
        return items[0].ref_spk_embedding

    def latents(self, voice):
        """Qwen3's speaker vector of a profile: cached in memory and as `qwen3.pt` next to the recording, with the
        recording's SHA-256 (another recording: computed again). A voice with `state` brings its own."""
        if (st := getattr(voice, 'state', None)) is not None:
            return st.spk
        key = str(folder(voice))
        if key not in self._xvec:
            import torch
            ref, f = folder(voice) / 'reference.wav', folder(voice) / 'qwen3.pt'
            digest = sha256(ref)
            cached = torch.load(f, map_location='cpu', weights_only=True) if f.exists() else {}
            if cached.get('sha256') != digest:
                pcm, sr = xtts.read_audio(ref)
                cached = {'sha256': digest, 'spk': self._vector(pcm[:REF_S * sr], sr).cpu()}
                torch.save(cached, f.with_suffix('.tmp'))  # a write cut short leaves no broken qwen3.pt
                f.with_suffix('.tmp').replace(f)
            self._xvec[key] = cached['spk'].to(self.device)
        return self._xvec[key]

    def condition(self, pcm: np.ndarray, sr: int) -> Sample:
        """The state of a voice from a sample, no profile folder: its vector from the first REF_S s, as latents()'s."""
        pcm = np.asarray(pcm, np.float32)[:int(REF_S * sr)]
        return Sample(self._vector(pcm, sr), pcm, sr)

    def forget(self, voice) -> None:
        """A new recording of the same profile: condition again (a voice Chatterbox took over stays with it)."""
        self._xvec.pop(str(folder(voice)), None)
        self._conds.pop(str(folder(voice)), None)

    def similarity(self, voice, pcm: np.ndarray, sr: int) -> float:
        """Cosine of Qwen3's own speaker vectors (its speaker encoder). Not XTTS's scale: its thresholds do not apply."""
        a, b = (t.float().cpu().numpy().ravel() for t in (self.latents(voice), self._vector(pcm, sr)))
        return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))

    def info(self) -> dict:
        """The runner's `info` adds this: runaways so far and the voices Chatterbox speaks."""
        return {'runaways': self.runaways, 'fallback_voices': sorted(self.fallen), 'fallback_loaded': self.fallback is not None}

    def synth(self, text: str, lang: str, voice=None) -> Iterator[np.ndarray]:
        if voice is None:
            raise ValueError('the clone engine speaks only in an enrolled voice')
        if lang not in LANGS:
            raise ValueError(f'Qwen3-TTS does not speak {lang!r}')
        text = ' '.join(text.split())
        if not re.search(r'\w', text):
            return
        # not XTTS's cap_pauses: Qwen3 speaks as loud as the recording, and a quiet voice under its -40 dB was all cut (C)
        yield from xtts.shape(self._say(text, lang, voice), bool(xtts.SENTENCE_END.search(text)))

    def _say(self, text: str, lang: str, voice) -> Iterator[np.ndarray]:
        name, budget = voice_key(voice), xtts.budget_samples(text)
        if name not in self.fallen:
            gen = self.model.generate_voice_clone_streaming(text=text, language=LANGS[lang], chunk_size=CHUNK,
                                                            voice_clone_prompt={'ref_spk_embedding': [self.latents(voice)]})
            chunks = (np.asarray(c, np.float32) for c, _sr, _timing in gen)
            try:
                yield from xtts.guard(chunks, budget)
                runaway = next(chunks, None) is not None  # still talking at the budget (the guard cut it)
            finally:
                gen.close()  # stops generation on a cut or when the caller stops listening (barge-in)
            if not runaway:
                return
            self.runaways += 1
            print(f'qwen3: runaway in {name}: cut at {budget / SR:.1f} s ({len(text)} chars); '
                  + ('Chatterbox says this clause and the rest of this voice' if self.fallback_dir else 'no fallback'),
                  file=sys.stderr, flush=True)
            if not self.fallback_dir:
                return
            self.fallen.add(name)
        yield from xtts.guard(self._chatterbox(text, lang, voice), budget)

    def _chatterbox(self, text: str, lang: str, voice) -> Iterator[np.ndarray]:
        """The whole clause at once (no streaming upstream)."""
        if self.fallback is None:
            from chatterbox.mtl_tts import ChatterboxMultilingualTTS
            self.fallback = ChatterboxMultilingualTTS.from_local(Path(self.fallback_dir), self.device)
        if (st := getattr(voice, 'state', None)) is not None:  # a sample: its conditionals live with it
            if st.conds is None:
                import soundfile as sf
                wav = io.BytesIO()  # Chatterbox reads a file (librosa.load): this one in memory
                sf.write(wav, st.pcm, st.sr, format='WAV', subtype='FLOAT')
                wav.seek(0)
                self.fallback.prepare_conditionals(wav)
                st.conds = self.fallback.conds
            self.fallback.conds = st.conds
        else:
            name = str(folder(voice))
            if name not in self._conds:
                self.fallback.prepare_conditionals(str(folder(voice) / 'reference.wav'))
                self._conds[name] = self.fallback.conds
            self.fallback.conds = self._conds[name]
        yield self.fallback.generate(text, language_id=lang, cfg_weight=CFG).squeeze(0).cpu().numpy().astype(np.float32)
