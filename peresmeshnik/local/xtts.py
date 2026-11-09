"""XTTS-v2 itself: model load, conditioning, streaming synth with its text and length guards, speaker similarity.

Imports only numpy, soundfile, torch and coqui-tts (TTS), never `peresmeshnik`: the voice pack's runner
(voice-runner/voice_runner.py) loads this file on its own Python, where the app's package is not importable;
`clone.XttsClone` uses the same code in-process. A voice is a profile folder (reference.wav, xtts.pt cache).
"""
from __future__ import annotations

import contextlib
import re
from collections.abc import Iterator
from pathlib import Path

import numpy as np

SR = 24_000
COND_SR, REF_S = 22_050, 30  # XTTS conditions on at most 30 s at 22.05 kHz
PAUSE_S, QUIET_DB = 0.4, -40.0  # XTTS sometimes stops for 1-2.5 s mid-sentence; longer pauses are cut to this
SLOW_CPS = 8.0  # spoken chars per second; real XTTS speech is 12-20 (BENCH): a safety net for runaway generation
FADE = SR // 100


def split_text(text: str, limit: int) -> list[str]:
    """Pieces of at most `limit` chars (XTTS's per-language limit), cut at sentence and clause ends."""
    pieces = []
    for p in re.split(r'(?<=[.!?;:,])\s+', ' '.join(text.split())):
        while len(p) > limit:  # no punctuation for too long: cut at a space
            cut = p.rfind(' ', 0, limit)
            cut = cut if cut > 0 else limit
            pieces.append(p[:cut])
            p = p[cut:].strip()
        pieces.append(p)
    parts: list[str] = []
    for p in pieces:
        if parts and len(parts[-1]) + 1 + len(p) <= limit:
            parts[-1] += ' ' + p
        else:
            parts.append(p)
    return [p for p in parts if re.search(r'\w', p)]


def spoken_len(text: str) -> int:
    """Text length as spoken: a digit reads as ~5 letters, an abbreviation letter by letter."""
    return len(re.sub(r'\d', 'xxxxx', re.sub(r'\b[A-ZА-ЯЁ]{2,}\b', lambda m: m.group() * 3, text)))


def budget_samples(text: str, sr: int = SR) -> int:
    """Length guard: XTTS sometimes keeps talking after the text (babble); cut past this."""
    return int((1.0 + spoken_len(text) / SLOW_CPS) * sr)


def cap_pauses(chunks: Iterator[np.ndarray], sr: int = SR, max_s: float = PAUSE_S) -> Iterator[np.ndarray]:
    """Shorten silences longer than `max_s` (20 ms frames below QUIET_DB) while the chunks stream by."""
    frame = sr // 50
    limit, run, rest = int(max_s * 50), 0, np.zeros(0, np.float32)
    for c in chunks:
        x = np.concatenate([rest, c])
        n = len(x) // frame * frame
        rest, f = x[n:], x[:n].reshape(-1, frame)
        keep = np.ones(len(f), bool)
        for i, quiet in enumerate(10 * np.log10(np.mean(f ** 2, axis=1) + 1e-10) < QUIET_DB):
            run = run + 1 if quiet else 0
            keep[i] = run <= limit
        if keep.any():
            yield f[keep].ravel()
    if len(rest):
        yield rest


def guard(chunks: Iterator[np.ndarray], limit: int) -> Iterator[np.ndarray]:
    """Pass chunks through until `limit` samples, then cut with a short fade."""
    for c in chunks:
        if len(c) >= limit:
            c = c[:limit].copy()
            n = min(FADE, len(c))
            c[len(c) - n:] *= np.linspace(1, 0, n, dtype=np.float32)
            yield c
            return
        limit -= len(c)
        yield c


def read_audio(path: Path) -> tuple[np.ndarray, int]:
    import soundfile as sf
    x, sr = sf.read(str(path), dtype='float32', always_2d=True)
    return x.mean(axis=1), sr


def folder(voice) -> Path:
    """A profile folder, or anything with `.path` to one (engines.Voice)."""
    return Path(getattr(voice, 'path', voice))


class Engine:
    sample_rate = SR

    def __init__(self, model_dir: Path | str, device: str | None = None, threads: int = 2,
                 temperature: float = 0.75, top_p: float = 0.85):
        import torch
        torch.set_num_threads(threads)
        with contextlib.suppress(RuntimeError):  # allowed once per process
            torch.set_num_interop_threads(1)
        from TTS.tts.configs.xtts_config import XttsConfig
        from TTS.tts.models.xtts import Xtts
        cfg = XttsConfig()
        cfg.load_json(str(Path(model_dir) / 'config.json'))
        self.model = Xtts.init_from_config(cfg)
        self.model.load_checkpoint(cfg, checkpoint_dir=str(model_dir), eval=True)
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.temperature, self.top_p = temperature, top_p
        self._latents: dict[str, tuple] = {}

    def latents(self, voice) -> tuple:
        """XTTS conditioning of a profile: cached in memory and as `xtts.pt` next to the recording."""
        key = str(folder(voice))
        if key not in self._latents:
            import torch
            f = folder(voice) / 'xtts.pt'
            if f.exists():
                lat = torch.load(f, map_location='cpu')
            else:
                lat = self._condition(*read_audio(folder(voice) / 'reference.wav'))
                torch.save(lat, f)
            self._latents[key] = (lat['gpt'].to(self.device), lat['spk'].to(self.device))
        return self._latents[key]

    def forget(self, voice) -> None:
        """A new recording of the same profile: condition again."""
        self._latents.pop(str(folder(voice)), None)

    def _audio(self, pcm: np.ndarray, sr: int):
        import torch
        import torchaudio.functional as F
        x = torch.from_numpy(np.asarray(pcm, dtype=np.float32))[None]
        return F.resample(x, sr, COND_SR)[:, :COND_SR * REF_S].to(self.device)

    def _condition(self, pcm: np.ndarray, sr: int) -> dict:
        x = self._audio(pcm, sr)
        gpt = self.model.get_gpt_cond_latents(x, COND_SR, length=6, chunk_length=6)
        spk = self.model.get_speaker_embedding(x, COND_SR)
        return {'gpt': gpt.cpu(), 'spk': spk.cpu()}

    def similarity(self, voice, pcm: np.ndarray, sr: int) -> float:
        """How much `pcm` sounds like the profile: cosine of XTTS's own speaker embeddings (its speaker encoder, no
        extra model or license). Thresholds and how it compares with an independent judge (ECAPA): BENCH."""
        import torch
        ref = self.latents(voice)[1].flatten().float()
        with torch.inference_mode():
            emb = self.model.get_speaker_embedding(self._audio(pcm, sr), COND_SR).flatten().float()
        return float(torch.nn.functional.cosine_similarity(ref, emb.to(ref.device), dim=0))

    def synth(self, text: str, lang: str, voice=None) -> Iterator[np.ndarray]:
        if voice is None:
            raise ValueError('the clone engine speaks only in an enrolled voice')
        gpt, spk = self.latents(voice)
        for part in split_text(text, self.model.tokenizer.char_limits.get(lang, 250)):
            gen = self.model.inference_stream(part, lang, gpt, spk, temperature=self.temperature, top_p=self.top_p,
                                              enable_text_splitting=False)
            try:
                yield from guard(cap_pauses(c.float().cpu().numpy() for c in gen), budget_samples(part))
            finally:
                gen.close()  # stops generation on a cut or when the caller stops listening (barge-in)


def cuda_info() -> dict:
    """What torch sees: the voice pack's runner reports it at start (and on `info`, with the peak memory so far)."""
    import torch
    info = {'torch': torch.__version__, 'cuda': torch.cuda.is_available()}
    if info['cuda']:
        free, total = torch.cuda.mem_get_info()
        gb = 2 ** 30
        info |= {'gpu': torch.cuda.get_device_name(), 'vram_total_gb': round(total / gb, 2), 'vram_free_gb': round(free / gb, 2),
                 'vram_peak_gb': round(torch.cuda.max_memory_allocated() / gb, 2),
                 'vram_reserved_peak_gb': round(torch.cuda.max_memory_reserved() / gb, 2)}
    return info


def cap_vram(gb: float) -> None:
    """At most `gb` of CUDA memory for this process (a bigger allocation fails as on a smaller card)."""
    import torch
    torch.cuda.set_per_process_memory_fraction(min(1.0, gb * 2 ** 30 / torch.cuda.mem_get_info()[1]))
