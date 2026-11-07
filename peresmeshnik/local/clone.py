"""Voice-clone TTS for "me -> them": XTTS-v2 behind `Tts` and `VoiceCloner`.

Part of the optional voice pack (torch + coqui-tts; weights under the noncommercial Coqui Public
Model License, docs/MODELS.md): the base app never imports this module. On NVIDIA it streams faster
than real time (docs/BENCH.md); on CPU it is slower than real time, so the direction falls behind
and its queue policy has to drop clauses. Nothing here pretends otherwise.

    python -m peresmeshnik.local.clone --voice ~/.local/share/Peresmeshnik/voices/me --model-dir DIR --lang en "Hello"

`model_dir` holds the XTTS-v2 files (config.json, model.pth, vocab.json, speakers_xtts.pth): the
registry's `xtts-v2` folder (`peresmeshnik models fetch xtts-v2`).
"""
from __future__ import annotations

import contextlib
import re
from collections.abc import Iterator
from pathlib import Path

import numpy as np

from peresmeshnik import voice as profiles
from peresmeshnik.engines import Voice

SR = 24_000
COND_SR, REF_S = 22_050, 30  # XTTS conditions on at most 30 s at 22.05 kHz
PAUSE_S, QUIET_DB = 0.4, -40.0  # XTTS sometimes stops for 1-2.5 s mid-sentence; longer pauses are cut to this
SLOW_CPS = 11.0  # spoken chars per second: real XTTS speech is 12-20 (BENCH); beyond the budget it babbles
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


class XttsClone:
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

    def latents(self, voice: Voice) -> tuple:
        """XTTS conditioning of a profile: cached in memory and as `xtts.pt` next to the recording."""
        key = str(voice.path)
        if key not in self._latents:
            import torch
            f = Path(voice.path) / 'xtts.pt'
            if f.exists():
                lat = torch.load(f, map_location='cpu')
            else:
                lat = self._condition(*profiles.read_audio(Path(voice.path) / 'reference.wav'))
                torch.save(lat, f)
            self._latents[key] = (lat['gpt'].to(self.device), lat['spk'].to(self.device))
        return self._latents[key]

    def _condition(self, pcm: np.ndarray, sr: int) -> dict:
        import torch
        import torchaudio.functional as F
        x = torch.from_numpy(np.asarray(pcm, dtype=np.float32))[None]
        x = F.resample(x, sr, COND_SR)[:, :COND_SR * REF_S].to(self.device)
        gpt = self.model.get_gpt_cond_latents(x, COND_SR, length=6, chunk_length=6)
        spk = self.model.get_speaker_embedding(x, COND_SR)
        return {'gpt': gpt.cpu(), 'spk': spk.cpu()}

    def enroll(self, pcm: np.ndarray, sample_rate: int, transcript: str | None = None, name: str = 'me',
               lang: str = 'ru', root: Path | None = None) -> Voice:
        v = profiles.save(pcm, sample_rate, name, lang, transcript, root)
        self.latents(v)
        return Voice(name=v.name, engine='xtts', path=v.path, meta=v.meta)

    def synth(self, text: str, lang: str, voice: Voice | None = None) -> Iterator[np.ndarray]:
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


def main(argv=None) -> None:
    import argparse
    import time

    import soundfile as sf
    ap = argparse.ArgumentParser(prog='python -m peresmeshnik.local.clone', description='Say a text in an enrolled voice.')
    ap.add_argument('text')
    ap.add_argument('--voice', type=Path, required=True, help='voice profile folder')
    ap.add_argument('--lang', default='en')
    ap.add_argument('--out', type=Path, default=Path('clone.wav'))
    ap.add_argument('--model-dir', type=Path, required=True, help='XTTS-v2 folder')
    ap.add_argument('--device')
    ap.add_argument('--threads', type=int, default=2)
    a = ap.parse_args(argv)
    eng = XttsClone(a.model_dir, a.device, a.threads)
    v = profiles.load(a.voice.name, a.voice.parent)
    t0, chunks = time.perf_counter(), []
    for c in eng.synth(a.text, a.lang, v):
        if not chunks:
            print(f'first audio {time.perf_counter() - t0:.2f} s', flush=True)
        chunks.append(c)
    wav = np.concatenate(chunks)
    dt = time.perf_counter() - t0
    sf.write(a.out, wav, eng.sample_rate)
    print(f'{len(wav) / eng.sample_rate:.2f} s of audio in {dt:.2f} s (RTF {dt * eng.sample_rate / len(wav):.2f}) -> {a.out}')


if __name__ == '__main__':
    main()
