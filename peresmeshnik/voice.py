"""Voice enrollment: a 30-60 s recording of the user's own voice -> a voice profile on disk.

    python -m peresmeshnik.voice enroll --record 45 --lang ru     # read the prompted text aloud
    python -m peresmeshnik.voice enroll --import me.wav --lang ru
    python -m peresmeshnik.voice check me.wav
    python -m peresmeshnik.voice list

A profile is a folder `<voices>/<name>/` with `reference.wav` (level-normalised) and `voice.json`
(language, transcript, quality numbers). Clone engines add their own cache next to it (XTTS:
`xtts.pt`). Profiles stay local and are never uploaded. This module is torch-free; the clone
engine is imported only for `--engine`.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from peresmeshnik.engines import Voice

MIN_S, MAX_S = 30.0, 60.0  # recording length
MIN_SPEECH_S = 15.0  # speech inside it (pauses and silence excluded)
MIN_SR = 16_000
CLIP = 0.99  # |sample| at or above counts as clipped
MAX_CLIPPED = 0.001  # share of clipped samples
QUIET_DB, LOW_DB = -45.0, -35.0  # speech level, dBFS: fail below QUIET_DB, warn below LOW_DB
NOISY_DB, FAIR_DB = 12.0, 20.0  # speech above the background, dB: fail / warn below (a LibriVox reader: 16)
TARGET_DB = -20.0  # saved reference: speech level, peak kept under -1 dBFS

PROMPTS = {  # read aloud for about 45 s; varied sounds, natural pauses at punctuation
    'ru': 'Привет! Я записываю этот короткий текст, чтобы приложение узнало, как звучит мой голос. '
          'Неделя выдалась насыщенной. В понедельник шёл дождь, и весь день пришлось работать дома. '
          'Во вторник мы встретились с друзьями, пили чай с вареньем и спорили о кино. '
          'В среду мы съездили за город: там тихо, пахнет хвоей, а у озера щебечут птицы. '
          'Четверг и пятница пролетели незаметно — письма, звонки, короткие встречи. '
          'А в выходные наконец-то удалось выспаться, потом был вкусный ужин и долгий разговор с родителями. Спасибо!',
    'en': 'Hello! I am recording this short passage so the app can learn how my voice sounds. '
          'Last week was busy but pleasant. On Monday it rained, so I worked from home all day. '
          'On Tuesday I met a few friends for coffee, and we argued about films for hours. '
          'On Wednesday I drove out of town: it was quiet, the air smelled of pine, and birds sang by the lake. '
          'Thursday and Friday flew by with emails, phone calls and quick meetings. '
          'At the weekend I finally slept in, cooked a proper dinner and called my parents. Thanks for listening!',
}


@dataclass
class Report:
    seconds: float
    speech_s: float
    level_db: float
    noise_db: float
    snr_db: float
    clipped: float
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors


def _frames_db(x: np.ndarray, sr: int, ms: int = 30) -> np.ndarray:
    n = sr * ms // 1000
    f = x[:len(x) // n * n].reshape(-1, n)
    return 10 * np.log10(np.mean(f.astype(np.float64) ** 2, axis=1) + 1e-12)


def check(pcm: np.ndarray, sr: int) -> Report:
    """Quality of an enrollment recording: length, speech time, level, clipping, noise."""
    x = np.asarray(pcm, dtype=np.float32)
    if x.ndim > 1:
        x = x.mean(axis=1)
    db = _frames_db(x, sr) if sr >= 100 and len(x) >= sr // 10 else np.array([-120.0])
    live = db[db > -100]  # digital silence (gaps, a muted mic) is no noise floor
    noise = float(np.percentile(live, 10)) if len(live) else -120.0  # pauses between phrases
    speech = db[db > max(noise + 6, -60)]
    level = float(10 * np.log10(np.mean(10 ** (speech / 10)))) if len(speech) else -120.0
    r = Report(seconds=round(len(x) / sr, 2), speech_s=round(len(speech) * 0.03, 2), level_db=round(level, 1),
               noise_db=round(noise, 1), snr_db=round(level - noise, 1),
               clipped=round(float(np.mean(np.abs(x) >= CLIP)) if len(x) else 0.0, 5))
    if sr < MIN_SR:
        r.errors.append(f'sample rate {sr} Hz is below {MIN_SR} Hz')
    if r.seconds < MIN_S:
        r.errors.append(f'{r.seconds:.0f} s recorded, need {MIN_S:.0f}-{MAX_S:.0f} s')
    elif r.seconds > MAX_S:
        r.warnings.append(f'{r.seconds:.0f} s recorded, only the first {MAX_S:.0f} s are used')
    if r.speech_s < MIN_SPEECH_S:
        r.errors.append(f'only {r.speech_s:.0f} s of speech, need at least {MIN_SPEECH_S:.0f} s: talk more, pause less')
    if r.clipped > MAX_CLIPPED:
        r.errors.append(f'clipping in {r.clipped:.2%} of samples: lower the mic gain or move back a little')
    if level < QUIET_DB:
        r.errors.append(f'speech level {level:.0f} dBFS is too quiet: raise the mic gain or come closer')
    elif level < LOW_DB:
        r.warnings.append(f'speech level {level:.0f} dBFS is low')
    if r.snr_db < NOISY_DB:
        r.errors.append(f'speech is only {r.snr_db:.0f} dB above the background: find a quieter place')
    elif r.snr_db < FAIR_DB:
        r.warnings.append(f'background noise is noticeable ({r.snr_db:.0f} dB below speech)')
    return r


def voices_dir() -> Path:
    from peresmeshnik.config import data_dir
    return data_dir() / 'voices'


def save(pcm: np.ndarray, sr: int, name: str, lang: str, transcript: str | None = None,
         root: Path | None = None) -> Voice:
    """Write `<root>/<name>/reference.wav` + `voice.json`; returns the profile (engine 'reference')."""
    import soundfile as sf
    x = np.asarray(pcm, dtype=np.float32)
    x = (x.mean(axis=1) if x.ndim > 1 else x)[:int(MAX_S * sr)]
    report = check(x, sr)  # of what is kept: at most MAX_S
    peak = float(np.max(np.abs(x))) or 1.0
    gain = min(10 ** ((TARGET_DB - report.level_db) / 20), 10 ** (-1 / 20) / peak)
    path = (root or voices_dir()) / name
    path.mkdir(parents=True, exist_ok=True)
    for old in path.glob('*.pt'):  # engine caches of an older recording
        old.unlink()
    sf.write(path / 'reference.wav', np.clip(x * gain, -1, 1), sr, subtype='PCM_16')
    meta = {'name': name, 'lang': lang, 'transcript': transcript or '', 'sample_rate': sr,
            'created': time.strftime('%Y-%m-%dT%H:%M:%S%z'), 'gain_db': round(20 * np.log10(gain), 1),
            'quality': {k: v for k, v in asdict(report).items() if k not in ('errors', 'warnings')}}
    (path / 'voice.json').write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding='utf-8')
    return Voice(name=name, engine='reference', path=path, meta=meta)


def load(name: str, root: Path | None = None) -> Voice:
    path = (root or voices_dir()) / name
    meta = json.loads((path / 'voice.json').read_text(encoding='utf-8'))
    return Voice(name=name, engine='reference', path=path, meta=meta)


def read_audio(path: Path) -> tuple[np.ndarray, int]:
    import soundfile as sf
    x, sr = sf.read(str(path), dtype='float32', always_2d=True)
    return x.mean(axis=1), sr


def _print_report(r: Report) -> None:
    print(f'length {r.seconds:.1f} s, speech {r.speech_s:.1f} s, level {r.level_db:.1f} dBFS, '
          f'noise {r.noise_db:.1f} dBFS (speech {r.snr_db:.1f} dB above it), clipped {r.clipped:.3%}')
    for w in r.warnings:
        print('  warning:', w)
    for e in r.errors:
        print('  problem:', e)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog='python -m peresmeshnik.voice', description='Record or import your own voice (30-60 s).')
    sub = ap.add_subparsers(dest='cmd', required=True)
    en = sub.add_parser('enroll', help='make a voice profile from a recording')
    src = en.add_mutually_exclusive_group(required=True)
    src.add_argument('--import', dest='file', type=Path, help='an existing recording (wav, flac, ogg, mp3)')
    src.add_argument('--record', type=float, nargs='?', const=45.0, metavar='SECONDS', help='record from the microphone (default 45 s)')
    en.add_argument('--name', default='me')
    en.add_argument('--lang', default='ru', help='language spoken in the recording')
    en.add_argument('--transcript', help='what is said (text or @file); --record uses the prompt text')
    en.add_argument('--device', help='input device name or index (see python -m sounddevice)')
    en.add_argument('--force', action='store_true', help='keep the profile even if the checks fail')
    en.add_argument('--engine', choices=['xtts'], help='also prepare the clone engine now (needs the voice pack)')
    en.add_argument('--model-dir', type=Path, help='XTTS-v2 model folder (config.json, model.pth, vocab.json)')
    ck = sub.add_parser('check', help='check a recording without saving')
    ck.add_argument('file', type=Path)
    sub.add_parser('list', help='list voice profiles')
    for p in (en, sub.choices['list']):
        p.add_argument('--voices', type=Path, help=f'profiles folder (default {voices_dir()})')
    a = ap.parse_args(argv)
    if getattr(a, 'engine', None) and not a.model_dir:
        ap.error('--engine needs --model-dir')

    if a.cmd == 'list':
        root = a.voices or voices_dir()
        for p in sorted(root.glob('*/voice.json')):
            m = json.loads(p.read_text(encoding='utf-8'))
            print(f"{m['name']}\t{m['lang']}\t{m['quality']['seconds']:.0f} s\t{m['created']}\t{p.parent}")
        return 0
    if a.cmd == 'check':
        r = check(*read_audio(a.file))
        _print_report(r)
        return 0 if r.ok else 1

    transcript = a.transcript
    if transcript and transcript.startswith('@'):
        transcript = Path(transcript[1:]).read_text(encoding='utf-8').strip()
    if a.file:
        pcm, sr = read_audio(a.file)
    else:
        from peresmeshnik.audio.mic import record
        if not transcript and a.lang not in PROMPTS:
            ap.error(f'no prompt text in {a.lang!r} ({", ".join(PROMPTS)}): pass --transcript with what you will read')
        transcript = transcript or PROMPTS[a.lang]
        print(f'Read this aloud at your normal pace, in a quiet room ({a.record:.0f} s):\n\n{transcript}\n')
        for i in (3, 2, 1):
            print(f'{i}...', end=' ', flush=True)
            time.sleep(1)
        print('recording', flush=True)
        sr = 24_000  # above the 16 kHz pipeline rate: the clone engine uses 22.05 kHz
        device = int(a.device) if a.device and a.device.isdigit() else a.device
        pcm = record(a.record, device=device, sample_rate=sr)
    r = check(pcm, sr)
    _print_report(r)
    if not r.ok and not a.force:
        print('Not saved. Fix the problems above and try again (or --force).')
        return 1
    v = save(pcm, sr, a.name, a.lang, transcript, a.voices)
    print(f'saved voice profile {v.name!r}: {v.path}')
    if a.engine == 'xtts':
        from peresmeshnik.local.clone import XttsClone
        XttsClone(a.model_dir).latents(v)
        print('XTTS-v2 conditioning ready')
    return 0


if __name__ == '__main__':
    sys.exit(main())
