"""XTTS-v2 itself: model load, conditioning, streaming synth with its text and length guards, speaker similarity.

Imports only numpy, soundfile, torch and coqui-tts (TTS), never `peresmeshnik`: the voice pack's runner
(voice-runner/voice_runner.py) loads this file on its own Python, where the app's package is not importable;
`clone.XttsClone` uses the same code in-process. A voice is a profile folder (reference.wav, xtts.pt cache), or
anything with `state`: condition()'s, a voice from a sample (the other person in a call, voices/live.py).
"""
from __future__ import annotations

import contextlib
import hashlib
import re
from collections.abc import Iterator
from pathlib import Path

import numpy as np

SR = 24_000
COND_SR, REF_S = 22_050, 30  # XTTS conditions on at most 30 s at 22.05 kHz
PAUSE_S, QUIET_DB = 0.4, -40.0  # XTTS sometimes stops for 1-2.5 s mid-sentence; longer pauses are cut to this
SLOW_CPS = 8.0  # spoken chars per second; real XTTS speech is 12-20 (BENCH): a safety net for runaway generation
FADE = SR // 100
FADE_IN = SR // 200  # XTTS starts mid-waveform: half of its utterances stepped -33..-10 dB under their peak (a click, BENCH)
SENTENCE_PAUSE_S = 0.35  # silence after a sentence: queued clauses played back to back had 0.1-0.16 s between sentences
QUOTES = '["»”)」』]'  # closing quotes and brackets after the end of a sentence or clause
SENTENCE_END = re.compile(rf'[.!?…。！？؟]{QUOTES}*$')  # zh / ja / ar too (stage 6)
# Han, kana, Hangul syllables: one is a syllable or two, spoken 4-8 a second against 12-20 letters (BENCH stage 6)
CJK = '\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uac00-\ud7af\uf900-\ufaff'
CJK_LEN = 3  # spoken length of one of them in letters
WIDE_STOP = '。！？；：，、'  # zh / ja punctuation: no space after it

# What XTTS's own text cleaning (num2words in its tokenizer) misreads, per language (BENCH stage 6 "Numbers"):
MONEY = {  # it drops the sign or names another currency: the word after the number instead (pt: € $ £ with cents it
    # reads as euros and cents itself). Letters before a sign make another currency: R$ the Brazilian real, E£ the
    # Egyptian pound (ar-EG: £ too), US$ the dollar everywhere
    'pt': {'R$': 'reais', '€': 'euros', '$': 'dólares', '£': 'libras'}, 'tr': {'€': 'avro', '$': 'dolar', '£': 'sterlin'},
    'ar': {'E£': 'جنيه', '€': 'يورو', '$': 'دولار', '£': 'جنيه'}, 'ko': {'€': '유로', '$': '달러', '£': '파운드'},
    'ja': {'€': 'ユーロ', '$': 'ドル', '£': 'ポンド'}, 'zh': {'€': '欧元', '$': '美元', '£': '英镑'}}
CLOCK = {  # 15:30 and 15:00 as "<hours><unit><joiner><minutes><minutes unit>" and "<hours><unit>"; else "15 30", "15"
    'de': (' Uhr', ' ', ''), 'fr': (' heures', ' ', ''), 'pt': (' horas', ' e ', ''), 'it': ('', ' e ', ''),
    'ja': ('時', '', '分'), 'ko': ('시', ' ', '분'), 'zh': ('点', '', '分')}
HOURS = {'fr': {1: 'une heure', 21: 'vingt et une heures'},  # "heure" / "hora" is feminine: not "un", "um", "dois"
         'pt': {1: 'uma hora', 2: 'duas horas', 21: 'vinte e uma horas', 22: 'vinte e duas horas'}}
# a final period: XTTS goes on after it ("punto" in Italian, 13 of 54) in these; with a comma 1 tail in 120 (BENCH)
COMMA_END = {'it', 'es', 'de', 'fr', 'pt', 'ko'}
ABBR_END = re.compile(r'(?:\b\w\.){2,}$|\b(?:etc|ecc|usw|bzw)\.$', re.I)  # p.m. / etc.: the period is the abbreviation's
POINT = {'tr': ('.', ',', 'virgül'), 'ar': (',', '.', 'فاصلة')}  # thousands, decimal mark, its word: num2words reads 12,5 as 12,50


def normalize(text: str, lang: str) -> str:
    """The text as XTTS should read it: money, clock times, decimals in tr / ar, Japanese numbers (its tokenizer
    leaves digits as they are for ja). The rest of the numbers, abbreviations and symbols: XTTS's own cleaning. The final
    period of each part XTTS says: ending()."""
    t = re.sub(r'(\d)\s+([€$£])', r'\1\2', re.sub(r'\bUS\$\s?', '$', text))  # "250,50 €": XTTS reads a sign only beside the number
    if lang in MONEY:
        signs = '|'.join(re.escape(x) for x in sorted(MONEY[lang], key=len, reverse=True))
        t = re.sub(rf'(?<![^\W\d_])({signs})\s?(\d[\d.,]*\d|\d)|(\d[\d.,]*\d|\d)({signs})', lambda m: _money(m, lang), t)
    if lang == 'de':  # 9.30 Uhr
        t = re.sub(r'(?<![\d.])(\d{1,2})\.(\d{2})(?=\s*Uhr\b)', r'\1:\2', t)
    sep = {'fr': '[:h]', 'pt': '[:h]', 'tr': '[:.]'}.get(lang, ':')  # tr 15.30, fr / pt 15h30 and 9h
    bare, after = r'|h(?!\w)' if lang in ('fr', 'pt') else '', r'(?:\s*Uhr\b)?' if lang == 'de' else ''
    t = re.sub(rf'(?<![\d:])(?<!\d[.,])(\d{{1,2}})(?:{sep}(\d{{2}}){bare})(?![\d:]|[.,]\d){after}', lambda m: _clock(m, lang), t)
    if lang in POINT or lang == 'ja':
        from num2words import num2words as n2w  # coqui-tts' own dependency: its tokenizer reads numbers with it

        def num2words(x, **kw):  # a number it cannot read (70+ digits: OverflowError) stays digits, the clause is said
            try:
                return n2w(x, **kw)
            except Exception:
                return str(x)
    if lang in POINT:
        thousands, mark, word = POINT[lang]
        t = re.sub(rf'(?<=\d)\{thousands}(?=\d{{3}}(?!\d))', '', t)  # 1.250.000
        t = re.sub(rf'(?<![\d.,])(\d+)\{mark}(\d+)(?![.,]?\d)', lambda m: _decimal(m, lang, word, num2words), t)  # not 2,3,4
    if lang == 'ja':
        t = re.sub(r'(?<=\d),(?=\d{3}(?!\d))', '', t).replace('%', 'パーセント').replace('％', 'パーセント')
        t = re.sub(r'\d+(?:\.\d+)?', lambda m: num2words(float(m[0]) if '.' in m[0] else int(m[0]), lang='ja'), t)
    return t


def _money(m: re.Match, lang: str) -> str:
    sign, amount = m[1] or m[4], m[2] or m[3]
    cents = re.fullmatch(r'(.+)[.,](\d\d)', amount) if lang == 'pt' else None
    if cents and len(sign) == 1:
        return m[0]  # 250,50€: XTTS reads euros and cents itself
    if cents:  # R$ 250,50
        return f'{cents[1]} {MONEY[lang][sign]}' + ('' if cents[2] == '00' else f' e {cents[2]} centavos')
    return f'{amount} {MONEY[lang][sign]}'


def _clock(m: re.Match, lang: str) -> str:
    minutes = m[2] or '00'
    if int(m[1]) > 24 or int(minutes) > 59:
        return m[0]
    unit, joiner, after = CLOCK.get(lang, ('', ' ', ''))
    hours = HOURS.get(lang, {}).get(int(m[1])) or m[1] + unit
    return hours if minutes == '00' else hours + joiner + minutes + after


def _decimal(m: re.Match, lang: str, word: str, num2words) -> str:
    """12,05 as "twelve point zero five": num2words alone reads its fraction 05 as five."""
    whole, frac = m[1], m[2]
    if not frac.strip('0'):
        return num2words(int(whole), lang=lang)
    zeros = [num2words(0, lang=lang)] * (len(frac) - len(frac.lstrip('0')))
    return ' '.join([num2words(int(whole), lang=lang), word, *zeros, num2words(int(frac), lang=lang)])


def ending(part: str, lang: str) -> str:
    """A part's final period as a comma where XTTS goes on after it (COMMA_END); an abbreviation's period stays."""
    part = part.rstrip()
    if lang not in COMMA_END or ABBR_END.search(re.sub(f'{QUOTES}+$', '', part)):
        return part
    return re.sub(rf'(?<!\.)\.({QUOTES}*)$', r',\1', part)


def split_text(text: str, limit: int) -> list[str]:
    """Pieces of at most `limit` chars (XTTS's per-language limit), cut at sentence and clause ends."""
    pieces = []
    for p in re.split(rf'(?<=[.!?;:,؟،؛])\s+|(?<=[{WIDE_STOP}])\s*', ' '.join(text.split())):
        while len(p) > limit:  # no punctuation for too long: cut at a space
            cut = p.rfind(' ', 0, limit)
            cut = cut if cut > 0 else limit
            pieces.append(p[:cut])
            p = p[cut:].strip()
        pieces.append(p)
    parts: list[str] = []
    for p in pieces:
        if parts and len(parts[-1]) + 1 + len(p) <= limit:
            parts[-1] += ('' if parts[-1][-1] in WIDE_STOP else ' ') + p
        else:
            parts.append(p)
    return [p for p in parts if re.search(r'\w', p)]


def closed(text: str) -> str:
    """A clause cut mid-sentence ("... opposed to the") ends with a comma for XTTS: without one it went on with a
    word or two of its own in 6 of 102 such clauses, with a comma in none (a period: 6; BENCH stage 5)."""
    text = ' '.join(text.split())
    return text if not text or re.search(rf'[.!?…,;:。！？，；：、؟،]{QUOTES}*$', text) else text + ','


def spoken_len(text: str) -> int:
    """Text length as spoken, in letters: a digit reads as ~5, an abbreviation letter by letter, a Han / kana / Hangul
    character as CJK_LEN."""
    t = re.sub(r'\d', 'xxxxx', re.sub(r'\b[A-ZА-ЯЁ]{2,}\b', lambda m: m.group() * 3, text))
    return len(t) + (CJK_LEN - 1) * len(re.findall(f'[{CJK}]', t))


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


def shape(chunks: Iterator[np.ndarray], sentence_end: bool, sr: int = SR) -> Iterator[np.ndarray]:
    """One generation's audio: faded in over FADE_IN (no click at its start), and after a sentence silence up to
    SENTENCE_PAUSE_S at its end (the next clause may follow back to back). Streams: nothing is held back."""
    frame, first, quiet = sr // 100, True, 0  # quiet: samples of silence at the end so far
    for c in chunks:
        if first and len(c):
            c, first = c.copy(), False
            n = min(FADE_IN, len(c))
            c[:n] *= np.linspace(0, 1, n + 1, dtype=np.float32)[1:]
        k = len(c) // frame
        blocks = c[len(c) - k * frame:].reshape(k, frame) if k else c[None]  # a chunk under 10 ms is one block
        f = 10 * np.log10(np.mean(blocks ** 2, axis=1) + 1e-10) if len(c) else np.zeros(0)
        loud = np.nonzero(f >= QUIET_DB)[0]
        quiet = quiet + len(c) if not len(loud) else (len(f) - 1 - loud[-1]) * frame
        yield c
    if sentence_end and (pad := int(SENTENCE_PAUSE_S * sr) - quiet) > 0:
        yield np.zeros(pad, np.float32)


def read_audio(path: Path) -> tuple[np.ndarray, int]:
    import soundfile as sf
    x, sr = sf.read(str(path), dtype='float32', always_2d=True)
    return x.mean(axis=1), sr


def folder(voice) -> Path:
    """A profile folder, or anything with `.path` to one (engines.Voice)."""
    return Path(getattr(voice, 'path', voice))


def _read_checkpoint(model_path) -> dict:
    """The checkpoint's weights, read only. coqui-tts's own loader writes the registry's model.pth again in place (its
    config is an XttsConfig object there): a write cut short (the runner killed, the power gone) left a broken file
    that `.verified` still vouches for (M-19)."""
    import torch
    from TTS.tts.models.xtts import _legacy_safe_globals
    with _legacy_safe_globals():
        return torch.load(model_path, map_location='cpu', weights_only=True)['model']


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
        self.model._load_checkpoint = _read_checkpoint  # not coqui's, which writes model.pth
        self.model.load_checkpoint(cfg, checkpoint_dir=str(model_dir), eval=True)
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.temperature, self.top_p = temperature, top_p
        self._latents: dict[str, tuple] = {}

    def latents(self, voice) -> tuple:
        """XTTS conditioning of a profile: cached in memory and as `xtts.pt` next to the recording, with the recording's
        SHA-256 (voice.json's ref_sha256): another recording there is conditioned again, never spoken from the old one.
        A voice with `state` brings its own."""
        if (state := getattr(voice, 'state', None)) is not None:
            return state
        key = str(folder(voice))
        if key not in self._latents:
            import torch
            f, ref = folder(voice) / 'xtts.pt', folder(voice) / 'reference.wav'
            digest = hashlib.sha256(ref.read_bytes()).hexdigest()
            lat = torch.load(f, map_location='cpu', weights_only=True) if f.exists() else {}  # tensors only: no code
            if lat.get('sha256') != digest:  # none yet, another recording, or a cache of 0.3.0 (without it)
                lat = {**self._condition(*read_audio(ref)), 'sha256': digest}
                torch.save(lat, f.with_suffix('.tmp'))  # a write cut short leaves no broken xtts.pt
                f.with_suffix('.tmp').replace(f)
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

    def condition(self, pcm: np.ndarray, sr: int) -> tuple:
        """The state of a voice from a sample, no profile folder: its conditioning on the device, (gpt, spk) as
        latents()'s; synth() speaks in it from anything with this as `state`."""
        lat = self._condition(pcm, sr)
        return lat['gpt'].to(self.device), lat['spk'].to(self.device)

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
        for part in split_text(normalize(closed(text), lang), self.model.tokenizer.char_limits.get(lang, 250)):
            pause = bool(SENTENCE_END.search(part))  # a sentence ends here: decided before ending() makes it a comma
            part = ending(part, lang)  # each part: a long text is cut into several
            gen = self.model.inference_stream(part, lang, gpt, spk, temperature=self.temperature, top_p=self.top_p,
                                              enable_text_splitting=False)
            try:
                yield from shape(guard(cap_pauses(c.float().cpu().numpy() for c in gen), budget_samples(part)), pause)
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
