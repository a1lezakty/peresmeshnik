"""The 14 languages: their names, and what a pair runs on — local engines (`Settings.local` + `langs` in models.toml)
or the cloud — with the reason when a side cannot be local. No Qt and no downloads here: the wizard, the tray, the app
and the tests all ask this module.

    from peresmeshnik import languages
    languages.LANGS                              # ('en', 'ru', 'uk', 'de', ...)
    languages.label('de')                        # 'Deutsch (German)'
    d = languages.direction(s, 'me_to_them')     # Direction: d.local, d.why, d.notes, d.asr/mt/tts, d.voice
    languages.clone_why(s.local.clone, 'de')     # '' = the clone of my voice speaks German live here, else why not
    languages.needed_models(s)                   # registry ids the chosen engines need, nothing more
    languages.optional_models(s)                 # cloud sides' fallback voices under a license to accept (asked)
"""
from __future__ import annotations

import shutil
import sys
from dataclasses import dataclass

# code: (English name, its own name)
NAMES = {
    'en': ('English', 'English'), 'ru': ('Russian', 'Русский'), 'uk': ('Ukrainian', 'Українська'),
    'de': ('German', 'Deutsch'), 'fr': ('French', 'Français'), 'es': ('Spanish', 'Español'),
    'it': ('Italian', 'Italiano'), 'pt': ('Portuguese', 'Português'), 'pl': ('Polish', 'Polski'),
    'tr': ('Turkish', 'Türkçe'), 'zh': ('Chinese', '中文'), 'ja': ('Japanese', '日本語'), 'ko': ('Korean', '한국어'),
    'ar': ('Arabic', 'العربية'),
}
LANGS = tuple(NAMES)
SIDES = ('them_to_me', 'me_to_them')
RTL = {'ar'}
# Qt's QFontDatabase.WritingSystem name per language: the UI checks that a font for it is installed
WRITING = {**dict.fromkeys(LANGS, 'Latin'), 'ru': 'Cyrillic', 'uk': 'Cyrillic', 'zh': 'SimplifiedChinese',
           'ja': 'Japanese', 'ko': 'Korean', 'ar': 'Arabic'}

# Gemini Live, checked 2026-11-09 — all 14 are in both models' tables:
# - gemini-3.5-live-translate-preview: `target_language_code`, BCP-47 (the source is detected from the speech),
#   https://ai.google.dev/gemini-api/docs/live-api/live-translate (Portuguese only as pt-BR / pt-PT, Chinese as zh-Hans)
# - gemini-3.5-transcribe-live: `language_codes`, BCP-47 with the region,
#   https://ai.google.dev/gemini-api/docs/live-api/live-transcribe (Spanish es-419 / es-US, Arabic only ar-EG)
# FLEURS (the bench) and the stock voices are Brazilian Portuguese, Latin American Spanish, Simplified Mandarin.
CLOUD_TRANSLATE = {**{c: c for c in LANGS}, 'pt': 'pt-BR', 'zh': 'zh-Hans'}
CLOUD_TRANSCRIBE = {'en': 'en-US', 'ru': 'ru-RU', 'uk': 'uk-UA', 'de': 'de-DE', 'fr': 'fr-FR', 'es': 'es-419',
                    'it': 'it-IT', 'pt': 'pt-BR', 'pl': 'pl-PL', 'tr': 'tr-TR', 'zh': 'cmn-Hans-CN', 'ja': 'ja-JP',
                    'ko': 'ko-KR', 'ar': 'ar-EG'}
CLOUD = set(CLOUD_TRANSLATE) & set(CLOUD_TRANSCRIBE)


def name(code: str) -> str:
    return NAMES[code][0] if code in NAMES else code


def label(code: str) -> str:
    """'Deutsch (German)'; English as is."""
    en, own = NAMES.get(code, (code, code))
    return own if own == en else f'{own} ({en})'


@dataclass(frozen=True)
class Direction:
    """One side for one pair. The ids are registry ids ('' = no local model for this language or pair)."""
    side: str  # 'them_to_me' | 'me_to_them'
    src: str
    tgt: str
    asr: str = ''
    mt: str = ''
    tts: str = ''  # them -> me: the stock voice; me -> them: the clone or the stock voice (see `voice`)
    voice: str = ''  # me -> them: 'clone' | 'stock' | '' (no local voice); them -> me: 'stock' | ''
    why: tuple[str, ...] = ()  # why this side cannot run locally; empty: it can
    notes: tuple[str, ...] = ()  # it can, but differently: a stock voice instead of the clone, slow on this computer
    cloud: bool = True  # Gemini Live has both languages

    @property
    def local(self) -> bool:
        return not self.why

    @property
    def subtitles(self) -> bool:
        """Local subtitles (ASR + MT) are possible: also beside the cloud voice, and the cloud's local fallback."""
        return bool(self.asr and self.mt)

    @property
    def models(self) -> tuple[str, ...]:
        return tuple(i for i in (self.asr, self.mt, self.tts) if i)


def covers(model_id: str, key: str) -> bool:
    """The registry model exists and its `langs` has the language (or the pair 'src-tgt', or both of its languages)."""
    from peresmeshnik import models
    m = models.registry().get(model_id)
    if m is None:
        return False
    parts = key.split('-')
    return key in m.langs or '*' in m.langs or (len(parts) == 2 and all(p in m.langs for p in parts))


def pick(table: dict[str, str], key: str) -> str:
    """The table's model for `key` (a language, or 'src-tgt' for MT), else its '*' entry (one model for every language
    or pair, e.g. a multilingual MT), when the registry's `langs` of that model have it."""
    return next((i for i in (table.get(key, ''), table.get('*', '')) if i and covers(i, key)), '')


def needs_gpu(model_id: str) -> bool:
    from peresmeshnik import models
    m = models.registry().get(model_id)
    return bool(m and m.needs == 'gpu')


ENGINE = {'asr': 'onnxruntime', 'mt': 'CTranslate2'}


def gpu(engine: str = '') -> bool:
    """An NVIDIA GPU (nvidia-smi). With `engine` ('asr': onnxruntime, 'mt': CTranslate2) also that this engine runs on it
    inside the app: the built .exe and a from-source install have a CPU-only onnxruntime (local/device.py)."""
    if not shutil.which('nvidia-smi'):
        return False
    from peresmeshnik.local import device
    return {'asr': device.onnx_cuda, 'mt': device.ct2_cuda}.get(engine, lambda: True)()


def clone_why(model_id: str, lang: str = '', has_gpu: bool | None = None) -> str:
    """Why the clone `model_id` cannot speak `lang` live on this computer ('' = it can): the one answer the wizard, the
    tray, the app and the cloud's fallback get (through `direction` or directly). It reads the registry — the model's
    `langs` and `needs = "gpu"` (XTTS-v2 keeps up only on an NVIDIA GPU, BENCH stage 6) — so another clone engine is a
    registry row, not a change here. `has_gpu` None: look for nvidia-smi."""
    if lang and not covers(model_id, lang):
        return f'your cloned voice cannot speak {name(lang)}'
    if needs_gpu(model_id) and not (gpu() if has_gpu is None else has_gpu):
        return 'no clone of your voice keeps up on a Mac' if sys.platform == 'darwin' else 'your cloned voice needs an NVIDIA GPU'
    return ''


def direction(s, side: str, has_gpu: bool | None = None) -> Direction:
    """What `side` of the pair in settings `s` runs on locally, and why not. A model with `needs = "gpu"` in the
    registry is not offered without an NVIDIA GPU (`has_gpu` None: look for nvidia-smi, and for the ASR and the MT also for
    CUDA in their engine in the app): for the ASR, the MT and a stock voice the side is then not local (`why`); for the
    clone (`clone_why`) the stock voice speaks instead (`notes`), if there is one."""
    from peresmeshnik.voice import STOCK
    src, tgt = (s.their_lang, s.my_lang) if side == 'them_to_me' else (s.my_lang, s.their_lang)
    loc, why, notes = s.local, [], []
    asked, has_gpu = has_gpu, gpu() if has_gpu is None else has_gpu

    def usable(i: str, missing: str, what: str, engine: str) -> str:
        if not i:
            why.append(missing)
        elif needs_gpu(i) and not (has_gpu if asked is not None else gpu(engine)):
            why.append(f'{what} needs an NVIDIA GPU ({i})' if not has_gpu else
                       f'{what} needs CUDA, and {ENGINE[engine]} in this app runs on the CPU only ({i})')
            return ''
        return i
    asr = usable(pick(loc.asr, src), f'no local speech recognition for {name(src)}', f'speech recognition for {name(src)}',
                 'asr')
    mt = usable(pick(loc.mt, f'{src}-{tgt}'), f'no local translation from {name(src)} to {name(tgt)}',
                f'translation from {name(src)} to {name(tgt)}', 'mt')
    stock, no_voice = pick(loc.tts, tgt), f'no local voice for {name(tgt)}'
    if stock and needs_gpu(stock) and not has_gpu:
        stock, no_voice = '', f'the voice for {name(tgt)} needs an NVIDIA GPU ({stock})'
    tts, voice = stock, 'stock' if stock else ''
    if side == 'me_to_them' and loc.clone != STOCK:
        if instead := clone_why(loc.clone, tgt, has_gpu):
            if stock:
                notes.append(f'{instead}: a stock voice speaks instead')
            else:
                no_voice = f'{instead}; {no_voice}'
        else:
            tts, voice = loc.clone, 'clone'
            if not stock:  # the clone's fallback (app.py) is the stock voice
                notes.append(f'no stock voice for {name(tgt)} takes over if your cloned voice fails')
    if not tts:
        why.append(no_voice)
    return Direction(side, src, tgt, asr, mt, tts, voice, tuple(why), tuple(notes), src in CLOUD and tgt in CLOUD)


def cloud_subtitles(s, side: str) -> bool:
    """Subtitles of a cloud side come from the cloud: when chosen, or when no local ASR + MT exists for the pair."""
    return s.cloud.subtitles or not direction(s, side).subtitles


def fallback_voice(s, side: str) -> str:
    """A cloud side's stock voice for when the cloud fails ('' = none: its subtitles come from the cloud, so there is no
    local ASR + MT to go on with, or its language has no stock voice)."""
    if getattr(s, side) != 'cloud' or cloud_subtitles(s, side):
        return ''
    d = direction(s, side)
    return d.tts if d.voice == 'stock' else pick(s.local.tts, d.tgt)


def _consent(model_id: str) -> bool:
    from peresmeshnik import models
    return models.get(model_id).consent


def needed_models(s) -> list[str]:
    """Registry ids the chosen setup runs on (ARCHITECTURE §6): a local side its ASR, MT and voice; a cloud side its
    local subtitles (unless they come from the cloud) and its fallback voice when that needs no consent (else it is
    optional: `optional_models`). Silero VAD is always there: it gates the cloud too."""
    ids = ['silero-vad']
    for side in SIDES:
        d = direction(s, side)
        if getattr(s, side) == 'local':
            ids += d.models
            if d.voice == 'clone':  # the clone's fallback (~60 MB): the stock voice when the clone cannot speak (S2)
                ids += [i for i in [pick(s.local.tts, d.tgt)] if i]
        elif not cloud_subtitles(s, side):  # local subtitles, and with its stock voice the side's way when the cloud fails
            ids += [d.asr, d.mt] + [i for i in [fallback_voice(s, side)] if i and not _consent(i)]
    return list(dict.fromkeys(ids))


def optional_models(s) -> list[str]:
    """Cloud sides' fallback voices under a license to accept (tr, ko, ja: noncommercial): downloaded only when it is
    accepted; without one that side stops while the cloud is down. Nobody chose them, so nothing waits for them."""
    need = set(needed_models(s))
    return list(dict.fromkeys(v for side in SIDES if (v := fallback_voice(s, side)) and _consent(v) and v not in need))


def table(s=None) -> str:
    """The README's table (Markdown) of what runs on this computer per language, from the settings' defaults and the
    registry: `python -m peresmeshnik.languages`."""
    from peresmeshnik import config, models
    loc = (s or config.Settings()).local
    reg = models.registry()

    def cell(i: str) -> str:
        return f'{i} (NVIDIA GPU)' if needs_gpu(i) else i or '— (cloud)'

    def asr(i: str) -> str:  # the apps' onnxruntime runs on the CPU only (gpu('asr')): such a model is never theirs
        return f'— (cloud; {i} from source with onnxruntime-gpu on an NVIDIA GPU)' if needs_gpu(i) else cell(i)
    rows = ['| Language | Speech recognition | Stock voice | Your cloned voice | Cloud (Gemini Live) |', '|---|---|---|---|---|']
    for c in LANGS:
        clone = loc.clone if covers(loc.clone, c) and reg[loc.clone].role == 'clone' else ''
        rows.append(f'| {label(c)} | {asr(pick(loc.asr, c))} | {cell(pick(loc.tts, c))} | '
                    f'{clone + " (NVIDIA GPU)" if clone else "— (cloud or a stock voice)"} | {"✔" if c in CLOUD else "—"} |')
    gpu_mark = {True: ' (NVIDIA GPU)', False: ''}
    pairs = [k.replace('-', ' → ') + gpu_mark[needs_gpu(loc.mt[k])] for k in sorted(loc.mt) if k != '*' and pick(loc.mt, k)]
    if (any_pair := loc.mt.get('*', '')) in reg:  # one model for every pair of its languages
        pairs.append(f'any other pair of {", ".join(c for c in LANGS if covers(any_pair, c))}{gpu_mark[needs_gpu(any_pair)]}')
    every = [f'{a}-{b}' for a in LANGS for b in LANGS if a != b]
    if all(pick(loc.mt, k) and not needs_gpu(pick(loc.mt, k)) for k in every):  # 182 pairs: not one by one
        via = sum(bool(reg[pick(loc.mt, k)].via) for k in every)
        rows.append(f'\nTranslation on this computer: any pair of the 14 ({via} of the {len(every)} through English).')
    else:
        rows.append(f'\nTranslation on this computer: {", ".join(pairs) or "none"}; every other pair of the 14 is translated '
                    'in the cloud.')
    return '\n'.join(rows)


if __name__ == '__main__':
    print(table())
