"""The 14 languages: their names, and what a pair runs on — local engines (`Settings.local` + `langs` in models.toml)
or the cloud — with the reason when a side cannot be local. No Qt and no downloads here: the wizard, the tray, the app
and the tests all ask this module.

    from peresmeshnik import languages
    languages.LANGS                              # ('en', 'ru', 'uk', 'de', ...)
    languages.label('de')                        # 'Deutsch (German)'
    d = languages.direction(s, 'me_to_them')     # Direction: d.local, d.why, d.notes, d.asr/mt/tts, d.voice
    languages.needed_models(s)                   # registry ids the chosen engines need, nothing more
"""
from __future__ import annotations

import shutil
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


def _pick(table: dict[str, str], key: str) -> str:
    i = table.get(key, '')
    return i if covers(i, key) else ''


def gpu() -> bool:
    return bool(shutil.which('nvidia-smi'))


def direction(s, side: str, has_gpu: bool | None = None) -> Direction:
    """What `side` of the pair in settings `s` runs on locally, and why not. `has_gpu` None: look for nvidia-smi."""
    from peresmeshnik.voice import STOCK
    src, tgt = (s.their_lang, s.my_lang) if side == 'them_to_me' else (s.my_lang, s.their_lang)
    loc, why, notes = s.local, [], []
    asr, mt, stock = _pick(loc.asr, src), _pick(loc.mt, f'{src}-{tgt}'), _pick(loc.tts, tgt)
    if not asr:
        why.append(f'no local speech recognition for {name(src)}')
    if not mt:
        why.append(f'no local translation from {name(src)} to {name(tgt)}')
    tts, voice = stock, 'stock' if stock else ''
    if side == 'me_to_them' and loc.clone != STOCK:
        if covers(loc.clone, tgt):
            tts, voice = loc.clone, 'clone'
            if not (gpu() if has_gpu is None else has_gpu):
                notes.append('your cloned voice falls behind the conversation without an NVIDIA GPU: '
                             'the cloud or a stock voice keeps up')
        elif stock:
            notes.append(f'your cloned voice cannot speak {name(tgt)}: a stock voice speaks instead')
    if not tts:
        why.append(f'no local voice for {name(tgt)}')
    return Direction(side, src, tgt, asr, mt, tts, voice, tuple(why), tuple(notes), src in CLOUD and tgt in CLOUD)


def cloud_subtitles(s, side: str) -> bool:
    """Subtitles of a cloud side come from the cloud: when chosen, or when no local ASR + MT exists for the pair."""
    return s.cloud.subtitles or not direction(s, side).subtitles


def needed_models(s) -> list[str]:
    """Registry ids the chosen setup runs on (ARCHITECTURE §6): a local side its ASR, MT and voice; a cloud side its
    local subtitles (unless they come from the cloud). Silero VAD is always there: it gates the cloud too."""
    ids = ['silero-vad']
    for side in SIDES:
        d = direction(s, side)
        if getattr(s, side) == 'local':
            ids += d.models
            if d.voice == 'clone':  # the clone's fallback (~60 MB): the stock voice when the clone cannot speak (S2)
                ids += [i for i in [_pick(s.local.tts, d.tgt)] if i]
        elif not cloud_subtitles(s, side):
            ids += [d.asr, d.mt]
    return list(dict.fromkeys(ids))
