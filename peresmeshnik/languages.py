"""The 14 languages: their names, and what a pair runs on — local engines (`Settings.local` + `langs` in models.toml)
or the cloud — with the reason when a side cannot be local. No Qt and no downloads here: the wizard, the tray, the app
and the tests all ask this module.

    from peresmeshnik import languages
    languages.LANGS                              # ('en', 'ru', 'uk', 'de', ...)
    languages.label('de')                        # 'Deutsch (German)'
    d = languages.direction(s, 'me_to_them')     # Direction: d.local, d.why, d.notes, d.asr/mt/tts, d.voice
    languages.clone_why(s.local.clone, 'de')     # '' = the clone of my voice speaks German live here, else why not
    languages.clone_engine('de', 'them_to_me')  # ('pocket-tts-de', '') = the engine of a cloned voice here, (None, why)
    languages.needed_models(s)                   # registry ids the chosen engines need, nothing more
    languages.optional_models(s)                 # cloud sides' fallback voices under a license to accept (asked)
"""
from __future__ import annotations

import importlib.util
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from typing import NamedTuple

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


# --- the clone's NVIDIA GPU (nvidia-smi, no torch): memory, driver, compute capability

MIN_VRAM_GB = 5.5  # the clone (XTTS-v2 ~2 GB) next to them -> me on CUDA (~0.6 GB): a 6 GB card (6144 MiB, some report less)


MIN_DRIVER = (570, 65)  # Windows: CUDA 12.8 GA's driver (the pack's torch 2.8.0+cu128; NVIDIA's CUDA 12.8.0 notes)
MIN_DRIVER_COMPAT = (528, 33)  # CUDA 12.x minor version compatibility on Windows: below 570.65 not checked, may work
MIN_CC = 6.1  # the pack's Windows torch is built for sm_61 .. sm_120 (its _cuda_getArchFlags, the voice-pack job)


class Gpu(NamedTuple):
    vram: float = 0.0  # GB
    driver: tuple[int, ...] = ()
    cc: float = 0.0  # compute capability; 0: not known (a driver too old to say)
    why: str = ''  # nvidia-smi is there but fails: the driver's own words

    @property
    def driver_text(self) -> str:
        return version_text(self.driver)


def _smi(exe: str, fields: str) -> list[list[str]] | str:
    try:
        r = subprocess.run([exe, f'--query-gpu={fields}', '--format=csv,noheader,nounits'], capture_output=True, text=True,
                           timeout=10, creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
    except (OSError, subprocess.SubprocessError) as e:
        return str(e)
    rows = [[x.strip() for x in ln.split(',')] for ln in (r.stdout or '').splitlines() if ln.strip()]
    if rows and all(len(row) == fields.count(',') + 1 and row[0].replace('.', '', 1).isdigit() for row in rows):
        return rows
    return ' '.join((r.stdout or '').split())[:200] or f'nvidia-smi exited with {getattr(r, "returncode", "?")}'


def nvidia() -> Gpu:
    """The biggest NVIDIA GPU by nvidia-smi (no torch, ~0.1 s): memory, driver, compute capability; Gpu() without one.
    A driver that does not answer is not "no GPU": `why` says what nvidia-smi said."""
    exe = shutil.which('nvidia-smi')
    if not exe:
        return Gpu()
    rows = _smi(exe, 'memory.total,driver_version,compute_cap')
    if isinstance(rows, str):  # a driver before compute_cap (R510) does not know the field
        rows = old if not isinstance(old := _smi(exe, 'memory.total,driver_version'), str) else rows
    if isinstance(rows, str):
        return Gpu(why=rows)
    best = max(rows, key=lambda row: float(row[0]))

    def num(x: str, kind=float):
        try:
            return kind(x)
        except ValueError:
            return kind()
    return Gpu(float(best[0]) / 1024, tuple(num(x, int) for x in best[1].split('.')), num(best[2]) if len(best) > 2 else 0.0)


def gpu_text(vram: float) -> str:
    return f'an NVIDIA GPU with {vram:.0f} GB' if vram else 'no NVIDIA GPU'


def version_text(v: tuple[int, ...]) -> str:
    return '.'.join(f'{x:02d}' if i else str(x) for i, x in enumerate(v))


_PROBED: dict = {}  # nvidia-smi's path -> (its answer, when): the card does not change while the app runs
FAILED_TTL_S = 60.0  # a driver that did not answer is asked again after this long (updated meanwhile), not at every call


def nvidia_once() -> Gpu:
    """nvidia() once per process (up to 2 x 10 s with a driver that hangs: the wizard's tier and every direction() share
    it); a driver that did not answer (`why`) is asked again after FAILED_TTL_S: "update it, then open this page again"."""
    exe = shutil.which('nvidia-smi')
    g, at = _PROBED.get(exe, (None, 0.0))
    if g is None or g.why and time.monotonic() - at > FAILED_TTL_S:
        g, at = _PROBED[exe] = nvidia(), time.monotonic()
    return g


def gpu_why(g: Gpu, windows: bool = True) -> str:
    """Why the clone cannot run on this NVIDIA GPU ('' = it can): its memory, and for the Windows voice pack's torch the
    driver and the compute capability."""
    if g.why:
        return f'the NVIDIA driver does not answer ({g.why})'
    if g.vram < MIN_VRAM_GB:
        return f'the clone of your voice needs an NVIDIA GPU with at least 6 GB of memory; this computer has {gpu_text(g.vram)}'
    if windows and g.driver < MIN_DRIVER_COMPAT:
        return (f'the clone of your voice needs NVIDIA driver {version_text(MIN_DRIVER)} or newer; this computer has '
                f'{g.driver_text or "an older one"}')
    if windows and g.cc and g.cc < MIN_CC:
        return (f'this NVIDIA GPU is too old for the voice pack (compute capability {g.cc:g}; it needs {MIN_CC:g} or newer: '
                'GeForce GTX 10xx and later)')
    return ''


def gpu(engine: str = '') -> bool:
    """An NVIDIA GPU (nvidia-smi). With `engine` ('asr': onnxruntime, 'mt': CTranslate2) also that this engine runs on it
    inside the app: the built .exe and a from-source install have a CPU-only onnxruntime (local/device.py)."""
    if not shutil.which('nvidia-smi'):
        return False
    from peresmeshnik.local import device
    return {'asr': device.onnx_cuda, 'mt': device.ct2_cuda}.get(engine, lambda: True)()


def clone_why(model_id: str, lang: str = '', has_gpu: bool | Gpu | None = None, os: str = '') -> str:
    """Why the clone `model_id` cannot speak `lang` live on this computer ('' = it can): the one answer the wizard, the
    tray, the app and the cloud's fallback get (through `direction` or directly). It reads the registry — the model's
    `langs` and `needs = "gpu"` (XTTS-v2 keeps up only on an NVIDIA GPU, BENCH stage 6) — and for such a model the GPU
    (`gpu_why`: memory; on Windows also the voice pack's driver and compute capability), so another clone engine is a
    registry row, not a change here. `has_gpu`: None looks (nvidia-smi, once per run), a `Gpu` is that card, True / False
    say there is a suitable one or none. `os`: sys.platform of the computer it runs on ('' = this one)."""
    os = os or sys.platform
    if lang and not covers(model_id, lang):
        return f'your cloned voice cannot speak {name(lang)}'
    if not needs_gpu(model_id) or has_gpu is True:
        return ''
    g = has_gpu if isinstance(has_gpu, Gpu) else nvidia_once() if has_gpu is None and gpu() else Gpu()
    if has_gpu is None and gpu() and not (g.vram or g.why):  # nvidia-smi said nothing (a stand-in): taken as able
        return ''
    if not (g.vram or g.why):
        return 'no clone of your voice keeps up on a Mac' if os == 'darwin' else 'your cloned voice needs an NVIDIA GPU'
    return gpu_why(g, windows=os == 'win32')


class Platform(NamedTuple):
    """The computer the voice engines run on: this one (`Platform()`), or a server's, from its hello (remote/)."""
    os: str = ''  # sys.platform there: 'darwin' | 'win32' | 'linux'; '' = this computer's
    gpu: Gpu | bool | None = None  # its NVIDIA card as `clone_why`'s has_gpu: None = look here (nvidia-smi, once per run)


def clone_engines() -> list[str]:
    """The registry's clone engines (`role = "clone"`), best first where each runs (RESEARCH-clone §7): those that need an
    NVIDIA GPU (Qwen3-TTS, then XTTS-v2 for the languages Qwen3 lacks: pl, tr, ar), then those on any CPU, Mac too
    (Pocket TTS, a model per language); models.toml's order within each. A new engine is a registry row."""
    from peresmeshnik import models
    reg = models.registry().values()
    backups = {m.fallback for m in reg}  # Chatterbox: only where Qwen3 breaks down, never chosen on its own
    return [m.id for m in sorted((m for m in reg if m.role == 'clone' and m.id not in backups), key=lambda m: m.needs != 'gpu')]


def pack_here() -> bool:
    """Whether an NVIDIA clone engine can start on this computer: the voice pack's runner, or torch in this Python."""
    from peresmeshnik.local.clone import voice_cmd
    return voice_cmd() is not None or importlib.util.find_spec('torch') is not None


def not_here(engine: str) -> str:
    """Why `engine` cannot start on this computer as it is now ('' = it can): a gated model without its weights and
    without the user's Hugging Face token, an NVIDIA engine without the voice pack (or torch from source)."""
    from peresmeshnik import models
    m = models.get(engine)
    if m.token == 'hf' and not models.hf_token() and not models.is_fetched(m):
        return 'your cloned voice needs a Hugging Face token (Voices…, setup wizard)'
    if needs_gpu(engine) and not pack_here():
        return 'the voice pack is not installed (setup wizard, "Your voice")'
    return ''


def clone_engine(lang: str, side: str = 'me_to_them', *, platform: Platform | None = None) -> tuple[str | None, str]:
    """(the clone engine that speaks `lang` live for `side` on `platform` (None: this computer), '') or (None, why not):
    then the stock voice of `lang` speaks, if there is one. The one answer to "can a cloned voice speak here, and which":
    the wizard, the tray, the app, the cloud's fallback and a server's client ask it, nobody checks on their own. Whether
    its weights are downloaded is not asked here (models). On this computer (`platform` without `os`) an engine that
    cannot start here (`not_here`) gives way to the next one that can: an NVIDIA engine without the voice pack to Pocket
    TTS; if none can, the NVIDIA engine stays (its pack is the wizard's offer, the app says why at start), a gated one
    without the user's token does not (the stock voice, said why; with the token it is chosen again)."""
    if side not in SIDES:
        raise ValueError(f'side {side!r}: one of {SIDES}')
    p = platform or Platform()
    whys = [(e, clone_why(e, lang, p.gpu, **({'os': p.os} if p.os else {}))) for e in clone_engines() if covers(e, lang)]
    fits = [e for e, why in whys if not why]  # its hardware is there
    if not p.os:  # this computer: also what is installed here
        late = {e: not_here(e) for e in fits}
        fits = [e for e in fits if not late[e]] or [e for e in fits if needs_gpu(e)]
        whys = [(e, late[e]) for e in late if late[e]] + whys
    if fits:
        return fits[0], ''
    why = next((why for _, why in whys if why), f'your cloned voice cannot speak {name(lang)}')  # the best engine's reason
    return None, why.replace('your', 'their') if side == 'them_to_me' else why  # the other person's voice


def direction(s, side: str, has_gpu: bool | None = None) -> Direction:
    """What `side` of the pair in settings `s` runs on locally, and why not. A model with `needs = "gpu"` in the
    registry is not offered without an NVIDIA GPU (`has_gpu` None: look for nvidia-smi, and for the ASR and the MT also for
    CUDA in their engine in the app): for the ASR, the MT and a stock voice the side is then not local (`why`); for the
    clone (`clone_engine`: its engine per platform and language) the stock voice speaks instead (`notes`), if there is one."""
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
        engine, instead = clone_engine(tgt, side, platform=Platform(gpu=asked))  # None: it looks at the card itself
        if instead:
            if stock:
                notes.append(f'{instead}: a stock voice speaks instead')
            else:
                no_voice = f'{instead}; {no_voice}'
        else:
            tts, voice = engine, 'clone'
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


def _fallback_engine(model_id: str) -> str:
    from peresmeshnik import models
    return models.get(model_id).fallback


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
                ids += [i for i in [_fallback_engine(d.tts)] if i]  # its own fallback engine (Qwen3: Chatterbox)
        elif getattr(s, side) == 'cloud' and not cloud_subtitles(s, side):  # local subtitles, and with its stock voice
            ids += [d.asr, d.mt] + [i for i in [fallback_voice(s, side)] if i and not _consent(i)]  # the side's way when
            # the cloud fails; remote: all of it on the server
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
        cpu, nvidia = (clone_engine(c, platform=Platform('linux', g))[0] for g in (False, True))
        clone = ' or '.join([f'{nvidia} (NVIDIA GPU)'] * (nvidia != cpu) + [cpu] * bool(cpu))
        rows.append(f'| {label(c)} | {asr(pick(loc.asr, c))} | {cell(pick(loc.tts, c))} | '
                    f'{clone or "— (cloud or a stock voice)"} | {"✔" if c in CLOUD else "—"} |')
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
