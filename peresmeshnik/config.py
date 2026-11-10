"""Settings: one dataclass with a section per area, stored as TOML in the user config folder.

    from peresmeshnik import config
    s = config.load()          # defaults + <config dir>/settings.toml (or $PERESMESHNIK_CONFIG)
    s.local.threads, s.cloud.translate_model, s.them_to_me ...
    config.save(s)

Unknown keys (from a newer version) are ignored with a warning; a value of the wrong type keeps its default, a file that
does not parse is kept aside as settings.toml.bad and the defaults are used. A file read only in part sends nothing to the
cloud: both directions are local, and `problem` says why (the app shows it at start and when translation goes on).
"""
from __future__ import annotations

import json
import os
import re
import sys
import tempfile
import time
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path

import platformdirs

APP = 'Peresmeshnik'
ENGINES = ('local', 'cloud')


def config_dir() -> Path:  # ~/.config/Peresmeshnik, ~/Library/Application Support/Peresmeshnik, %LOCALAPPDATA%\Peresmeshnik
    return platformdirs.user_config_path(APP, appauthor=False)


def cache_dir() -> Path:  # ~/.cache/Peresmeshnik, ~/Library/Caches/Peresmeshnik, %LOCALAPPDATA%\Peresmeshnik\Cache
    return platformdirs.user_cache_path(APP, appauthor=False)


def data_dir() -> Path:  # voice profiles live here and are never uploaded
    return platformdirs.user_data_path(APP, appauthor=False)


LANGS = ('en', 'ru', 'uk', 'de', 'fr', 'es', 'it', 'pt', 'pl', 'tr', 'zh', 'ja', 'ko', 'ar')  # = languages.LANGS (a test)


DIRECT_RU = ('uk', 'de', 'fr', 'es', 'it', 'pt', 'pl')  # X<->ru by OPUS-MT of their own: through English the speaker's
# gender and ты / вы get lost, and they win on short everyday sentences (BENCH stage 6, MT)
OPUS_TO_EN = ('ru', 'uk', 'de', 'fr', 'es', 'it', 'pt', 'pl', 'tr')  # X->en by OPUS-MT: quickmt makes up a continuation of a
# one- or two-word reply ("Да." -> "Yes, I am very"); = the `pivot` rows of models.toml (a test)


def _mt() -> dict[str, str]:
    """Every direction of the 14 (BENCH stage 6): X<->ru direct for DIRECT_RU; X->en OPUS-MT for OPUS_TO_EN, else quickmt;
    en->X quickmt; the rest through English (models.registry: mt-X-Y-via-en)."""
    def one(a, b):
        if 'ru' in (a, b) and {a, b} - {'ru'} <= set(DIRECT_RU):
            return f'opus-mt-{a}-{b}'
        if b == 'en':
            return f'opus-mt-{a}-en' if a in OPUS_TO_EN else f'quickmt-{a}-en'
        return f'quickmt-en-{b}' if a == 'en' else f'mt-{a}-{b}-via-en'
    return {f'{a}-{b}': one(a, b) for a in LANGS for b in LANGS if a != b}


@dataclass
class Local:
    threads: int = 2  # per runtime: onnxruntime intra_op, CTranslate2 intra_threads, torch set_num_threads
    # ru: GigaAM, not Parakeet: Parakeet has no language setting and drifts into Ukrainian on short windows (BENCH stage 1).
    # tr / zh / ko: Whisper large-v3-turbo, live only on an NVIDIA GPU (its registry row: needs = "gpu"; BENCH stage 6)
    asr: dict[str, str] = field(default_factory=lambda: {
        'en': 'parakeet-tdt-0.6b-v3', 'ru': 'gigaam-v3-e2e-ctc',
        **dict.fromkeys(('uk', 'de', 'es', 'it', 'pt', 'pl'), 'parakeet-tdt-0.6b-v3'),
        # fr: Parakeet v3 drifts into English mid-monologue (BENCH stage 6: WER 0.31 against 0.17 in a monologue)
        'fr': 'fastconformer-fr-pc',
        'ja': 'parakeet-tdt_ctc-0.6b-ja', 'ar': 'fastconformer-ar-pcd',
        **dict.fromkeys(('tr', 'zh', 'ko'), 'whisper-large-v3-turbo')})
    mt: dict[str, str] = field(default_factory=lambda: _mt())
    # stock voices (BENCH stage 6); tr / ko / ja: noncommercial voices (consent); zh / ar: none with a usable license
    tts: dict[str, str] = field(default_factory=lambda: {
        'en': 'piper-en_US-john-medium', 'ru': 'piper-ru_RU-dmitri-medium', 'uk': 'piper-uk_UA-ukrainian_tts-medium',
        'de': 'piper-de_DE-thorsten-medium', 'fr': 'piper-fr_FR-siwis-medium', 'es': 'piper-es_MX-ald-medium',
        'it': 'piper-it_IT-paola-medium', 'pt': 'piper-pt_BR-faber-medium', 'pl': 'piper-pl_PL-gosia-medium',
        'tr': 'piper-tr_TR-dfki-medium', 'ko': 'piper-ko_KR-kss-medium', 'ja': 'piper-ja_JP-hi_fi_captain-medium'})
    clone: str = 'xtts-v2'  # clone TTS for me -> them
    voice: str = ''  # enrolled voice name ('' = none yet)
    # them -> me: a long clause's stable words get a translated subtitle while it is still said (BENCH stage 5)
    partial_mt: bool = True
    # the echo gates look for our own speech coming back this late, s (a call's round trip; up to 2.8: the reference
    # keeps 4 s). 1.0 / 1.5 / 2.0 cost 0.21 / 0.28 / 0.34 ms of CPU per 16 ms block while we speak (BENCH stage 5)
    echo_max_delay_s: float = 2.0
    # the microphone's gate (speakers, no headphones): how closely the capture must follow what the speakers played; a
    # room's reverb smears it, and at 0.5 a reverberant room (RT60 0.8 s) let 2 of 36 echoes through (BENCH stage 5).
    # The call side's gate keeps audio/echo.py's 0.5
    mic_echo_rho_on: float = 0.4


@dataclass
class Cloud:
    key_file: str = ''  # '' = <config dir>/gemini-key; $GEMINI_API_KEY wins
    translate_model: str = 'gemini-3.5-live-translate-preview'
    transcribe_model: str = 'gemini-3.5-transcribe-live'
    subtitles: bool = False  # cloud subtitles from transcribe-live; default: local ASR (faster, BENCH)
    reconnect_s: float = 540.0  # hand over to a fresh connection before the server's ~10 min limit
    # "too slow": no translated voice this long after a phrase began -> the side goes local until the cloud is back
    # (the first translated voice after a phrase began: p90 3.3-4.6 s, max 4.7 s measured, BENCH). A drop, an error or
    # the quota: at once
    fallback_after_s: float = 6.0


@dataclass
class Ui:
    hotkey: str = 'Ctrl+Shift+Space' if sys.platform == 'darwin' else 'Ctrl+Alt+Space'  # translation <-> live voice (Qt text; mac ⇧⌘Space)
    mic: str = ''  # me -> them input ('' = system default)
    headphones: str = ''  # them -> me output sink ('' = system default)
    call_app: str = ''  # them -> me: the call app moved to "Peresmeshnik Call" (process binary, else app name; '' = none)
    show_latency: bool = False  # the overlay shows per clause: end of its speech -> its subtitle, -> its voice


@dataclass
class Settings:
    my_lang: str = 'ru'
    their_lang: str = 'en'
    them_to_me: str = 'local'  # 'local' | 'cloud'
    me_to_them: str = 'cloud'  # default for an 8 GB Mac (ARCHITECTURE §6): no live local clone there
    # the sides ('them_to_me', 'me_to_them') the app itself moved to the cloud, for a pair they cannot run locally: they
    # go back to 'local' with a pair that can (a side the user set to 'cloud' is not here)
    forced_cloud: list[str] = field(default_factory=list)
    models_dir: str = ''  # '' = <cache dir>/models; $PERESMESHNIK_MODELS wins
    local: Local = field(default_factory=Local)
    cloud: Cloud = field(default_factory=Cloud)
    ui: Ui = field(default_factory=Ui)


def settings_path() -> Path:
    return Path(os.environ.get('PERESMESHNIK_CONFIG') or config_dir() / 'settings.toml')


# defaults that changed since a release that saved its [local] tables whole (0.2.0, bac8639): such a value goes on
# load, so the new default comes in. A whole table is told by a key equal to today's default (save() no longer writes
# those), so a value chosen again since is kept
OLD_DEFAULTS = {'local.mt': {'en-ru': 'opus-mt-en-ru'}}


problem = ''  # what the last load() could not use, for the user ('' = nothing)


def _fill(obj, data: dict, bad: list[str], where: str = ''):
    for k, v in data.items():
        if not hasattr(obj, k):
            print(f'settings: unknown key {where}{k} ignored', file=sys.stderr)
            continue
        cur = getattr(obj, k)
        if is_dataclass(cur):
            if not isinstance(v, dict):
                bad.append(f'[{where}{k}] must be a table')
                continue
            _fill(cur, v, bad, f'{where}{k}.')
            continue
        if isinstance(cur, float) and type(v) is int:
            v = float(v)
        if type(v) is not type(cur) or (isinstance(v, dict | list) and not all(isinstance(x, str) for x in
                                                                               (v.values() if isinstance(v, dict) else v))):
            bad.append(f'{where}{k} must be {type(cur).__name__}, got {v!r}')
            continue
        if isinstance(cur, dict) and any(cur.get(x) == y for x, y in v.items()):  # a table saved whole
            v = {x: y for x, y in v.items() if OLD_DEFAULTS.get(f'{where}{k}', {}).get(x) != y}
        setattr(obj, k, {**cur, **v} if isinstance(cur, dict) else v)


def load(path: Path | None = None) -> Settings:
    """Never raises on what the file holds: a broken file must not stop the app (H-01); `problem` names what was not used."""
    import tomllib
    global problem
    s, path, bad = Settings(), path or settings_path(), []
    try:
        _fill(s, tomllib.loads(path.read_text('utf-8-sig')), bad)  # -sig: Notepad's BOM
    except FileNotFoundError:  # none yet (the wizard writes it)
        pass
    except ValueError as e:  # TOML or UTF-8: kept aside, not overwritten by the next save
        keep = path.with_name(path.name + '.bad')
        try:
            os.replace(path, keep)
            bad.append(f'it does not parse ({e}), it is kept as {keep.name}')
        except OSError as e2:  # held by an antivirus, or gone meanwhile
            bad.append(f'it does not parse ({e}) and could not be kept aside ({e2})')
    except OSError as e:  # cannot be opened now (an antivirus holds it): not broken, not moved
        bad.append(f'it cannot be opened ({e})')
    for k in ('them_to_me', 'me_to_them'):
        if getattr(s, k) not in ENGINES:  # local, not the default: a typo must not send the voice to Google unasked
            bad.append(f'{k} must be one of {ENGINES}, got {getattr(s, k)!r}: local is used')
            setattr(s, k, 'local')
    if wrong := set(s.forced_cloud) - {'them_to_me', 'me_to_them'}:
        bad.append(f'forced_cloud may hold only them_to_me, me_to_them, got {sorted(wrong)}')
        s.forced_cloud = []
    if not 0.2 <= (r := s.local.mic_echo_rho_on) <= 0.8:  # near 0 it zeroes my speech, near 1 lets every echo through
        s.local.mic_echo_rho_on = min(max(r, 0.2), 0.8)
        print(f'settings: local.mic_echo_rho_on {r} is outside 0.2..0.8, {s.local.mic_echo_rho_on} used', file=sys.stderr)
    if bad:  # read in part: nothing goes to the cloud by a mistake in the file (the key is kept apart, H-01)
        s.them_to_me = s.me_to_them = 'local'
        bad.insert(0, f'{path.name} could not be read whole, both directions are local now')
    for b in bad:
        print(f'settings: {b}', file=sys.stderr)
    problem = f'{bad[0]}: {"; ".join(bad[1:])}' if bad else ''
    return s


def _toml(v) -> str:
    if isinstance(v, bool):
        return 'true' if v else 'false'
    if isinstance(v, int | float):
        return repr(v)
    if isinstance(v, str):  # JSON string escapes are valid TOML basic strings; TOML also wants DEL escaped and cannot
        # hold a lone surrogate (a device name from the OS not in UTF-16 norm): U+FFFD, and the file stays readable
        return json.dumps(re.sub('[\ud800-\udfff]', '\ufffd', v), ensure_ascii=False).replace('\x7f', '\\u007f')
    if isinstance(v, list):
        return '[' + ', '.join(_toml(x) for x in v) + ']'
    if isinstance(v, dict):
        return '{ ' + ', '.join(f'{_toml(k)} = {_toml(x)}' for k, x in v.items()) + ' }'
    raise TypeError(f'cannot write {v!r} to TOML')


def _changed(v, default):
    return {k: x for k, x in v.items() if default.get(k) != x} if isinstance(v, dict) else v


def save(s: Settings, path: Path | None = None) -> Path:
    path = path or settings_path()
    top, tables = [], []
    for f in fields(s):
        v = getattr(s, f.name)
        if is_dataclass(v):  # a table's dict: only its keys that differ from the default, so a new default reaches
            base = type(v)()  # saved settings too (load merges the saved keys over the defaults)
            tables.append(f'\n[{f.name}]\n' + ''.join(f'{g.name} = {_toml(_changed(getattr(v, g.name), getattr(base, g.name)))}\n'
                                                     for g in fields(v)))
        else:
            top.append(f'{f.name} = {_toml(v)}\n')
    data = ''.join(top + tables).encode('utf-8')
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=path.name, suffix='.tmp', dir=path.parent)  # whole or not at all: a crash or
    try:  # a full disk mid-write leaves the old file (H-01)
        with os.fdopen(fd, 'wb') as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        for wait in (0.05, 0.2, 0.5, None):  # Windows: an antivirus or a backup tool holds the file a moment
            try:
                os.replace(tmp, path)
                break
            except PermissionError:
                if wait is None:
                    raise
                time.sleep(wait)
    except BaseException:
        os.unlink(tmp)
        raise
    return path
