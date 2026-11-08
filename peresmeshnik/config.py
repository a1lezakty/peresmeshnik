"""Settings: one dataclass with a section per area, stored as TOML in the user config folder.

    from peresmeshnik import config
    s = config.load()          # defaults + <config dir>/settings.toml (or $PERESMESHNIK_CONFIG)
    s.local.threads, s.cloud.translate_model, s.them_to_me ...
    config.save(s)

Unknown keys (from a newer version) are ignored with a warning; a wrong type is an error.
"""
from __future__ import annotations

import json
import os
import sys
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


@dataclass
class Local:
    threads: int = 2  # per runtime: onnxruntime intra_op, CTranslate2 intra_threads, torch set_num_threads
    # ru: GigaAM, not Parakeet: Parakeet has no language setting and drifts into Ukrainian on short windows (BENCH stage 1)
    asr: dict[str, str] = field(default_factory=lambda: {'en': 'parakeet-tdt-0.6b-v3', 'ru': 'gigaam-v3-e2e-ctc'})
    mt: dict[str, str] = field(default_factory=lambda: {'en-ru': 'opus-mt-en-ru', 'ru-en': 'opus-mt-ru-en'})
    tts: dict[str, str] = field(default_factory=lambda: {'en': 'piper-en_US-john-medium', 'ru': 'piper-ru_RU-dmitri-medium'})
    clone: str = 'xtts-v2'  # clone TTS for me -> them
    voice: str = ''  # enrolled voice name ('' = none yet)


@dataclass
class Cloud:
    key_file: str = ''  # '' = <config dir>/gemini-key; $GEMINI_API_KEY wins
    translate_model: str = 'gemini-3.5-live-translate-preview'
    transcribe_model: str = 'gemini-3.5-transcribe-live'
    subtitles: bool = False  # cloud subtitles from transcribe-live; default: local ASR (faster, BENCH)
    reconnect_s: float = 540.0  # hand over to a fresh connection before the server's ~10 min limit


@dataclass
class Ui:
    hotkey: str = 'Ctrl+Alt+Space'  # translation <-> live voice; Qt key-sequence text
    mic: str = ''  # me -> them input ('' = system default)
    headphones: str = ''  # them -> me output sink ('' = system default)
    call_app: str = ''  # them -> me: the call app moved to "Peresmeshnik Call" (process binary, else app name; '' = none)


@dataclass
class Settings:
    my_lang: str = 'ru'
    their_lang: str = 'en'
    them_to_me: str = 'local'  # 'local' | 'cloud'
    me_to_them: str = 'cloud'  # default for an 8 GB Mac (ARCHITECTURE §6): no live local clone there
    models_dir: str = ''  # '' = <cache dir>/models; $PERESMESHNIK_MODELS wins
    local: Local = field(default_factory=Local)
    cloud: Cloud = field(default_factory=Cloud)
    ui: Ui = field(default_factory=Ui)


def settings_path() -> Path:
    return Path(os.environ.get('PERESMESHNIK_CONFIG') or config_dir() / 'settings.toml')


def _fill(obj, data: dict, where: str = ''):
    for k, v in data.items():
        if not hasattr(obj, k):
            print(f'settings: unknown key {where}{k} ignored', file=sys.stderr)
            continue
        cur = getattr(obj, k)
        if is_dataclass(cur):
            if not isinstance(v, dict):
                raise ValueError(f'settings: [{where}{k}] must be a table')
            _fill(cur, v, f'{where}{k}.')
            continue
        if isinstance(cur, float) and type(v) is int:
            v = float(v)
        if type(v) is not type(cur) or (isinstance(v, dict) and not all(isinstance(x, str) for x in v.values())):
            raise ValueError(f'settings: {where}{k} must be {type(cur).__name__}, got {v!r}')
        setattr(obj, k, {**cur, **v} if isinstance(cur, dict) else v)


def load(path: Path | None = None) -> Settings:
    import tomllib
    s, path = Settings(), path or settings_path()
    if path.exists():
        _fill(s, tomllib.loads(path.read_text('utf-8')))
    for k in ('them_to_me', 'me_to_them'):
        if getattr(s, k) not in ENGINES:
            raise ValueError(f'settings: {k} must be one of {ENGINES}, got {getattr(s, k)!r}')
    return s


def _toml(v) -> str:
    if isinstance(v, bool):
        return 'true' if v else 'false'
    if isinstance(v, int | float):
        return repr(v)
    if isinstance(v, str):
        return json.dumps(v, ensure_ascii=False)  # JSON string escapes are valid TOML basic strings
    if isinstance(v, dict):
        return '{ ' + ', '.join(f'{json.dumps(k)} = {_toml(x)}' for k, x in v.items()) + ' }'
    raise TypeError(f'cannot write {v!r} to TOML')


def save(s: Settings, path: Path | None = None) -> Path:
    path = path or settings_path()
    top, tables = [], []
    for f in fields(s):
        v = getattr(s, f.name)
        if is_dataclass(v):
            tables.append(f'\n[{f.name}]\n' + ''.join(f'{g.name} = {_toml(getattr(v, g.name))}\n' for g in fields(v)))
        else:
            top.append(f'{f.name} = {_toml(v)}\n')
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(''.join(top + tables), 'utf-8')
    return path
