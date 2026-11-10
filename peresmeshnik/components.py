"""What the app can download after it is installed, one row each: the Components page of the setup wizard and the tray's
Components… show this list. The installer puts only the app on the computer; nothing here downloads until the user
ticks it and presses Download, and anything installed can be removed again.

The rows:
- the models of the chosen pair (models.toml: speech recognition, translation, the stock voice, Silero VAD, the speaker
  vector that confirms a voice as mine) and the models of other pairs still on disk (to remove);
- the clone engines that speak the other person's language (`role = "clone"`: Kyutai Pocket on any CPU, its weights
  behind the user's Hugging Face token; XTTS-v2 on an NVIDIA GPU);
- the NVIDIA voice pack (Windows: its own Python with torch, local/voicepack.py);
- the pack's addon "Qwen3 + Chatterbox" (NVIDIA only, installed into the same pack only when ticked). The addon is a
  section `addon` of pack-windows.json ({wheels, archives, installed_bytes, models, vram_gb}) installed by
  `voicepack.install(..., addon=True)` with `voicepack.state(addon=True)` and removed by
  `voicepack.uninstall(addon=True)` (S1); without that section there is no row.

    from peresmeshnik import components
    for c in components.listing(settings):   # ticked by default: c.recommended
        ...
    c.fetch(progress, cancel)                 # only after Download
    c.remove()                                # installed ones: its folder goes
"""
from __future__ import annotations

import json
import os
import shutil
import sys
import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from peresmeshnik import languages, models
from peresmeshnik.languages import Gpu
from peresmeshnik.local import voicepack

VOICE_PACK = 'voice-pack'
ADDON = 'voice-pack-addon'
ARROW = {'them_to_me': 'them → me', 'me_to_them': 'me → them'}
ROLE = {'vad': 'Speech detector', 'speaker': 'Voice check', 'asr': 'Speech recognition', 'mt': 'Translation',
        'tts': 'Stock voice', 'clone': 'Your cloned voice'}
PACK_LICENSE = 'BSD-3-Clause (PyTorch) + NVIDIA CUDA terms + MPL-2.0 (coqui-tts)'
PACK_URL = 'https://docs.nvidia.com/cuda/eula/'


@dataclass
class Component:
    id: str  # a registry id (models.toml), VOICE_PACK or ADDON
    title: str  # what the user reads: "Speech recognition: gigaam-v3-e2e-ctc"
    size: int  # bytes to download
    license: str  # 'MIT', 'CC-BY-4.0 + Apache-2.0' (both bind)
    license_url: str
    why: str  # what it does for this user, one line
    installed: bool
    on_disk: int  # bytes its folder takes now (0: nothing there)
    recommended: bool  # ticked by default: the chosen pair runs on it on this computer
    instead: str = ''  # not installed: what the pair does without it ("a stock voice speaks instead")
    unavailable: str = ''  # why it is not offered on this computer ('' = it is): no NVIDIA GPU, too little GPU memory
    consent: bool = False  # a restricted license (noncommercial, territory): accepted before the download
    token: bool = False  # a gated Hugging Face repo: the download needs the user's token (models.hf_token)
    note: str = ''  # the registry's note: authors, what the license asks
    _fetch: Callable = field(default=None, repr=False)
    _remove: Callable = field(default=None, repr=False)

    def fetch(self, progress=lambda stage, done, total, text='': None, cancel: threading.Event | None = None) -> None:
        """Download and verify it (resumes what stopped): `progress` as voicepack.install's (stage, done bytes, total
        bytes, text); `cancel` is checked between files (models) or as the pack goes."""
        self._fetch(progress, cancel or threading.Event())

    def remove(self) -> None:
        """Delete it from disk; a pair that needs it says so, and the wizard downloads it again."""
        if self._remove:
            self._remove()


def du(p: Path) -> int:
    return sum(f.stat().st_size for f in p.rglob('*') if f.is_file()) if p.is_dir() else 0


def _rmtree(folder: Path) -> None:
    """The marker first: a model whose files the running app still holds (Windows) is "missing", not half there."""
    (folder / models.MARKER).unlink(missing_ok=True)
    shutil.rmtree(folder, ignore_errors=True)


def _model(m: models.Model, base: Path, recommended: bool, why: str, instead: str = '', unavailable: str = '') -> Component:
    def fetch(progress, cancel):
        done = [0]

        def step(n):
            done[0] += n
            progress('downloading', done[0], m.size, m.id)
        if not cancel.is_set():
            models.fetch(m.id, base, step)
    return Component(m.id, f'{ROLE.get(m.role, m.role)}: {m.id}', 0 if models.is_fetched(m, base) else m.size, m.license,
                     m.license_url, why, models.is_fetched(m, base), du(base / m.id), recommended and not unavailable,
                     instead, unavailable, m.consent, getattr(m, 'token', '') == 'hf', m.note, fetch,
                     lambda: _rmtree(base / m.id))


def _uses(s) -> dict[str, tuple[str, str]]:
    """What the chosen pair runs each model on, per part: id -> (why, instead)."""
    reg = models.registry()
    use = {'silero-vad': ('tells speech from silence: every setup needs it', 'nothing runs')}
    for side in languages.SIDES:
        d, arrow, engine = languages.direction(s, side), ARROW[side], getattr(s, side)  # as needed_models asks it
        if not (engine == 'local' or engine == 'cloud' and not languages.cloud_subtitles(s, side)):  # remote: the server's
            continue
        lang, cloud = languages.name(d.tgt), engine == 'cloud'
        away = 'the cloud makes the subtitles instead' if cloud else \
            f'{arrow} does not run here' + (' (the cloud can, with your Gemini key)' if d.cloud else '')
        for i, why in ((d.asr, f'{arrow}: hears {languages.name(d.src)}' + (' (the subtitles)' if cloud else '')),
                       (d.mt, f'{arrow}: translates {languages.name(d.src)} → {lang}' + (' (the subtitles)' if cloud else ''))):
            for p in models.parts(reg[i]) if i in reg else []:  # '' (none), or not a registry row: nothing to list
                use.setdefault(p.id, (why, away))
        stock = languages.fallback_voice(s, side) if cloud else languages.pick(s.local.tts, d.tgt)
        if stock:
            use.setdefault(stock, (f'{arrow}: speaks {lang} when the cloud fails', 'that side stops while the cloud is down')
                           if cloud else (f'{arrow}: speaks {lang} in a stock voice', 'subtitles only')
                           if side == 'them_to_me' else (f'{arrow}: takes over in a stock voice if your cloned voice fails',
                                                         'nothing takes over if your cloned voice fails')
                           if d.voice == 'clone' else (f'{arrow}: speaks {lang} in a stock voice',
                                                       'nothing speaks for you: choose cloud for me → them'))
        if d.voice == 'clone' and not cloud:
            use.setdefault(d.tts, (f'{arrow}: speaks {lang} in your own voice',
                                   'a stock voice speaks instead' if stock else 'nothing speaks: choose cloud for me → them'))
            use.setdefault('wespeaker-resnet34', ('confirms at the microphone that the voice is yours',
                                                  'your voice cannot be confirmed: a stock voice speaks into the call'))
    return use


def _pack(s, gpu: Gpu, needed: set[str]) -> list[Component]:
    """The voice pack and its addon: on Windows, with the pack's files in this build, without a runner of one's own."""
    if sys.platform != 'win32' or os.environ.get('PERESMESHNIK_VOICE_CMD') or not voicepack.lock_file().is_file():
        return []
    spec = json.loads(voicepack.lock_file().read_text())
    no = 'needs an NVIDIA GPU' if not (gpu.vram or gpu.why) else languages.gpu_why(gpu)
    st, gpu_clone = voicepack.state(), any(languages.needs_gpu(i) for i in needed if models.get(i).role == 'clone')
    dl = sum(f['size'] for f in [spec['python'], spec['pip'], *spec['wheels'], *spec.get('archives', [])])
    folder = voicepack.folder()
    out = [Component(VOICE_PACK, 'NVIDIA voice pack (XTTS-v2)', 0 if st == 'ready' else dl, PACK_LICENSE, PACK_URL,
                     'your own voice on an NVIDIA GPU: its own Python with torch', st == 'ready',
                     (spec['installed_bytes'] if st == 'ready' else du(folder))  # not a walk over ~8 GB of torch each time
                     + du(folder.with_name('voice-pack.download')), gpu_clone and not no,
                     'a stock voice speaks instead of yours' if gpu_clone else '', no, True, False,
                     lambda progress, cancel: voicepack.install(progress, cancel),
                     lambda: [shutil.rmtree(p, ignore_errors=True) for p in (folder, folder.with_name('voice-pack.download'))])]
    if add := spec.get('addon'):
        low = f'the addon needs {add["vram_gb"]} GB of GPU memory; this computer has {languages.gpu_text(gpu.vram)}' \
            if gpu.vram < add.get('vram_gb', 0) else ''
        ms, base = [models.get(i) for i in add['models']], models.root()
        ready = voicepack.state(addon=True) == 'ready' and all(models.is_fetched(m, base) for m in ms)

        def fetch(progress, cancel):  # into the same pack, then its models (their rows are this one)
            voicepack.install(progress, cancel, addon=True)
            for m in ms:
                if not cancel.is_set():
                    _model(m, base, False, '').fetch(progress, cancel)

        def remove():
            voicepack.uninstall(addon=True)
            for m in ms:
                _rmtree(base / m.id)
        size = sum(f['size'] for f in [*add['wheels'], *add.get('archives', [])]) + sum(m.size for m in ms)
        out.append(Component(ADDON, 'Qwen3 + Chatterbox (addon to the voice pack)', 0 if ready else size,
                             ' + '.join(dict.fromkeys(models.get(i).license for i in add['models'])),
                             models.get(add['models'][0]).license_url,
                             'a faster, closer clone of your voice on an NVIDIA GPU', ready, 0, False, '', no or low, True,
                             False, fetch, remove))
    return out


def listing(s, gpu: Gpu | None = None) -> list[Component]:
    """Every component for settings `s` on this computer (`gpu`: None looks once, nvidia-smi): the chosen pair's first,
    then the clone engines and the voice pack, then what other pairs left on disk."""
    base, reg = models.root(), models.registry()
    gpu = gpu if gpu is not None else languages.nvidia_once() if languages.gpu() else Gpu()
    needed = set(languages.needed_models(s))
    use = _uses(s)
    out = {}
    for i in [*languages.needed_models(s), *languages.optional_models(s)]:
        for p in models.parts(models.get(i)):
            why, instead = use.get(p.id, ('a cloud direction falls back on it when the cloud fails',
                                          'that side stops while the cloud is down'))
            out.setdefault(p.id, _model(p, base, i in needed, why, instead))
    if (i := 'wespeaker-resnet34') in use and i not in out:  # not a pair's model, but the clone speaks only after it
        out[i] = _model(reg[i], base, True, *use[i])
    for m in reg.values():
        if m.role == 'clone' and m.id not in out and languages.covers(m.id, s.their_lang):
            out[m.id] = _model(m, base, False, f'speaks {languages.name(s.their_lang)} in your own voice',
                               unavailable=languages.clone_why(m.id, s.their_lang, gpu))
    pack = _pack(s, gpu, needed)
    with_addon = set(json.loads(voicepack.lock_file().read_text()).get('addon', {}).get('models', ())) \
        if any(c.id == ADDON for c in pack) else set()
    rows = [c for c in out.values() if c.id not in with_addon] + pack  # the addon's models come with it
    for m in reg.values():  # what other pairs left: to remove
        if not m.via and m.id not in out and (base / m.id).is_dir():
            rows.append(_model(m, base, False, 'not used by the chosen pair'))
    return rows
