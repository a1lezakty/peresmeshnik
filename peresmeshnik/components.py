"""What the app can download after it is installed, one row each: the Components page of the setup wizard and the tray's
Components… show this list. The installer puts only the app on the computer; nothing here downloads until the user
ticks it and presses Download, and anything installed can be removed again.

The rows:
- the models of the chosen pair (models.toml: speech recognition, translation, the stock voice, Silero VAD, the speaker
  vector that confirms a voice as mine) and the models of other pairs still on disk (to remove);
- the clone engines that speak my language (`role = "clone"`: Kyutai Pocket on any CPU, its weights behind the user's
  Hugging Face token; XTTS-v2 on an NVIDIA GPU);
- the NVIDIA voice pack (Windows: its own Python with torch, local/voicepack.py);
- the pack's addon "Qwen3 + Chatterbox" (NVIDIA only, installed into the same pack only when ticked). The addon is a
  section `addon` of pack-windows.json ({wheels, archives, installed_bytes, models, vram_gb}) installed by
  `voicepack.install(..., addon=True)` with `voicepack.state(addon=True)` (S1); without that section there is no row.

    from peresmeshnik import components
    for c in components.listing(settings):   # ticked by default: c.recommended
        ...
    c.fetch(progress, cancel)                 # only after Download
    c.remove()                                # installed ones: its folder goes
"""
from __future__ import annotations

import threading
from collections.abc import Callable
from dataclasses import dataclass, field

from peresmeshnik.languages import Gpu

VOICE_PACK = 'voice-pack'
ADDON = 'voice-pack-addon'


@dataclass
class Component:
    id: str  # a registry id (models.toml), VOICE_PACK or ADDON
    title: str  # what the user reads: "Speech recognition: Russian (GigaAM v3)"
    size: int  # bytes to download: what is not here yet
    license: str  # 'MIT', 'CC-BY-4.0 + Apache-2.0' (both bind)
    license_url: str
    why: str  # what it does for this user, one line
    installed: bool
    on_disk: int  # bytes its folder takes now (0: not installed)
    recommended: bool  # ticked by default: the chosen pair runs on it on this computer
    instead: str = ''  # not installed: what the pair does without it ("a stock voice speaks", "the cloud translates")
    unavailable: str = ''  # why it is not offered on this computer ('' = it is): no NVIDIA GPU, too little GPU memory
    consent: bool = False  # a restricted license (noncommercial, territory): accepted before the download
    token: bool = False  # a gated Hugging Face repo: the download needs the user's token (models.hf_token)
    _fetch: Callable = field(default=None, repr=False)
    _remove: Callable = field(default=None, repr=False)

    def fetch(self, progress: Callable[[int], None] = lambda n: None, cancel: threading.Event | None = None) -> None:
        """Download and verify it (resumes what stopped); `progress` gets bytes as they come."""
        self._fetch(progress, cancel or threading.Event())

    def remove(self) -> None:
        """Delete it from disk; the pair that needs it says so the next time (and the wizard downloads it again)."""
        self._remove()


def listing(s, gpu: Gpu | None = None) -> list[Component]:
    """Every component for settings `s` on this computer (`gpu`: None looks once, nvidia-smi), the chosen pair's first."""
    raise NotImplementedError
