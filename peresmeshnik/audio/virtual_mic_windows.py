"""Windows virtual microphone: VB-Audio Virtual Cable (the user installs it; the call app takes "CABLE Output" as its
microphone).

    the app -> peresmeshnik-audio play -> CABLE Input == CABLE Output -> the call app

The same `VirtualMic` as on Linux and macOS (one playback process from the first play() until stop() / drain(), fed
by a writer thread, so play() never blocks), with the helper in place of pacat and nothing to create or remove.
"""
from __future__ import annotations

import subprocess

from peresmeshnik.audio import windows
from peresmeshnik.audio.virtual_mic_linux import VirtualMic as _VirtualMic

NAME = windows.CABLE_IN


class VirtualMic(_VirtualMic):
    """AudioOut into CABLE Input; CableMissing (with how to install it) when VB-Cable is not there. `call_mic` is the
    endpoint the call app records (CABLE Output)."""
    to_call = True  # what plays here goes into the call (app.Side: never the other person's cloned voice)
    name = NAME
    reopen = (7,)  # the helper ends with 7 when Windows resets the stream (a format change of the cable): play again

    def _make(self) -> list[str]:
        self.device, self.call_mic = windows.cable()
        return []  # nothing of ours to remove at exit

    def _open(self, sample_rate: int) -> None:
        # --latency: as pacat's: a cushion in the player, so a late frame from the microphone's clock is no gap
        self._proc = windows.spawn('play', '--to', self.device['uid'], '--rate', str(sample_rate), '--latency', str(self.latency_ms),
                                   stdout=subprocess.DEVNULL)
        self._sr = sample_rate
