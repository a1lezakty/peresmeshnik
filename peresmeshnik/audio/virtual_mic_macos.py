"""macOS virtual microphone: BlackHole 2ch (the user installs it; the call app takes it as its microphone).

    the app -> peresmeshnik-audio play -> BlackHole 2ch output == BlackHole 2ch input -> the call app

The same `VirtualMic` as on Linux (one playback process from the first play() until stop() / drain(), fed by a
writer thread, so play() never blocks), with the helper in place of pacat and nothing to create or remove.
"""
from __future__ import annotations

import subprocess

from peresmeshnik.audio import macos
from peresmeshnik.audio.virtual_mic_linux import VirtualMic as _VirtualMic

NAME = macos.BLACKHOLE


class VirtualMic(_VirtualMic):
    """AudioOut into BlackHole 2ch; BlackHoleMissing (with how to install it) when it is not there."""

    def _make(self) -> list[str]:
        self.device = macos.blackhole()
        return []  # nothing of ours to remove at exit

    @staticmethod
    def _stop_proc(p: subprocess.Popen) -> None:
        """SIGTERM, not SIGKILL: the helper stops its IOProc first. Killed with it running, its last 10 ms buffer may
        stay in coreaudiod's mix and loop in BlackHole 2ch, the call hearing it (seen on the CI runner)."""
        p.terminate()
        try:
            p.wait(2)
        except subprocess.TimeoutExpired:
            p.kill()
            p.wait()

    def _open(self, sample_rate: int) -> None:
        # --latency: as pacat's: a cushion in the player, so a late frame from the microphone's clock is no gap
        self._proc = macos.spawn('play', '--to', self.device['uid'], '--rate', str(sample_rate), '--latency', str(self.latency_ms),
                                 stdout=subprocess.DEVNULL)
        self._sr = sample_rate
