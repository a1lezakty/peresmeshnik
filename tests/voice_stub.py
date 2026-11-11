"""A stand-in for peresmeshnik/local/xtts.py inside voice-runner/voice_runner.py (tests/test_voice_pack.py): no torch,
no model. Only the runner's hidden --engine option loads it; nothing in the app passes that.

A synth chunk holds len(text) / 100 in every sample, so a test sees whose audio it got; in a voice of condition() (its
state: the sample's mean) it holds that mean. Texts: 'slow' streams 100
chunks 50 ms apart, 'boom' fails like CUDA out of memory, 'die' kills the runner like a driver crash, 'hang' sends
nothing for a minute like a hung CUDA call."""
import os
import time
from pathlib import Path

import numpy as np


def cuda_info() -> dict:
    return {'torch': 'stub', 'cuda': False}


def cap_vram(gb: float) -> None:
    pass


class Engine:
    sample_rate = 24_000

    def __init__(self, model_dir, device=None, threads=2):
        if Path(model_dir).name == 'broken':
            raise FileNotFoundError(f'{model_dir}/model.pth')
        self.conditioned: list[str] = []
        self.count = 0

    def synth(self, text, lang, voice):
        if text == 'boom':
            raise RuntimeError('CUDA out of memory')
        if text == 'die':
            os._exit(3)
        if text == 'hang':
            time.sleep(60)
        n, pause = (100, 0.05) if text == 'slow' else (3, 0.0)
        for _ in range(n):
            time.sleep(pause)
            yield np.full(2400, len(text) / 100 if getattr(voice, 'state', None) is None else voice.state, np.float32)

    def condition(self, pcm, sr):
        return float(np.mean(pcm))

    def latents(self, voice):  # as xtts.Engine: cached in memory, written as xtts.pt (the count of conditionings)
        if str(voice) not in self.conditioned:
            self.conditioned.append(str(voice))
            self.count += 1
            (Path(voice) / 'xtts.pt').write_text(str(self.count))

    def forget(self, voice):
        if str(voice) in self.conditioned:
            self.conditioned.remove(str(voice))

    def similarity(self, voice, pcm, sr):
        return float(np.sum(pcm)) / sr
