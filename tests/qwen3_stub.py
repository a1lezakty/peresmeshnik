"""Stand-ins for torch, faster_qwen3_tts and chatterbox under peresmeshnik/local/qwen3.py (tests/test_qwen3.py): no GPU,
no weights. `fakes()` are the modules to put in sys.modules; run as a script, this puts them there and runs the voice
pack's runner with the rest of the arguments (`python tests/qwen3_stub.py --kind qwen3 ...`).

Qwen3 says 0.1 in every sample, Chatterbox 0.3, so a test hears whose audio it got. Qwen3 streams 12 chunks of 2000
samples (1 s), and never stops on a text with 'loop' in it (a runaway); Chatterbox says 1 s at once."""
import pickle
import sys
import types
from pathlib import Path

import numpy as np

QWEN, CHATTERBOX, CHUNK = 0.1, 0.3, 2000


class FakeTensor:
    def __init__(self, x):
        self.x = np.asarray(x, np.float32)

    def to(self, *a, **kw):
        return self

    def cpu(self):
        return self

    def float(self):
        return self

    def numpy(self):
        return self.x

    def squeeze(self, dim):
        return FakeTensor(self.x.squeeze(dim))


class Base:  # qwen_tts' Qwen3TTSModel: the speaker vector
    def __init__(self):
        self.vectors: list[float] = []  # seconds of audio of each one computed

    def create_voice_clone_prompt(self, ref_audio, ref_text, x_vector_only_mode):
        assert x_vector_only_mode and ref_text == ''
        pcm, sr = ref_audio
        self.vectors.append(len(pcm) / sr)
        return [types.SimpleNamespace(ref_spk_embedding=FakeTensor([float(np.mean(np.abs(pcm))), 1.0]))]


class FasterQwen3TTS:
    LANGS = {'English', 'Russian', 'German', 'French', 'Spanish', 'Italian', 'Portuguese', 'Chinese', 'Japanese', 'Korean'}

    def __init__(self, path):
        self.path, self.model, self.calls, self.chunks, self.closed = path, Base(), [], 0, 0

    @classmethod
    def from_pretrained(cls, model_name, device='cuda'):
        return cls(model_name)

    def warmup(self):
        pass

    def generate_voice_clone_streaming(self, text, language, chunk_size, voice_clone_prompt, **kw):
        assert language in self.LANGS, language
        self.calls.append((text, language, voice_clone_prompt['ref_spk_embedding'][0]))
        try:
            n = 0
            while n < 12 or 'loop' in text:
                n += 1
                self.chunks += 1
                yield np.full(CHUNK, QWEN, np.float32), 24_000, {}
        finally:
            self.closed += 1


class ChatterboxMultilingualTTS:
    sr = 24_000

    def __init__(self, ckpt_dir, device):
        self.ckpt_dir, self.device, self.conds, self.said = ckpt_dir, device, None, []

    @classmethod
    def from_local(cls, ckpt_dir, device):
        if not Path(ckpt_dir).is_dir():
            raise FileNotFoundError(f'{ckpt_dir}/ve.pt')
        return cls(ckpt_dir, device)

    def prepare_conditionals(self, wav_fpath, exaggeration=0.5):  # a file in memory: ('sample', its seconds, its level)
        if isinstance(wav_fpath, str):
            self.conds = wav_fpath
        else:
            import soundfile as sf
            x, sr = sf.read(wav_fpath, dtype='float32')
            self.conds = ('sample', len(x) / sr, round(float(np.mean(x)), 3))

    def generate(self, text, language_id, cfg_weight=0.5, **kw):
        self.said.append((text, language_id, self.conds, cfg_weight))
        return FakeTensor(np.full((1, 24_000), CHATTERBOX, np.float32))


def fakes() -> dict:
    torch = types.ModuleType('torch')
    torch.__version__ = 'stub'
    torch.set_num_threads = torch.set_num_interop_threads = lambda n: None
    torch.save = lambda obj, f: Path(f).write_bytes(pickle.dumps(obj))
    def load(f, map_location=None, weights_only=False):
        assert weights_only, 'a cache next to the recording loads tensors only'
        return pickle.loads(Path(f).read_bytes())
    torch.load = load
    torch.cuda = types.SimpleNamespace(is_available=lambda: False)
    fq = types.ModuleType('faster_qwen3_tts')
    fq.FasterQwen3TTS = FasterQwen3TTS
    cb, mtl = types.ModuleType('chatterbox'), types.ModuleType('chatterbox.mtl_tts')
    mtl.ChatterboxMultilingualTTS = ChatterboxMultilingualTTS
    cb.mtl_tts = mtl
    return {'torch': torch, 'faster_qwen3_tts': fq, 'chatterbox': cb, 'chatterbox.mtl_tts': mtl}


if __name__ == '__main__':
    import runpy

    import qwen3_stub  # this file as a module: torch.save pickles its FakeTensor, and the runner replaces __main__
    sys.modules.update(qwen3_stub.fakes())
    runner = str(Path(__file__).resolve().parents[1] / 'voice-runner' / 'voice_runner.py')
    sys.argv = [runner, *sys.argv[1:]]
    runpy.run_path(runner, run_name='__main__')
