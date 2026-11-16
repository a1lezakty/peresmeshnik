"""R4: intelligibility of r4_tts_cpu.py's wavs, as BENCH "TTS without cloning": faster-whisper small int8, beam 1,
CER against the input text after tts_clone.norm. Prints one JSON line per wav folder.

    python r4/cer.py wav/kitten2mini-ru wav/supertonic3-ru ...
"""
import json, sys
from pathlib import Path

import numpy as np
import soundfile as sf
from faster_whisper import WhisperModel

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'bench'))
from tts_clone import edits, norm  # noqa: E402

HERE = Path(__file__).resolve().parent
sents = json.loads((HERE / 'sents.json').read_text())
m = WhisperModel('small', device='cpu', compute_type='int8', cpu_threads=4)
for d in map(Path, sys.argv[1:]):
    lang = d.name.rsplit('-', 1)[1]
    e = n = 0
    hyps = []
    for i, text in enumerate(sents[lang]):
        f = d / f'{i:02d}.wav'
        if not f.exists():
            continue
        x, sr = sf.read(f, dtype='float32')
        x = np.interp(np.arange(0, len(x), sr / 16000), np.arange(len(x)), x).astype(np.float32)
        hyp = norm(' '.join(s.text for s in m.transcribe(x, language=lang, beam_size=1)[0]))
        ref = norm(text)
        e += edits(ref, hyp); n += len(ref); hyps.append(hyp)
    print(json.dumps(dict(dir=d.name, lang=lang, judge='whisper-small-int8', cer=round(e / max(n, 1), 4),
                          n=len(hyps), sample=hyps[0][:80] if hyps else ''), ensure_ascii=False), flush=True)
