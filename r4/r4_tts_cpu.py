"""R4: CPU time-to-first-audio and RTF of light TTS without published numbers (server A, onnxruntime, no torch).

    taskset -c 0-3 python r4_tts_cpu.py ENGINE LANG THREADS   # ENGINE: kitten2mini | supertonic3 | kokororu
Same 10 shortest bench sentences as BENCH.md "TTS without cloning". None of these stream inside a sentence,
so first audio = synthesis of the whole sentence. One warm-up sentence. Prints one JSON line.
"""
import json, os, resource, sys, time
from pathlib import Path

import numpy as np
import soundfile as sf

eng, lang, threads = sys.argv[1], sys.argv[2], int(sys.argv[3])
HERE = Path(__file__).resolve().parent
sents = json.loads((HERE / 'sents.json').read_text())[lang]
out = HERE / 'wav' / f'{eng}-{lang}'
out.mkdir(parents=True, exist_ok=True)

t0 = time.perf_counter()
if eng == 'kitten2mini':
    from kittenml import KittenTTS
    m = KittenTTS('KittenML/kitten-tts-2-mini')
    m.threads = threads  # sessions are created lazily, so this sets intra_op_num_threads
    voice = m.available_voices[lang][0]
    sr = 24000
    def synth(t):
        return np.asarray(m.generate(t, voice=voice, language=lang)).reshape(-1)
elif eng == 'supertonic3':
    from supertonic import TTS
    md = HERE / 'models/supertonic3'  # server A keeps it here; elsewhere the SDK downloads it
    kw = dict(model_dir=md, auto_download=False) if md.exists() else {}
    m = TTS(model='supertonic-3', intra_op_num_threads=threads, inter_op_num_threads=1, **kw)
    voice = 'F1'
    style = m.get_voice_style(voice_name=voice)
    sr = m.sample_rate if hasattr(m, 'sample_rate') else 44100
    def synth(t):
        return np.asarray(m.synthesize(t, voice_style=style, lang=lang)[0]).reshape(-1)
elif eng == 'kokororu':
    import onnxruntime as ort
    from huggingface_hub import snapshot_download
    d = Path(snapshot_download('zaakirio/kokoro-ru', allow_patterns=[
        'config.json', 'ru_g2p.py', 'onnx/model_quantized.onnx', 'voices/sveta.bin', 'espeak-data/*', 'espeak-data/**']))
    sys.path.insert(0, str(d))
    from ru_g2p import RuG2P
    g2p = RuG2P(espeak_data=d / 'espeak-data')  # snapshot files are links into blobs/, so pass the folder
    vocab = json.loads((d / 'config.json').read_text())['vocab']
    so = ort.SessionOptions(); so.intra_op_num_threads = threads; so.inter_op_num_threads = 1
    sess = ort.InferenceSession(str(d / 'onnx/model_quantized.onnx'), so, providers=['CPUExecutionProvider'])
    style = np.fromfile(d / 'voices/sveta.bin', dtype=np.float32).reshape(510, 256)
    voice, sr = 'sveta q8', 24000
    g2p_s = []
    def synth(t):
        a = time.perf_counter()
        ipa, _ = g2p(t)
        g2p_s.append(time.perf_counter() - a)
        ids = [vocab[c] for c in ipa if c in vocab][:510]
        return sess.run(None, dict(input_ids=np.array([[0, *ids, 0]], np.int64),
                                   style=style[len(ids) - 1][None], speed=np.ones(1, np.float32)))[0].reshape(-1)
elif eng == 'piper':  # anchor: the voices BENCH.md measured on macos-15 and vast-a-cpu4 (ORT default threads)
    from huggingface_hub import hf_hub_download
    from piper import PiperVoice
    name = {'en': 'en/en_US/lessac/medium/en_US-lessac-medium', 'ru': 'ru/ru_RU/irina/medium/ru_RU-irina-medium'}[lang]
    onnx = hf_hub_download('rhasspy/piper-voices', name + '.onnx'); hf_hub_download('rhasspy/piper-voices', name + '.onnx.json')
    m = PiperVoice.load(onnx)
    voice, sr = name.rsplit('/', 1)[1], m.config.sample_rate
    def synth(t):
        return np.concatenate([c.audio_float_array for c in m.synthesize(t)])
else:
    raise SystemExit(f'unknown engine {eng}')
synth(sents[0])  # warm-up
load_s = time.perf_counter() - t0

ttfa, dur = [], []
def throttled():
    try:
        return int(dict(l.split() for l in open('/sys/fs/cgroup/cpu.stat'))['throttled_usec'])
    except Exception:
        return -1
thr0 = throttled()
cpu0 = time.process_time(); wall0 = time.perf_counter()
for i, t in enumerate(sents):
    a = time.perf_counter()
    wav = synth(t)
    ttfa.append(time.perf_counter() - a)
    dur.append(len(wav) / sr)
    sf.write(out / f'{i:02d}.wav', wav, sr)
wall = time.perf_counter() - wall0
res = dict(engine=eng, lang=lang, voice=voice, threads=threads, n=len(sents),
           ttfa_ms_median=round(1000 * float(np.median(ttfa))), ttfa_ms_max=round(1000 * max(ttfa)),
           ttfa_ms_short=round(1000 * ttfa[0]), rtf=round(sum(ttfa) / sum(dur), 4),
           cpu_cores=round((time.process_time() - cpu0) / wall, 2),
           peak_rss_mb=round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 * 1024 if sys.platform == 'darwin' else 1024)), load_s=round(load_s, 1),
           audio_s=round(sum(dur), 1), sr=sr, loadavg1=os.getloadavg()[0],
           cgroup_throttled_ms=round((throttled() - thr0) / 1000))
if eng == 'kokororu':
    res['g2p_ms_median'] = round(1000 * float(np.median(g2p_s[1:])))
print(json.dumps(res, ensure_ascii=False))
