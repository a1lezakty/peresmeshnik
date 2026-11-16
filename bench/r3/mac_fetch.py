"""GitHub macOS runner: 30 clips each of Golos crowd, CV17-ru, LibriSpeech clean (no ffmpeg: soundfile + soxr),
the latency list, and the R3 models. Writes data/ and models/ next to bench.py."""
import json, os, sys, urllib.parse, urllib.request, io, time

import numpy as np
import soundfile as sf
import soxr

API = 'https://datasets-server.huggingface.co'
SETS = {'golos_crowd': ('bond005/sberdevices_golos_10h_crowd', 'default', 'test', 'transcription', 'ru'),
        'cv17_ru': ('fixie-ai/common_voice_17_0', 'ru', 'test', 'sentence', 'ru'),
        'ls_clean': ('openslr/librispeech_asr', 'clean', 'test', 'text', 'en')}


def get(url):
    for i in range(5):
        try:
            with urllib.request.urlopen(url, timeout=60) as r:
                return r.read()
        except Exception as e:
            print('retry', e, file=sys.stderr)
            time.sleep(3 * (i + 1))
    raise RuntimeError(url)


os.makedirs('data', exist_ok=True)
man = open('data/manifest.jsonl', 'w')
for name, (ds, cfg, split, field, lang) in SETS.items():
    os.makedirs(f'data/{name}', exist_ok=True)
    q = urllib.parse.urlencode({'dataset': ds, 'config': cfg, 'split': split})
    k = 0
    for off in (7, 1007, 2007):
        for r in json.loads(get(f'{API}/rows?{q}&offset={off}&length=10'))['rows']:
            row = r['row']
            if not row[field]:
                continue
            src = row['audio'][0]['src'] if isinstance(row['audio'], list) else row['audio']['src']
            a, sr = sf.read(io.BytesIO(get(src)), dtype='float32')
            a = a.mean(1) if a.ndim > 1 else a
            a = soxr.resample(a, sr, 16000) if sr != 16000 else a
            wav = f'data/{name}/{k:03d}.wav'
            sf.write(wav, a, 16000, subtype='PCM_16')
            man.write(json.dumps({'set': name, 'id': f'{name}/{k:03d}', 'wav': wav, 'text': row[field],
                                  'dur': round(len(a) / 16000, 3), 'lang': lang}, ensure_ascii=False) + '\n')
            k += 1
    print(name, k, flush=True)
man.close()
rows = [json.loads(l) for l in open('data/manifest.jsonl')]
with open('data/lat_ids.txt', 'w') as f:
    for s, n, lang in (('golos_crowd', 8, 'ru'), ('cv17_ru', 7, 'ru'), ('ls_clean', 10, 'en')):
        for r in [r for r in rows if r['set'] == s and 3.0 <= r['dur'] <= 5.5][:n]:
            f.write(f"{r['id']} {lang}\n")

import onnx_asr
from huggingface_hub import snapshot_download
for name in ('gigaam-v3-e2e-ctc', 'gigaam-v3-e2e-rnnt', 'gigaam-multilingual-large-ctc', 'nemo-parakeet-tdt-0.6b-v3',
             'alphacep/vosk-model-ru'):
    onnx_asr.load_model(name, 'models/' + name.replace('/', '__'), quantization='int8')
snapshot_download('t-tech/T-one', local_dir='models/t-one', allow_patterns=['model.onnx', '*.json', '*.yaml', '*.txt'])
snapshot_download('pantinor/nemotron-3.5-asr-streaming-multilingual-0.6b-int8-sherpa', local_dir='models/nemotron-stream-int8-sherpa')
snapshot_download('pisar/nemotron-unified-asr-0.6b-sherpa-onnx', local_dir='models/nemotron-unified-sherpa')
print('models ok', flush=True)
