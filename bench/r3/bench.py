"""R3 STT bench on server A: WER per set x condition, RTF on N CPU threads, app-style pseudo-streaming latency.

python bench.py MODEL_KEY [--threads 4] [--conds clean,opus64,opus16,cs] [--lat]
Writes results/<key>.jsonl (one row per clip) and results/<key>.lat.jsonl. Scoring: score.py.
"""
import argparse, json, os, sys, time

import numpy as np
import soundfile as sf

# key: (onnx-asr model type, local dir, quantization, languages, takes `language`)
MODELS = {
    'gigaam-v3-e2e-ctc': ('gigaam-v3-e2e-ctc', 'models/gigaam-v3-e2e-ctc', 'int8', {'ru'}, False),
    'gigaam-v3-e2e-rnnt': ('gigaam-v3-e2e-rnnt', 'models/gigaam-v3-e2e-rnnt', 'int8', {'ru'}, False),
    'gigaam-ml-large-ctc': ('gigaam-multilingual-large-ctc', 'models/gigaam-multilingual-large-ctc', 'int8', {'ru', 'en'}, False),
    'gigaam-ml-ctc': ('gigaam-multilingual-ctc', 'models/gigaam-multilingual-ctc', 'int8', {'ru', 'en'}, False),
    'parakeet-tdt-0.6b-v3': ('nemo-parakeet-tdt-0.6b-v3', 'models/nemo-parakeet-tdt-0.6b-v3', 'int8', {'ru', 'en'}, False),
    'canary-1b-v2': ('nemo-canary-1b-v2', 'models/nemo-canary-1b-v2', 'int8', {'ru', 'en'}, True),
    'fastconformer-ru-rnnt': ('nemo-fastconformer-ru-rnnt', 'models/nemo-fastconformer-ru-rnnt', 'int8', {'ru'}, False),
    'vosk-ru': ('alphacep/vosk-model-ru', 'models/alphacep__vosk-model-ru', 'int8', {'ru'}, False),
    'vosk-small-ru': ('alphacep/vosk-model-small-ru', 'models/alphacep__vosk-model-small-ru', 'int8', {'ru'}, False),
    't-one': ('t-one-ctc', 'models/t-one', None, {'ru'}, False),
    'nemotron-stream-0.6b': ('sherpa-online', 'models/nemotron-stream-int8-sherpa/', None, {'ru', 'en'}, False),  # auto-LID
    'nemotron-unified-0.6b': ('sherpa', 'models/nemotron-unified-sherpa/', None, {'ru', 'en'}, False),  # offline, language auto
    'qwen3-asr-1.7b': ('qwen', 'models/qwen3-asr-1.7b', None, {'ru', 'en'}, True),  # GPU bf16, transformers backend
    'qwen3-asr-1.7b-auto': ('qwen', 'models/qwen3-asr-1.7b', None, {'ru', 'en'}, False),  # same, language detected
}


class Sherpa:
    def __init__(self, d, threads):
        import sherpa_onnx
        self.r = sherpa_onnx.OfflineRecognizer.from_transducer(
            tokens=d + 'tokens.txt', encoder=d + 'encoder.onnx', decoder=d + 'decoder.onnx', joiner=d + 'joiner.onnx',
            num_threads=threads, sample_rate=SR, feature_dim=128, decoding_method='greedy_search', model_type='nemo_transducer')

    def recognize(self, pcm, sample_rate=16000):
        s = self.r.create_stream()
        s.accept_waveform(sample_rate, pcm)
        self.r.decode_stream(s)
        return s.result.text


class SherpaOnline:
    """Native cache-aware streaming (sherpa-onnx OnlineRecognizer): 100 ms blocks in, text as it comes."""
    online = True

    def __init__(self, d, threads):
        import sherpa_onnx
        self.r = sherpa_onnx.OnlineRecognizer.from_transducer(
            tokens=d + 'tokens.txt', encoder=d + 'encoder.onnx', decoder=d + 'decoder.onnx', joiner=d + 'joiner.onnx',
            num_threads=threads, sample_rate=16000, feature_dim=128, decoding_method='greedy_search')

    def recognize(self, pcm, sample_rate=16000):
        s = self.r.create_stream()
        s.accept_waveform(sample_rate, pcm)
        s.accept_waveform(sample_rate, np.zeros(16000, np.float32))
        s.input_finished()
        while self.r.is_ready(s):
            self.r.decode_stream(s)
        return self.r.get_result(s)

    def stream_lat(self, a, on, end, blk=1600):
        """(first text s after onset, last text change s after speech end, text, compute ms per 100 ms block)"""
        s, avail, first, changed, prev, cs = self.r.create_stream(), 0.0, None, None, '', []
        for i in range(0, len(a) + 16000, blk):  # the clip, then 1 s of silence
            x = a[i:i + blk] if i < len(a) else np.zeros(blk, np.float32)
            t0 = time.perf_counter()
            s.accept_waveform(16000, x)
            while self.r.is_ready(s):
                self.r.decode_stream(s)
            c = time.perf_counter() - t0
            cs.append(c)
            avail = max((i + blk) / 16000, avail) + c
            txt = self.r.get_result(s)
            if first is None and txt.strip():
                first = avail - on
            if txt != prev:
                changed, prev = avail, txt
        return first, (changed - end) if changed is not None else None, prev, cs


class Qwen:
    NAMES = {'ru': 'Russian', 'en': 'English'}

    def __init__(self, d):
        import torch
        from qwen_asr import Qwen3ASRModel
        self.m = Qwen3ASRModel.from_pretrained(d, dtype=torch.bfloat16, device_map='cuda:0', max_inference_batch_size=1,
                                               max_new_tokens=256)
        self.torch = torch

    def recognize(self, pcm, sample_rate=16000, language=None):
        r = self.m.transcribe(audio=(pcm, sample_rate), language=self.NAMES.get(language))
        self.torch.cuda.synchronize()
        return r[0].text
SR = 16000
LEAD = np.zeros(SR // 2, np.float32)  # the app puts 0.5 s of silence before a window that starts at a VAD onset
HANGOVER = 0.3  # the app's Silero VAD closes a segment after 300 ms below threshold
STEP = 0.3  # the app re-runs ASR every 300 ms of new audio


def load(key, threads):
    typ, path, q, langs, takes_lang = MODELS[key]
    if typ == 'sherpa':
        return Sherpa(path, threads), langs, takes_lang
    if typ == 'sherpa-online':
        return SherpaOnline(path, threads), langs, takes_lang
    if typ == 'qwen':
        return Qwen(path), langs, takes_lang
    import onnx_asr
    import onnxruntime as ort
    o = ort.SessionOptions()
    o.intra_op_num_threads, o.inter_op_num_threads = threads, 1
    m = onnx_asr.load_model(typ, path, quantization=q, sess_options=o, providers=['CPUExecutionProvider'])
    return m, langs, takes_lang


def read(path):
    a, sr = sf.read(path, dtype='float32')
    assert sr == SR, (path, sr)
    return a


def rec(m, pcm, lang, takes_lang):
    kw = {'language': lang} if takes_lang else {}
    t = time.perf_counter()
    r = m.recognize(pcm, sample_rate=SR, **kw)
    return (r if isinstance(r, str) else getattr(r, 'text', str(r))), time.perf_counter() - t


def bounds(a):
    """Speech onset / end by frame energy (20 ms): first and last frame within 35 dB of the peak."""
    f = a[: len(a) // 320 * 320].reshape(-1, 320)
    db = 10 * np.log10((f ** 2).mean(1) + 1e-10)
    on = np.where(db >= db.max() - 35)[0]  # never empty: the peak frame itself
    return on[0] * 0.02, (on[-1] + 1) * 0.02


def wer_run(key, m, langs, takes_lang, conds, out):
    man = [r for r in (json.loads(l) for l in open('data/manifest.jsonl')) if r['text']]
    cs = [json.loads(l) for l in open('data/cs_manifest.jsonl')] if os.path.exists('data/cs_manifest.jsonl') else []
    with open(out, 'a') as f:
        for cond in conds:
            rows = cs if cond == 'cs' else [r for r in man if r['lang'] in langs]
            for r in rows:
                wav = r['wav'] if cond in ('clean', 'cs') else r['wav'].replace('data/', f'data_{cond}/', 1)
                a = read(wav)
                lang = r.get('lang', 'ru') if cond != 'cs' else ('ru' if 'ru' in langs else 'en')
                hyp, s = rec(m, a, lang, takes_lang)
                f.write(json.dumps({'model': key, 'cond': cond, 'set': r['set'], 'id': r['id'], 'dur': len(a) / SR,
                                    'ref': r['text'], 'hyp': hyp, 'compute_s': round(s, 4), 'lang': r.get('lang')},
                                   ensure_ascii=False) + '\n')
            f.flush()
            print(key, cond, len(rows), 'done', flush=True)


def lat_run(key, m, langs, takes_lang, out):
    ids = [l.split() for l in open('data/lat_ids.txt')]
    man = {json.loads(l)['id']: json.loads(l) for l in open('data/manifest.jsonl')}
    with open(out, 'a') as f:
        for cid, lang in ids:
            if lang not in langs:
                continue
            r = man[cid]
            a = read(r['wav'])
            on, end = bounds(a)
            if getattr(m, 'online', False):
                first, last, txt, cs = m.stream_lat(a, on, end)
                f.write(json.dumps({'model': key, 'id': cid, 'lang': lang, 'speech_s': round(end - on, 2),
                                    'first_partial_s': None if first is None else round(first, 3), 'first_text': '',
                                    'final_after_end_s': None if last is None else round(last, 3), 'final_compute_s': 0.0,
                                    'partial_compute_ms_median': round(1000 * float(np.median(cs)), 1),
                                    'partial_compute_ms_max': round(1000 * max(cs), 1), 'final': txt, 'ref': r['text'],
                                    'native_stream': True}, ensure_ascii=False) + '\n')
                continue
            start = max(0.0, on - 0.2)  # VAD onset with a little pre-roll
            avail, first, first_text, runs, cpu = 0.0, None, '', [], []
            t = on + STEP
            while t < end + HANGOVER - 1e-6:
                p0 = time.process_time()
                hyp, c = rec(m, np.concatenate([LEAD, a[int(start * SR):int(t * SR)]]), lang, takes_lang)
                cpu.append(time.process_time() - p0)  # CPU s of all threads: less sensitive to other tenants than wall
                avail = max(t, avail) + c
                runs.append(c)
                if first is None and hyp.strip():
                    first, first_text = avail - on, hyp
                t += STEP
            tf = end + HANGOVER
            hyp, c = rec(m, np.concatenate([LEAD, a[int(start * SR):min(len(a), int(tf * SR))]]), lang, takes_lang)
            final_lag = max(tf, avail) + c - end
            f.write(json.dumps({'model': key, 'id': cid, 'lang': lang, 'speech_s': round(end - on, 2),
                                'first_partial_s': None if first is None else round(first, 3), 'first_text': first_text,
                                'final_after_end_s': round(final_lag, 3), 'final_compute_s': round(c, 4),
                                'partial_compute_ms_median': round(1000 * float(np.median(runs)), 1) if runs else None,
                                'partial_compute_ms_max': round(1000 * max(runs), 1) if runs else None,
                                'partial_cpu_ms_median': round(1000 * float(np.median(cpu)), 1) if cpu else None,
                                'load1': round(os.getloadavg()[0], 2),
                                'final': hyp, 'ref': r['text']}, ensure_ascii=False) + '\n')
    print(key, 'lat done', flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('key')
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--conds', default='clean,opus16,cs')  # opus64 only for the current models: near-transparent
    p.add_argument('--lat', action='store_true')
    p.add_argument('--no-wer', action='store_true')
    a = p.parse_args()
    os.makedirs('results', exist_ok=True)
    t = time.perf_counter()
    m, langs, tl = load(a.key, a.threads)
    print(a.key, 'loaded in', round(time.perf_counter() - t, 1), 's', flush=True)
    if a.lat:
        lat_run(a.key, m, langs, tl, f'results/{a.key}.lat.jsonl')
    if not a.no_wer:
        wer_run(a.key, m, langs, tl, a.conds.split(','), f'results/{a.key}.jsonl')
