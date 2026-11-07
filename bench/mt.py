"""MT benchmark on FLEURS parallel sentences: latency per sentence, CPU, peak RAM, chrF / BLEU.

    python bench/mt.py opus en-ru               # OPUS-MT via CTranslate2 int8 (converted once, needs torch+transformers)
    python bench/mt.py nllb en-ru               # NLLB-200 distilled 600M, CTranslate2 int8
    python bench/mt.py llama ru-en --gguf PATH  # any GGUF translation LLM through llama-server (e.g. HY-MT1.5)

Source text is the gold FLEURS sentence, so MT is measured without ASR errors.
"""
import argparse
import json
import os
import subprocess
import time
import urllib.request
from pathlib import Path

import numpy as np

from common import RESULTS, Timer, manifest, record

MODELS = Path(os.environ.get('PERESMESHNIK_MODELS', 'models'))
NLLB = {'en': 'eng_Latn', 'ru': 'rus_Cyrl'}
NAMES = {'en': 'English', 'ru': 'Russian'}


def opus(src, tgt, threads, device):
    import ctranslate2
    import sentencepiece as spm
    from huggingface_hub import snapshot_download
    repo = f'Helsinki-NLP/opus-mt-{src}-{tgt}'
    out = MODELS / 'ct2' / f'opus-mt-{src}-{tgt}-int8'
    hf = Path(snapshot_download(repo, allow_patterns=['*.json', '*.spm', '*.bin', '*.safetensors', '*.txt']))
    if not out.exists():
        ctranslate2.converters.TransformersConverter(str(hf)).convert(str(out), quantization='int8')
    tr = ctranslate2.Translator(str(out), device=device, inter_threads=1, intra_threads=threads)
    sp_src, sp_tgt = spm.SentencePieceProcessor(str(hf / 'source.spm')), spm.SentencePieceProcessor(str(hf / 'target.spm'))

    def run(text):
        toks = sp_src.encode(text, out_type=str) + ['</s>']
        r = tr.translate_batch([toks], beam_size=1, max_decoding_length=256)[0].hypotheses[0]
        return sp_tgt.decode(r)
    return run


def nllb(src, tgt, threads, device):
    import ctranslate2
    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer
    path = snapshot_download('JustFrederik/nllb-200-distilled-600M-ct2-int8')
    tok = AutoTokenizer.from_pretrained('facebook/nllb-200-distilled-600M', src_lang=NLLB[src])
    tr = ctranslate2.Translator(path, device=device, inter_threads=1, intra_threads=threads)

    def run(text):
        toks = tok.convert_ids_to_tokens(tok.encode(text))
        r = tr.translate_batch([toks], target_prefix=[[NLLB[tgt]]], beam_size=1, max_decoding_length=256)[0].hypotheses[0]
        return tok.decode(tok.convert_tokens_to_ids(r[1:]))
    return run


def llama(src, tgt, threads, device, gguf):
    """llama-server from a llama.cpp release; $LLAMA_SERVER points at the binary."""
    port = 8089
    log = RESULTS / f'llama-server-{Path(gguf).stem}.log'  # shows whether Metal/GPU offload was used
    RESULTS.mkdir(exist_ok=True)
    srv = subprocess.Popen([os.environ.get('LLAMA_SERVER', 'llama-server'), '-m', gguf, '--port', str(port), '-t', str(threads),
                            '-c', '1024', '-np', '1', '-ngl', '0' if device == 'cpu' else '99'],
                           stdout=subprocess.DEVNULL, stderr=open(log, 'w'))
    for _ in range(600):
        try:
            urllib.request.urlopen(f'http://127.0.0.1:{port}/health', timeout=1)
            break
        except OSError:
            time.sleep(0.5)

    def run(text):
        # prompt from the HY-MT1.5 model card (non-Chinese pairs)
        msg = f'Translate the following segment into {NAMES[tgt]}, without additional explanation.\n\n{text}'
        body = json.dumps({'messages': [{'role': 'user', 'content': msg}], 'temperature': 0, 'max_tokens': 256}).encode()
        req = urllib.request.Request(f'http://127.0.0.1:{port}/v1/chat/completions', body, {'Content-Type': 'application/json'})
        return json.load(urllib.request.urlopen(req))['choices'][0]['message']['content'].strip()
    run.server = srv
    return run


def server_stats(pid):
    """Peak RSS (MB) and CPU seconds of the llama-server process (Linux /proc, else psutil)."""
    try:
        status = Path(f'/proc/{pid}/status').read_text()
        hwm = int(next(ln.split()[1] for ln in status.splitlines() if ln.startswith('VmHWM'))) // 1024
        f = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()
        return hwm, (int(f[11]) + int(f[12])) / os.sysconf('SC_CLK_TCK')
    except FileNotFoundError:
        import psutil
        p = psutil.Process(pid)
        t = p.cpu_times()
        return p.memory_info().rss // 2**20, t.user + t.system


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('engine', choices=['opus', 'nllb', 'llama'])
    ap.add_argument('direction', choices=['en-ru', 'ru-en'])
    ap.add_argument('--gguf')
    ap.add_argument('--threads', type=int, default=4)
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--n', type=int, default=30)
    a = ap.parse_args()
    import sacrebleu
    src, tgt = a.direction.split('-')
    items = manifest()[:a.n]
    t0 = time.perf_counter()
    run = llama(src, tgt, a.threads, a.device, a.gguf) if a.engine == 'llama' else globals()[a.engine](src, tgt, a.threads, a.device)
    load_s = time.perf_counter() - t0
    srv = getattr(run, 'server', None)
    try:
        run('Hello.' if src == 'en' else 'Привет.')  # warm-up
        cpu_srv0 = server_stats(srv.pid)[1] if srv else 0
        lat, hyps, cpu0, wall0 = [], [], time.process_time(), time.perf_counter()
        for it in items:
            with Timer() as t:
                hyps.append(run(it[src]['text']))
            lat.append(t.s)
        wall, cpu = time.perf_counter() - wall0, time.process_time() - cpu0
        extra = {}
        if srv:
            rss, cpu_srv = server_stats(srv.pid)
            cpu += cpu_srv - cpu_srv0
            extra = {'server_peak_rss_mb': rss}
    finally:
        if srv:
            srv.terminate()
    refs = [it[tgt]['text'] for it in items]
    name = Path(a.gguf).stem if a.gguf else {'opus': 'opus-mt', 'nllb': 'nllb-200-distilled-600M'}[a.engine]
    record('mt', name, direction=a.direction, device=a.device, threads=a.threads, n=len(items),
           load_s=round(load_s, 1), latency_ms_median=round(1000 * float(np.median(lat))),
           latency_ms_p90=round(1000 * float(np.percentile(lat, 90))), cpu_cores=round(cpu / wall, 2),
           chrf=round(sacrebleu.corpus_chrf(hyps, [refs]).score, 1), bleu=round(sacrebleu.corpus_bleu(hyps, [refs]).score, 1),
           samples=[{'src': it[src]['text'], 'hyp': h, 'ref': r} for it, h, r in list(zip(items, hyps, refs))[:3]], **extra)


if __name__ == '__main__':
    main()
