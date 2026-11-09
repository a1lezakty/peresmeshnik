"""MT benchmark on FLEURS parallel sentences: latency per sentence, CPU, peak RAM, chrF / chrF++ / BLEU.

    python bench/mt.py opus en-ru               # OPUS-MT via CTranslate2 int8 (converted once, needs torch+transformers)
    python bench/mt.py opus de-en --path DIR    # a CTranslate2 conversion with source.spm / target.spm
    python bench/mt.py pivot de-ru --path DIR1 --path2 DIR2  # two OPUS-MT through English: de-en, then en-ru
    python bench/mt.py nllb en-ru [--path DIR]  # NLLB-200 distilled 600M, CTranslate2 int8 (sentencepiece.bpe.model in DIR)
    python bench/mt.py m2m de-ja --path DIR     # M2M100 418M, CTranslate2 (sentencepiece.bpe.model in DIR)
    python bench/mt.py llama ru-en --gguf PATH  # any GGUF translation LLM through llama-server (e.g. HY-MT1.5)

Source text is the gold FLEURS sentence, so MT is measured without ASR errors. Any of the 14 languages: `--fleurs`
(default for a pair not en-ru / ru-en) takes the first `--n` sentences of FLEURS test (FLoRes devtest) in both
languages; the hypotheses go to $BENCH_DATA/mt-hyps/ for COMET (bench/mt_comet.py, its own torch venv). `--tatoeba`:
`--n` short everyday sentences of the Tatoeba-Challenge test set instead (a call's register; extra_text: the share of
translations more than twice their reference, a continuation the model made up).
"""
import argparse
import json
import os
import subprocess
import time
import urllib.request
from pathlib import Path

import numpy as np
from data import CODES, fleurs, tatoeba

from common import RESULTS, ROOT, Timer, manifest, record

MODELS = Path(os.environ.get('PERESMESHNIK_MODELS', 'models'))
NLLB = {'en': 'eng_Latn', 'ru': 'rus_Cyrl', 'uk': 'ukr_Cyrl', 'de': 'deu_Latn', 'fr': 'fra_Latn', 'es': 'spa_Latn',
        'it': 'ita_Latn', 'pt': 'por_Latn', 'pl': 'pol_Latn', 'tr': 'tur_Latn', 'zh': 'zho_Hans', 'ja': 'jpn_Jpan',
        'ko': 'kor_Hang', 'ar': 'arb_Arab'}
NAMES = {'en': 'English', 'ru': 'Russian', 'uk': 'Ukrainian', 'de': 'German', 'fr': 'French', 'es': 'Spanish',
         'it': 'Italian', 'pt': 'Portuguese', 'pl': 'Polish', 'tr': 'Turkish', 'zh': 'Chinese', 'ja': 'Japanese',
         'ko': 'Korean', 'ar': 'Arabic'}
ZH_NAMES = {'en': '英语', 'ru': '俄语', 'uk': '乌克兰语', 'de': '德语', 'fr': '法语', 'es': '西班牙语', 'it': '意大利语',
            'pt': '葡萄牙语', 'pl': '波兰语', 'tr': '土耳其语', 'zh': '中文', 'ja': '日语', 'ko': '韩语', 'ar': '阿拉伯语'}
CHARS = {'zh', 'ja'}  # BLEU on characters there (no spaces)


def ct2(path, threads, device):
    import ctranslate2
    return ctranslate2.Translator(str(path), device=device, inter_threads=1, intra_threads=threads,
                                  compute_type='int8' if device == 'cpu' else 'default')  # a float16 conversion: int8 at load


def eos(path):
    """Whether the conversion appends </s> to the source itself (ours of the Marian weights: add_source_eos)."""
    cfg = Path(path) / 'config.json'
    return cfg.exists() and json.loads(cfg.read_text()).get('add_source_eos', False)


def opus_dir(path, threads, device):
    """An OPUS-MT CTranslate2 directory as the app runs it (local/mt.py: length cap, no 3-gram twice)."""
    import sentencepiece as spm
    tr = ct2(path, threads, device)
    p = Path(path)
    names = next(n for n in (('source.spm', 'target.spm'), ('src.spm.model', 'tgt.spm.model'), ('joint.spm.model', 'joint.spm.model'))
                 if (p / n[0]).exists())  # OPUS-MT, quickmt, quickmt en-fr
    sp_src, sp_tgt = (spm.SentencePieceProcessor(str(p / n)) for n in names)
    tail = [] if eos(path) else ['</s>']
    pre = Path(path) / 'source_prefix.txt'  # a multi-target model's >>xxx<< token
    head = [pre.read_text().strip()] if pre.exists() else []

    from peresmeshnik.local.mt import sentences

    def run(text):  # sentence by sentence in one batch, as the app
        batch = [sp_src.encode(x, out_type=str) for x in sentences(text)] or [[]]
        out = tr.translate_batch([head + t + tail for t in batch], beam_size=1, max_decoding_length=min(256, 2 * max(map(len, batch)) + 10),
                                 no_repeat_ngram_size=3)
        return ' '.join(sp_tgt.decode(r.hypotheses[0]) for r in out)
    return run


def pivot(src, tgt, threads, device, path, path2):
    first, second = opus_dir(path, threads, device), opus_dir(path2, threads, device)
    return lambda text: second(first(text))


def multi(path, threads, device, src_tok, tgt_tok):
    """NLLB-200 / M2M100 on CTranslate2: [src token] pieces </s>, target prefix [tgt token]."""
    import sentencepiece as spm
    tr = ct2(path, threads, device)
    sp = spm.SentencePieceProcessor(str(next(Path(path).glob('*.model'))))

    def run(text):
        toks = [src_tok] + sp.encode(text, out_type=str) + ['</s>']
        r = tr.translate_batch([toks], target_prefix=[[tgt_tok]], beam_size=1,
                               max_decoding_length=min(256, 2 * len(toks) + 10), no_repeat_ngram_size=3)[0].hypotheses[0]
        return sp.decode(r[1:])
    return run


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


def m2m(src, tgt, threads, device, path):
    return multi(path, threads, device, f'__{src}__', f'__{tgt}__')


def nllb(src, tgt, threads, device, path=None):
    if path:
        return multi(path, threads, device, NLLB[src], NLLB[tgt])
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
        if 'Hy-MT2' in gguf:  # its model card: Chinese names in a Chinese prompt for pairs with Chinese
            msg = (f'将以下文本翻译为{ZH_NAMES[tgt]}，注意只需要输出翻译后的结果，不要额外解释：\n\n{text}' if 'zh' in (src, tgt) else
                   f'Translate the following text into {NAMES[tgt]}. Note that you should only output the translated result '
                   f'without any additional explanation:\n\n{text}')
        else:  # prompt from the HY-MT1.5 model card (non-Chinese pairs)
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
    ap.add_argument('engine', choices=['opus', 'pivot', 'nllb', 'm2m', 'llama'])
    ap.add_argument('direction', help='src-tgt, two of: ' + ' '.join(CODES))
    ap.add_argument('--gguf')
    ap.add_argument('--path', help='the CTranslate2 model directory (opus: default the HF conversion once)')
    ap.add_argument('--path2', help='pivot: the second model (en-tgt)')
    ap.add_argument('--name', help='the result row name (default: from the engine)')
    ap.add_argument('--fleurs', action='store_true', help='FLEURS test text, not the en/ru bench set')
    ap.add_argument('--tatoeba', action='store_true', help="Tatoeba's short everyday sentences (Tatoeba-Challenge test)")
    ap.add_argument('--threads', type=int, default=4)
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--n', type=int, default=30)
    a = ap.parse_args()
    import sacrebleu
    src, tgt = a.direction.split('-')
    if a.tatoeba:
        items, fl = [{src: {'text': s}, tgt: {'text': t}} for s, t in tatoeba(src, tgt, a.n)], False
    elif fl := a.fleurs or a.direction not in ('en-ru', 'ru-en'):
        s_rows, t_rows = fleurs(src, split='test', audio=False), fleurs(tgt, split='test', audio=False)
        items = [{src: {'text': s_rows[i][1]}, tgt: {'text': t_rows[i][1]}} for i in sorted(set(s_rows) & set(t_rows))[:a.n]]
    else:
        items = manifest()[:a.n]
    t0 = time.perf_counter()
    if a.engine == 'llama':
        run = llama(src, tgt, a.threads, a.device, a.gguf)
    elif a.engine == 'opus' and a.path:
        run = opus_dir(a.path, a.threads, a.device)
    elif a.engine == 'pivot':
        run = pivot(src, tgt, a.threads, a.device, a.path, a.path2)
    elif a.engine in ('nllb', 'm2m') and a.path:
        run = globals()[a.engine](src, tgt, a.threads, a.device, a.path)
    else:
        run = globals()[a.engine](src, tgt, a.threads, a.device)
    load_s = time.perf_counter() - t0
    srv = getattr(run, 'server', None)
    try:
        run(items[-1][src]['text'])  # warm-up
        cpu_srv0 = server_stats(srv.pid)[1] if srv else 0
        lat, hyps, cpu0, wall0 = [], [], time.process_time(), time.perf_counter()
        for it in items:
            with Timer() as t:
                hyps.append(run(it[src]['text']))
            lat.append(t.s)
        wall, cpu = time.perf_counter() - wall0, time.process_time() - cpu0
        clause = []  # a clause of a call: the first 10 words of a sentence (zh / ja: 15 characters)
        for it in items[:30]:
            text = it[src]['text']
            with Timer() as t:
                run(text[:15] if src in CHARS else ' '.join(text.split()[:10]))
            clause.append(t.s)
        extra = {'clause_ms_median': round(1000 * float(np.median(clause))),
                 'clause_ms_p90': round(1000 * float(np.percentile(clause, 90)))}
        if srv:
            rss, cpu_srv = server_stats(srv.pid)
            cpu += cpu_srv - cpu_srv0
            extra['server_peak_rss_mb'] = rss
    finally:
        if srv:
            srv.terminate()
    refs = [it[tgt]['text'] for it in items]
    name = a.name or (Path(a.gguf).stem if a.gguf else {'opus': 'opus-mt', 'nllb': 'nllb-200-distilled-600M',
                                                         'm2m': 'm2m100-418M', 'pivot': 'opus-mt-via-en'}[a.engine])
    if fl or a.tatoeba:
        out = ROOT / ('mt-hyps-tatoeba' if a.tatoeba else 'mt-hyps') / f'{name}-{a.direction}-{a.device}.json'
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({'src': [it[src]['text'] for it in items], 'hyp': hyps, 'ref': refs}, ensure_ascii=False))
    words = sum(len(it[src]['text'].split()) for it in items)
    record('mt', name, direction=a.direction, device=a.device, threads=a.threads, n=len(items),
           data='tatoeba-test' if a.tatoeba else 'fleurs-test' if fl else 'set', load_s=round(load_s, 1),
           latency_ms_median=round(1000 * float(np.median(lat))),
           latency_ms_p90=round(1000 * float(np.percentile(lat, 90))), ms_per_src_word=round(1000 * sum(lat) / max(words, 1), 1),
           cpu_cores=round(cpu / wall, 2),
           extra_text=round(sum(len(h) > 2 * len(r) + 10 for h, r in zip(hyps, refs)) / len(refs), 3),  # a continuation of its own
           chrf=round(sacrebleu.corpus_chrf(hyps, [refs]).score, 1),
           chrfpp=round(sacrebleu.corpus_chrf(hyps, [refs], word_order=2).score, 1),
           bleu=round(sacrebleu.corpus_bleu(hyps, [refs], tokenize='char' if tgt in CHARS else '13a').score, 1),
           samples=[{'src': it[src]['text'], 'hyp': h, 'ref': r} for it, h, r in list(zip(items, hyps, refs))[:3]], **extra)


if __name__ == '__main__':
    main()
