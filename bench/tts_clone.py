"""TTS with voice cloning: XTTS-v2, Chatterbox Multilingual, Fun-CosyVoice3.

Two steps, each in its own process (and venv), so the synth peak RSS is clean:

    python bench/tts_clone.py synth --model xtts --device cpu   # model venv: wavs + timings
    python bench/tts_clone.py eval --model xtts                 # tts-eval venv: speaker-sim + CER, records rows
    python bench/tts_clone.py baseline                          # tts-eval venv: real-speaker reference rows

Cases: reference voice {en, ru} x output language {en, ru}, the N shortest manifest
sentences per language. TTFA = time to the first audio chunk (streaming models) or to
the whole utterance. Wavs go to $BENCH_OUT/tts_clone/<model>/ (default <BENCH_DATA>/../out).
"""
import argparse
import json
import os
import re
import sys
import time
import unicodedata
from pathlib import Path

import numpy as np
import soundfile as sf

from common import ROOT, SET, Timer, manifest, peak_rss_mb, record

OUT = Path(os.environ.get('BENCH_OUT', ROOT.parent / 'out')) / 'tts_clone'
THREADS = int(os.environ.get('BENCH_THREADS', 4))
LANGS = ('en', 'ru')
COSY_PREFIX = 'You are a helpful assistant.<|endofprompt|>'


def sentences(n):
    m = manifest()
    return {lang: sorted(m, key=lambda x: x[lang]['sec'])[:n] for lang in LANGS}


def ref(lang):
    return str(SET / f'ref_{lang}.wav'), (SET / f'ref_{lang}.txt').read_text(encoding='utf-8').strip()


# --- model adapters: load(device) -> (clone(ref_lang) -> cond, synth(text, lang, cond, ref_lang) -> iter of chunks, sr)

def load_xtts(device, stream):
    from TTS.api import TTS
    m = TTS('tts_models/multilingual/multi-dataset/xtts_v2').to(device).synthesizer.tts_model

    def clone(lang):
        return m.get_conditioning_latents(audio_path=[ref(lang)[0]])

    def synth(text, lang, cond, ref_lang):
        if not stream:
            yield m.inference(text, lang, *cond)['wav']
            return
        for c in m.inference_stream(text, lang, *cond, enable_text_splitting=False):
            yield c.cpu().numpy()
    return clone, synth, 24000


def load_chatterbox(device, stream):  # no streaming upstream
    from chatterbox.mtl_tts import ChatterboxMultilingualTTS
    m = ChatterboxMultilingualTTS.from_pretrained(device=device)

    def clone(lang):
        m.prepare_conditionals(ref(lang)[0])
        return m.conds

    def synth(text, lang, cond, ref_lang):
        m.conds = cond
        # cfg_weight=0 is the upstream advice when the reference language differs (less accent transfer)
        yield m.generate(text, language_id=lang, cfg_weight=0.5 if lang == ref_lang else 0.0).squeeze(0).cpu().numpy()
    return clone, synth, m.sr


def load_cosyvoice(device, stream):
    sys.path[:0] = [os.environ['COSYVOICE_SRC'], os.environ['COSYVOICE_SRC'] + '/third_party/Matcha-TTS']
    from cosyvoice.cli.cosyvoice import AutoModel
    m = AutoModel(model_dir=os.environ.get('COSYVOICE_MODEL', 'FunAudioLLM/Fun-CosyVoice3-0.5B-2512'))

    def clone(lang):
        # CosyVoice refuses prompts > 30 s: rebuild a shorter reference (same reader, matching text), ~17 s en / ~26 s ru
        wav = OUT / 'cosyvoice' / f'ref_{lang}_short.wav'
        if not wav.exists():  # built once; or copy ref_*_short.{wav,txt} from another host
            import data
            x, text = data.reference(lang, 15)
            sf.write(wav, x, data.SR)
            wav.with_suffix('.txt').write_text(text, encoding='utf-8')
        text, wav = wav.with_suffix('.txt').read_text(encoding='utf-8'), str(wav)
        m.add_zero_shot_spk(COSY_PREFIX + text, wav, f'ref_{lang}')
        return f'ref_{lang}'

    hop0 = m.model.token_hop_len

    def synth(text, lang, cond, ref_lang):
        # upstream doubles token_hop_len per streamed chunk (stream_scale_factor=2) and never resets it between
        # calls, so after the first utterance the first chunk waits for ~4 s of tokens; reset per utterance
        m.model.token_hop_len = hop0
        # same language: zero-shot with the prompt transcript; other language: cross-lingual (prompt text dropped)
        f = m.inference_zero_shot if lang == ref_lang else m.inference_cross_lingual
        args = (text, '', '') if lang == ref_lang else (COSY_PREFIX + text, '')
        for c in f(*args, zero_shot_spk_id=cond, stream=stream):
            yield c['tts_speech'].squeeze(0).cpu().numpy()
    return clone, synth, m.sample_rate


MODELS = {'xtts': load_xtts, 'chatterbox': load_chatterbox, 'cosyvoice': load_cosyvoice}


def synth_main(a):
    import torch
    torch.set_num_threads(THREADS)
    if a.device == 'cuda':
        torch.cuda.set_per_process_memory_fraction(a.gpu_mem_frac)  # shared GPU: stay inside the cap
    out = OUT / a.model
    out.mkdir(parents=True, exist_ok=True)
    with Timer() as tl:
        clone, synth, sr = MODELS[a.model](a.device, not a.no_stream)
    run = {'model': a.model, 'device': a.device, 'stream': not a.no_stream and a.model != 'chatterbox',
           'threads': THREADS, 'torch': torch.__version__, 'load_s': round(tl.s, 2),
           'rss_after_load_mb': peak_rss_mb(), 'cases': []}
    sents = sentences(a.n)
    for ref_lang in LANGS:
        with Timer() as tc:
            cond = clone(ref_lang)
        for i, out_lang in enumerate(LANGS):
            if i == 0 and ref_lang == LANGS[0]:  # warm-up, not timed
                for _ in synth(sents[out_lang][0][out_lang]['text'], out_lang, cond, ref_lang):
                    pass
            items = []
            for s in sents[out_lang]:
                text, chunks = s[out_lang]['text'], []
                with Timer() as ts:
                    for c in synth(text, out_lang, cond, ref_lang):
                        if not chunks:
                            ttfa = time.perf_counter() - ts.t
                        chunks.append(c)
                wav = np.concatenate(chunks).astype('float32')
                path = out / f'{ref_lang}2{out_lang}_{s["id"]}.wav'
                sf.write(path, wav, sr)
                items.append({'id': s['id'], 'text': text, 'wav': str(path), 'ttfa_s': round(ttfa, 3),
                              'synth_s': round(ts.s, 3), 'audio_s': round(len(wav) / sr, 3)})
                print(ref_lang, out_lang, items[-1], flush=True)
            run['cases'].append({'ref_lang': ref_lang, 'out_lang': out_lang, 'clone_s': round(tc.s, 2), 'items': items})
    run['peak_rss_mb'] = peak_rss_mb()
    if a.device == 'cuda':
        run['gpu_peak_mb'] = round(torch.cuda.max_memory_allocated() / 2**20)
    (out / 'run.json').write_text(json.dumps(run, ensure_ascii=False, indent=1), encoding='utf-8')
    print('peak RSS MB', run['peak_rss_mb'])


# --- evaluation (tts-eval venv): ECAPA speaker similarity + faster-whisper small int8 CER

def norm(t):
    t = unicodedata.normalize('NFKC', t).lower().replace('ё', 'е')
    return ' '.join(re.sub(r'[^\w\s]|_', ' ', t).split())


def edits(a, b):
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


class Judge:
    def __init__(self):
        import torch
        torch.set_num_threads(THREADS)
        from faster_whisper import WhisperModel
        from speechbrain.inference.speaker import EncoderClassifier
        self.torch = torch
        self.spk = EncoderClassifier.from_hparams(
            'speechbrain/spkrec-ecapa-voxceleb', run_opts={'device': 'cpu'},
            savedir=str(Path(os.environ.get('HF_HOME', 'models')) / 'speechbrain-ecapa'))
        self.asr = WhisperModel('small', device='cpu', compute_type='int8', cpu_threads=THREADS)

    def emb(self, path, start=0.0, end=None):
        import torchaudio.functional as F
        x, sr = sf.read(path, dtype='float32', always_2d=True)
        x = self.torch.from_numpy(x.mean(1))
        x = F.resample(x, sr, 16000) if sr != 16000 else x
        x = x[int(start * 16000):int(end * 16000) if end else None]
        e = self.spk.encode_batch(x[None]).squeeze()
        return e / e.norm()

    def cer(self, path, text, lang):
        hyp = ' '.join(s.text for s in self.asr.transcribe(path, language=lang, beam_size=5)[0])
        ref_, hyp = norm(text), norm(hyp)
        return edits(ref_, hyp), len(ref_), hyp


def eval_main(a):
    run = json.loads((OUT / a.model / 'run.json').read_text(encoding='utf-8'))
    j = Judge()
    refs = {lang: j.emb(ref(lang)[0]) for lang in LANGS}
    for c in run['cases']:
        it, e, n, sims = c['items'], 0, 0, []
        for x in it:
            if not Path(x['wav']).exists():  # wavs copied from another host
                x['wav'] = str(OUT / a.model / Path(x['wav']).name)
            sims.append(float(j.emb(x['wav']) @ refs[c['ref_lang']]))
            de, dn, x['hyp'] = j.cer(x['wav'], x['text'], c['out_lang'])
            e, n = e + de, n + dn
            print(c['ref_lang'], c['out_lang'], round(sims[-1], 3), round(de / max(dn, 1), 3), x['hyp'], flush=True)
        audio, synth = sum(x['audio_s'] for x in it), sum(x['synth_s'] for x in it)
        record('tts_clone', run['model'], ref_lang=c['ref_lang'], out_lang=c['out_lang'], device=run['device'],
               threads=run['threads'], n=len(it), load_s=run['load_s'], clone_s=c['clone_s'],
               ttfa_s=round(np.mean([x['ttfa_s'] for x in it]), 3), ttfa_max_s=max(x['ttfa_s'] for x in it),
               synth_s=round(synth / len(it), 3), audio_s=round(audio, 2), rtf=round(synth / audio, 3),
               peak_rss_mb=run['peak_rss_mb'], rss_after_load_mb=run['rss_after_load_mb'], gpu_peak_mb=run.get('gpu_peak_mb'),
               torch=run.get('torch'), stream=run.get('stream'),
               spk_sim=round(np.mean(sims), 3), spk_sim_min=round(min(sims), 3), cer=round(e / n, 4))
    (OUT / a.model / 'eval.json').write_text(json.dumps(run, ensure_ascii=False, indent=1), encoding='utf-8')


def baseline_main(a):
    """Reference points: same speaker (two halves of a reference), different speakers (ref_en vs
    ref_ru; ref vs real FLEURS speakers), and whisper CER on the real FLEURS recordings."""
    j = Judge()
    refs = {lang: j.emb(ref(lang)[0]) for lang in LANGS}
    record('tts_clone', 'baseline', case='diff_speakers_ref_en_vs_ref_ru', spk_sim=round(float(refs['en'] @ refs['ru']), 3))
    for lang in LANGS:
        half = float(j.emb(ref(lang)[0], 0, 20) @ j.emb(ref(lang)[0], 20))
        record('tts_clone', 'baseline', case=f'same_speaker_ref_{lang}_halves', spk_sim=round(half, 3))
    for lang, items in sentences(a.n).items():
        e = n = 0
        sims = {r: [] for r in LANGS}
        for s in items:
            path = str(SET / s[lang]['wav'])
            for r in LANGS:
                sims[r].append(float(j.emb(path) @ refs[r]))
            de, dn, _ = j.cer(path, s[lang]['text'], lang)
            e, n = e + de, n + dn
        record('tts_clone', 'baseline', case=f'real_fleurs_{lang}', out_lang=lang, n=len(items), cer=round(e / n, 4),
               **{f'spk_sim_vs_ref_{r}': round(np.mean(v), 3) for r, v in sims.items()})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('step', choices=['synth', 'eval', 'baseline'])
    ap.add_argument('--model', choices=list(MODELS))
    ap.add_argument('--device', default='cpu', choices=['cpu', 'mps', 'cuda'])
    ap.add_argument('--n', type=int, default=6, help='sentences per language')
    ap.add_argument('--no-stream', action='store_true', help='full utterances (CosyVoice streaming on CPU is ~3x slower)')
    ap.add_argument('--gpu-mem-frac', type=float, default=0.6, help='CUDA memory cap for this process')
    a = ap.parse_args()
    {'synth': synth_main, 'eval': eval_main, 'baseline': baseline_main}[a.step](a)


if __name__ == '__main__':
    main()
