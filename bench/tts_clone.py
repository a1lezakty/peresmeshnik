"""TTS with voice cloning: XTTS-v2, Chatterbox Multilingual, Fun-CosyVoice3.

Two steps, each in its own process (and venv), so the synth peak RSS is clean:

    python bench/tts_clone.py synth --model xtts --device cpu   # model venv: wavs + timings
    python bench/tts_clone.py eval --model xtts                 # tts-eval venv: speaker-sim + CER, records rows
    python bench/tts_clone.py baseline                          # tts-eval venv: real-speaker reference rows

Cases: reference voice {en, ru} x output language {en, ru}, the N shortest manifest
sentences per language. TTFA = time to the first audio chunk (streaming models) or to
the whole utterance. Wavs go to $BENCH_OUT/tts_clone/<model>/ (default <BENCH_DATA>/../out).

More output languages (stage 6): `--langs de,ja,...` on every step, after

    python bench/tts_clone.py set --langs en,ru,de,...          # tts-eval venv: the same FLEURS sentences in each

in a BENCH_DATA of its own (it writes that set's manifest.json; the references come from data.py). Each output
language gets an untimed first synth (`first_s`: it loads that language's text tools, e.g. pypinyin, cutlet). The
judge: $JUDGE_ASR (faster-whisper, default small) on $JUDGE_DEVICE (default cpu); CER without spaces for zh / ja.
XTTS (`xtts`, `xtts-engine`) loads from $XTTS_DIR, whose model.pth must match the app's registry (its sha256 goes
into the rows).
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
REFS = ('en', 'ru')  # the voices: the LibriSpeech reader (en), the LibriVox reader (ru, "me")
LANGS = ('en', 'ru')  # output languages (--langs)
NO_SPACES = {'zh', 'ja'}
COSY_PREFIX = 'You are a helpful assistant.<|endofprompt|>'


def sentences(n):
    m = manifest()
    return {lang: sorted(m, key=lambda x: x[lang]['sec'])[:n] for lang in LANGS}


def ref(lang):
    return str(SET / f'ref_{lang}.wav'), (SET / f'ref_{lang}.txt').read_text(encoding='utf-8').strip()


# --- model adapters: load(device) -> (clone(ref_lang) -> cond, synth(text, lang, cond, ref_lang) -> iter of chunks, sr)

def xtts_checked() -> str:
    """$XTTS_DIR's model.pth must be the one the app downloads (peresmeshnik/models.toml): another copy measures another
    model (coqui's own downloader once replaced it through a link with the HF main one). Its sha256 goes into the rows."""
    from peresmeshnik import models
    got, want = models.sha256(Path(os.environ['XTTS_DIR']) / 'model.pth'), models.get('xtts-v2').files['model.pth']['sha256']
    if got != want:
        sys.exit(f'$XTTS_DIR/model.pth: sha256 {got[:12]}..., the registry has {want[:12]}...: not the model the app ships')
    return got


def load_xtts(device, stream):
    """XTTS-v2 by itself, from $XTTS_DIR (not coqui's downloader: it fetches the HF main files, not the registry's)."""
    from TTS.tts.configs.xtts_config import XttsConfig
    from TTS.tts.models.xtts import Xtts
    cfg = XttsConfig()
    cfg.load_json(os.path.join(os.environ['XTTS_DIR'], 'config.json'))
    m = Xtts.init_from_config(cfg)
    m.load_checkpoint(cfg, checkpoint_dir=os.environ['XTTS_DIR'], eval=True)
    m.to(device)

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


def load_engine(device, stream):
    """The app's clone engine (peresmeshnik.local.clone): enrollment profile, text split, length guard."""
    from peresmeshnik import voice
    from peresmeshnik.local.clone import XttsClone
    e = XttsClone(os.environ['XTTS_DIR'], device=device, threads=THREADS,
                  temperature=float(os.environ.get('XTTS_TEMPERATURE', 0.75)), top_p=float(os.environ.get('XTTS_TOP_P', 0.85)))

    def clone(lang):
        x, sr = voice.read_audio(ref(lang)[0])
        return e.enroll(x, sr, ref(lang)[1], name=f'ref_{lang}', lang=lang, root=OUT / 'voices')

    def synth(text, lang, cond, ref_lang):
        yield from e.synth(text, lang, cond)
    return clone, synth, e.sample_rate


MODELS = {'xtts': load_xtts, 'chatterbox': load_chatterbox, 'cosyvoice': load_cosyvoice, 'xtts-engine': load_engine}


def synth_main(a):
    import torch
    torch.set_num_threads(THREADS)
    if a.device == 'cuda':
        torch.cuda.set_per_process_memory_fraction(a.gpu_mem_frac)  # shared GPU: stay inside the cap
    out = OUT / (a.label or a.model)
    out.mkdir(parents=True, exist_ok=True)
    sha = xtts_checked() if a.model in ('xtts', 'xtts-engine') else None
    with Timer() as tl:
        clone, synth, sr = MODELS[a.model](a.device, not a.no_stream)
    run = {'model': a.label or a.model, 'device': a.device, 'stream': not a.no_stream and a.model != 'chatterbox',
           'threads': THREADS, 'torch': torch.__version__, 'load_s': round(tl.s, 2),
           'rss_after_load_mb': peak_rss_mb(), 'cases': [], 'model_sha256': sha,
           'temperature': float(os.environ.get('XTTS_TEMPERATURE', 0.75)) if a.model == 'xtts-engine' else None,
           'top_p': float(os.environ.get('XTTS_TOP_P', 0.85)) if a.model == 'xtts-engine' else None}
    sents, run['first_s'] = sentences(a.n), {}
    for ref_lang in REFS:
        with Timer() as tc:
            cond = clone(ref_lang)
        for out_lang in LANGS:
            if out_lang not in run['first_s']:  # warm-up per language, not timed: the first loads its text tools
                with Timer() as tw:
                    for _ in synth(sents[out_lang][0][out_lang]['text'], out_lang, cond, ref_lang):
                        pass
                run['first_s'][out_lang] = round(tw.s, 2)
            items = []
            for s in sents[out_lang]:
                text, chunks, cpu0 = s[out_lang]['text'], [], time.process_time()
                with Timer() as ts:
                    for c in synth(text, out_lang, cond, ref_lang):
                        if not chunks:
                            ttfa = time.perf_counter() - ts.t
                        chunks.append(c)
                wav = np.concatenate(chunks).astype('float32')
                path = out / f'{ref_lang}2{out_lang}_{s["id"]}.wav'
                sf.write(path, wav, sr)
                items.append({'id': s['id'], 'text': text, 'wav': str(path), 'ttfa_s': round(ttfa, 3),
                              'synth_s': round(ts.s, 3), 'audio_s': round(len(wav) / sr, 3),
                              'cpu_s': round(time.process_time() - cpu0, 3)})
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
    t = re.sub('[\u064b-\u0652\u0670]', '', t)  # Arabic vowel marks: Whisper writes none
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
        dev = os.environ.get('JUDGE_DEVICE', 'cpu')
        self.name = os.environ.get('JUDGE_ASR', 'small')
        self.asr = WhisperModel(self.name, device=dev, compute_type='float16' if dev == 'cuda' else 'int8', cpu_threads=THREADS)

    def emb(self, path, start=0.0, end=None):
        import torchaudio.functional as F
        x, sr = sf.read(path, dtype='float32', always_2d=True)
        x = self.torch.from_numpy(x.mean(1))
        x = F.resample(x, sr, 16000) if sr != 16000 else x
        x = x[int(start * 16000):int(end * 16000) if end else None]
        e = self.spk.encode_batch(x[None]).squeeze()
        return e / e.norm()

    def cer(self, path, text, lang):
        """(char edits, ref chars, hyp, word edits, ref words): chars without spaces in zh / ja, no words there."""
        hyp = ' '.join(s.text for s in self.asr.transcribe(path, language=lang, beam_size=5)[0])
        ref_, hyp = norm(text), norm(hyp)
        if lang in NO_SPACES:
            return edits(ref_.replace(' ', ''), hyp.replace(' ', '')), len(ref_.replace(' ', '')), hyp, 0, 0
        return edits(ref_, hyp), len(ref_), hyp, edits(ref_.split(), hyp.split()), len(ref_.split())


def eval_main(a):
    name = a.label or a.model
    run = json.loads((OUT / name / 'run.json').read_text(encoding='utf-8'))
    j = Judge()
    refs = {lang: j.emb(ref(lang)[0]) for lang in REFS}
    for c in run['cases']:
        it, e, n, we, wn, sims = c['items'], 0, 0, 0, 0, []
        for x in it:
            if not Path(x['wav']).exists():  # wavs copied from another host
                x['wav'] = str(OUT / name / Path(x['wav']).name)
            sims.append(float(j.emb(x['wav']) @ refs[c['ref_lang']]))
            de, dn, x['hyp'], dw, nw = j.cer(x['wav'], x['text'], c['out_lang'])
            e, n, we, wn = e + de, n + dn, we + dw, wn + nw
            print(c['ref_lang'], c['out_lang'], round(sims[-1], 3), round(de / max(dn, 1), 3), x['hyp'], flush=True)
        audio, synth = sum(x['audio_s'] for x in it), sum(x['synth_s'] for x in it)
        record('tts_clone', run['model'], ref_lang=c['ref_lang'], out_lang=c['out_lang'], device=run['device'],
               threads=run['threads'], n=len(it), load_s=run['load_s'], clone_s=c['clone_s'],
               ttfa_s=round(np.mean([x['ttfa_s'] for x in it]), 3), ttfa_max_s=max(x['ttfa_s'] for x in it),
               synth_s=round(synth / len(it), 3), audio_s=round(audio, 2), rtf=round(synth / audio, 3),
               peak_rss_mb=run['peak_rss_mb'], rss_after_load_mb=run['rss_after_load_mb'], gpu_peak_mb=run.get('gpu_peak_mb'),
               torch=run.get('torch'), stream=run.get('stream'), temperature=run.get('temperature'), top_p=run.get('top_p'),
               cpu_per_audio_s=round(sum(x['cpu_s'] for x in it) / audio, 3) if 'cpu_s' in it[0] else None,
               spk_sim=round(np.mean(sims), 3), spk_sim_min=round(min(sims), 3), cer=round(e / n, 4),
               wer=round(we / wn, 4) if wn else None, judge=j.name, first_s=run.get('first_s', {}).get(c['out_lang']),
               cps=round(sum(len(norm(x['text']).replace(' ', '')) for x in it) / audio, 2), model_sha256=run.get('model_sha256'))
    (OUT / name / 'eval.json').write_text(json.dumps(run, ensure_ascii=False, indent=1), encoding='utf-8')


def baseline_main(a):
    """Reference points: same speaker (two halves of a reference), different speakers (ref_en vs
    ref_ru; ref vs real FLEURS speakers), and whisper CER on the real FLEURS recordings."""
    j = Judge()
    refs = {lang: j.emb(ref(lang)[0]) for lang in REFS}
    record('tts_clone', 'baseline', case='diff_speakers_ref_en_vs_ref_ru', spk_sim=round(float(refs['en'] @ refs['ru']), 3))
    for lang in REFS:
        half = float(j.emb(ref(lang)[0], 0, 20) @ j.emb(ref(lang)[0], 20))
        record('tts_clone', 'baseline', case=f'same_speaker_ref_{lang}_halves', spk_sim=round(half, 3))
    for lang, items in sentences(a.n).items():
        e = n = we = wn = 0
        sims = {r: [] for r in REFS}
        for s in items:
            path = str(SET / s[lang]['wav'])
            for r in REFS:
                sims[r].append(float(j.emb(path) @ refs[r]))
            de, dn, _, dw, nw = j.cer(path, s[lang]['text'], lang)
            e, n, we, wn = e + de, n + dn, we + dw, wn + nw
        record('tts_clone', 'baseline', case=f'real_fleurs_{lang}', out_lang=lang, n=len(items), cer=round(e / n, 4),
               wer=round(we / wn, 4) if wn else None, judge=j.name,
               cps=round(sum(len(norm(s[lang]['text']).replace(' ', '')) for s in items) / sum(s[lang]['sec'] for s in items), 2),
               **{f'spk_sim_vs_ref_{r}': round(np.mean(v), 3) for r, v in sims.items()})


def set_main(a):
    """The bench set for --langs: the --n sentences of FLEURS dev that all of them have, shortest first by their longest
    recording (at least 3 s), with a manifest like data.py's; the references (ref_en / ref_ru) stay as data.py made them."""
    import data
    rows = {lang: data.fleurs(lang) for lang in LANGS}
    secs = {lang: {i: sf.info(str(w)).duration for i, (w, _) in r.items()} for lang, r in rows.items()}
    ids = set.intersection(*(set(r) for r in rows.values()))
    longest = {i: max(secs[lang][i] for lang in LANGS) for i in ids if min(secs[lang][i] for lang in LANGS) >= 3}
    items = []
    for sid in sorted(longest, key=longest.get)[:a.n]:
        item = {'id': sid}
        for lang in LANGS:
            wav, text = rows[lang][sid]
            x, sr = sf.read(wav, dtype='float32')
            dst = SET / lang / f'{sid}.wav'
            dst.parent.mkdir(parents=True, exist_ok=True)
            sf.write(dst, x, sr, subtype='PCM_16')
            item[lang] = {'wav': str(dst.relative_to(SET)), 'text': text, 'sec': round(len(x) / sr, 2)}
        items.append(item)
    (SET / 'manifest.json').write_text(json.dumps(items, ensure_ascii=False, indent=1), encoding='utf-8')
    print(f'{len(items)} sentences x {len(LANGS)} languages, the longest recording {max(longest[i["id"]] for i in items):.1f} s '
          f'-> {SET}')


def main():
    global LANGS
    ap = argparse.ArgumentParser()
    ap.add_argument('step', choices=['synth', 'eval', 'baseline', 'set'])
    ap.add_argument('--model', choices=list(MODELS))
    ap.add_argument('--device', default='cpu', choices=['cpu', 'mps', 'cuda'])
    ap.add_argument('--n', type=int, default=6, help='sentences per language')
    ap.add_argument('--no-stream', action='store_true', help='full utterances (CosyVoice streaming on CPU is ~3x slower)')
    ap.add_argument('--label', help='results name (default: --model), e.g. one per setting')
    ap.add_argument('--gpu-mem-frac', type=float, default=0.6, help='CUDA memory cap for this process')
    ap.add_argument('--langs', default=','.join(LANGS), help='output languages, e.g. en,ru,de,ja')
    a = ap.parse_args()
    LANGS = tuple(a.langs.split(','))
    {'synth': synth_main, 'eval': eval_main, 'baseline': baseline_main, 'set': set_main}[a.step](a)


if __name__ == '__main__':
    main()
