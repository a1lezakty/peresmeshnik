"""Light voice-clone engines for both directions (research for stage 8): the BENCH tts_clone method (same references,
the 6 shortest FLEURS sentences, ECAPA similarity against the full ~40 s reference, faster-whisper small judge), plus the
reference cut to 3 / 5 / 10 s: how soon a voice picked up in a call starts to sound like its owner.

    python bench/clone_light.py refs                                    # judge venv: short references cut at a pause
    python bench/clone_light.py synth --model moss --ref-s 5 --label moss-4t-r5   # the model's venv
    python bench/clone_light.py eval --label moss-4t-r5                 # judge venv: rows family=clone_light

Cases: reference {en, ru} x output {en, ru}, as BENCH; ru -> en is "me -> them", en -> ru is "them -> me" for a Russian
speaker. A model without Russian output skips those cases. Threads: $BENCH_THREADS (pin the cores with taskset).
Engines and what they need: see each load_*.
"""
import argparse
import json
import os
import queue
import re
import sys
import threading
import time
from pathlib import Path

import numpy as np
import soundfile as sf

import tts_clone as tc
from common import Timer, peak_rss_mb, record

THREADS = tc.THREADS
OUT = tc.OUT
REFS = OUT / 'refs'


def ref(lang, sec):
    """(wav, transcript) of the reference: the full ~40 s one (sec 0) or its first `sec` seconds, cut at a pause."""
    if not sec:
        return tc.ref(lang)
    p = REFS / f'ref_{lang}_{sec}s.wav'
    return str(p), p.with_suffix('.txt').read_text(encoding='utf-8').strip()


def stream_cb(run):
    """A generator over what `run(emit)` emits from its own thread (engines that stream through a callback)."""
    q = queue.Queue()

    def work():
        try:
            run(q.put)
        except BaseException as e:  # noqa: BLE001 - re-raised in the consumer
            q.put(e)
        q.put(None)
    threading.Thread(target=work, daemon=True).start()
    while (c := q.get()) is not None:
        if isinstance(c, BaseException):
            raise c
        yield c


def quiet_ort(spin):
    """onnxruntime sessions whose idle intra-op threads sleep instead of spinning: these engines make hundreds of small
    calls per second, and spinning threads fight each other for the cores (MOSS on 2 threads: RTF 0.97 -> 0.47)."""
    import onnxruntime as ort
    if spin or getattr(ort.SessionOptions, 'quiet', False):
        return
    base = ort.SessionOptions

    def options():
        o = base()
        o.add_session_config_entry('session.intra_op.allow_spinning', '0')
        return o
    options.quiet = True
    ort.SessionOptions = options


# --- adapters: load(device, args) -> clone(ref_lang) -> cond; synth(text, lang, cond, ref_lang) -> iter of chunks; sr; langs

def load_moss(device, a):
    """MOSS-TTS-Nano (100M, Apache-2.0), upstream's ONNX runtime: $MOSS_SRC = a checkout of OpenMOSS/MOSS-TTS-Nano,
    $MOSS_MODELS = a folder with MOSS-TTS-Nano-100M-ONNX and MOSS-Audio-Tokenizer-Nano-ONNX. No torch: upstream imports it
    only to read the reference file, which is done here with soundfile + scipy."""
    import types
    from scipy.signal import resample_poly
    stubs = [n for n in ('torch', 'torchaudio') if sys.modules.setdefault(n, s := types.ModuleType(n)) is s]
    sys.path.insert(0, os.environ['MOSS_SRC'])
    quiet_ort(a.spin)
    import onnx_tts_runtime as rt
    for name in stubs:  # scipy looks at sys.modules['torch']
        del sys.modules[name]
    m = rt.OnnxTtsRuntime(model_dir=os.environ['MOSS_MODELS'], thread_count=THREADS, execution_provider=device,
                          output_dir=OUT / 'moss-tmp')
    cfg = m.codec_meta['codec_config']

    def read_ref(path):
        x, sr = sf.read(str(path), dtype='float32')
        x = resample_poly(x, int(cfg['sample_rate']), sr).astype('float32')
        return np.tile(x, (int(cfg['channels']), 1))[None]
    m._load_reference_audio = read_ref

    def clone(lang):
        return m.encode_reference_audio(ref(lang, a.ref_s)[0])

    def synth(text, lang, cond, ref_lang):
        text = m.prepare_synthesis_text(text=text, voice='', enable_wetext=False, enable_normalize_tts_text=True)['text']
        rows = m.build_voice_clone_request_rows(cond, m.encode_text(text))

        def run(emit):
            pend = []

            def flush():
                if pend:
                    d = m.codec_streaming_session.run_frames(pend[:])
                    pend.clear()
                    if d is not None and d[1] > 0:
                        emit(d[0][0, :, :d[1]].mean(0))

            def on_frame(_frames, _step, frame):
                pend.append(list(frame))
                if len(pend) >= a.hop:
                    flush()
            m.codec_streaming_session.reset()
            m.generate_audio_frames(rows, on_frame=on_frame)
            flush()
            m.codec_streaming_session.reset()
        yield from stream_cb(run)
    return clone, synth, int(cfg['sample_rate']), {'en', 'ru'}


def load_pocket(device, a):
    """Kyutai Pocket TTS (100M, CC-BY-4.0) through the community ONNX export: $POCKET_DIR = a snapshot of
    KevinAHM/pocket-tts-onnx (pocket_tts_onnx.py + onnx/<bundle>). English bundle here; no Russian upstream."""
    sys.path.insert(0, os.environ['POCKET_DIR'])
    quiet_ort(a.spin)
    import pocket_tts_onnx as p

    def opts(self):
        o = p.ort.SessionOptions()
        o.intra_op_num_threads, o.inter_op_num_threads = THREADS, 1
        return o
    p.PocketTTSOnnx._make_session_options = opts
    m = p.PocketTTSOnnx(models_dir=os.environ['POCKET_DIR'] + '/onnx', language=a.bundle or 'english_2026-04',
                        precision=a.precision, device=device)

    def clone(lang):
        return m.encode_voice(ref(lang, a.ref_s)[0])

    def synth(text, lang, cond, ref_lang):
        yield from m.stream(text, cond)
    return clone, synth, m.sample_rate, {'en'}


def proc_stat(pid):
    """(CPU seconds, peak RSS MB) of another process: engines that run as a local server. The peak is the kernel's on
    Linux; elsewhere (psutil) it is the RSS at the time of the call."""
    try:
        f = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()
        hwm = re.search(r'VmHWM:\s+(\d+)', Path(f'/proc/{pid}/status').read_text())
        return (int(f[11]) + int(f[12])) / os.sysconf('SC_CLK_TCK'), round(int(hwm.group(1)) / 1024)
    except FileNotFoundError:
        import psutil
        p = psutil.Process(pid)
        t = p.cpu_times()
        return t.user + t.system, round(p.memory_info().rss / 2**20)


def load_zonos(device, a):
    """ZONOS3-Mini (Apache-2.0; 800M weights, ~100M active per token) on its C++/ggml server: $ZONOS_BIN = the build
    folder (zonos3-server, spk-encoder-cli), $ZONOS_GGUF = zonos3-ptq.gguf + zac-keep2.gguf + spk-encoder.gguf.
    The voice is one 192-d embedding from the reference; text goes in raw (no NeMo worker)."""
    import atexit
    import base64
    import subprocess
    import urllib.request
    bin_, gguf, port = os.environ['ZONOS_BIN'], os.environ['ZONOS_GGUF'], int(os.environ.get('ZONOS_PORT', 19190))
    env = dict(os.environ, OMP_NUM_THREADS=str(THREADS))
    srv = subprocess.Popen([f'{bin_}/zonos3-server', f'{gguf}/zonos3-ptq.gguf', '--port', str(port),
                            '--gpu' if device != 'cpu' else '--cpu', *os.environ.get('ZONOS_ARGS', '').split()],
                           env=env, stdout=open(OUT / a.label / 'server.log', 'w'), stderr=subprocess.STDOUT)
    atexit.register(srv.terminate)
    url = f'http://127.0.0.1:{port}'
    for _ in range(600):  # wait for the model
        try:
            urllib.request.urlopen(url + '/v2/capabilities', timeout=2).read()
            break
        except OSError:
            if srv.poll() is not None:
                sys.exit(f'zonos3-server exited {srv.returncode}: see server.log')
            time.sleep(0.5)
    a.server_pid = srv.pid

    def clone(lang):
        emb = OUT / a.label / f'emb_{lang}.npy'
        subprocess.run([f'{bin_}/spk-encoder-cli', f'{gguf}/spk-encoder.gguf', '--audio', ref(lang, a.ref_s)[0], str(emb),
                        '--cpu'], check=True, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return base64.b64encode(emb.read_bytes()).decode()

    def synth(text, lang, cond, ref_lang):
        body = {'text': text, 'stream': True, 'speaker_embedding_base64': cond, 'speaker_mode': 'embedding',
                'text_normalization': False, 'seed': 1, **json.loads(os.environ.get('ZONOS_BODY', '{}'))}
        req = urllib.request.Request(url + '/v2/generate', json.dumps(body).encode(), {'Content-Type': 'application/json'})
        with urllib.request.urlopen(req, timeout=600) as r:
            tail = b''
            while (b := r.read1(65536)):
                b = tail + b
                n = len(b) // 4 * 4
                tail = b[n:]
                if n:
                    yield np.frombuffer(b[:n], '<f4')
    return clone, synth, 44100, {'en'}  # also fr de es it pt ja zh ko; no Russian: its text comes out as babble


PIPER = {'en': 'en/en_US/joe/medium/en_US-joe-medium', 'ru': 'ru/ru_RU/dmitri/medium/ru_RU-dmitri-medium'}  # the app's


def resample(x, sr, to):
    from scipy.signal import resample_poly
    g = np.gcd(int(sr), int(to))
    return x if sr == to else resample_poly(x, to // g, sr // g).astype('float32')


def load_pipervc(device, a):
    """The app's stock Piper voice, then zero-shot voice conversion to the reference speaker: any language Piper has.
    --vc openvoice | facodec | knnvc | chatterbox (voiceclonnx's ONNX exports) or none (Piper alone: the floor).
    $PIPER_DIR = rhasspy/piper-voices files. OpenVoice and FACodec keep one 256-d vector per voice; the other engines
    get the reference file on every call (as voiceclonnx does it)."""
    os.cpu_count = lambda: THREADS  # voiceclonnx sizes its sessions by it
    quiet_ort(a.spin)
    from piper import PiperVoice
    voices = {lang: PiperVoice.load(f"{os.environ['PIPER_DIR']}/{v}.onnx") for lang, v in PIPER.items()}

    def tts(text, lang):
        ch = list(voices[lang].synthesize(text))
        return np.concatenate([c.audio_float_array for c in ch]).astype('float32'), ch[0].sample_rate

    def read(lang, sr):
        x, xsr = sf.read(ref(lang, a.ref_s)[0], dtype='float32')
        return resample(x, xsr, sr)
    if a.vc == 'none':
        return (lambda lang: None), (lambda text, lang, cond, ref_lang: iter([tts(text, lang)[0]])), 22050, set(PIPER)
    from voiceclonnx import VoiceCloner
    vc = VoiceCloner(engine=a.vc, quantized=a.precision == 'int8')
    ad, sr = vc._adapter, vc.sample_rate
    if a.vc == 'openvoice':
        from voiceclonnx.engines import openvoice as ov
        ad._ensure_models()
        own = {}  # the stock voice's own tone embedding: once per Piper voice

        def clone(lang):
            return ad._extract_tone_embedding(read(lang, sr))

        def synth(text, lang, cond, ref_lang):
            x = resample(*tts(text, lang), sr)
            if lang not in own:
                own[lang] = ad._extract_tone_embedding(x)
            spec = ov._compute_linear_spec(x)
            yield ad._converter_sess.run(None, {'spec': spec[None], 'spec_lengths': np.array([spec.shape[1]], 'int64'),
                                                'src_g': own[lang][:, :, None], 'tgt_g': cond[:, :, None]})[0][0, 0]
    elif a.vc == 'facodec':
        from voiceclonnx.engines import facodec as fa
        ad._ensure_models()

        def clone(lang):
            return ad._get_timbre(ad._encode(read(lang, sr)))

        def synth(text, lang, cond, ref_lang):
            x = resample(*tts(text, lang), sr)
            feats, mel = ad._encode(x), fa._compute_prosody_mel(x)
            t = feats.shape[2]  # as voiceclonnx: the prosody mel cut or padded to the encoder's length
            mel = mel[:, :, :t] if mel.shape[2] >= t else np.pad(mel, ((0, 0), (0, 0), (0, t - mel.shape[2])),
                                                                  constant_values=np.log(1e-5))
            yield ad._decode(ad._quantize(feats, mel), cond)
    else:
        tmp = OUT / a.label

        def clone(lang):
            return ref(lang, a.ref_s)[0]

        def synth(text, lang, cond, ref_lang):
            x, xsr = tts(text, lang)
            sf.write(tmp / 'src.wav', x, xsr, subtype='PCM_16')
            vc.clone_voice(str(tmp / 'src.wav'), cond, str(tmp / 'vc.wav'))
            yield sf.read(tmp / 'vc.wav', dtype='float32')[0]
    return clone, synth, sr, set(PIPER)


def load_qwen(device, a):
    """Qwen3-TTS 0.6B Base (Apache-2.0), the official qwen-tts package: no audio streaming there, so the first sound is
    the whole clause. --xvec: the voice is one speaker vector; without it the reference and its transcript are the prompt."""
    import torch
    from qwen_tts import Qwen3TTSModel
    m = Qwen3TTSModel.from_pretrained(os.environ.get('QWEN_MODEL', 'Qwen/Qwen3-TTS-12Hz-0.6B-Base'),
                                      device_map='cuda:0' if device == 'cuda' else device,
                                      dtype=torch.bfloat16 if device == 'cuda' else torch.float32)
    names = {'en': 'English', 'ru': 'Russian'}

    def clone(lang):
        wav, text = ref(lang, a.ref_s)
        return m.create_voice_clone_prompt(ref_audio=wav, ref_text=text, x_vector_only_mode=a.xvec)

    def synth(text, lang, cond, ref_lang):
        wavs, sr = m.generate_voice_clone(text=text, language=names[lang], voice_clone_prompt=cond)
        assert sr == 24000, sr
        yield wavs[0]
    return clone, synth, 24000, set(names)


MODELS = {'qwen': load_qwen, 'moss': load_moss, 'pocket': load_pocket, 'zonos': load_zonos, 'pipervc': load_pipervc}


def synth_main(a):
    try:
        import torch
        torch.set_num_threads(THREADS)
        if a.device == 'cuda':
            torch.cuda.set_per_process_memory_fraction(a.gpu_mem_frac)  # shared GPU: stay inside the cap
    except ImportError:
        torch = None
    out = OUT / a.label
    out.mkdir(parents=True, exist_ok=True)
    with Timer() as tl:
        clone, synth, sr, langs = MODELS[a.model](a.device, a)
    run = {'model': a.label, 'engine': a.model, 'device': a.device, 'threads': THREADS, 'ref_s': a.ref_s,
           'load_s': round(tl.s, 2), 'rss_after_load_mb': peak_rss_mb(), 'cases': []}
    run['args'] = {k: v for k, v in vars(a).items() if k != 'server_pid'}
    pid = getattr(a, 'server_pid', None)  # the engine is a local server: its CPU time and memory count
    cpu, rss = ((lambda: proc_stat(pid)[0]) if pid else time.process_time), 0
    sents, warm = tc.sentences(a.n), set()
    for case in a.cases.split(','):
        ref_lang, out_lang = case.split('2')
        if out_lang not in langs:
            continue
        with Timer() as tcl:
            cond = clone(ref_lang)
        if out_lang not in warm:  # warm-up per output language, not timed
            for _ in synth(sents[out_lang][0][out_lang]['text'], out_lang, cond, ref_lang):
                pass
            warm.add(out_lang)
        items = []
        for s in sents[out_lang]:
            text, chunks, cpu0 = s[out_lang]['text'], [], cpu()
            with Timer() as ts:
                for c in synth(text, out_lang, cond, ref_lang):
                    if not chunks:
                        ttfa = time.perf_counter() - ts.t
                    chunks.append(np.asarray(c, dtype='float32').reshape(-1))
            wav = np.concatenate(chunks)
            path = out / f'{ref_lang}2{out_lang}_{s["id"]}.wav'
            sf.write(path, wav, sr)
            items.append({'id': s['id'], 'text': text, 'wav': str(path), 'ttfa_s': round(ttfa, 3), 'synth_s': round(ts.s, 3),
                          'audio_s': round(len(wav) / sr, 3), 'cpu_s': round(cpu() - cpu0, 3)})
            print(ref_lang, out_lang, items[-1], flush=True)
            rss = max(rss, proc_stat(pid)[1]) if pid else 0
        run['cases'].append({'ref_lang': ref_lang, 'out_lang': out_lang, 'clone_s': round(tcl.s, 2), 'items': items})
    run['peak_rss_mb'] = rss if pid else peak_rss_mb()
    if a.device == 'cuda' and torch:
        run['gpu_peak_mb'] = round(torch.cuda.max_memory_allocated() / 2**20)
    (out / 'run.json').write_text(json.dumps(run, ensure_ascii=False, indent=1), encoding='utf-8')
    print('peak RSS MB', run['peak_rss_mb'], 'GPU MB', run.get('gpu_peak_mb'))


def eval_main(a):
    run = json.loads((OUT / a.label / 'run.json').read_text(encoding='utf-8'))
    j = tc.Judge()
    refs = {lang: j.emb(tc.ref(lang)[0]) for lang in tc.REFS}  # always against the full reference
    for c in run['cases']:
        it, ce, cn, we, wn, sims = c['items'], 0, 0, 0, 0, []
        for x in it:
            if not Path(x['wav']).exists():  # wavs copied from another host
                x['wav'] = str(OUT / a.label / Path(x['wav']).name)
            sims.append(float(j.emb(x['wav']) @ refs[c['ref_lang']]))
            de, dn, x['hyp'], dw, nw = j.cer(x['wav'], x['text'], c['out_lang'])
            x['spk_sim'] = round(sims[-1], 3)
            ce, cn, we, wn = ce + de, cn + dn, we + dw, wn + nw
            print(c['ref_lang'], c['out_lang'], x['spk_sim'], round(de / max(dn, 1), 3), x['hyp'], flush=True)
        audio, synth = sum(x['audio_s'] for x in it), sum(x['synth_s'] for x in it)
        record('clone_light', run['model'], engine=run['engine'], ref_lang=c['ref_lang'], out_lang=c['out_lang'],
               device=run['device'], threads=run['threads'], ref_s=run['ref_s'], n=len(it), load_s=run['load_s'],
               clone_s=c['clone_s'], ttfa_s=round(float(np.mean([x['ttfa_s'] for x in it])), 3),
               ttfa_max_s=max(x['ttfa_s'] for x in it), rtf=round(synth / audio, 3), audio_s=round(audio, 2),
               cpu_per_audio_s=round(sum(x['cpu_s'] for x in it) / audio, 3), synth_rss_mb=run['peak_rss_mb'],
               rss_after_load_mb=run['rss_after_load_mb'], gpu_peak_mb=run.get('gpu_peak_mb'),
               spk_sim=round(float(np.mean(sims)), 3), spk_sim_min=round(min(sims), 3), cer=round(ce / cn, 4),
               wer=round(we / wn, 4) if wn else None, judge=j.name,
               cps=round(sum(len(tc.norm(x['text']).replace(' ', '')) for x in it) / audio, 2))
    (OUT / a.label / 'eval.json').write_text(json.dumps(run, ensure_ascii=False, indent=1), encoding='utf-8')


def refs_main(a):
    """Cut each ~40 s reference at the quietest 100 ms of the last 2 s before N s; the judge writes the transcript."""
    j = tc.Judge()
    REFS.mkdir(parents=True, exist_ok=True)
    for lang in tc.REFS:
        x, sr = sf.read(tc.ref(lang)[0], dtype='float32')
        f = int(0.02 * sr)
        db = 20 * np.log10(np.sqrt(np.mean(x[:len(x) // f * f].reshape(-1, f) ** 2, 1)) + 1e-9)
        sm = np.convolve(db, np.ones(5) / 5, mode='same')
        for sec in a.secs:
            lo, hi = int((sec - 2) / 0.02), int(sec / 0.02)
            cut = (lo + int(np.argmin(sm[lo:hi]))) * f
            p = REFS / f'ref_{lang}_{sec}s.wav'
            sf.write(p, x[:cut], sr, subtype='PCM_16')
            hyp = ' '.join(s.text for s in j.asr.transcribe(str(p), language=lang, beam_size=5)[0]).strip()
            p.with_suffix('.txt').write_text(hyp, encoding='utf-8')
            print(lang, sec, round(cut / sr, 2), hyp, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('step', choices=['refs', 'synth', 'eval'])
    ap.add_argument('--model', choices=list(MODELS))
    ap.add_argument('--device', default='cpu', choices=['cpu', 'cuda', 'mps'])
    ap.add_argument('--n', type=int, default=6)
    ap.add_argument('--label')
    ap.add_argument('--cases', default='ru2en,en2ru,en2en,ru2ru')
    ap.add_argument('--ref-s', type=int, default=5, help='reference seconds (0 = the full ~40 s)')
    ap.add_argument('--secs', type=int, nargs='+', default=[3, 5, 10])
    ap.add_argument('--hop', type=int, default=2, help='codec frames per streamed chunk (moss)')
    ap.add_argument('--precision', default='int8', choices=['int8', 'fp32'])
    ap.add_argument('--bundle', help="the engine's model variant (pocket: english_2026-04, ...)")
    ap.add_argument('--vc', default='openvoice', help='pipervc: openvoice, facodec, knnvc, chatterbox, none')
    ap.add_argument('--xvec', action='store_true', help='qwen: the voice as one speaker vector')
    ap.add_argument('--spin', action='store_true', help="onnxruntime's default: idle threads spin (see quiet_ort)")
    ap.add_argument('--gpu-mem-frac', type=float, default=0.6)
    a = ap.parse_args()
    {'refs': refs_main, 'synth': synth_main, 'eval': eval_main}[a.step](a)


if __name__ == '__main__':
    main()
