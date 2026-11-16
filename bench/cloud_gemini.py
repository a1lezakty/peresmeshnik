"""Gemini Live cloud bench: speech-to-speech translation and streaming transcription
over the Live API, streamed at real-time pace like a call.

    GEMINI_API_KEY=... python bench/cloud_gemini.py run   [--model translate|transcribe] [--dir en-ru|ru-en] [--n 10]
    GEMINI_API_KEY=... python bench/cloud_gemini.py run   --continuous 5     # 5 utterances back to back, no pauses
    GEMINI_API_KEY=... python bench/cloud_gemini.py run   --model transcribe --stream-end 0.25   # hybrid VAD
    python bench/cloud_gemini.py score   # latency, WER/CER, chrF -> one record() row per model x direction (x mode)

`run` saves raw events (jsonl) and output audio to $GEMINI_OUT (default $BENCH_DATA/../out/gemini).
`score` reads every raw file there and writes back_index.json (translated wavs + reference); if a
back.json {wav: text} from an external ASR run (faster-whisper small) is present, it also scores that.
"""
import argparse
import asyncio
import json
import os
import re
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).parent))
from common import SET, manifest, record  # noqa: E402

MODELS = {'translate': 'gemini-3.5-live-translate-preview', 'transcribe': 'gemini-3.5-transcribe-live'}
LANG = {'en': 'en-US', 'ru': 'ru-RU'}
SR, OUT_SR = 16000, 24000
OUT = Path(os.environ.get('GEMINI_OUT', SET.parent.parent / 'out' / 'gemini'))


def speech_bounds(x, sr, frame=0.02, db=-20, min_run=5):
    """(first, last) second of speech: runs of >= min_run frames within `db` of the loudest frame
    (FLEURS noise floor sits at -30..-45 dB, so a laxer threshold fires on noise)."""
    n = int(sr * frame)
    e = np.array([np.sqrt(np.mean(x[i:i + n].astype(float) ** 2)) + 1e-9 for i in range(0, len(x) - n, n)])
    loud = np.r_[False, 20 * np.log10(e / e.max()) > db, False]
    edges = np.flatnonzero(np.diff(loud.astype(int)))
    runs = [(a, b) for a, b in zip(edges[::2], edges[1::2]) if b - a >= min_run]
    return runs[0][0] * frame, runs[-1][1] * frame


def source(items, src, uid):
    """Input pcm of one job: an utterance id, or 'contK' = first K utterances with 0.15 s gaps."""
    if str(uid).startswith('cont'):  # a speaker who does not pause
        gap = np.zeros(int(0.15 * SR), 'int16')
        return np.concatenate([np.r_[sf.read(SET / it[src]['wav'], dtype='int16')[0], gap] for it in items[:int(uid[4:])]])
    return sf.read(SET / next(it for it in items if it['id'] == uid)[src]['wav'], dtype='int16')[0]


def f0_median(x, sr):
    """Median pitch (Hz) over voiced 40 ms frames, plain autocorrelation."""
    x = x.astype(float)
    n, f0 = int(sr * 0.04), []
    thr = 0.1 * np.sqrt(np.mean(x ** 2))
    for i in range(0, len(x) - n, n):
        f = x[i:i + n] - x[i:i + n].mean()
        if np.sqrt(np.mean(f ** 2)) < thr:
            continue
        r = np.correlate(f, f, 'full')[n - 1:]
        lo, hi = int(sr / 400), int(sr / 60)
        k = lo + int(np.argmax(r[lo:hi]))
        if r[k] > 0.4 * r[0]:
            f0.append(sr / k)
    return float(np.median(f0)) if f0 else None


def config(kind, src, tgt):
    from google.genai import types
    if kind == 'translate':
        return types.LiveConnectConfig(
            response_modalities=['AUDIO'],
            input_audio_transcription=types.AudioTranscriptionConfig(),
            output_audio_transcription=types.AudioTranscriptionConfig(),
            translation_config=types.TranslationConfig(target_language_code=tgt))
    return types.LiveConnectConfig(
        response_modalities=['TEXT'],
        input_audio_transcription=types.AudioTranscriptionConfig(language_codes=[LANG[src]]))


async def stream(client, kind, src, tgt, pcm, chunk, tail, end_at=None):
    """One Live session: send `pcm` (+`tail` s silence) paced like a microphone, log every event.
    end_at: second at which a client VAD would send audio_stream_end (hybrid VAD); then just wait."""
    from google.genai import types
    cfg, model = config(kind, src, tgt), MODELS[kind]
    for attempt in range(5):
        t0 = time.perf_counter()
        try:
            cm = client.aio.live.connect(model=model, config=cfg)
            s = await cm.__aenter__()
            break
        except Exception as e:  # 429 / quota / transient: back off
            msg = str(e)
            print(f'connect failed ({msg[:120]}), retry {attempt + 1}', flush=True)
            await asyncio.sleep(15 * 2 ** attempt if ('429' in msg or 'RESOURCE' in msg or 'quota' in msg.lower()) else 5)
    else:
        raise RuntimeError('cannot connect')
    setup = time.perf_counter() - t0
    ev, audio = [], []
    pcm = np.concatenate([pcm, np.zeros(int(tail * SR), 'int16')])
    T = time.perf_counter()

    async def recv():
        while True:
            async for m in s.receive():
                t = round(time.perf_counter() - T, 3)
                sc = m.server_content
                if m.go_away:
                    ev.append([t, 'go_away', str(m.go_away.time_left)])
                if m.usage_metadata:
                    ev.append([t, 'usage', m.usage_metadata.prompt_token_count, m.usage_metadata.response_token_count])
                if m.voice_activity:
                    ev.append([t, 'vad', str(m.voice_activity.voice_activity_type), m.voice_activity.audio_offset])
                if not sc:
                    continue
                if sc.model_turn:
                    for p in sc.model_turn.parts:
                        if p.inline_data and p.inline_data.data:
                            a = np.frombuffer(p.inline_data.data, 'int16')
                            audio.append(a)
                            ev.append([t, 'audio', len(a) / OUT_SR, round(float(np.sqrt(np.mean(a.astype(float) ** 2))))])
                for k in ('interim_input_transcription', 'input_transcription', 'output_transcription'):
                    v = getattr(sc, k)
                    if v and v.text:
                        ev.append([t, k, v.text])
                for k in ('turn_complete', 'generation_complete', 'interrupted'):
                    if getattr(sc, k):
                        ev.append([t, k])

    rt = asyncio.create_task(recv())
    n = int(chunk * SR)
    for i in range(0, len(pcm), n):  # a chunk leaves when its last sample would have been captured
        await asyncio.sleep(max(0.0, T + min(i + n, len(pcm)) / SR - time.perf_counter()))
        await s.send_realtime_input(audio=types.Blob(data=pcm[i:i + n].tobytes(), mime_type=f'audio/pcm;rate={SR}'))
        if rt.done():
            break
        if end_at is not None and (i + n) / SR >= end_at:
            await s.send_realtime_input(audio_stream_end=True)
            ev.append([round(time.perf_counter() - T, 3), 'stream_end_sent'])
            await asyncio.sleep(max(0.0, T + len(pcm) / SR - time.perf_counter()))
            break
    await asyncio.sleep(0.3)
    err = None if not rt.done() else repr(rt.exception())[:300]
    rt.cancel()
    try:
        await cm.__aexit__(None, None, None)
    except Exception:
        pass
    return setup, ev, (np.concatenate(audio) if audio else np.zeros(0, 'int16')), err


async def run(a):
    from google import genai
    from google.genai import types
    client = genai.Client(http_options=types.HttpOptions(api_version='v1beta'))  # Live is v1beta-only; SDK 3.x defaults to v1
    items = manifest()
    tag = time.strftime('%Y%m%d-%H%M%S')
    for kind in a.model:
        for d in a.dir:
            src, tgt = d.split('-')
            raw = OUT / f'{kind}_{d}_{tag}.jsonl'
            (OUT / kind / d).mkdir(parents=True, exist_ok=True)
            if a.continuous:
                sel, uid = items[:a.continuous], f'cont{a.continuous}'
                jobs = [(uid, source(items, src, uid), ' '.join(it[src]['text'] for it in sel),
                         ' '.join(it[tgt]['text'] for it in sel))]
            else:
                jobs = [(it['id'], source(items, src, it['id']), it[src]['text'], it[tgt]['text']) for it in items[:a.n]]
            for uid, pcm, ref_src, ref_tgt in jobs:
                on, off = speech_bounds(pcm, SR)
                setup, ev, out, err = await stream(client, kind, src, tgt, pcm, a.chunk, a.tail,
                                                   None if a.stream_end is None else off + a.stream_end)
                wav = None
                if len(out):
                    wav = str(OUT / kind / d / f'{uid}_{tag}.wav')
                    sf.write(wav, out, OUT_SR)
                mode = 'continuous' if a.continuous else 'utterance' if a.stream_end is None else 'stream_end'
                row = dict(id=uid, kind=kind, dir=d, mode=mode, tag=tag, setup=round(setup, 3), speech_on=on, speech_off=off,
                           sec=len(pcm) / SR, tail=a.tail, chunk=a.chunk, ref_src=ref_src, ref_tgt=ref_tgt,
                           wav=wav, err=err, events=ev)
                with open(raw, 'a', encoding='utf-8') as f:
                    f.write(json.dumps(row, ensure_ascii=False) + '\n')
                m = metrics(row)
                print(kind, d, uid, {k: v for k, v in m.items() if not isinstance(v, str)}, err or '', flush=True)


def metrics(r):
    """Per-utterance latencies (s). Times are from stream start = first input sample."""
    ev, on, off = r['events'], r['speech_on'], r['speech_off']
    m = {'setup': r['setup']}
    voiced = [e for e in ev if e[1] == 'audio' and e[3] > 200]
    if voiced:
        first, last = voiced[0][0], voiced[-1][0] + voiced[-1][2]  # chunks arrive at playback pace: end = arrival + length
        m.update(audio_from_start=first - on, audio_from_end=first - off, audio_done_after_end=last - off,
                 out_speech_sec=sum(e[2] for e in voiced))
        m['cut'] = last > r['sec'] + r['tail'] - 0.5  # still talking when we hung up
    key = 'output_transcription' if r['kind'] == 'translate' else 'interim_input_transcription'
    fin = 'output_transcription' if r['kind'] == 'translate' else 'input_transcription'
    txt = [e for e in ev if e[1] in (key, fin)]
    if txt:
        m.update(text_from_start=txt[0][0] - on, text_from_end=txt[0][0] - off)
    fins = [e for e in ev if e[1] == fin]
    if fins:
        m['final_after_end'] = fins[-1][0] - off
    m['hyp'] = ''.join(e[2] for e in fins) if r['kind'] == 'translate' else ' '.join(e[2].strip() for e in fins)
    m['hyp_src'] = ''.join(e[2] for e in ev if e[1] == 'input_transcription')
    se = [e[0] for e in ev if e[1] == 'stream_end_sent']
    if se:
        m['stream_end_after_end'] = se[0] - off
    u = [e for e in ev if e[1] == 'usage']
    m['tok_in'], m['tok_out'] = sum(e[2] or 0 for e in u), sum(e[3] or 0 for e in u)
    return m


def norm(s):
    s = re.sub(r'[^\w\s]', ' ', s.lower().replace('ё', 'е'))
    return ' '.join(s.split())


def stats(v):
    v = sorted(x for x in v if x is not None)
    if not v:
        return None
    return {'med': round(statistics.median(v), 3), 'p90': round(v[min(len(v) - 1, int(0.9 * len(v)))], 3)}


def score(a):
    import jiwer
    import sacrebleu
    rows = [json.loads(line) for f in sorted(OUT.glob('*.jsonl')) for line in open(f, encoding='utf-8')]
    items = manifest()
    index, back_txt = [], json.loads((OUT / 'back.json').read_text()) if (OUT / 'back.json').exists() else {}
    mode = lambda r: r.get('mode') or ('continuous' if str(r['id']).startswith('cont') else 'utterance')  # noqa: E731
    for kind, d, md in sorted({(r['kind'], r['dir'], mode(r)) for r in rows}):
        rs = [r for r in rows if (r['kind'], r['dir'], mode(r)) == (kind, d, md)]
        cont = md == 'continuous'
        src, tgt = d.split('-')
        for r in rs:  # bounds from the current speech_bounds(), not the one the run used
            r['speech_on'], r['speech_off'] = speech_bounds(source(items, src, r['id']), SR)
        ms = [metrics(r) for r in rs]
        lat = {k: stats([m.get(k) for m in ms]) for k in
               ('setup', 'audio_from_start', 'audio_from_end', 'audio_done_after_end',
                'text_from_start', 'text_from_end', 'final_after_end', 'stream_end_after_end')}
        lat = {k: v for k, v in lat.items() if v}
        per_run = {}
        for r, m in zip(rs, ms):
            per_run.setdefault(r['tag'], []).append(m.get('audio_done_after_end', m.get('final_after_end')))
        q = {}
        if kind == 'transcribe':
            refs, hyps = [norm(r['ref_src']) for r in rs], [norm(m['hyp']) for m in ms]
            q = dict(wer=round(jiwer.wer(refs, hyps), 4), cer=round(jiwer.cer(refs, hyps), 4))
        else:
            q['chrf_out_transcript'] = round(sacrebleu.corpus_chrf([m['hyp'] for m in ms], [[r['ref_tgt'] for r in rs]]).score, 1)
            q['src_transcript_wer'] = round(jiwer.wer([norm(r['ref_src']) for r in rs], [norm(m['hyp_src']) for m in ms]), 4)
            # back-transcription runs elsewhere (no local models): OUT/back_index.json -> OUT/back.json {wav: text}
            index += [dict(id=r['id'], dir=d, lang=tgt, wav=r['wav'], ref=r['ref_tgt']) for r in rs if r['wav']]
            back = {r['wav']: back_txt.get(r['wav'], '') for r in rs}
            if back_txt:
                q['chrf_back_whisper'] = round(sacrebleu.corpus_chrf(list(back.values()), [[r['ref_tgt'] for r in rs]]).score, 1)
            f0 = []
            for r in rs:
                if r['wav'] and not cont:
                    f0.append((f0_median(source(items, src, r['id']), SR), f0_median(sf.read(r['wav'], dtype='int16')[0], OUT_SR)))
            f0 = [(i, o) for i, o in f0 if i and o]
            if len(f0) > 2:
                fi, fo = np.array(f0).T
                q['voice_f0_in_out_corr'] = round(float(np.corrcoef(fi, fo)[0, 1]), 2)
                q['voice_f0_semitone_diff_med'] = round(float(np.median(np.abs(12 * np.log2(fo / fi)))), 1)
                q['voice_gender_match'] = round(float(np.mean((fi < 165) == (fo < 165))), 2)
            q['cut_at_hangup'] = sum(bool(m.get('cut')) for m in ms)
            for r, m in zip(rs, ms):
                print(f"{r['id']}\n  ref {r['ref_tgt']}\n  out {m['hyp']}\n  whs {back.get(r['wav'], '')}")
        record('cloud', MODELS[kind], direction=d, mode=md, n=len(rs),
               runs=len(per_run), lat=lat, **q,
               lag_med_per_run={k: round(statistics.median([x for x in v if x is not None]), 2) for k, v in per_run.items()
                                if any(x is not None for x in v)},
               tok_in=sum(m['tok_in'] for m in ms), tok_out=sum(m['tok_out'] for m in ms),
               speech_sec=round(sum(r['sec'] for r in rs), 1),
               errors=[r['err'] for r in rs if r['err']][:3])
    (OUT / 'back_index.json').write_text(json.dumps(index, ensure_ascii=False, indent=1), encoding='utf-8')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=['run', 'score'])
    ap.add_argument('--model', nargs='+', default=list(MODELS), choices=list(MODELS))
    ap.add_argument('--dir', nargs='+', default=['en-ru', 'ru-en'])
    ap.add_argument('--n', type=int, default=10)
    ap.add_argument('--continuous', type=int, default=0)
    ap.add_argument('--chunk', type=float, default=0.1)
    ap.add_argument('--tail', type=float, default=6.0)
    ap.add_argument('--stream-end', type=float, help='send audio_stream_end this many s after speech end (hybrid VAD)')
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    asyncio.run(run(a)) if a.cmd == 'run' else score(a)


if __name__ == '__main__':
    x = np.r_[np.zeros(SR), np.ones(SR) * 1000, np.zeros(SR), np.ones(40) * 3000, np.zeros(SR)].astype('int16')  # click ignored
    assert speech_bounds(x, SR) == (1.0, 2.0), speech_bounds(x, SR)
    t = np.arange(SR) / SR
    assert abs(f0_median((3000 * np.sin(2 * np.pi * 120 * t)).astype('int16'), SR) - 120) < 3
    main()
