"""The local pipeline on conversational speech (stage 2): clause segmentation settings, and the voice's
lag behind the speaker with and without the queue policy.

Data (only numbers go to the repo): AMI meetings (CC-BY-4.0; English, four people, overlaps, "um"),
YODAS2 ru000 (CC-BY-3.0; Russian YouTube videos with uploaded subtitles), FLEURS (CC-BY-4.0) put back
to back as continuous read speech, the worst case for the voice queue.

    python bench/conversation.py prep --src DIR        # AMI wavs + annotation zip, YODAS2 shard -> $BENCH_DATA/conv
    python bench/conversation.py segment en --vad 0.5 --silence 300 --max-clause 5   # WER, chrF, clause lengths
    python bench/conversation.py lag ru --max-lag 3 --max-clause 5 [--continuous]    # real time with Piper

`segment` scores the streamed clauses against the reference words (WER) and their translations against
the translation of the reference segments (chrF; a pseudo-reference: there are no human translations),
next to the offline ASR of the reference segments. `lag` feeds the recording in real time and reads the
per-clause latency log: when each clause's voice starts after the clause ended (back to back playback).
"""
import argparse
import json
import logging
import re
import sys
import tarfile
import threading
import unicodedata
import zipfile
from pathlib import Path
from xml.etree import ElementTree

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
# isort: off
from peresmeshnik.pipeline import LocalDirection, make_local, read_audio, run  # noqa: E402  first: it pins BLAS threads
import numpy as np  # noqa: E402
# isort: on

from common import ROOT, SET, manifest, record  # noqa: E402

SR = 16000
CONV = ROOT / 'conv'
AMI = ('ES2002a', 'IS1009a')
FILLERS = {'um', 'uh', 'hmm', 'mm', 'mhm', 'mm-hmm', 'uh-huh', 'eh', 'ah', 'er', 'erm', 'э', 'ээ', 'эээ', 'эм', 'мм', 'ммм', 'хм', 'а-а'}


def norm(s: str) -> str:
    s = unicodedata.normalize('NFKC', s).lower().replace('ё', 'е')
    s = re.sub(r"[^\w\s'-]|_", ' ', s)
    return ' '.join(w for w in s.split() if w.strip("'-") and w not in FILLERS)


class NoTts:
    sample_rate = SR

    def synth(self, text, lang, voice=None):
        yield from ()


def pct(xs, q):
    return round(float(np.percentile(xs, q)), 2) if len(xs) else None


# data
def save(lang, name, pcm, meta):
    import soundfile as sf
    d = CONV / lang
    d.mkdir(parents=True, exist_ok=True)
    pcm = pcm * (0.05 / max(np.sqrt(np.mean(pcm ** 2)), 1e-6))  # -26 dBFS RMS, a call app's AGC (as tests/data)
    sf.write(d / f'{name}.wav', np.clip(pcm, -1, 1).astype(np.float32), SR)
    (d / f'{name}.json').write_text(json.dumps(meta, ensure_ascii=False), 'utf-8')
    print(lang, name, f'{len(pcm) / SR:.0f} s', len(meta['words']), 'words', len(meta['segments']), 'segments')


def prep_ami(src: Path, seconds: float):
    z = zipfile.ZipFile(src / 'ami_public_manual_1.6.2.zip')
    for m in AMI:
        words = []
        for name in z.namelist():
            if name.startswith(f'words/{m}.') and name.endswith('.words.xml'):
                spk = name.split('.')[1]
                for w in ElementTree.fromstring(z.read(name)).iter('w'):
                    if w.get('starttime') and w.text:
                        words.append((float(w.get('starttime')), float(w.get('endtime')), spk, w.text, w.get('punc') == 'true'))
        words.sort()
        t0 = next(w[0] for w in words if not w[4]) - 0.5
        t1 = t0 + seconds
        pcm = read_audio(str(src / f'{m}.Mix-Headset.wav'))[int(t0 * SR):int(t1 * SR)]
        keep = [w for w in words if t0 <= w[0] and w[1] <= t1]
        segs = []  # a speaker's run of words with gaps < 1 s
        for s, e, spk, text, punc in keep:
            if segs and segs[-1][3] == spk and (punc or s - segs[-1][1] < 1.0):
                segs[-1][1] = max(segs[-1][1], e)
                segs[-1][2] += text if punc else ' ' + text
            elif not punc:
                segs.append([s, e, text, spk])
        save('en', m, pcm, {'source': f'AMI {m} Mix-Headset (CC-BY-4.0)', 'offset_s': round(t0, 2),
                            'words': [[round(w[0] - t0, 2), w[3]] for w in keep if not w[4]],
                            'segments': [[round(s - t0, 2), round(e - t0, 2), t] for s, e, t, _ in segs]})


def prep_yodas(src: Path, seconds: float, keep_n: int, candidates: int):
    """Videos with subtitles that cover most of the first `seconds` and look hand-made (punctuation), the
    `keep_n` whose offline ASR agrees best with the subtitles (some uploaded subtitles are loose)."""
    import jiwer

    from peresmeshnik.local.asr import OnnxAsr
    videos = json.loads((src / 'yodas2-ru000-text-00000000.json').read_text('utf-8'))
    durs = dict(line.split() for line in (src / 'yodas2-ru000-duration-00000000.txt').read_text().splitlines())
    cands = []
    for v in videos:
        if float(durs.get(v['audio_id'], 0)) < seconds + 30:
            continue
        segs = sorted((int(k.split('-')[-2]) / 100, int(k.split('-')[-1]) / 100, t) for k, t in v['text'].items())
        if not segs:
            continue
        t0 = segs[0][0]
        win = [s for s in segs if s[0] >= t0 and s[1] <= t0 + seconds]
        cover = sum(e - s for s, e, _ in win) / seconds
        punct = sum(bool(re.search(r'[.,?!]', t)) for _, _, t in win) / max(len(win), 1)
        if cover > 0.6 and punct > 0.5:
            cands.append((cover, v['audio_id'], t0, win))
    cands = sorted(cands, reverse=True)[:candidates]
    want = {f'./{c[1]}.wav' for c in cands}
    audio = {}
    with tarfile.open(src / 'yodas2-ru000-audio-00000000.tar.gz', 'r:gz') as tar:
        for member in tar:
            if member.name in want:
                import soundfile as sf
                x, sr = sf.read(tar.extractfile(member), dtype='float32', always_2d=True)
                x = x.mean(axis=1)
                audio[member.name[2:-4]] = np.interp(np.arange(0, len(x), sr / SR), np.arange(len(x)), x).astype(np.float32)
    asr = OnnxAsr('gigaam-v3-e2e-ctc', threads=4)
    scored = []
    for cover, vid, t0, win in cands:
        pcm = audio[vid][int(t0 * SR):int((t0 + seconds) * SR)]
        ref = norm(' '.join(t for _, _, t in win))
        hyp = norm(' '.join(asr.recognize(pcm[int(max(0, s - t0 - 0.2) * SR):int((e - t0 + 0.2) * SR)]).text for s, e, _ in win))
        wer = jiwer.wer(ref, hyp)
        print(vid, f'cover {cover:.2f} offline WER {wer:.3f}', ref[:80])
        scored.append((wer, vid, t0, win, pcm))
    for wer, vid, t0, win, pcm in sorted(scored, key=lambda s: s[0])[:keep_n]:
        words = [[round(s - t0 + (e - s) * i / max(len(t.split()), 1), 2), w] for s, e, t in win for i, w in enumerate(t.split())]
        save('ru', vid, pcm, {'source': f'YODAS2 ru000 {vid} (CC-BY-3.0)', 'offset_s': t0, 'subtitle_offline_wer': round(wer, 3),
                              'words': words, 'segments': [[round(s - t0, 2), round(e - t0, 2), t] for s, e, t in win]})


def prep(a):
    prep_ami(a.src, a.seconds)
    prep_yodas(a.src, a.seconds, a.keep, a.candidates)


def items(lang):
    for j in sorted((CONV / lang).glob('*.json')):
        yield j.stem, read_audio(str(j.with_suffix('.wav'))), json.loads(j.read_text('utf-8'))


def continuous(lang, n=30):
    """FLEURS utterances back to back (0.15 s apart, under the VAD pause): continuous read speech."""
    xs = []
    for it in manifest()[:n]:
        x = read_audio(str(SET / it[lang]['wav']))
        xs += [x * (0.05 / np.sqrt(np.mean(x ** 2))), np.zeros(int(0.15 * SR), np.float32)]
    return np.concatenate(xs)


class Rows(logging.Handler):
    def __init__(self):
        super().__init__()
        self.rows = []

    def emit(self, r):
        self.rows.append(json.loads(r.getMessage()))

    def __enter__(self):
        log = logging.getLogger('peresmeshnik.latency')
        log.addHandler(self)
        log.setLevel(logging.INFO)
        return self

    def __exit__(self, *e):
        logging.getLogger('peresmeshnik.latency').removeHandler(self)


def drain(d):
    t = threading.Thread(target=lambda: list(d.events()), daemon=True)
    t.start()
    return t


def segment(a):
    import jiwer
    import sacrebleu

    from peresmeshnik.local.vad import SileroVad
    src, tgt = a.lang, 'ru' if a.lang == 'en' else 'en'
    base = make_local(src, tgt, a.asr, a.threads, tts=NoTts())
    hyps, refs, offl, trs, otrs, prefs, rows = [], [], [], [], [], [], []
    for name, pcm, meta in items(src):
        d = LocalDirection(SileroVad(threshold=a.vad, min_silence_ms=a.silence), base.asr, base.mt, base.tts, src, tgt,
                           max_clause_s=a.max_clause)
        d.asr.context_s = a.context
        d.realtime = False
        with Rows() as r:
            t = drain(d)
            run(d, np.concatenate([pcm, np.zeros(SR, np.float32)]))
            d.finish()
            t.join()
        rows += r.rows
        hyps.append(norm(' '.join(x['text'] for x in r.rows)))
        trs.append(' '.join(x['translation'] for x in r.rows))
        refs.append(norm(' '.join(w for _, w in meta['words'])))
        segs = meta['segments']
        prefs.append(' '.join(base.mt.translate(t, src, tgt) for _, _, t in segs))  # pseudo-reference translation
        texts = [base.asr.recognize(pcm[int(max(0, s - 0.2) * SR):int((e + 0.2) * SR)]).text for s, e, _ in segs]
        offl.append(norm(' '.join(texts)))  # offline: each reference segment in one window, translated whole
        otrs.append(' '.join(base.mt.translate(t, src, tgt) for t in texts if t.strip()))
    dur = [x['end_s'] - x['start_s'] for x in rows]
    nw = [len(x['text'].split()) for x in rows]
    # how long a word waits for its clause to end (words spread evenly over the clause): the voice can start no earlier
    wait = [x['end_s'] - (x['start_s'] + (x['end_s'] - x['start_s']) * (i + 0.5) / n) for x, n in zip(rows, nw) for i in range(n)]
    reasons = {k: round(sum(x['reason'] == k for x in rows) / len(rows), 3) for k in ('pause', 'punct', 'max', 'eof')}
    record('pipeline', f'{base.asr.name}+opus-mt', lang=src, mode='segment', data='+'.join(n for n, _, _ in items(src)),
           audio_s=round(sum(len(p) for _, p, _ in items(src)) / SR), vad_threshold=a.vad, min_silence_ms=a.silence,
           max_clause_s=a.max_clause, context_s=a.context, threads=a.threads, clauses=len(rows),
           wer=round(jiwer.wer(refs, hyps), 4), wer_offline=round(jiwer.wer(refs, offl), 4),
           chrf=round(sacrebleu.corpus_chrf(trs, [prefs]).score, 1), chrf_offline=round(sacrebleu.corpus_chrf(otrs, [prefs]).score, 1),
           clause_s_p50=pct(dur, 50), clause_s_p95=pct(dur, 95), short_share=round(sum(n <= 2 for n in nw) / len(nw), 3),
           wait_s_p50=pct(wait, 50), wait_s_p95=pct(wait, 95), reasons=reasons,
           samples=[x['text'] for x in rows[:6]])


def lag(a):
    from peresmeshnik import pipeline
    pipeline.FAST_AFTER_S, pipeline.MAX_SPEED = a.fast_after, a.max_speed
    src, tgt = a.lang, 'ru' if a.lang == 'en' else 'en'
    d = make_local(src, tgt, a.asr, a.threads)
    d.warm_up()
    data = [('fleurs-continuous', continuous(src), None)] if a.continuous else list(items(src))
    rows, audio = [], 0.0
    for name, pcm, _ in data:
        d = LocalDirection(type(d.vad)(), d.asr, d.mt, d.tts, src, tgt, max_clause_s=a.max_clause,
                           max_lag_s=a.max_lag or None)
        with Rows() as r:
            t = drain(d)
            run(d, np.concatenate([pcm, np.zeros(SR * 3 // 2, np.float32)]), realtime=True)
            d.finish()
            t.join()
        rows += r.rows
        audio += len(pcm) / SR
    spoken = [x for x in rows if 'dropped' not in x]
    lags = [x['voice_lag_s'] for x in spoken if 'voice_lag_s' in x]
    nw = lambda x: len(x['text'].split())  # noqa: E731
    # ear-voice span per word (words spread evenly over the clause) up to the start of the clause's voice
    evs = [x['end_s'] - (x['start_s'] + (x['end_s'] - x['start_s']) * (i + 0.5) / nw(x)) + x['voice_lag_s']
           for x in spoken if 'voice_lag_s' in x for i in range(nw(x))]
    record('pipeline', f'{d.asr.name}+opus-mt+piper', lang=src, mode='lag', data='fleurs-continuous' if a.continuous else 'conversation',
           audio_s=round(audio), max_lag_s=a.max_lag or None, max_clause_s=a.max_clause, threads=a.threads,
           clauses=len(rows), dropped_clauses=round(1 - len(spoken) / len(rows), 3),
           dropped_words=round(1 - sum(map(nw, spoken)) / sum(map(nw, rows)), 3),
           voice_lag_s_p50=pct(lags, 50), voice_lag_s_p95=pct(lags, 95), voice_lag_s_max=round(max(lags), 2),
           evs_s_p50=pct(evs, 50), evs_s_p95=pct(evs, 95),
           fast_after_s=a.fast_after, max_speed=a.max_speed, speed_mean=round(float(np.mean([x.get('speed', 1.0) for x in spoken])), 3),
           speech_per_audio=round(sum(x['speech_s'] for x in spoken) / audio, 3))
    d.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('mode', choices=['prep', 'segment', 'lag'])
    ap.add_argument('lang', nargs='?', choices=['en', 'ru'])
    ap.add_argument('--src', type=Path, help='prep: folder with the downloads')
    ap.add_argument('--seconds', type=float, default=300.0, help='prep: excerpt length')
    ap.add_argument('--keep', type=int, default=2, help='prep: YODAS videos kept')
    ap.add_argument('--candidates', type=int, default=8, help='prep: YODAS videos scored')
    ap.add_argument('--asr', help='ASR model id (default: settings)')
    ap.add_argument('--threads', type=int, default=2)
    ap.add_argument('--vad', type=float, default=0.5, help='Silero speech threshold')
    ap.add_argument('--silence', type=int, default=300, help='VAD pause that ends a clause, ms')
    ap.add_argument('--max-clause', type=float, default=5.0)
    ap.add_argument('--context', type=float, default=1.0)
    ap.add_argument('--max-lag', type=float, default=4.0, help='lag: queue policy cap, s (0 = no policy)')
    ap.add_argument('--continuous', action='store_true', help='lag: FLEURS back to back instead of the conversations')
    ap.add_argument('--fast-after', type=float, default=0.5, help='lag: speed up from this lag, s (pipeline.FAST_AFTER_S)')
    ap.add_argument('--max-speed', type=float, default=1.4, help='lag: up to this speed at the cap (pipeline.MAX_SPEED)')
    a = ap.parse_args()
    globals()[a.mode](a)


if __name__ == '__main__':
    main()
