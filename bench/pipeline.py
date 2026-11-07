"""The local pipeline as it runs in the app (VAD, ASR partials, clause cuts, MT, TTS), on recordings.

    python bench/pipeline.py quality en   # bench set: each utterance alone; WER of the clause texts, chrF of their translations
    python bench/pipeline.py work en      # tests/data/call-en.ogg, 30 s, with Piper: RTF and CPU s per audio s per link

`quality` compares the streamed result with the offline rows (asr: whole utterance; mt: gold source),
so it shows what clause cutting costs. Engines use --threads each, as in the app (default 2).
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
# isort: off
from peresmeshnik.pipeline import cpu_s, make_local, read_audio, run  # noqa: E402  first: it pins BLAS threads before numpy loads
import numpy as np  # noqa: E402
# isort: on
from peresmeshnik.engines import Caption  # noqa: E402

from common import SET, manifest, record  # noqa: E402

DATA = Path(__file__).resolve().parents[1] / 'tests' / 'data'


class NoTts:
    sample_rate = 16000

    def synth(self, text, lang, voice=None):
        yield from ()


def finals(d):
    return [e for e in d.events() if isinstance(e, Caption) and e.final]


def quality(src, tgt, a):
    import jiwer
    import sacrebleu
    from asr import norm
    d = make_local(src, tgt, a.asr, a.threads, tts=NoTts())
    d.max_clause_s, d.asr.context_s = a.max_clause, a.context
    hyps, trs, refs, trefs, offline, clauses = [], [], [], [], [], 0
    pad = np.zeros(16000, np.float32)
    for it in manifest()[:a.n]:
        d = d if not hyps else type(d)(d.vad, d.asr, d.mt, d.tts, src, tgt, max_clause_s=a.max_clause)  # same engines
        d.vad.reset()
        d.realtime = False
        x = read_audio(SET / it[src]['wav'])
        x *= 0.05 / np.sqrt(np.mean(x ** 2))  # -26 dBFS RMS like tests/data: a call app's AGC; some FLEURS takes are at -50 dBFS
        run(d, np.concatenate([pad, x, pad]))
        d.finish()
        cs = finals(d)
        clauses += len(cs)
        hyps.append(norm(' '.join(c.original for c in cs)))
        trs.append(' '.join(c.translation for c in cs))
        refs.append(norm(it[src]['text']))
        trefs.append(it[tgt]['text'])
        offline.append(norm(d.asr.recognize(x).text))  # the same audio in one window: what the cuts cost
    record('pipeline', f'{d.asr.name}+opus-mt', lang=src, mode='quality', threads=a.threads, n=len(refs), clauses=clauses,
           max_clause_s=a.max_clause, context_s=a.context,
           wer=round(jiwer.wer(refs, hyps), 4), wer_offline=round(jiwer.wer(refs, offline), 4),
           chrf=round(sacrebleu.corpus_chrf(trs, [trefs]).score, 1),
           samples=[{'hyp': h, 'tr': t} for h, t in list(zip(hyps, trs))[:2]])


def work(src, tgt, a):
    d = make_local(src, tgt, a.asr, a.threads)
    pcm = read_audio(DATA / f'call-{src}.ogg')
    d.asr.recognize(pcm[:16000])  # warm-up
    d.mt.translate('Hello.' if src == 'en' else 'Привет.', src, tgt)
    list(d.tts.synth('Привет.' if tgt == 'ru' else 'Hello.', tgt))
    d.realtime = False
    cpu0, t0 = cpu_s(d), time.perf_counter()
    run(d, pcm)
    d.finish()
    wall, cpu = time.perf_counter() - t0, cpu_s(d) - cpu0
    d.close()
    audio = len(pcm) / 16000
    record('pipeline', f'{d.asr.name}+opus-mt+piper', lang=src, mode='work', threads=a.threads, **d.summary(),
           rtf=round(wall / audio, 3), cpu_s_per_audio_s=round(cpu / audio, 3))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('mode', choices=['quality', 'work'])
    ap.add_argument('lang', choices=['en', 'ru'])
    ap.add_argument('--asr', default='parakeet', choices=['parakeet', 'gigaam'])
    ap.add_argument('--threads', type=int, default=2)
    ap.add_argument('--n', type=int, default=30)
    ap.add_argument('--max-clause', type=float, default=5.0, help='hard clause limit, s')
    ap.add_argument('--context', type=float, default=1.0, help='left context after a cut, s')
    a = ap.parse_args()
    globals()[a.mode](a.lang, 'ru' if a.lang == 'en' else 'en', a)


if __name__ == '__main__':
    main()
