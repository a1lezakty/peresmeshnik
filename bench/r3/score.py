"""Score results/*.jsonl: corpus WER per model x set x condition, RTF; latency medians from *.lat.jsonl.
Normalisation (as in BENCH.md): lowercase, ё→е, punctuation removed, hyphens split; digits spelled out with num2words
(ru / en), so "3" and "три" match. Prints markdown tables; --csv writes summary CSVs."""
import glob, json, re, sys
from collections import defaultdict

import jiwer
from num2words import num2words


def norm(t, lang):
    t = (t or '').lower().replace('ё', 'е')
    t = re.sub(r'<[^>]*>', ' ', t)  # language tags some models emit
    t = re.sub(r'\d+', lambda m: ' ' + num2words(int(m.group()), lang=lang if lang in ('ru', 'en') else 'ru') + ' ', t)
    t = re.sub(r"[^\w\s']|_", ' ', t).replace("'", '')
    return ' '.join(t.split())


def wer(refs, hyps):
    pairs = [(r, h) for r, h in zip(refs, hyps) if r]
    return jiwer.wer([r for r, _ in pairs], [h if h else '<empty>' for _, h in pairs])


def main(csv=False):
    rows = [json.loads(l) for f in sorted(glob.glob('results/*.jsonl')) if not f.endswith('.lat.jsonl') for l in open(f)]
    g = defaultdict(list)
    for r in rows:
        lang = r['lang'] if r['lang'] in ('ru', 'en') else 'ru'
        if r['set'] == 'cs_en_ru':  # mixed: normalise each script's words with its own number language
            ref, hyp = norm(r['ref'], 'ru'), norm(r['hyp'], 'ru')
        else:
            ref, hyp = norm(r['ref'], lang), norm(r['hyp'], lang)
        g[(r['model'], r['set'], r['cond'])].append((ref, hyp, r['dur'], r['compute_s']))
    sets = ['golos_crowd', 'golos_farfield', 'cv17_ru', 'ls_clean', 'ls_other', 'cs_en_ru']
    models = sorted({k[0] for k in g})
    out = []
    print('| model | cond | ' + ' | '.join(sets) + ' | RTF 4cpu |')
    print('|---' * (len(sets) + 3) + '|')
    for m in models:
        for c in ('clean', 'opus64', 'opus16', 'cs'):
            cells, d, s = [], 0, 0
            for st in sets:
                v = g.get((m, st, c))
                if not v:
                    cells.append('—')
                    continue
                w = wer([x[0] for x in v], [x[1] for x in v])
                cells.append(f'{100 * w:.1f}')
                d += sum(x[2] for x in v)
                s += sum(x[3] for x in v)
                out.append({'model': m, 'set': st, 'cond': c, 'n': len(v), 'wer': round(w, 4)})
            if d:
                print(f'| {m} | {c} | ' + ' | '.join(cells) + f' | {s / d:.3f} |')
    lat = [json.loads(l) for f in sorted(glob.glob('results/*.lat.jsonl')) for l in open(f)]
    if lat:
        import statistics as st
        print('\n| model | lang | n | first partial s (med / p90) | final after end s (med / p90) | final compute ms med |')
        print('|---|---|---|---|---|---|')
        by = defaultdict(list)
        for r in lat:
            by[(r['model'], r['lang'])].append(r)
        for (m, lang), v in sorted(by.items()):
            fp = sorted(x['first_partial_s'] for x in v if x['first_partial_s'] is not None)
            fe = sorted(x['final_after_end_s'] for x in v)
            fc = st.median(x['final_compute_s'] for x in v) * 1000
            p90 = lambda a: a[min(len(a) - 1, int(0.9 * len(a)))]
            print(f'| {m} | {lang} | {len(v)} | {st.median(fp):.2f} / {p90(fp):.2f} | {st.median(fe):.2f} / {p90(fe):.2f} | {fc:.0f} |')
    if csv:
        with open('results/summary_wer.json', 'w') as f:
            json.dump(out, f, ensure_ascii=False, indent=0)


if __name__ == '__main__':
    main('--csv' in sys.argv)
