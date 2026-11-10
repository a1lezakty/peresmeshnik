"""Prepare the benchmark set: parallel EN/RU utterances from FLEURS dev (CC-BY-4.0)
and ~40 s single-speaker voice references (LibriSpeech test-clean, CC-BY-4.0;
Russian LibriSpeech / LibriVox, public domain).

    python bench/data.py [--n 30]      # writes $BENCH_DATA/set (default ./bench-data)
"""
import argparse
import csv
import io
import json
import os
import re
import tarfile
import urllib.request
from pathlib import Path

import numpy as np
import soundfile as sf

HF = 'https://huggingface.co/datasets'
# each dataset at a fixed commit, not `main`: BENCH's numbers come from these very bytes
FLEURS = HF + '/google/fleurs/resolve/70bb2e84b976b7e960aa89f1c648e09c59f894dd/data/{code}/{f}'
REFS = {
    'en': HF + '/openslr/librispeech_asr/resolve/71cacbfb7e2354c4226d01e70d77d5fca3d04ba1/all/test.clean/0000.parquet',
    'ru': HF + '/bond005/rulibrispeech/resolve/b15916bc0ee9a35dc8cb83e6c83d242ddb9e453c/data/'
               'test-00000-of-00001-0e9859b3235b0392.parquet',
}
SR = 16000
ROOT = Path(os.environ.get('BENCH_DATA', 'bench-data'))


def fetch(url, dst):
    if not dst.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        print('download', url)
        urllib.request.urlretrieve(url, dst.with_suffix('.part'))
        dst.with_suffix('.part').rename(dst)
    return dst


# FLEURS configs of the languages; their sentences (FLoRes) are the same in every language under the same id
CODES = {'en': 'en_us', 'ru': 'ru_ru', 'uk': 'uk_ua', 'de': 'de_de', 'fr': 'fr_fr', 'es': 'es_419', 'it': 'it_it',
         'pt': 'pt_br', 'pl': 'pl_pl', 'tr': 'tr_tr', 'zh': 'cmn_hans_cn', 'ja': 'ja_jp', 'ko': 'ko_kr', 'ar': 'ar_eg'}


def fleurs(lang, country=None, split='dev', audio=True):
    """{sentence id: (wav path, raw text)} of a FLEURS split; `lang` an ISO code of CODES or FLEURS' own with `country`."""
    code = f'{lang}_{country}' if country else CODES[lang]
    d = ROOT / 'fleurs' / lang
    tsv = fetch(FLEURS.format(code=code, f=f'{split}.tsv'), d / f'{split}.tsv')
    if audio and not (d / split).exists():
        with tarfile.open(fetch(FLEURS.format(code=code, f=f'audio/{split}.tar.gz'), d / f'{split}.tar.gz')) as t:
            t.extractall(d, filter='data')
    rows = {}
    for r in csv.reader(open(tsv, encoding='utf-8'), delimiter='\t', quoting=csv.QUOTE_NONE):
        rows.setdefault(int(r[0]), (d / split / r[1], r[2]))  # first recording of each sentence
    return rows


# Tatoeba-Challenge test sets (Tatoeba sentences, CC-BY 2.0 FR): short everyday sentences, the call's register
TATOEBA = 'https://raw.githubusercontent.com/Helsinki-NLP/Tatoeba-Challenge/6faf2dab0b7b01a0e08a114dbacbb7deac54988d/data/test/{pair}/test.txt'
ISO3 = {'en': 'eng', 'ru': 'rus', 'uk': 'ukr', 'de': 'deu', 'fr': 'fra', 'es': 'spa', 'it': 'ita', 'pt': 'por', 'pl': 'pol',
        'tr': 'tur', 'zh': 'zho', 'ja': 'jpn', 'ko': 'kor', 'ar': 'ara'}


def tatoeba(src, tgt, n):
    """[(source, reference)] of a Tatoeba test set: `n` spread over it (sorted by the source)."""
    a, b = sorted((ISO3[src], ISO3[tgt]))
    rows = [r for r in csv.reader(open(fetch(TATOEBA.format(pair=f'{a}-{b}'), ROOT / 'tatoeba' / f'{a}-{b}.txt'), encoding='utf-8'),
                                  delimiter='\t', quoting=csv.QUOTE_NONE) if len(r) == 4]
    if 'zh' in (src, tgt):  # Mandarin in simplified characters (not cmn_Hant, yue, lzh)
        rows = [r for r in rows if 'cmn_Hans' in r[:2]]
    pairs = sorted((r[2], r[3]) if ISO3[src] == a else (r[3], r[2]) for r in rows)
    return pairs[::max(1, len(pairs) // n)][:n]


def reference(lang, seconds=40.0):
    """Concatenate utterances of one reader until `seconds` of speech."""
    import pyarrow.parquet as pq
    t = pq.read_table(fetch(REFS[lang], ROOT / 'refs' / f'{lang}.parquet'))
    audio, text = t.column('audio').to_pylist(), t.column('text' if lang == 'en' else 'transcription').to_pylist()
    if lang == 'en':  # the speaker with the most material
        spk = t.column('speaker_id').to_pylist()
        best = max(set(spk), key=spk.count)
        keep = [i for i, s in enumerate(spk) if s == best]
    else:  # one LibriVox chapter = one reader
        keep = [i for i, a in enumerate(audio) if re.match(r'poemi_01_', a['path'])]
    parts, words, total = [], [], 0.0
    for i in keep:
        x, sr = sf.read(io.BytesIO(audio[i]['bytes']), dtype='float32')
        assert sr == SR, sr
        parts += [x, np.zeros(int(0.3 * SR), 'float32')]
        words.append(text[i].strip())
        total += len(x) / SR
        if total >= seconds:
            break
    return np.concatenate(parts), ' '.join(words)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=30)
    n = ap.parse_args().n
    en, ru = fleurs('en', 'us'), fleurs('ru', 'ru')
    out = ROOT / 'set'
    items = []
    for sid in sorted(set(en) & set(ru))[:n]:
        item = {'id': sid}
        for lang, rows in (('en', en), ('ru', ru)):
            wav, text = rows[sid]
            x, sr = sf.read(wav, dtype='float32')
            assert sr == SR, sr
            dst = out / lang / f'{sid}.wav'
            dst.parent.mkdir(parents=True, exist_ok=True)
            sf.write(dst, x, SR, subtype='PCM_16')
            item[lang] = {'wav': str(dst.relative_to(out)), 'text': text, 'sec': round(len(x) / SR, 2)}
        items.append(item)
    for lang in ('en', 'ru'):
        x, text = reference(lang)
        sf.write(out / f'ref_{lang}.wav', x, SR, subtype='PCM_16')
        (out / f'ref_{lang}.txt').write_text(text + '\n', encoding='utf-8')
    (out / 'manifest.json').write_text(json.dumps(items, ensure_ascii=False, indent=1), encoding='utf-8')
    print(f'{len(items)} pairs, {sum(i["en"]["sec"] for i in items):.0f} s EN, '
          f'{sum(i["ru"]["sec"] for i in items):.0f} s RU -> {out}')


if __name__ == '__main__':
    main()
