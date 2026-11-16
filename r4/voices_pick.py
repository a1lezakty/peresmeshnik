"""R4 copy (16.11): eight FLEURS dev speakers as voices, en 2 M + 2 F, ru 2 M + 2 F (paths from $BENCH_DATA, $VOICES); 10-14 s utterances, the cleanest by
a crude SNR, pairwise ECAPA similarity < 0.35 (FLEURS has no speaker ids). One data dir per voice: the voice is the
reference under both names (ref_en, ref_ru) so every engine adapter takes it as 'en'."""
import csv, json, os
from pathlib import Path
import numpy as np, soundfile as sf, torch, torchaudio.functional as F
from speechbrain.inference.speaker import EncoderClassifier
D = Path(os.environ['BENCH_DATA']); V = Path(os.environ['VOICES'])
spk = EncoderClassifier.from_hparams('speechbrain/spkrec-ecapa-voxceleb', run_opts={'device': 'cpu'},
                                     savedir=str(Path(os.environ['HF_HOME']) / 'speechbrain-ecapa'))
def trim(x, sr):
    f = int(0.02 * sr)
    db = 20 * np.log10(np.sqrt(np.mean(x[:len(x) // f * f].reshape(-1, f) ** 2, 1)) + 1e-9)
    loud = np.nonzero(db > np.percentile(db, 90) - 30)[0]
    return x[max(0, loud[0] * f - int(0.1 * sr)):min(len(x), (loud[-1] + 1) * f + int(0.1 * sr))], float(np.percentile(db, 90) - np.percentile(db, 10))
def emb(x, sr):
    e = spk.encode_batch(F.resample(torch.from_numpy(x), sr, 16000)[None]).squeeze()
    return e / e.norm()
used = {m['id'] for m in json.loads((D / 'set/manifest.json').read_text())}
chosen, meta = [], []
for lang, gender, n in (('en', 'MALE', 2), ('en', 'FEMALE', 2), ('ru', 'MALE', 2), ('ru', 'FEMALE', 2)):
    cand = []
    for row in csv.reader(open(D / f'fleurs/{lang}/dev.tsv', encoding='utf-8'), delimiter='\t', quoting=csv.QUOTE_NONE):
        p = D / f'fleurs/{lang}/dev/{row[1]}'
        if row[-1] != gender or int(row[0]) in used or not p.exists() or not 10 <= sf.info(p).duration <= 14:
            continue
        x, sr = sf.read(p, dtype='float32')
        if np.abs(x).max() >= 0.99:
            continue
        x, snr = trim(x, sr)
        if len(x) / sr >= 8:
            cand.append((snr, row[0], row[1], row[2], x, sr))
    cand.sort(key=lambda c: -c[0])
    got = 0
    for snr, sid, name, text, x, sr in cand[:25]:
        e = emb(x, sr)
        if any(float(e @ c) > 0.35 for c in chosen):
            continue
        chosen.append(e); got += 1
        v = f'v{len(chosen):02d}'; d = V / v / 'set'; d.mkdir(parents=True, exist_ok=True)
        for nm in ('en', 'ru', 'manifest.json'):
            if not (d / nm).exists():
                os.symlink(D.resolve() / 'set' / nm, d / nm)
        for l in ('en', 'ru'):
            sf.write(d / f'ref_{l}.wav', x, sr, subtype='PCM_16'); (d / f'ref_{l}.txt').write_text(text, encoding='utf-8')
        meta.append({'voice': v, 'lang': lang, 'gender': gender, 'fleurs_id': sid, 'file': name, 'sec': round(len(x) / sr, 1), 'snr_db': round(snr, 1)})
        print(meta[-1], flush=True)
        if got == n:
            break
(V / 'voices.json').write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding='utf-8')
