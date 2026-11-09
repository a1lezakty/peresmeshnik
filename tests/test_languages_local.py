"""Them -> me in the 12 new languages on the real models (stage 6), one short phrase each: FLEURS sentence 1515 ("It is
Martelly's fifth CEP in four years.", CC-BY-4.0), the setup wizard's test phrase (peresmeshnik/ui/samples).

- ASR (the language's default model) through the stream as the app runs it: partials, the final, CER against FLEURS
  (the clause cuts of zh / ja / ar / ko: tests/test_pipeline.py on stand-ins). tr / zh / ko run Whisper large-v3-turbo,
  live only on an NVIDIA GPU: not on the runners (BENCH stage 6, on an A4000).
- MT into Russian on the FLEURS text: de direct (OPUS-MT tc-big), ja / ar / tr through English; a one-word reply stays one,
  du / Sie stays ты / вы (through English both get lost or made up: BENCH stage 6, MT).
- The stock voice says the FLEURS text (zh / ar have none).

Needs the models of `MODELS` below (CI: the work-gate job on Linux, cached); else skipped.
"""
import os
import re
import unicodedata
from pathlib import Path

import numpy as np
import pytest

GATE = os.environ.get('PERESMESHNIK_WORK_GATE') == '1'
pytestmark = pytest.mark.skipif(not GATE and os.environ.get('GITHUB_JOB') != 'work-gate', reason='needs models: PERESMESHNIK_WORK_GATE=1')
SAMPLES = Path(os.environ.get('PERESMESHNIK_SAMPLES') or Path(__file__).parents[1] / 'peresmeshnik' / 'ui' / 'samples')
FLEURS_1515 = {
    'uk': "Це п'ята Тимчасова виборча рада Мартеллі за чотири роки.",
    'de': 'Es ist Martellys fünftes CEP innerhalb von vier Jahren.',
    'fr': 'C’est le cinquième PEC de Martelly en quatre ans.',
    'es': 'En cuatro años, es el quinto CEP de Martelly.',
    'it': 'Si tratta del quinto Consiglio elettorale provvisorio di Martelly in quattro anni.',
    'pt': 'É o quinto CEP do Martelly em quatro anos.',
    'pl': "To jest piąty CEP Martelly'ego w cztery lata.",
    'tr': "Bu Martelly'nin dört yıl içindeki beşinci Geçici Seçim Konseyi olacak.",
    'zh': '这是马特利 (Martelly) 四年来第五次入选海地临时选举委员会 (CEP)。',
    'ja': 'マルテリーにとっては4年間で5度目のCEPです。',
    'ko': '마텔리가 4년 만에 다섯 번째 하는 CEP다.',
    'ar': 'إنه المجلس الانتخابي المؤقت الخامس لمارتيلي خلال أربع سنوات.',
}
CPU_ASR = ('uk', 'de', 'fr', 'es', 'it', 'pt', 'pl', 'ja', 'ar')
MT_PAIRS = ('de-ru', 'ja-ru', 'ar-ru', 'tr-ru')  # de: its own model; the others through English (OPUS-MT tr-en, quickmt)
# what CI fetches for this file (ci.yml, work-gate, Linux)
MODELS = ('parakeet-tdt-0.6b-v3', 'parakeet-tdt_ctc-0.6b-ja', 'fastconformer-ar-pcd', 'fastconformer-fr-pc', 'opus-mt-de-ru',
          'quickmt-ja-en', 'quickmt-ar-en', 'opus-mt-tr-en', 'quickmt-en-ru', 'piper-uk_UA-ukrainian_tts-medium',
          'piper-de_DE-thorsten-medium', 'piper-fr_FR-siwis-medium', 'piper-es_MX-ald-medium', 'piper-it_IT-paola-medium',
          'piper-pt_BR-faber-medium', 'piper-pl_PL-gosia-medium', 'piper-tr_TR-dfki-medium', 'piper-ko_KR-kss-medium',
          'piper-ja_JP-hi_fi_captain-medium')


def norm(text, lang):
    t = re.sub(r"[^\w\s']|_", ' ', unicodedata.normalize('NFKC', text).lower())
    return ''.join(t.split()) if lang in ('zh', 'ja') else ' '.join(t.split())


def cer(ref, hyp):
    """Character edit distance over the reference's length."""
    row = list(range(len(hyp) + 1))
    for i, a in enumerate(ref, 1):
        prev, row[0] = row[0], i
        for j, b in enumerate(hyp, 1):
            prev, row[j] = row[j], min(row[j] + 1, row[j - 1] + 1, prev + (a != b))
    return row[-1] / max(len(ref), 1)


def sample(lang):
    import soundfile as sf
    if not (SAMPLES / f'{lang}.ogg').exists():  # the wizard's samples come with the shell stream's #16
        pytest.skip(f'no {lang}.ogg in {SAMPLES}')
    x, sr = sf.read(SAMPLES / f'{lang}.ogg', dtype='float32')
    assert sr == 16000
    return x if x.ndim == 1 else x.mean(axis=1)


@pytest.mark.parametrize('lang', CPU_ASR)
def test_their_speech_is_recognised_in_the_stream(lang):
    from peresmeshnik import config
    from peresmeshnik.local.asr import OnnxAsr
    model = config.Local().asr[lang]
    asr = OnnxAsr(model, threads=2)
    s = asr.stream(lang)
    x, partials = sample(lang), []
    for i in range(0, len(x), 1600):  # 0.1 s at a time, as the VAD hands it on
        if (p := s.feed(x[i:i + 1600])) is not None:
            partials.append(p)
    text = s.finish()
    got = cer(norm(FLEURS_1515[lang], lang), norm(text, lang))
    print(f'{lang} {model}: {got:.3f} {text!r} ({len(partials)} partials)')
    assert partials and got <= 0.35, (text, partials)  # a name and an acronym in 5 s of speech: ~0.05-0.25 measured


@pytest.mark.parametrize('pair', MT_PAIRS)
def test_their_words_are_translated_into_russian(pair):
    from peresmeshnik import config, models
    from peresmeshnik.local.mt import OpusMt
    src, tgt = pair.split('-')
    model = config.Local().mt[pair]
    assert bool(models.get(model).via) == (src != 'de'), model
    out = OpusMt(src, tgt, model, threads=2).translate(FLEURS_1515[src], src, tgt)
    print(pair, model, out)
    assert re.search('[а-яё]{3}', out.lower()) and re.search(r'4|четыр', out) and re.search('Март|Martelly', out), out  # the name kept


def test_a_short_reply_stays_short_and_du_sie_stays():
    """quickmt X->en makes up a continuation of a one-word reply ("Ja." -> "I think we're going to get a job."): the
    defaults (de-ru direct, tr-ru through OPUS-MT tr-en) do not; du / Sie come out as ты / вы."""
    from peresmeshnik import config
    from peresmeshnik.local.mt import OpusMt
    mt = config.Local().mt
    de, tr = OpusMt('de', 'ru', mt['de-ru'], threads=2), OpusMt('tr', 'ru', mt['tr-ru'], threads=2)
    for e, src, text in ((de, 'de', 'Ja.'), (de, 'de', 'Nein.'), (de, 'de', 'Danke.'), (tr, 'tr', 'Evet.'), (tr, 'tr', 'Teşekkürler.')):
        out = e.translate(text, src, 'ru')
        print(src, text, out)
        assert len(out.split()) <= 2, (text, out)
    du = de.translate('Kannst du mir die Datei schicken?', 'de', 'ru').lower()
    sie = de.translate('Können Sie mir die Datei schicken?', 'de', 'ru').lower()
    print(du, '|', sie)
    assert re.search(r'\b(ты|можешь)\b', du) and re.search(r'\b(вы|можете)\b', sie), (du, sie)


@pytest.mark.parametrize('lang', [k for k in FLEURS_1515 if k not in ('zh', 'ar')])
def test_the_stock_voice_says_it(lang):
    from peresmeshnik import config
    from peresmeshnik.local.tts import PiperTts
    tts = PiperTts(config.Local().tts[lang], threads=2)
    try:
        audio = np.concatenate(list(tts.synth(FLEURS_1515[lang], lang)))
    finally:
        tts.close()
    seconds = len(audio) / tts.sample_rate
    print(lang, f'{seconds:.1f} s')
    assert 1.5 < seconds < 15 and float(np.abs(audio).max()) > 0.05
