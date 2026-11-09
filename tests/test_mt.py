"""OPUS-MT engine on stand-ins for CTranslate2 and SentencePiece: a pair through English runs its two models one after
the other; a multi-target model gets its >>xxx<< token in front; </s> is added unless the conversion adds it."""
import json
from pathlib import Path

import pytest


def test_a_pair_through_english_and_the_target_token(monkeypatch, tmp_path):
    ctranslate2 = pytest.importorskip('ctranslate2')
    spm = pytest.importorskip('sentencepiece')
    from peresmeshnik import models
    from peresmeshnik.local import device, mt
    seen = []

    class Translator:
        def __init__(self, path, device, **kw):
            self.path = path

        def translate_batch(self, batch, **kw):
            seen.append((Path(self.path).name, batch[0]))
            out = [f'<{self.path.rsplit("-", 1)[-1]}>'] + [t for t in batch[0] if not t.startswith('>>')]
            return [type('R', (), {'hypotheses': [out]})()]

    class Sp:
        def __init__(self, path):
            pass

        def encode(self, text, out_type=str):
            return text.split()

        def decode(self, toks):
            return ' '.join(t for t in toks if t != '</s>')
    for name, cfg, prefix in (('opus-mt-de-en', {}, None), ('opus-mt-en-ru', {'add_source_eos': True}, '>>rus<<')):
        (tmp_path / name).mkdir()
        (tmp_path / name / 'config.json').write_text(json.dumps(cfg))
        if prefix:
            (tmp_path / name / 'source_prefix.txt').write_text(prefix)
    reg = {n: models.Model(n, 'mt', (n[8:],), 'CC-BY-4.0', 'https://x', 'https://x/', {}) for n in ('opus-mt-de-en', 'opus-mt-en-ru')}
    reg['opus-mt-de-ru-via-en'] = models.Model('opus-mt-de-ru-via-en', 'mt', ('de-ru',), 'CC-BY-4.0', 'https://x', '', {},
                                               via=('opus-mt-de-en', 'opus-mt-en-ru'))
    monkeypatch.setattr(models, 'registry', lambda: reg)
    monkeypatch.setattr(models, 'path', lambda i, base=None: tmp_path / (reg[i].via[0] if reg[i].via else i))
    monkeypatch.setattr(device, 'ct2_cuda', lambda: False)
    monkeypatch.setattr(ctranslate2, 'Translator', Translator)
    monkeypatch.setattr(spm, 'SentencePieceProcessor', Sp)
    e = mt.OpusMt('de', 'ru', 'opus-mt-de-ru-via-en')
    assert e.translate('Guten Tag', 'de', 'ru') == '<ru> <en> Guten Tag'
    assert seen == [('opus-mt-de-en', ['Guten', 'Tag', '</s>']),  # an HF conversion: </s> added here
                    ('opus-mt-en-ru', ['>>rus<<', '<en>', 'Guten', 'Tag'])]  # ours adds it itself; the token in front


def test_a_clause_is_translated_sentence_by_sentence():
    """Tatoeba's ru-uk stops after the first of two sentences: each goes on its own (one batch); "U.N. also" is one."""
    from peresmeshnik.local.mt import sentences
    assert sentences('Привет, как дела? Я позвоню завтра.') == ['Привет, как дела?', 'Я позвоню завтра.']
    assert sentences('The U.N. also hopes. then') == ['The U.N. also hopes. then']
    assert sentences('今日は晴れです。明日も') == ['今日は晴れです。', '明日も'] and sentences('مرحبا. كيف') == ['مرحبا.', 'كيف']
    assert sentences('Mr. Smith said yes. No. Fine.') == ['Mr. Smith said yes.', 'No.', 'Fine.']  # a title, not an end
