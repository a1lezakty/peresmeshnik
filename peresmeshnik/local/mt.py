"""OPUS-MT (Helsinki-NLP) or quickmt (both CC-BY-4.0) on CTranslate2 int8, one model per language pair; a pair without
one goes through English (a registry row with `via`: two models one after the other)."""
from __future__ import annotations

import json
import re

# a sentence ends at . ! ? ; and a space (not "U.N. also": no space, or a lowercase word after), or at 。！？ alone
_END = re.compile(r'[.!?;؟…]+\s+|[。！？；]+\s*')
# ponytail: titles before a name only ("Mr. Smith said"); add one when a clause shows it cut
_TITLES = {'mr', 'mrs', 'ms', 'dr', 'prof', 'st', 'jr', 'sr', 'sra', 'srta', 'hr', 'fr', 'mme', 'mlle', 'sig', 'д-р'}


def sentences(text: str) -> list[str]:
    """A clause's sentences: a model trained on single sentences (Tatoeba's ru-uk) stops after the first of two."""
    from peresmeshnik.local.asr import _starts_sentence
    out, start = [], 0
    for m in _END.finditer(text):
        word = text[start:m.start()].rsplit(None, 1)[-1:] or ['']
        if m.end() < len(text) and _starts_sentence(text[m.end():]) and \
                not (m.group().strip() == '.' and word[0].lower() in _TITLES):
            out.append(text[start:m.end()].strip())
            start = m.end()
    out.append(text[start:].strip())
    return [x for x in out if x]


class OpusMt:
    def __init__(self, src: str, tgt: str, model_id: str | None = None, threads: int = 2):
        """`model_id`: a registry entry (default opus-mt-<src>-<tgt>, our conversion of the original Marian weights)."""
        import ctranslate2
        import sentencepiece as spm

        from peresmeshnik import models
        from peresmeshnik.local import device
        self.src, self.tgt = src, tgt
        m = models.get(model_id or f'opus-mt-{src}-{tgt}')
        if m.via:  # de -> en -> ru
            models.path(m.id)  # all of it there, or ModelMissing
            hops = [models.get(i).langs[0].split('-') for i in m.via]
            self._steps = [OpusMt(a, b, i, threads) for (a, b), i in zip(hops, m.via)]
            self.device = self._steps[0].device
            return
        self._steps = []
        path = models.path(m.id)
        cfg = json.loads((path / 'config.json').read_text()) if (path / 'config.json').exists() else {}
        self._eos = [] if cfg.get('add_source_eos') else ['</s>']  # ours of the Marian weights adds it; HF conversions not
        pre = path / 'source_prefix.txt'  # a multi-target model: >>rus<< etc. in front
        self._head = [pre.read_text(encoding='utf-8').strip()] if pre.exists() else []
        self.device = 'cuda' if device.ct2_cuda() else 'cpu'
        names = next((n for n in (('src.spm.model', 'tgt.spm.model'), ('joint.spm.model', 'joint.spm.model'))  # quickmt
                      if (path / n[0]).exists()), ('source.spm', 'target.spm'))  # OPUS-MT
        self._sp_src, self._sp_tgt = (spm.SentencePieceProcessor(str(path / n)) for n in names)
        try:
            self._tr = ctranslate2.Translator(str(path), device=self.device, inter_threads=1, intra_threads=threads,
                                              compute_type='int8' if self.device == 'cpu' else 'default')  # quickmt: float16
            if self.device == 'cuda':  # cuBLAS loads at the first translation: a driver without the CUDA libraries fails there
                self.translate('Hello.', src, tgt)
        except Exception as e:  # CUDA there but the model would not load or run on it: them -> me falls back to CPU
            if self.device != 'cuda':
                raise
            import logging
            logging.getLogger('peresmeshnik.device').warning('MT on CUDA failed (%s); using CPU', e)
            self.device = 'cpu'
            self._tr = ctranslate2.Translator(str(path), device='cpu', inter_threads=1, intra_threads=threads, compute_type='int8')

    def translate(self, text: str, src: str, tgt: str, context: tuple[str, ...] = ()) -> str:
        assert (src, tgt) == (self.src, self.tgt), f'model is {self.src}-{self.tgt}'
        if self._steps:
            for step in self._steps:
                text = step.translate(text, step.src, step.tgt)
            return text
        if self.src == 'ru':  # a lowercase fragment ending in "." makes ru-en invent UN boilerplate; en-ru loses chrF (BENCH stage 1)
            text = text[:1].upper() + text[1:]
        from peresmeshnik.local.asr import NO_SPACES
        batch = [self._sp_src.encode(x, out_type=str) for x in sentences(text)] or [[]]
        # a clause fragment makes OPUS-MT ramble ("I'm not sure if I'm not sure..."): cap the length, no 3-gram twice
        out = self._tr.translate_batch([self._head + t + self._eos for t in batch], beam_size=1,
                                       max_decoding_length=min(256, 2 * max(map(len, batch)) + 10), no_repeat_ngram_size=3)
        return ('' if tgt in NO_SPACES else ' ').join(self._sp_tgt.decode(r.hypotheses[0]) for r in out)
