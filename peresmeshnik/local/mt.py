"""OPUS-MT (Helsinki-NLP, CC-BY-4.0) on CTranslate2 int8, one model per language pair."""
from __future__ import annotations


class OpusMt:
    def __init__(self, src: str, tgt: str, model_id: str | None = None, threads: int = 2):
        """`model_id`: a registry entry (default opus-mt-<src>-<tgt>, our conversion of the original Marian weights)."""
        import ctranslate2
        import sentencepiece as spm

        from peresmeshnik import models
        path = models.path(model_id or f'opus-mt-{src}-{tgt}')
        self.src, self.tgt = src, tgt
        self._tr = ctranslate2.Translator(str(path), device='cpu', inter_threads=1, intra_threads=threads)
        self._sp_src = spm.SentencePieceProcessor(str(path / 'source.spm'))
        self._sp_tgt = spm.SentencePieceProcessor(str(path / 'target.spm'))

    def translate(self, text: str, src: str, tgt: str, context: tuple[str, ...] = ()) -> str:
        assert (src, tgt) == (self.src, self.tgt), f'model is {self.src}-{self.tgt}'
        toks = self._sp_src.encode(text, out_type=str)  # the converted model appends </s> itself
        hyp = self._tr.translate_batch([toks], beam_size=1, max_decoding_length=256)[0].hypotheses[0]
        return self._sp_tgt.decode(hyp)
