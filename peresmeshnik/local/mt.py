"""OPUS-MT (Helsinki-NLP, CC-BY-4.0) on CTranslate2 int8, one model per language pair."""
from __future__ import annotations


class OpusMt:
    def __init__(self, src: str, tgt: str, model_id: str | None = None, threads: int = 2):
        """`model_id`: a registry entry (default opus-mt-<src>-<tgt>, our conversion of the original Marian weights)."""
        import ctranslate2
        import sentencepiece as spm

        from peresmeshnik import models
        from peresmeshnik.local import device
        path = models.path(model_id or f'opus-mt-{src}-{tgt}')
        self.src, self.tgt = src, tgt
        self.device = 'cuda' if device.ct2_cuda() else 'cpu'
        try:
            self._tr = ctranslate2.Translator(str(path), device=self.device, inter_threads=1, intra_threads=threads)
        except Exception as e:  # CUDA there but the model would not load on it (VRAM): them -> me falls back to CPU
            if self.device != 'cuda':
                raise
            import logging
            logging.getLogger('peresmeshnik.device').warning('MT on CUDA failed (%s); using CPU', e)
            self.device = 'cpu'
            self._tr = ctranslate2.Translator(str(path), device='cpu', inter_threads=1, intra_threads=threads)
        self._sp_src = spm.SentencePieceProcessor(str(path / 'source.spm'))
        self._sp_tgt = spm.SentencePieceProcessor(str(path / 'target.spm'))

    def translate(self, text: str, src: str, tgt: str, context: tuple[str, ...] = ()) -> str:
        assert (src, tgt) == (self.src, self.tgt), f'model is {self.src}-{self.tgt}'
        if self.src == 'ru':  # a lowercase fragment ending in "." makes ru-en invent UN boilerplate; en-ru loses chrF (BENCH stage 1)
            text = text[:1].upper() + text[1:]
        toks = self._sp_src.encode(text, out_type=str)  # the converted model appends </s> itself
        # a clause fragment makes OPUS-MT ramble ("I'm not sure if I'm not sure..."): cap the length, no 3-gram twice
        hyp = self._tr.translate_batch([toks], beam_size=1, max_decoding_length=min(256, 2 * len(toks) + 10),
                                       no_repeat_ngram_size=3)[0].hypotheses[0]
        return self._sp_tgt.decode(hyp)
