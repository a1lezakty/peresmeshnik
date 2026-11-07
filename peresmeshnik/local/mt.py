"""OPUS-MT (Helsinki-NLP, CC-BY-4.0) on CTranslate2 int8, one model per language pair."""
from __future__ import annotations

import os
import shutil
import tempfile
import urllib.request
import zipfile
from pathlib import Path

OPUS_ZIP = 'https://object.pouta.csc.fi/OPUS-MT-models/{src}-{tgt}/opus-2020-02-11.zip'  # original Marian weights


def models_dir() -> Path:
    # ponytail: stand-in until the registry (models.py) lands; then its cache dir and fetch replace this
    return Path(os.environ.get('PERESMESHNIK_MODELS') or Path.home() / '.cache' / 'Peresmeshnik' / 'models')


def convert_opus_mt(src: str, tgt: str, out: Path) -> Path:
    """Original Marian checkpoint -> CTranslate2 int8 with numpy only (no torch); keeps the tokenizers and license."""
    import ctranslate2
    with tempfile.TemporaryDirectory() as tmp:
        z = Path(tmp) / 'model.zip'
        urllib.request.urlretrieve(OPUS_ZIP.format(src=src, tgt=tgt), z)
        zipfile.ZipFile(z).extractall(tmp)
        ctranslate2.converters.OpusMTConverter(tmp).convert(str(out), quantization='int8', force=True)
        for f in ('source.spm', 'target.spm', 'LICENSE', 'README.md'):
            shutil.copy(Path(tmp) / f, out / f)
    return out


class OpusMt:
    def __init__(self, src: str, tgt: str, path: str | Path | None = None, threads: int = 2):
        import ctranslate2
        import sentencepiece as spm
        path = Path(path or models_dir() / 'ct2' / f'opus-mt-{src}-{tgt}-int8')
        if not (path / 'model.bin').exists():
            convert_opus_mt(src, tgt, path)
        self.src, self.tgt = src, tgt
        self._tr = ctranslate2.Translator(str(path), device='cpu', inter_threads=1, intra_threads=threads)
        self._sp_src = spm.SentencePieceProcessor(str(path / 'source.spm'))
        self._sp_tgt = spm.SentencePieceProcessor(str(path / 'target.spm'))

    def translate(self, text: str, src: str, tgt: str, context: tuple[str, ...] = ()) -> str:
        assert (src, tgt) == (self.src, self.tgt), f'model is {self.src}-{self.tgt}'
        toks = self._sp_src.encode(text, out_type=str)  # the converted model appends </s> itself
        hyp = self._tr.translate_batch([toks], beam_size=1, max_decoding_length=256)[0].hypotheses[0]
        return self._sp_tgt.decode(hyp)
