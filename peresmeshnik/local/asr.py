"""ASR through onnx-asr: Parakeet TDT 0.6B v3 int8 (en, ru) or GigaAM v3 e2e CTC (ru, punctuation).

Neither model streams natively, so the stream re-runs the model on the current clause every
`partial_every_s` of new audio (ARCHITECTURE §4: partials on a window that grows to the clause end).
The last partial keeps token timestamps, which lets the segmenter cut a long clause at punctuation
or between words without another model run (`cut`).
"""
from __future__ import annotations

import re

import numpy as np

from peresmeshnik.engines import SAMPLE_RATE
from peresmeshnik.local.vad import ort_options

ONNX_ASR = {'parakeet-tdt-0.6b-v3': ('nemo-parakeet-tdt-0.6b-v3', 'int8'),  # registry id -> onnx-asr model, quantization
            'gigaam-v3-e2e-ctc': ('gigaam-v3-e2e-ctc', 'int8')}
SENTENCE_END = tuple('.?!;')
LEAD_S = 0.5
TAIL_S = 0.8  # timestamps of the last ~0.8 s of a window run ahead of the audio


def _on_cuda(model) -> bool:
    """Whether the loaded onnx-asr model's onnxruntime sessions actually run on CUDA (not a silent CPU fallback).
    Walks the model for InferenceSessions; if none are found (a layout we do not know), trusts the request."""
    try:
        import onnxruntime as rt
    except Exception:
        return True
    seen = []

    def walk(o, depth=0):
        if depth > 3 or len(seen) > 30:
            return
        for v in vars(o).values() if hasattr(o, '__dict__') else ():
            if isinstance(v, rt.InferenceSession):
                seen.append(v)
            else:
                walk(v, depth + 1)
    walk(model)
    return not seen or any('CUDAExecutionProvider' in s.get_providers() for s in seen)


def _text(tokens: list[str]) -> str:
    return ' '.join(''.join(tokens).split())


def _key(word: str) -> str:
    return re.sub(r'\W', '', word.lower())


class OnnxAsr:
    def __init__(self, model_id: str = 'parakeet-tdt-0.6b-v3', threads: int = 2, partial_every_s: float = 0.3,
                 context_s: float = 1.0):
        import onnx_asr

        from peresmeshnik import models
        from peresmeshnik.local import device
        name, quant = ONNX_ASR[model_id]
        self.name = model_id
        self.partial_every = int(partial_every_s * SAMPLE_RATE)
        self.context_s = context_s
        self._lead = np.zeros(int(LEAD_S * SAMPLE_RATE), np.float32)
        path, opts = models.path(model_id), ort_options(threads)
        providers = device.onnx_providers()
        self.device = 'cuda' if providers[0] == 'CUDAExecutionProvider' else 'cpu'
        try:
            model = onnx_asr.load_model(name, path, quantization=quant, sess_options=opts, providers=providers)
        except Exception as e:  # CUDA listed but could not start (driver, VRAM): them -> me falls back to CPU
            if self.device != 'cuda':
                raise
            import logging
            logging.getLogger('peresmeshnik.device').warning('ASR on CUDA failed (%s); using CPU', e)
            self.device = 'cpu'
            model = onnx_asr.load_model(name, path, quantization=quant, sess_options=opts, providers=['CPUExecutionProvider'])
        # onnxruntime does not raise when the CUDA provider cannot init; it falls back to CPU silently. Report the truth.
        if self.device == 'cuda' and not _on_cuda(model):
            import logging
            logging.getLogger('peresmeshnik.device').warning('ASR: onnxruntime has no usable CUDA; running on CPU')
            self.device = 'cpu'
        self.model = model.with_timestamps()

    def stream(self, lang: str) -> OnnxAsrStream:
        return OnnxAsrStream(self, lang, self.context_s)

    def recognize(self, pcm: np.ndarray, lead: bool = True):
        """Parakeet (int8 and fp32 alike) now and then returns '' for a short window. On windows that start
        at a VAD onset 0.5 s of digital silence in front fixes it (tests/data: 10 of 126 windows empty ->
        1, +19 % time); on windows that start mid-speech after a cut it does harm, so those go as they are."""
        if not lead:
            return self.model.recognize(pcm, sample_rate=SAMPLE_RATE)
        r = self.model.recognize(np.concatenate([self._lead, pcm]), sample_rate=SAMPLE_RATE)
        r.timestamps = [t - LEAD_S for t in r.timestamps or ()]
        return r


class OnnxAsrStream:
    """Audio of the current clause; after a cut, with up to `context_s` of the previous clause in front:
    a window that starts mid-speech loses words, so the model hears the end of the previous clause
    and its tokens (timestamps before the clause start) are dropped."""

    def __init__(self, asr: OnnxAsr, lang: str, context_s: float = 1.0):
        self.asr, self.lang = asr, lang  # both models detect the language themselves
        self.context = int(context_s * SAMPLE_RATE)
        self.pcm = np.zeros(0, np.float32)  # [left context | clause]
        self.ctx = 0  # samples of left context at the start of self.pcm
        self.offset = 0  # where the current clause starts, samples from the segment start
        self._since = 0  # samples fed since the last partial
        self._last = None  # (tokens, timestamps in s from self.pcm[0]) of the clause in the last good partial
        self._last_at = 0  # len(self.pcm) when it was made
        self._tail = ''  # last word emitted by cut(): window timestamps are approximate, the next clause may repeat it

    @property
    def duration(self) -> float:
        """Seconds of the clause, without the left context."""
        return (len(self.pcm) - self.ctx) / SAMPLE_RATE

    def _run(self) -> tuple[list[str], list[float]]:
        r = self.asr.recognize(self.pcm, self.offset == 0)
        toks, ts = r.tokens or [], r.timestamps or []
        if self.ctx:  # drop the words of the context
            start = self.ctx / SAMPLE_RATE - 0.08
            i = next((i for i, (tok, t) in enumerate(zip(toks, ts)) if t >= start and tok.startswith(' ')), len(toks))
            toks, ts = toks[i:], ts[i:]
        return toks, ts

    def feed(self, pcm: np.ndarray, partial: bool = True) -> str | None:
        """Add audio; every `partial_every_s` re-run the model and return the new partial text (if it changed)."""
        self.pcm = np.concatenate([self.pcm, pcm])
        self._since += len(pcm)
        if self._since < self.asr.partial_every or not partial:
            return None
        prev = _text(self._last[0]) if self._last else ''
        self._since = 0
        toks, ts = self._run()
        if not (text := _text(toks)):  # Parakeet now and then returns '' for a window; keep the last good partial
            return None
        self._last, self._last_at = (toks, ts), len(self.pcm)
        return text if text != prev else None

    def cut(self, force: bool = False) -> str | None:
        """Final text of the clause up to the last sentence end in the last partial, if there is a
        word after it; with `force` (clause too long) also at a comma or between words.
        The audio after the cut stays as the start of the next clause."""
        if not self._last:
            return self.finish() if force else None
        toks, ts = self._last
        # word starts far enough from the window end: there the model squeezes the last words together
        starts = [k for k in range(1, len(toks)) if toks[k].startswith(' ') and ts[k] <= len(self.pcm) / SAMPLE_RATE - TAIL_S]
        # a sentence end is punctuation before a capitalised word ("U.N. also" is not one)
        k = next((k for k in reversed(starts) if toks[k - 1].endswith(SENTENCE_END) and toks[k][1:2].isupper()), None)
        if k is None and force:  # a comma in the second half of the clause, else between the last words
            half = self.ctx / SAMPLE_RATE + self.duration / 2
            k = next((k for k in reversed(starts) if toks[k - 1].endswith(',') and ts[k] >= half), starts[-1] if starts else None)
        if k is None:
            return self.finish() if force else None
        at = int(max((ts[k - 1] + ts[k]) / 2, ts[k] - 0.16) * SAMPLE_RATE)  # between the two tokens
        ctx = min(at, self.context)
        self.offset += at - self.ctx
        self.pcm, self.ctx = self.pcm[at - ctx:], ctx
        self._last, self._since = None, self.asr.partial_every  # the next feed() makes a partial of the rest
        text = self._dedup(_text(toks[:k]))
        self._tail = _key(text.split()[-1]) if text else ''
        return text

    def _dedup(self, text: str) -> str:
        words, tail, self._tail = text.split(), self._tail, ''
        return ' '.join(words[1:]) if tail and words and _key(words[0]) == tail else text

    def finish(self, settled_s: float = 0.0) -> str:
        """Final text of the segment. `settled_s`: the segment ends in that much known silence (the VAD
        pause), so a partial made within it already saw all the speech and is reused as final."""
        fresh = self._last is not None and len(self.pcm) - self._last_at <= settled_s * SAMPLE_RATE
        toks = self._last[0] if fresh else (self._run()[0] if len(self.pcm) > self.ctx else [])
        text = _text(toks) or (_text(self._last[0]) if self._last else '')
        if not text and self.duration > 0.3:  # still nothing: the clause alone, with the other lead-in
            text = self.asr.recognize(self.pcm[self.ctx:], self.offset != 0).text.strip()
        self.offset += len(self.pcm) - self.ctx
        self.pcm, self.ctx, self._last, self._since = np.zeros(0, np.float32), 0, None, 0
        return self._dedup(text)
