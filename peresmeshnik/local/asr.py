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
            'gigaam-v3-e2e-ctc': ('gigaam-v3-e2e-ctc', 'int8'),
            'parakeet-tdt_ctc-0.6b-ja': ('nemo-conformer-tdt', 'int8'),
            'fastconformer-ar-pcd': ('nemo-conformer-ctc', 'int8'),
            'fastconformer-fr-pc': ('nemo-conformer-ctc', 'int8'),
            'whisper-large-v3-turbo': ('whisper', 'fp16')}  # no timestamps: see OnnxAsrStream._cut_quiet
SENTENCE_END = tuple('.?!;。？！；؟')
COMMAS = tuple(',，、،')
NO_SPACES = {'zh', 'ja'}  # no spaces between words: every token boundary is a word boundary, words are characters
LEAD_S = 0.5
TAIL_S = 0.8  # timestamps of the last ~0.8 s of a window run ahead of the audio (0.5: BENCH stage 5, not taken)
MIN_PART_S = 1.0  # an early cut at a comma leaves at least this much speech before it
# the next clause may repeat up to this many of a cut's last words ("to cope with" | "cope with the"): dropped when
# its own window did not hear them in the context before the cut (there "Thank you. | Thank you very much" is said twice)
REPEAT_WORDS = 3
REPEAT_CHARS = 4  # the same in zh / ja, in characters (a word is 1-3 of them)
# Whisper decodes up to 448 tokens; a window it hears as a loop takes them all (7 s on a CPU, BENCH stage 6). A clause
# is <= 5 s plus its context: ~60 tokens even in Chinese
WHISPER_MAX_TOKENS = 128


def _on_cuda(model) -> bool:
    """Whether the loaded onnx-asr model's onnxruntime sessions actually run on CUDA (not a silent CPU fallback).
    Walks the model for objects with get_providers() (an InferenceSession, however onnx-asr wraps it); if none are
    found (a layout we do not know), reports CPU — it does not claim CUDA it cannot confirm."""
    seen = []

    def walk(o, depth=0):
        if depth > 3 or len(seen) > 30:
            return
        for v in vars(o).values() if hasattr(o, '__dict__') else ():
            if callable(getattr(v, 'get_providers', None)):
                seen.append(v)
            else:
                walk(v, depth + 1)
    walk(model)
    return bool(seen) and any('CUDAExecutionProvider' in s.get_providers() for s in seen)


def cap_whisper(model, max_tokens: int = WHISPER_MAX_TOKENS) -> None:
    """Cap the tokens an onnx-asr Whisper decodes per window (its language guess asks for 3: kept)."""
    w = getattr(model, 'asr', None)
    dec = getattr(w, '_decoding', None)
    if dec is not None:
        w._decoding = lambda enc, toks, max_length=448: dec(enc, toks, min(max_length, max_tokens))


def _text(tokens: list[str]) -> str:
    return ' '.join(''.join(tokens).split())


def _key(word: str) -> str:
    return re.sub(r'\W', '', word.lower())


def units(text: str, lang: str) -> list[str]:
    """The words of a text, or its characters in a language written without spaces (zh, ja)."""
    return text.split() if lang not in NO_SPACES else [c for c in text if not c.isspace()]


def join(words: list[str], lang: str) -> str:
    return ('' if lang in NO_SPACES else ' ').join(words)


def _starts_sentence(tok: str) -> bool:
    """A sentence may start at this token: a capital, or a letter of a script without case (CJK, Hangul, Arabic)."""
    c = tok.strip()[:1]
    return c.isupper() or (c.isalpha() and c.lower() == c.upper())


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
        self.timed = name != 'whisper'  # Whisper through onnx-asr: text only
        if not self.timed:
            cap_whisper(self.model)

    def stream(self, lang: str) -> OnnxAsrStream:
        return OnnxAsrStream(self, lang, self.context_s)

    def recognize(self, pcm: np.ndarray, lead: bool = True, lang: str | None = None):
        """Parakeet (int8 and fp32 alike) now and then returns '' for a short window. On windows that start
        at a VAD onset 0.5 s of digital silence in front fixes it (tests/data: 10 of 126 windows empty ->
        1, +19 % time); on windows that start mid-speech after a cut it does harm, so those go as they are."""
        kw = {} if self.timed or lang is None else {'language': lang}  # Whisper: the call's language, no guessing
        if not lead:
            return self.model.recognize(pcm, sample_rate=SAMPLE_RATE, **kw)
        r = self.model.recognize(np.concatenate([self._lead, pcm]), sample_rate=SAMPLE_RATE, **kw)
        r.timestamps = [t - LEAD_S for t in r.timestamps or ()]
        return r


class OnnxAsrStream:
    """Audio of the current clause; after a cut, with up to `context_s` of the previous clause in front:
    a window that starts mid-speech loses words, so the model hears the end of the previous clause
    and its tokens (timestamps before the clause start) are dropped."""

    def __init__(self, asr: OnnxAsr, lang: str, context_s: float = 1.0):
        self.asr, self.lang = asr, lang  # Parakeet and GigaAM detect the language themselves
        self.spaced = lang not in NO_SPACES
        self.repeat = REPEAT_WORDS if self.spaced else REPEAT_CHARS
        self.context = int(context_s * SAMPLE_RATE)
        self.pcm = np.zeros(0, np.float32)  # [left context | clause]
        self.ctx = 0  # samples of left context at the start of self.pcm
        self.offset = 0  # where the current clause starts, samples from the segment start
        self._since = 0  # samples fed since the last partial
        self._last = None  # (tokens, timestamps in s from self.pcm[0]) of the clause in the last good partial
        self._last_at = 0  # len(self.pcm) when it was made
        self._heard = self._last_heard = ()  # last words of the context in the last run / in the last good partial
        self._tail = ()  # last words emitted by cut(): window timestamps are approximate, the next clause may repeat them
        self.cut_at = ''  # where the last cut() cut: 'sentence', 'comma' or 'word'

    @property
    def duration(self) -> float:
        """Seconds of the clause, without the left context."""
        return (len(self.pcm) - self.ctx) / SAMPLE_RATE

    def _word(self, tok: str) -> bool:
        """A word starts at this token (in zh / ja: at any token with text)."""
        return tok.startswith(' ') if self.spaced else bool(tok.strip())

    def _run(self) -> tuple[list[str], list[float]]:
        r = self.asr.recognize(self.pcm, self.offset == 0, self.lang)
        if not getattr(self.asr, 'timed', True):  # text only: its words as tokens, no times (and no context: ctx is 0)
            return [w if not self.spaced else ' ' + w for w in units(r.text, self.lang)], []
        toks, ts = r.tokens or [], r.timestamps or []
        self._heard = ()
        if self.ctx:  # drop the words of the context
            start = self.ctx / SAMPLE_RATE - 0.08
            i = next((i for i, (tok, t) in enumerate(zip(toks, ts)) if t >= start and self._word(tok)), len(toks))
            self._heard = tuple(_key(w) for w in units(_text(toks[:i]), self.lang)[-self.repeat:])
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
        self._last, self._last_at, self._last_heard = (toks, ts), len(self.pcm), self._heard
        return text if text != prev else None

    def cut(self, force: bool = False, comma: bool = False) -> str | None:
        """Final text of the clause up to the last sentence end in the last partial, if there is a
        word after it; with `comma` (a long clause) also at a comma at least `MIN_PART_S` in; with `force`
        (clause too long) at a comma in its second half or between words. `cut_at` says where.
        The audio after the cut stays as the start of the next clause."""
        self.cut_at = 'word'  # forced with no partial yet: the whole clause
        if not self._last:
            return self.finish() if force else None
        if not getattr(self.asr, 'timed', True):
            return self._cut_quiet() if force else None
        toks, ts = self._last
        # word starts far enough from the window end: there the model squeezes the last words together
        starts = [k for k in range(1, len(toks)) if self._word(toks[k]) and ts[k] <= len(self.pcm) / SAMPLE_RATE - TAIL_S]
        # a sentence end is punctuation before a capitalised word ("U.N. also" is not one; scripts without case: a letter)
        k = next((k for k in reversed(starts) if toks[k - 1].endswith(SENTENCE_END) and _starts_sentence(toks[k])), None)
        self.cut_at = 'sentence'
        if k is None and (force or comma):  # a comma (in the second half when forced), else between the last words
            lo = self.ctx / SAMPLE_RATE + (self.duration / 2 if force else MIN_PART_S)
            k = next((k for k in reversed(starts) if toks[k - 1].endswith(COMMAS) and ts[k] >= lo), None)
            self.cut_at = 'comma'
            if k is None and force and starts:
                k, self.cut_at = starts[-1], 'word'
        if k is None:
            return self.finish() if force else None
        at = int(max((ts[k - 1] + ts[k]) / 2, ts[k] - 0.16) * SAMPLE_RATE)  # between the two tokens
        ctx = min(at, self.context)
        self.offset += at - self.ctx
        self.pcm, self.ctx = self.pcm[at - ctx:], ctx
        self._last, self._since = None, self.asr.partial_every  # the next feed() makes a partial of the rest
        text = self._dedup(_text(toks[:k]), self._last_heard)
        self._tail = tuple(_key(w) for w in units(text, self.lang)[-self.repeat:])
        return text

    def _cut_quiet(self) -> str:
        """A forced cut without timestamps: at the quietest 0.1 s of the clause's second half (short of the last
        TAIL_S), the words before it from one more run; the rest starts the next clause, with no context."""
        f = SAMPLE_RATE // 10
        lo, hi = len(self.pcm) // 2 // f, (len(self.pcm) - int(TAIL_S * SAMPLE_RATE)) // f
        if hi <= lo:
            return self.finish()
        rms = [float(np.mean(self.pcm[i * f:(i + 1) * f] ** 2)) for i in range(lo, hi)]
        at = (lo + int(np.argmin(rms))) * f + f // 2
        text = self._dedup(self.asr.recognize(self.pcm[:at], self.offset == 0, self.lang).text.strip(), ())
        self.offset += at
        self.pcm, self._last, self._since = self.pcm[at:], None, self.asr.partial_every
        self._tail = tuple(_key(w) for w in units(text, self.lang)[-self.repeat:])
        return text

    def _dedup(self, text: str, heard: tuple) -> str:
        """Without the longest run of the cut's last words this text starts with, unless this window heard that run
        in its context too (then it is said again, not the same audio)."""
        words, tail, self._tail = units(text, self.lang), self._tail, ()
        head = tuple(_key(w) for w in words[:len(tail)])
        k = next((k for k in range(len(tail), 0, -1) if head[:k] == tail[-k:] and heard[-k:] != tail[-k:]), 0)
        return join(words[k:], self.lang) if k else text

    def finish(self, settled_s: float = 0.0) -> str:
        """Final text of the segment. `settled_s`: the segment ends in that much known silence (the VAD
        pause), so a partial made within it already saw all the speech and is reused as final."""
        fresh = self._last is not None and len(self.pcm) - self._last_at <= settled_s * SAMPLE_RATE
        toks, heard = (self._last[0], self._last_heard) if fresh else \
            (self._run()[0], self._heard) if len(self.pcm) > self.ctx else ([], ())
        text = _text(toks)
        if not text and self._last:
            text, heard = _text(self._last[0]), self._last_heard
        if not text and self.duration > 0.3:  # still nothing: the clause alone, with the other lead-in
            text = self.asr.recognize(self.pcm[self.ctx:], self.offset != 0, self.lang).text.strip()
        self.offset += len(self.pcm) - self.ctx
        self.pcm, self.ctx, self._last, self._since = np.zeros(0, np.float32), 0, None, 0
        return self._dedup(text, heard)
