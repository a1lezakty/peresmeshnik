# TTS with voice cloning: notes

Script: `bench/tts_clone.py` (synth in the model venv, eval in the `tts-eval` venv). Results: `bench/results/<host>.jsonl`,
rows `family=tts_clone`. Wavs: `$BENCH_OUT/tts_clone/<model>/<ref>2<out>_<id>.wav` (+ `run.json`, `eval.json` with per-sentence
timings and back-transcriptions).

Method: reference voice `ref_en` (LibriSpeech reader, ~40 s) / `ref_ru` (Russian LibriVox reader, Pushkin poetry, ~40 s),
output = 6 shortest FLEURS dev sentences per language (same for every model). One untimed warm-up sentence after load.
TTFA = first audio chunk (streaming models: XTTS `inference_stream`, CosyVoice `stream=True`) or the whole utterance
(Chatterbox). RTF = synth time / audio seconds. Speaker similarity = cosine of SpeechBrain ECAPA (`spkrec-ecapa-voxceleb`)
embeddings, output vs the full 40 s reference. CER = faster-whisper `small` int8, language forced, NFKC + lowercase,
punctuation stripped, ё→е, corpus-level over the 6 sentences.

## Results (07.11.2026)

Hosts: `linux-x86_64-4c` = project server (AMD EPYC-Genoa, 4 threads, measured while the box was swapping:
~1 GB free, 9 GB swap in use, so its timings are pessimistic); `vast-c-cpu8` = Vast server C (i7-13700, 8 threads,
CUDA hidden), a stand-in for an x86 laptop; `vast-c-a4000` = the same box on one RTX A4000 (memory cap 0.6).
macOS M1 numbers are still to come from the `macos-14` runner. TTFA/synth are per sentence (mean, max).
Chatterbox never streams. CosyVoice streams on CUDA only: on CPU, streaming made it ~3x slower (first sentence
TTFA 60 s, RTF 23 vs 35 s / RTF 6.8 without), so its CPU row is `--no-stream`.

| host | model | ref→out | stream | load s | TTFA s (max) | synth s/sent | RTF | peak RSS MB | GPU MB | spk-sim (min) | CER |
|---|---|---|---|---|---|---|---|---|---|---|---|
| vast-c-a4000 | chatterbox | en→en | n | 8.46 | 2.152 (3.093) | 2.152 | 0.404 | 5042 | 3704 | 0.692 (0.645) | 0.008 |
| vast-c-a4000 | chatterbox | en→ru | n | 8.46 | 2.684 (3.13) | 2.684 | 0.395 | 5042 | 3704 | 0.553 (0.509) | 0.0382 |
| vast-c-a4000 | chatterbox | ru→en | n | 8.46 | 2.506 (3.162) | 2.506 | 0.402 | 5042 | 3704 | 0.695 (0.612) | 0.0683 |
| vast-c-a4000 | chatterbox | ru→ru | n | 8.46 | 2.509 (2.852) | 2.509 | 0.423 | 5042 | 3704 | 0.759 (0.627) | 0.0191 |
| vast-c-cpu8 | chatterbox | en→en | n | 7.78 | 18.187 (21.635) | 18.187 | 3.516 | 6990 |  | 0.707 (0.67) | 0.0181 |
| vast-c-cpu8 | chatterbox | en→ru | n | 7.78 | 22.689 (25.897) | 22.689 | 3.288 | 6990 |  | 0.518 (0.483) | 0.0573 |
| vast-c-cpu8 | chatterbox | ru→en | n | 7.78 | 19.489 (23.463) | 19.489 | 3.497 | 6990 |  | 0.681 (0.643) | 0.0382 |
| vast-c-cpu8 | chatterbox | ru→ru | n | 7.78 | 19.764 (23.601) | 19.764 | 3.504 | 6990 |  | 0.778 (0.663) | 0.0191 |
| vast-c-a4000 | cosyvoice | en→en | y | 98.29 | 2.206 (2.259) | 6.074 | 1.221 | 8424 | 3891 | 0.721 (0.663) | 0.0341 |
| vast-c-a4000 | cosyvoice | en→ru | y | 98.29 | 2.234 (2.281) | 6.268 | 1.204 | 8424 | 3891 | 0.502 (0.412) | 0.0361 |
| vast-c-a4000 | cosyvoice | ru→en | y | 98.29 | 3.057 (3.091) | 10.279 | 1.831 | 8424 | 3891 | 0.589 (0.532) | 0.008 |
| vast-c-a4000 | cosyvoice | ru→ru | y | 98.29 | 3.328 (3.376) | 8.976 | 2.015 | 8424 | 3891 | 0.783 (0.76) | 0.2781 |
| vast-c-cpu8 | cosyvoice | en→en | n | 62.66 | 35.176 (41.969) | 35.176 | 6.817 | 8841 |  | 0.731 (0.663) | 0.0422 |
| vast-c-cpu8 | cosyvoice | en→ru | n | 62.66 | 35.113 (36.751) | 35.114 | 6.4 | 8841 |  | 0.433 (0.335) | 0.0361 |
| vast-c-cpu8 | cosyvoice | ru→en | n | 62.66 | 49.389 (54.045) | 49.39 | 8.352 | 8841 |  | 0.571 (0.493) | 0.0221 |
| vast-c-cpu8 | cosyvoice | ru→ru | n | 62.66 | 46.5 (52.09) | 46.501 | 11.196 | 8841 |  | 0.676 (0.059) | 0.2251 |
| linux-x86_64-4c | xtts | en→en | y | 208.23 | 3.478 (4.26) | 26.686 | 4.342 | 3849 |  | 0.605 (0.564) | 0.1305 |
| linux-x86_64-4c | xtts | en→ru | y | 208.23 | 3.546 (5.358) | 26.679 | 4.176 | 3849 |  | 0.385 (0.303) | 0.0594 |
| linux-x86_64-4c | xtts | ru→en | y | 208.23 | 3.102 (3.579) | 17.686 | 3.671 | 3849 |  | 0.421 (0.363) | 0.012 |
| linux-x86_64-4c | xtts | ru→ru | y | 208.23 | 3.342 (4.057) | 23.028 | 4.034 | 3849 |  | 0.68 (0.533) | 0.0764 |
| vast-c-a4000 | xtts | en→en | y | 7.55 | 0.144 (0.148) | 1.007 | 0.179 | 4499 | 2070 | 0.557 (0.454) | 0.0622 |
| vast-c-a4000 | xtts | en→ru | y | 7.55 | 0.143 (0.144) | 1.026 | 0.176 | 4499 | 2070 | 0.38 (0.322) | 0.0488 |
| vast-c-a4000 | xtts | ru→en | y | 7.55 | 0.144 (0.148) | 0.796 | 0.172 | 4499 | 2070 | 0.396 (0.278) | 0.0141 |
| vast-c-a4000 | xtts | ru→ru | y | 7.55 | 0.144 (0.145) | 0.95 | 0.176 | 4499 | 2070 | 0.657 (0.543) | 0.0616 |
| vast-c-cpu8 | xtts | en→en | y | 9.79 | 1.264 (1.588) | 12.362 | 1.894 | 4406 |  | 0.602 (0.489) | 0.1586 |
| vast-c-cpu8 | xtts | en→ru | y | 9.79 | 1.303 (1.605) | 11.883 | 1.858 | 4406 |  | 0.353 (0.202) | 0.0616 |
| vast-c-cpu8 | xtts | ru→en | y | 9.79 | 1.169 (1.522) | 8.458 | 1.747 | 4406 |  | 0.44 (0.338) | 0.006 |
| vast-c-cpu8 | xtts | ru→ru | y | 9.79 | 1.407 (1.648) | 10.042 | 1.806 | 4406 |  | 0.677 (0.618) | 0.0701 |

Reference points (same judge): different real speakers `ref_en` vs `ref_ru` 0.085; real FLEURS speakers vs the refs
-0.03..0.10; same speaker (two halves of one reference) 0.90 en / 0.93 ru. Whisper-small CER on the real FLEURS
recordings: 0.022 en / 0.013 ru, which is the floor for CER here.

Findings:
- **Clone similarity:** Chatterbox is best on every case, and it holds up cross-lingually (ru→en 0.68-0.70,
  en→ru 0.52-0.55). CosyVoice3 is close in the same language (0.68-0.78) and weaker cross-lingually (ru→en 0.57-0.59).
  XTTS is weakest cross-lingually (ru→en 0.40-0.44, en→ru 0.35-0.38); in the same language it reaches 0.56-0.68.
  The en→ru case is the lowest for all three models.
- **Intelligibility:** all three are fine for ru→en (CER 0.006-0.07). Two failure modes:
  - XTTS often adds a babbling tail after the sentence ("... шоу оттай", "... детьми пальцем", and one long English
    hallucination on the shortest sentence). That tail is most of its en→en CER of 0.13-0.16, so production needs
    a tail trim or length guard.
  - CosyVoice3's same-language Russian zero-shot with the 26 s poetry prompt sometimes stops early or drops clauses
    (one 0.24 s output), giving CER 0.22-0.28. Cross-lingual mode does not show it.
- **Speed:**
  - XTTS is the only one with usable streaming: TTFA 0.14 s, RTF 0.18 on the A4000; TTFA 1.2-1.4 s, RTF 1.75-1.9
    on 8 x86 threads (not real time).
  - Chatterbox: RTF 0.4 on GPU (one sentence ≈ 2-3 s before any audio), RTF 3.5 on CPU.
  - CosyVoice3 (fp32, no JIT/TensorRT): RTF 1.2-2.0 on GPU, 6-11 on CPU, load 60-100 s. Not real time anywhere here.
- **Memory (peak RSS):** XTTS 3.8-4.5 GB (the peak is at load; `rss_after_load` = peak). Chatterbox 7.0 GB on CPU
  (5.0 GB host + 3.7 GB GPU on CUDA). CosyVoice3 8.4-8.8 GB.
  So on an 8 GB M1 only XTTS fits; Chatterbox and CosyVoice would swap.

## XTTS-v2 (coqui-tts 0.28.0, idiap fork)

- License: code MPL-2.0; weights Coqui Public Model License 1.0.0 (non-commercial only; OK for PolyForm Noncommercial).
  Weights now on HF `tts-hub/XTTS-v2` (downloaded by coqui-tts on first run, `COQUI_TOS_AGREED=1`).
- Size on disk: 1.8 GB weights (HF `tts-hub/XTTS-v2`); venv ~1.5 GB with CPU torch, ~5 GB with CUDA torch.
- Languages: 17 incl. Russian (yes) and English; cross-lingual cloning works out of the box (speaker latents are language-free).
- Streaming: yes, `inference_stream` (chunks of 20 GPT tokens); this bench times the streaming path.
- Gotchas: coqui-tts 0.28.0 imports `six` without declaring it (pinned in requirements); transformers 5.x warns about
  bos/eos token ids (harmless). Text > ~180 chars for ru must be split (`enable_text_splitting`); bench sentences are short.
  Reference: uses at most 30 s of it (`max_ref_length`), GPT conditioning on 6 s chunks.
- MPS: `--device mps` is wired, but XTTS on MPS is known to be partial (some ops fall back; set
  `PYTORCH_ENABLE_MPS_FALLBACK=1`); CPU is the safe default on M1. Untested here.
- macos-14 runner (3 cores, 7 GB): fits (peak ~4 GB). Expect ~15 min for synth + eval incl. the 1.8 GB download,
  assuming RTF 2.5-4 on 3 M1 cores (an estimate, not measured).
- Verdict: the only candidate that fits an 8 GB M1 and streams, but CPU RTF is ~1.8 even on 8 fast x86 threads,
  cross-lingual similarity is the weakest of the three, and the babbling tail needs a guard.

## Chatterbox Multilingual (chatterbox-tts 0.1.7)

- License: code MIT; weights MIT (`ResembleAI/chatterbox`). Every output carries an imperceptible Perth watermark
  (resemble-perth), applied inside `generate()`.
- Size on disk: ~3.2 GB downloaded (`t3_mtl23ls_v2.safetensors` 2.1 GB fp32 0.5B Llama, `s3gen.pt` 1.1 GB, `ve.pt`);
  venv ~1.5 GB (it pulls gradio 6.8 as a hard dependency).
- Languages: 23 incl. Russian (yes, `language_id='ru'`) and English.
- Streaming: not in the official package (only full utterances); TTFA = full synth time.
- Gotchas: hard pins torch/torchaudio 2.6.0, transformers 5.2.0, numpy<2, so it needs its own venv. For a reference in a
  different language than the output, upstream advises `cfg_weight=0` to reduce accent transfer. The bench does this for
  cross-lingual cases and uses the default 0.5 otherwise. Uses only the first ~10 s of the reference.
- More gotchas:
  - resemble-perth imports `pkg_resources`; without setuptools the watermarker silently becomes None and loading fails
    with `'NoneType' object is not callable` (pinned `setuptools<70`).
  - spacy-pkuseg downloads a 35 MB model to `~/.pkuseg` on first load (`PKUSEG_HOME` overrides it).
  - Its alignment analyzer forces EOS on long tails, so it does not babble like XTTS.
- MPS: supported upstream (`device='mps'`, checkpoints are mapped to CPU first). Untested here.
- macos-14 runner: CPU peak RSS 7.0 GB will not fit in 7 GB without heavy swapping. Try `--device mps`, or `--n 2`.
  At CPU RTF 3.5 on 8 x86 threads, the full 24 sentences on 3 M1 cores are likely > 30 min.
- Verdict: best voice similarity (incl. ru→en) and intelligibility. GPU only (RTF 0.4); no streaming, so the latency
  floor is about one sentence of synthesis.

## Fun-CosyVoice3-0.5B-2512 (FunAudioLLM/CosyVoice)

- CosyVoice2-0.5B: no Russian (zh/en/ja/ko + Chinese dialects), so it is skipped in favour of CosyVoice3.
- License: code Apache-2.0; weights Apache-2.0 (`FunAudioLLM/Fun-CosyVoice3-0.5B-2512` on HF / ModelScope).
- Languages: 9 incl. Russian (yes): zh, en, ja, ko, de, es, fr, it, ru, plus 18 Chinese dialects; zero-shot and cross-lingual.
- Size on disk: the HF repo is 9.7 GB, but inference needs only 5.1 GB: `llm.pt` 2.0 GB, `flow.pt` 1.3 GB,
  `speech_tokenizer_v3.onnx` 1.0 GB, `CosyVoice-BlankEN/` 1.0 GB (Qwen tokenizer/config), `hift.pt`, `campplus.onnx`, yaml.
  Skip `llm.rl.pt`, `*.batch.onnx`, and `flow.decoder.estimator.fp32.onnx` (TensorRT only). Venv ~1.5 GB.
- Install: not on PyPI. Clone the repo with its `third_party/Matcha-TTS` submodule. Upstream requirements are CUDA-centric
  (deepspeed, tensorrt, onnxruntime-gpu, gradio). `requirements-tts-clone-cosyvoice.txt` is the CPU inference subset.
- Gotchas: the prompt must be <= 30 s (assert in the frontend), so the bench rebuilds a shorter reference with matching text
  via `data.reference(lang, 15)`: ~17 s en / ~26 s ru (needs `$BENCH_DATA/refs/*.parquet` from `data.py` and pyarrow).
  CosyVoice3 requires `You are a helpful assistant.<|endofprompt|>` in the prompt text (zero-shot) or in the tts text
  (cross-lingual). Same language: zero-shot with the transcript; other language: cross-lingual (no prompt text).
- More gotchas:
  - Runtime imports beyond upstream's inference set: `setuptools<70` (lightning imports `pkg_resources`), `gdown`,
    `wget`, `matplotlib` (via matcha). `openai-whisper==20231117` needs `--build-constraints` with `setuptools<70`.
  - It writes to `~/.cache/gdown`, `~/.config/matplotlib`, and a modelscope wetext asset; set `HOME`/`MPLCONFIGDIR`
    if that matters.
  - Streaming bug: `CosyVoice3Model` doubles `token_hop_len` per streamed chunk and never resets it, so after the first
    utterance TTFA ≈ full synth. `tts_clone.py` resets it per call.
  - On CPU, streaming makes it ~3x slower (the flow re-runs over the whole token prefix per chunk while the LLM thread
    competes for the cores).
- Device: CosyVoice picks `cuda` if available, otherwise CPU; there is no MPS path, so `--device` is ignored.
- macos-14 runner: does not fit (peak RSS 8.8 GB) and is too slow (CPU RTF 6-11 on 8 x86 threads). Skip it in CI,
  or run `--n 1` only as a smoke test.
- Verdict: good same-language clone, but too heavy and too slow for this product locally; not a candidate.

## Recommendation ("me → them": my RU voice speaking EN)

- **M1 8 GB:** XTTS-v2 is the only one that fits (~4 GB). It is unlikely to be real time on CPU, so measure it on
  macos-14 (CPU and MPS). If RTF stays above 1, the 8 GB tier needs cloud mode or a lighter path (fast TTS + voice
  converter, not benchmarked here).
- **M1 Pro 16 GB:** XTTS (streaming, TTFA ~1 s if MPS/CPU RTF < 1). Chatterbox only if MPS brings RTF well below 1,
  since it gives a clearly better clone (ru→en similarity 0.69 vs 0.42).
- **This 4-core CPU:** none is usable live. XTTS ran at RTF ~4 here (under swap) and 1.8 even on 8 i7 threads.
- **NVIDIA:** Chatterbox for quality (RTF 0.4, sentence-level latency ~2-3 s), or XTTS for latency
  (TTFA 0.14 s, RTF 0.18).

## Rerun

One venv per stack (Python 3.11, `uv`). Linux: install torch/torchaudio first from `https://download.pytorch.org/whl/cpu`
(or `/whl/cu128` xtts, `/whl/cu124` chatterbox, `/whl/cu121` cosyvoice for CUDA), then the requirements file.
macOS arm64: plain PyPI torch (it ships MPS).

```sh
export BENCH_DATA=$PWD/bench-data HF_HOME=$PWD/models/hf TTS_HOME=$PWD/models/tts COQUI_TOS_AGREED=1
export BENCH_THREADS=4 OMP_NUM_THREADS=4   # 3 on the macos-14 runner
uv venv -p 3.11 venvs/xtts && uv pip install -p venvs/xtts/bin/python -r bench/requirements-tts-clone-xtts.txt
uv venv -p 3.11 venvs/chatterbox && uv pip install -p venvs/chatterbox/bin/python -r bench/requirements-tts-clone-chatterbox.txt
uv venv -p 3.11 venvs/tts-eval && uv pip install -p venvs/tts-eval/bin/python -r bench/requirements-tts-clone-eval.txt
git clone https://github.com/FunAudioLLM/CosyVoice src/CosyVoice && git -C src/CosyVoice checkout 074ca6dc9e80a2f424f1f74b48bdd7d3fea531cc \
  && git -C src/CosyVoice submodule update --init --recursive
echo 'setuptools<70' > /tmp/bc.txt
uv venv -p 3.11 venvs/cosyvoice && uv pip install -p venvs/cosyvoice/bin/python --build-constraints /tmp/bc.txt \
  -r bench/requirements-tts-clone-cosyvoice.txt
export COSYVOICE_SRC=$PWD/src/CosyVoice COSYVOICE_MODEL=$PWD/models/cosyvoice3
venvs/xtts/bin/python -c "from huggingface_hub import snapshot_download as d; d('FunAudioLLM/Fun-CosyVoice3-0.5B-2512', \
  local_dir='$COSYVOICE_MODEL', allow_patterns=['CosyVoice-BlankEN/*', 'campplus.onnx', 'cosyvoice3.yaml', 'flow.pt', 'hift.pt', \
  'llm.pt', 'speech_tokenizer_v3.onnx', 'config*.json'])"

export PKUSEG_HOME=$PWD/models/pkuseg PYTORCH_ENABLE_MPS_FALLBACK=1
for m in xtts chatterbox cosyvoice; do                 # one model per process; --device cpu|mps|cuda
  venvs/$m/bin/python bench/tts_clone.py synth --model $m --device cpu   # cosyvoice on CPU: add --no-stream
  venvs/tts-eval/bin/python bench/tts_clone.py eval --model $m
done
venvs/tts-eval/bin/python bench/tts_clone.py baseline
```

`--n` limits the sentences per language (default 6). `--gpu-mem-frac` (default 0.6) caps CUDA memory. Wavs go to
`$BENCH_OUT/tts_clone/` (default `$BENCH_DATA/../out`). `BENCH_HOST` overrides the results file tag.
