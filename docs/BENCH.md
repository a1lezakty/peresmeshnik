# Benchmarks (stages 0–2)

Scripts: [`bench/`](../bench). Raw rows: `bench/results/<host>.jsonl`; the tables below are made
by `python bench/report.py`. Measured 2026-11-07.

## Where it was measured

| Host tag | Machine | Used for |
|---|---|---|
| `macos-15` | GitHub Actions `macos-15` runner: Apple M1 (Virtual), 3 cores, 7 GB; torch MPS unavailable ("paravirtual device does not support Metal 2.0"), onnxruntime lists CoreML | the **M1** column: lower bound for a real M1 8 GB (which has 4 performance + 4 efficiency cores) |
| `vast-a-cpu4` | AMD Ryzen 5 5600X, **4 threads**, CUDA hidden (shared Vast server) | **x86 CPU** laptop stand-in |
| `vast-a-rtx3060` | same server, NVIDIA RTX 3060 12 GB | the **NVIDIA** tier |
| `vast-c-cpu8`, `vast-c-a4000` | Intel i7-13700, 8 threads / NVIDIA RTX A4000 16 GB (shared Vast server) | voice-clone TTS (too big for the other hosts) |
| `linux-x86_64-4c` | the dev server, AMD EPYC-Genoa, 4 cores, 7.6 GB (shared, at times swapping) | Gemini Live runs (network only); Piper voices; one early XTTS run (pessimistic); stage 1 PipeWire runs |
| `vast-b-pipeline` | AMD Ryzen 9 7950X3D (shared Vast server), the app's 2 threads per engine | stage 1 pipeline: cost, quality, real-time lags |
| `ubuntu-latest`, `windows-latest` | GitHub Actions runners, 4 vCPU | stage 1 working-mode gate (with `macos-15`) |

All CPU runs pin every thread pool (`--threads`, `OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS`) and
run under `nice`; the Vast servers host other services, so absolute times can be a few percent
noisy. One model per process, so `peak_rss_mb` is that model's own footprint.

## Data

- **FLEURS dev** (CC-BY-4.0): 30 sentences that exist in both `en_us` and `ru_ru`, first
  recording of each: 315 s of English, 364 s of Russian read speech. Because FLEURS is built on
  FLoRes, the Russian text is the reference translation of the English one and vice versa.
- **Voice references** for cloning, ~40 s each: LibriSpeech test-clean (one reader, EN) and
  Russian LibriSpeech test (one LibriVox reader of Pushkin's poems, RU).
- Prepared by `python bench/data.py` (downloads ~1 GB once).

## Metrics

- **WER / CER**: lowercase, `ё→е`, punctuation removed; FLEURS raw transcripts keep digits, so a
  model writing numbers as words pays a small penalty.
- **utt latency**: time to transcribe one whole utterance (median 10 s of audio).
- **2 s window**: time to transcribe a 2 s window — the cost of one subtitle refresh when an
  offline model is re-run on the growing clause; Whisper pads every window to 30 s, so this is its
  floor.
- **chunk latency**: true streaming models (T-one) — compute per 300 ms chunk.
- **RTF**: processing time / audio time. **CPU s per audio s**: process CPU seconds per second of
  audio — how many cores the link keeps busy in real time.
- **MT**: chrF and BLEU (sacrebleu) against the FLEURS reference, source = gold text (no ASR
  errors); latency per sentence.
- **TTS**: time to first audio (TTFA), RTF, CER of a Whisper-small back-transcription
  (intelligibility). Clone TTS adds speaker similarity (cosine of speaker embeddings, reference vs
  output; the same metric between two different real speakers is given as a baseline).
- **idle_loaded**: process CPU while sleeping right after an inference, with a model loaded —
  checks that runtime thread pools do not spin when nothing is happening.

## Results

### ASR

FLEURS dev, 30 utterances per language (315 s EN, 364 s RU). `fw:` = faster-whisper int8 (float16 on
GPU), greedy; `onnx:` = onnx-asr (Parakeet int8, GigaAM and T-one fp32). Whisper's `win2s` is its
floor because it pads to 30 s. T-one is an 8 kHz telephony model (input resampled).

| host | model | lang | wer | cer | utt_latency_ms_median | win2s_latency_ms | chunk_latency_ms | rtf | cpu_s_per_audio_s | peak_rss_mb | load_s |
|---|---|---|---|---|---|---|---|---|---|---|---|
| macos-15 | fw:large-v3-turbo | en | 0.0507 | 0.0202 | 12607 | 10433 | — | 1.202 | 3.27 | 1848 | 22.0 |
| macos-15 | fw:large-v3-turbo | ru | 0.0563 | 0.0223 | 10059 | 9419 | — | 0.832 | 2.329 | 1808 | 5.18 |
| macos-15 | fw:medium | en | 0.0627 | 0.0257 | 8838 | 8083 | — | 0.871 | 2.327 | 2157 | 21.68 |
| macos-15 | fw:medium | ru | 0.0613 | 0.0223 | 8528 | 6311 | — | 0.736 | 2.019 | 2046 | 5.64 |
| macos-15 | fw:small | en | 0.0701 | 0.0287 | 3516 | 3399 | — | 0.365 | 0.935 | 1430 | 7.0 |
| macos-15 | fw:small | ru | 0.0877 | 0.0319 | 3554 | 2675 | — | 0.295 | 0.787 | 905 | 1.42 |
| macos-15 | onnx:gigaam-v3-ctc | ru | 0.0778 | 0.0446 | 688 | 151 | — | 0.075 | 0.206 | 1624 | 7.67 |
| macos-15 | onnx:gigaam-v3-e2e-ctc | ru | 0.0563 | 0.0148 | 623 | 150 | — | 0.056 | 0.158 | 1741 | 5.94 |
| macos-15 | onnx:nemo-parakeet-tdt-0.6b-v3 | en | 0.0463 | 0.0194 | 342 | 106 | — | 0.037 | 0.106 | 1792 | 7.02 |
| macos-15 | onnx:nemo-parakeet-tdt-0.6b-v3 | ru | 0.0762 | 0.027 | 387 | 84 | — | 0.034 | 0.097 | 1375 | 2.37 |
| macos-15 | onnx:t-tech/t-one | ru | 0.1308 | 0.0547 | 827 | 180 | 18.8 | 0.076 | 0.213 | 708 | 4.97 |
| vast-a-cpu4 | fw:large-v3-turbo | en | 0.0478 | 0.0179 | 5550 | 5357 | — | 0.527 | 2.059 | 1661 | 2.12 |
| vast-a-cpu4 | fw:large-v3-turbo | ru | 0.0596 | 0.023 | 5723 | 5357 | — | 0.473 | 1.848 | 1661 | 2.15 |
| vast-a-cpu4 | fw:medium | en | 0.0612 | 0.0247 | 3697 | 3136 | — | 0.353 | 1.401 | 1592 | 1.98 |
| vast-a-cpu4 | fw:medium | ru | 0.0695 | 0.0263 | 4223 | 3194 | — | 0.356 | 1.411 | 1592 | 2.06 |
| vast-a-cpu4 | fw:small | en | 0.0687 | 0.0287 | 1284 | 1077 | — | 0.127 | 0.518 | 622 | 0.86 |
| vast-a-cpu4 | fw:small | ru | 0.0927 | 0.0315 | 1503 | 1040 | — | 0.124 | 0.504 | 621 | 0.95 |
| vast-a-cpu4 | onnx:gigaam-v3-ctc | ru | 0.0778 | 0.0446 | 382 | 92 | — | 0.032 | 0.147 | 1827 | 12.83 |
| vast-a-cpu4 | onnx:gigaam-v3-e2e-ctc | ru | 0.0563 | 0.0148 | 388 | 89 | — | 0.033 | 0.15 | 1831 | 15.07 |
| vast-a-cpu4 | onnx:nemo-parakeet-tdt-0.6b-v3 | en | 0.0478 | 0.0209 | 392 | 169 | — | 0.039 | 0.184 | 1446 | 2.32 |
| vast-a-cpu4 | onnx:nemo-parakeet-tdt-0.6b-v3 | ru | 0.0828 | 0.0268 | 444 | 114 | — | 0.037 | 0.172 | 1371 | 2.33 |
| vast-a-cpu4 | onnx:t-tech/t-one | ru | 0.1308 | 0.0547 | 548 | 116 | 12.3 | 0.045 | 0.192 | 705 | 4.9 |
| vast-a-rtx3060 | fw:large-v3-turbo | en | 0.0478 | 0.0192 | 298 | 256 | — | 0.028 | 0.049 | 1730 | 1.85 |
| vast-a-rtx3060 | fw:large-v3-turbo | ru | 0.0579 | 0.0225 | 323 | 252 | — | 0.027 | 0.045 | 1729 | 1.73 |
| vast-a-rtx3060 | fw:medium | en | 0.0642 | 0.0265 | 322 | 196 | — | 0.031 | 0.052 | 1644 | 1.61 |
| vast-a-rtx3060 | fw:medium | ru | 0.0712 | 0.0261 | 435 | 184 | — | 0.037 | 0.055 | 1644 | 1.61 |
| vast-a-rtx3060 | fw:small | en | 0.0701 | 0.0287 | 133 | 76 | — | 0.013 | 0.034 | 710 | 0.93 |
| vast-a-rtx3060 | fw:small | ru | 0.0861 | 0.0315 | 174 | 67 | — | 0.015 | 0.033 | 704 | 0.8 |

### MT

Gold FLEURS source sentences, references from the other language. OPUS-MT on `macos-15`
includes the one-time CTranslate2 conversion (torch loaded) in `peak_rss_mb`; the clean CPU
footprint is the `vast-a-cpu4` row (0.17 GB). HY-MT runs in `llama-server` (its RAM in
`server_peak_rss_mb`); on `macos-15` llama.cpp offloads to the runner's paravirtual Metal device,
which is far slower than a real M1 GPU.

| host | model | direction | chrf | bleu | latency_ms_median | latency_ms_p90 | cpu_cores | peak_rss_mb | server_peak_rss_mb | load_s |
|---|---|---|---|---|---|---|---|---|---|---|
| macos-15 | HY-MT1.5-1.8B-Q4_K_M | en-ru | 54.5 | 18.4 | 33432 | 54584 | 2.2 | 42 | 2242 | 44.1 |
| macos-15 | HY-MT1.5-1.8B-Q4_K_M | ru-en | 61.9 | 25.9 | 8563 | 12981 | 2.28 | 44 | 2341 | 3.8 |
| macos-15 | HY-MT1.5-7B-Q4_K_M | en-ru | 58.2 | 24.2 | 56241 | 131779 | 2.12 | 42 | 4489 | 72.2 |
| macos-15 | HY-MT1.5-7B-Q4_K_M | ru-en | 65.7 | 32.7 | 31619 | 45243 | 2.28 | 41 | 4743 | 69.4 |
| macos-15 | nllb-200-distilled-600M | en-ru | 56.5 | 27.8 | 1795 | 3151 | 2.78 | 1699 | — | 15.7 |
| macos-15 | nllb-200-distilled-600M | ru-en | 62.8 | 34.6 | 942 | 1771 | 2.81 | 1499 | — | 8.8 |
| macos-15 | opus-mt | en-ru | 58.8 | 30.8 | 226 | 349 | 2.84 | 1206 | — | 22.3 |
| macos-15 | opus-mt | ru-en | 60.7 | 33.6 | 112 | 198 | 2.9 | 1181 | — | 8.6 |
| vast-a-cpu4 | HY-MT1.5-1.8B-Q4_K_M | en-ru | 54.8 | 18.9 | 2063 | 3719 | 4.0 | 47 | 2105 | 1.5 |
| vast-a-cpu4 | HY-MT1.5-1.8B-Q4_K_M | ru-en | 62.2 | 25.8 | 1457 | 2025 | 4.0 | 46 | 2112 | 1.5 |
| vast-a-cpu4 | HY-MT1.5-7B-Q4_K_M | en-ru | 58.8 | 24.2 | 9743 | 15314 | 4.0 | 47 | 8228 | 4.1 |
| vast-a-cpu4 | HY-MT1.5-7B-Q4_K_M | ru-en | 65.7 | 32.8 | 6018 | 9361 | 4.0 | 46 | 8227 | 4.1 |
| vast-a-cpu4 | nllb-200-distilled-600M | en-ru | 57.7 | 29.5 | 753 | 1191 | 4.0 | 1636 | — | 15.4 |
| vast-a-cpu4 | nllb-200-distilled-600M | ru-en | 63.6 | 35.0 | 604 | 985 | 4.0 | 1599 | — | 3.3 |
| vast-a-cpu4 | opus-mt | en-ru | 59.5 | 32.5 | 67 | 105 | 4.0 | 173 | — | 0.3 |
| vast-a-cpu4 | opus-mt | ru-en | 59.9 | 32.5 | 57 | 91 | 4.0 | 170 | — | 0.4 |
| vast-a-rtx3060 | HY-MT1.5-1.8B-Q4_K_M | en-ru | 55.4 | 19.2 | 309 | 484 | 1.0 | 47 | 1387 | 1.5 |
| vast-a-rtx3060 | HY-MT1.5-1.8B-Q4_K_M | ru-en | 62.1 | 26.2 | 176 | 249 | 1.0 | 46 | 1387 | 1.0 |
| vast-a-rtx3060 | HY-MT1.5-7B-Q4_K_M | en-ru | 57.6 | 22.7 | 1125 | 1734 | 1.0 | 47 | 4737 | 2.7 |
| vast-a-rtx3060 | HY-MT1.5-7B-Q4_K_M | ru-en | 65.7 | 32.8 | 493 | 799 | 1.0 | 46 | 4737 | 2.7 |
| vast-a-rtx3060 | nllb-200-distilled-600M | en-ru | 56.6 | 28.0 | 137 | 212 | 1.0 | 1657 | — | 3.4 |
| vast-a-rtx3060 | nllb-200-distilled-600M | ru-en | 63.3 | 34.9 | 104 | 166 | 1.0 | 1656 | — | 3.6 |
| vast-a-rtx3060 | opus-mt | en-ru | 59.4 | 32.6 | 38 | 58 | 1.0 | 494 | — | 0.4 |
| vast-a-rtx3060 | opus-mt | ru-en | 60.6 | 32.7 | 35 | 49 | 1.0 | 493 | — | 0.5 |

Two sentences per direction by eye (x86 rows):

**en-ru** — source: *Soon, officers equipped with riot gear entered the yard and cornered the inmates with tear gas.*  
reference: *Вскоре надзиратели в защитном снаряжении прошли во двор и загнали заключенных в угол, применив слезоточивый газ.*

| model | translation |
|---|---|
| opus-mt | Вскоре полицейские, оснащенные снаряжением для подавления беспорядков, вошли во двор и загнали заключенных в угол слезоточивым газом. |
| nllb-200-distilled-600M | Вскоре на дворе вошли офицеры, обладающие оружием для борьбы с беспорядками, и загнали заключенных в угол слезоточивым газом. |
| HY-MT1.5-1.8B-Q4_K_M | Вскоре офицеры, вооруженные противопожарным оборудованием, вошли в двор и с помощью слезоточивого газа вынудили заключенных покинуть место. |
| HY-MT1.5-7B-Q4_K_M | Вскоре во двор вошли офицеры, вооружённые средствами для подавления беспорядков, и с помощью слезоточивого газа загнали заключённых в угол. |

**en-ru** — source: *This new environment has different resources and different competitors, so the new population will need different features or adaptations to be a strong competitor than what they had needed before.*  
reference: *У этой новой среды есть другие ресурсы и другие конкуренты, поэтому новой популяции будут необходимы другие характеристики или изменения, по сравнению с теми, которые у неё были раньше, чтобы стать сильным соперником.*

| model | translation |
|---|---|
| opus-mt | Эта новая среда имеет разные ресурсы и различные конкуренты, поэтому новое население нуждается в различных особенностях или адаптации, чтобы стать сильным конкурентом, чем это было раньше. |
| nllb-200-distilled-600M | Эта новая среда имеет разные ресурсы и разные конкуренты, поэтому новому населению понадобятся разные особенности или адаптации, чтобы быть сильным конкурентом, чем раньше. |
| HY-MT1.5-1.8B-Q4_K_M | В этом новом окружении существуют разные ресурсы и разные конкуренты. Поэтому новое население потребует наличия определенных особенностей или адаптаций, чтобы стать сильным конкурентом по сравнению с тем, что было необходимо ранее. |
| HY-MT1.5-7B-Q4_K_M | В этой новой среде имеются другие ресурсы и другие конкуренты, поэтому новым представителям этой группы потребуются иные характеристики или адаптации, чтобы стать сильными конкурентами — в отличие от того, что было необходимо ранее. |

**ru-en** — source: *Вскоре надзиратели в защитном снаряжении прошли во двор и загнали заключенных в угол, применив слезоточивый газ.*  
reference: *Soon, officers equipped with riot gear entered the yard and cornered the inmates with tear gas.*

| model | translation |
|---|---|
| opus-mt | The guards in protective equipment soon went into the yard and cornered the prisoners using tear gas. |
| nllb-200-distilled-600M | Soon, guards in protective gear entered the courtyard and drove the prisoners into the corner using tear gas. |
| HY-MT1.5-1.8B-Q4_K_M | Soon, the guards in protective gear entered the courtyard and forced the prisoners into a corner, using tear-inducing gas. |
| HY-MT1.5-7B-Q4_K_M | Soon after, guards in protective gear entered the yard and cornered the prisoners, using tear gas in the process. |

**ru-en** — source: *У этой новой среды есть другие ресурсы и другие конкуренты, поэтому новой популяции будут необходимы другие характеристики или изменения, по сравнению с теми, которые у неё были раньше, чтобы стать сильным соперником.*  
reference: *This new environment has different resources and different competitors, so the new population will need different features or adaptations to be a strong competitor than what they had needed before.*

| model | translation |
|---|---|
| opus-mt | This new environment has other resources and other competitors, so the new population will need different characteristics or changes than the ones it had before to become a strong rival. |
| nllb-200-distilled-600M | This new environment has other resources and other competitors, so a new population will need different characteristics or changes from those it had before to become a strong competitor. |
| HY-MT1.5-1.8B-Q4_K_M | In this new environment, there are other resources and competitors as well. Therefore, the new population will need other characteristics or changes compared to those it had before, in order to become a strong competitor. |
| HY-MT1.5-7B-Q4_K_M | In this new environment, there are different resources and different competitors, so the new population will need to possess different characteristics or undergo different changes compared to before in order to become a strong competitor. |

### TTS without cloning

10 shortest sentences per language. `cer` = Whisper-small back-transcription vs the input text.
Piper (GPL) runs as its own process through the upstream CLI, never imported; that CLI has no
thread option and its onnxruntime pool takes one thread per physical core (`taskset` does not
bound it), so `threads` for Piper rows is the host's core count. The chosen voices (`john`,
`joe`, `dmitri`; `denis` rejected for intelligibility) were measured on the dev server
(`linux-x86_64-4c`, 4 cores, shared): one run there under load gave 380–470 ms for
`irina`/`dmitri`, the rerun 231–267 ms is kept. On that host the chosen voices are within ~30 % of
`lessac`/`irina`, which are the Piper voices measured on `macos-15` and `vast-a-cpu4` (those rows
come from an earlier in-process version of the script).

| host | model | lang | voice | ttfa_ms_median | rtf | cpu_cores | peak_rss_mb | cer | load_s |
|---|---|---|---|---|---|---|---|---|---|
| linux-x86_64-4c | piper | en | en_US-joe-medium | 208 | 0.045 | 3.72 | 295 | 0.0303 | 1.9 |
| linux-x86_64-4c | piper | en | en_US-john-medium | 269 | 0.051 | 3.36 | 370 | 0.0182 | 4.3 |
| linux-x86_64-4c | piper | en | en_US-lessac-medium | 211 | 0.048 | 3.58 | 293 | 0.0145 | 4.1 |
| linux-x86_64-4c | piper | ru | ru_RU-denis-medium | 212 | 0.06 | 3.26 | 324 | 0.1029 | 3.9 |
| linux-x86_64-4c | piper | ru | ru_RU-dmitri-medium | 231 | 0.061 | 3.24 | 317 | 0.0593 | 1.9 |
| linux-x86_64-4c | piper | ru | ru_RU-irina-medium | 267 | 0.04 | 3.79 | 356 | 0.047 | 1.9 |
| macos-15 | piper | en | en_US-lessac-medium | 232 | 0.057 | 2.95 | 476 | 0.0097 | 5.8 |
| macos-15 | piper | ru | ru_RU-irina-medium | 265 | 0.043 | 2.95 | 487 | 0.0526 | 5.3 |
| macos-15 | silero | en | v3_en/en_0 | 410 | 0.102 | 1.85 | 1322 | 0.0218 | 7.7 |
| macos-15 | silero | ru | v5_6_ru/xenia | 252 | 0.048 | 1.54 | 983 | 0.0481 | 11.3 |
| vast-a-cpu4 | piper | en | en_US-lessac-medium | 125 | 0.027 | 3.95 | 384 | 0.0157 | 4.1 |
| vast-a-cpu4 | piper | ru | ru_RU-irina-medium | 172 | 0.027 | 3.98 | 425 | 0.047 | 5.0 |
| vast-a-cpu4 | silero | en | v3_en/en_0 | 201 | 0.051 | 2.98 | 1045 | 0.0218 | 6.1 |
| vast-a-cpu4 | silero | ru | v5_6_ru/xenia | 224 | 0.045 | 0.98 | 1166 | 0.0649 | 13.8 |

### TTS with voice cloning

6 sentences per output language (4 on `macos-15`), ~40 s references. `spk_sim` = cosine of
SpeechBrain ECAPA embeddings, reference vs output (mean; `spk_sim_min` the worst sentence);
baselines: two different real speakers 0.085, two halves of one speaker 0.90–0.93; Whisper-small
CER on the real FLEURS recordings 0.013–0.022. XTTS and Chatterbox stream / synthesize per
sentence (`ttfa_s` = full synthesis when not streaming); CosyVoice3 on CPU ran with streaming off
(its streaming is ~3× slower on CPU). Chatterbox uses `cfg_weight=0` for cross-lingual cases;
CosyVoice3 gets a ≤ 30 s prompt (its limit). The `linux-x86_64-4c` rows ran while that server was
swapping (pessimistic). XTTS often appends a garbage tail after the sentence — most of its CER.
Details: [`bench/notes/tts_clone.md`](../bench/notes/tts_clone.md).

| host | model | case | ref_lang | out_lang | device | stream | ttfa_s | ttfa_max_s | rtf | peak_rss_mb | gpu_peak_mb | spk_sim | spk_sim_min | cer | load_s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| linux-x86_64-4c | xtts | — | en | en | cpu | — | 3.478 | 4.26 | 4.342 | 3849 | — | 0.605 | 0.564 | 0.1305 | 208.23 |
| linux-x86_64-4c | xtts | — | en | ru | cpu | — | 3.546 | 5.358 | 4.176 | 3849 | — | 0.385 | 0.303 | 0.0594 | 208.23 |
| linux-x86_64-4c | xtts | — | ru | en | cpu | — | 3.102 | 3.579 | 3.671 | 3849 | — | 0.421 | 0.363 | 0.012 | 208.23 |
| linux-x86_64-4c | xtts | — | ru | ru | cpu | — | 3.342 | 4.057 | 4.034 | 3849 | — | 0.68 | 0.533 | 0.0764 | 208.23 |
| macos-15 | xtts | — | en | en | cpu | True | 5.275 | 6.037 | 8.202 | 3585 | — | 0.632 | 0.563 | 0.0199 | 66.45 |
| macos-15 | xtts | — | en | ru | cpu | True | 5.217 | 5.831 | 7.673 | 3585 | — | 0.352 | 0.281 | 0.0712 | 66.45 |
| macos-15 | xtts | — | ru | en | cpu | True | 5.274 | 5.721 | 6.869 | 3585 | — | 0.44 | 0.427 | 0.0228 | 66.45 |
| macos-15 | xtts | — | ru | ru | cpu | True | 4.453 | 4.953 | 6.519 | 3585 | — | 0.682 | 0.633 | 0.1085 | 66.45 |
| vast-c-a4000 | chatterbox | — | en | en | cuda | — | 2.152 | 3.093 | 0.404 | 5042 | 3704 | 0.692 | 0.645 | 0.008 | 8.46 |
| vast-c-a4000 | chatterbox | — | en | ru | cuda | — | 2.684 | 3.13 | 0.395 | 5042 | 3704 | 0.553 | 0.509 | 0.0382 | 8.46 |
| vast-c-a4000 | chatterbox | — | ru | en | cuda | — | 2.506 | 3.162 | 0.402 | 5042 | 3704 | 0.695 | 0.612 | 0.0683 | 8.46 |
| vast-c-a4000 | chatterbox | — | ru | ru | cuda | — | 2.509 | 2.852 | 0.423 | 5042 | 3704 | 0.759 | 0.627 | 0.0191 | 8.46 |
| vast-c-a4000 | cosyvoice | — | en | en | cuda | True | 2.206 | 2.259 | 1.221 | 8424 | 3891 | 0.721 | 0.663 | 0.0341 | 98.29 |
| vast-c-a4000 | cosyvoice | — | en | ru | cuda | True | 2.234 | 2.281 | 1.204 | 8424 | 3891 | 0.502 | 0.412 | 0.0361 | 98.29 |
| vast-c-a4000 | cosyvoice | — | ru | en | cuda | True | 3.057 | 3.091 | 1.831 | 8424 | 3891 | 0.589 | 0.532 | 0.008 | 98.29 |
| vast-c-a4000 | cosyvoice | — | ru | ru | cuda | True | 3.328 | 3.376 | 2.015 | 8424 | 3891 | 0.783 | 0.76 | 0.2781 | 98.29 |
| vast-c-a4000 | xtts | — | en | en | cuda | — | 0.144 | 0.148 | 0.179 | 4499 | 2070 | 0.557 | 0.454 | 0.0622 | 7.55 |
| vast-c-a4000 | xtts | — | en | ru | cuda | — | 0.143 | 0.144 | 0.176 | 4499 | 2070 | 0.38 | 0.322 | 0.0488 | 7.55 |
| vast-c-a4000 | xtts | — | ru | en | cuda | — | 0.144 | 0.148 | 0.172 | 4499 | 2070 | 0.396 | 0.278 | 0.0141 | 7.55 |
| vast-c-a4000 | xtts | — | ru | ru | cuda | — | 0.144 | 0.145 | 0.176 | 4499 | 2070 | 0.657 | 0.543 | 0.0616 | 7.55 |
| vast-c-cpu8 | baseline | diff_speakers_ref_en_vs_ref_ru | — | — | — | — | — | — | — | 1817 | — | 0.085 | — | — | — |
| vast-c-cpu8 | baseline | real_fleurs_en | — | en | — | — | — | — | — | 1817 | — | — | — | 0.0221 | — |
| vast-c-cpu8 | baseline | real_fleurs_ru | — | ru | — | — | — | — | — | 1817 | — | — | — | 0.0127 | — |
| vast-c-cpu8 | baseline | same_speaker_ref_en_halves | — | — | — | — | — | — | — | 1817 | — | 0.904 | — | — | — |
| vast-c-cpu8 | baseline | same_speaker_ref_ru_halves | — | — | — | — | — | — | — | 1817 | — | 0.93 | — | — | — |
| vast-c-cpu8 | chatterbox | — | en | en | cpu | — | 18.187 | 21.635 | 3.516 | 6990 | — | 0.707 | 0.67 | 0.0181 | 7.78 |
| vast-c-cpu8 | chatterbox | — | en | ru | cpu | — | 22.689 | 25.897 | 3.288 | 6990 | — | 0.518 | 0.483 | 0.0573 | 7.78 |
| vast-c-cpu8 | chatterbox | — | ru | en | cpu | — | 19.489 | 23.463 | 3.497 | 6990 | — | 0.681 | 0.643 | 0.0382 | 7.78 |
| vast-c-cpu8 | chatterbox | — | ru | ru | cpu | — | 19.764 | 23.601 | 3.504 | 6990 | — | 0.778 | 0.663 | 0.0191 | 7.78 |
| vast-c-cpu8 | cosyvoice | — | en | en | cpu | False | 35.176 | 41.969 | 6.817 | 8841 | — | 0.731 | 0.663 | 0.0422 | 62.66 |
| vast-c-cpu8 | cosyvoice | — | en | ru | cpu | False | 35.113 | 36.751 | 6.4 | 8841 | — | 0.433 | 0.335 | 0.0361 | 62.66 |
| vast-c-cpu8 | cosyvoice | — | ru | en | cpu | False | 49.389 | 54.045 | 8.352 | 8841 | — | 0.571 | 0.493 | 0.0221 | 62.66 |
| vast-c-cpu8 | cosyvoice | — | ru | ru | cpu | False | 46.5 | 52.09 | 11.196 | 8841 | — | 0.676 | 0.059 | 0.2251 | 62.66 |
| vast-c-cpu8 | xtts | — | en | en | cpu | — | 1.264 | 1.588 | 1.894 | 4406 | — | 0.602 | 0.489 | 0.1586 | 9.79 |
| vast-c-cpu8 | xtts | — | en | ru | cpu | — | 1.303 | 1.605 | 1.858 | 4406 | — | 0.353 | 0.202 | 0.0616 | 9.79 |
| vast-c-cpu8 | xtts | — | ru | en | cpu | — | 1.169 | 1.522 | 1.747 | 4406 | — | 0.44 | 0.338 | 0.006 | 9.79 |
| vast-c-cpu8 | xtts | — | ru | ru | cpu | — | 1.407 | 1.648 | 1.806 | 4406 | — | 0.677 | 0.618 | 0.0701 | 9.79 |

### Stage 1: the clone engine and the "me → them" path

Measured 2026-11-07 for stage 1. Hosts: `rtx3060-laptop` = a GPU host with an NVIDIA RTX 3060
Laptop 12 GB (shared with other work: CUDA memory cap 0.6, 2 CPU threads); `ryzen-7950x3d-8t` /
`ryzen-7950x3d-2t` = the same host's AMD Ryzen 9 7950X3D with 8 / 2 threads, CUDA hidden;
`linux-x86_64-4c` = the 4-core dev server (PipeWire, pipeline, virtual mic), which was also
running other heavy work, so its pipeline times are pessimistic. Same references, sentences and
judge as the table above.

**The engine as the app runs it** (`peresmeshnik.local.clone.XttsClone`: profile from
`peresmeshnik.voice`, streaming, text split, pause cap, length guard; `bench/tts_clone.py --model
xtts-engine`; the CPU rows ran before the pause cap, which does not change speed). Mean of the
four ref → out cases; `cpu s / audio s` = process CPU per second of speech made.

| host | sentences / lang | TTFA s (max) | RTF | CPU s / audio s | peak RSS MB | GPU MB | load s | conditioning s | spk-sim | CER |
|---|---|---|---|---|---|---|---|---|---|---|
| rtx3060-laptop (final: with the pause cap) | 15 | 0.18 (0.20) | 0.23–0.25 | 0.23–0.25 | 4539 | 2133 | 7.9 | 0.05–0.17 | 0.541 | 0.029 |
| ryzen-7950x3d-8t | 6 | 1.10 (1.14) | 1.35–1.51 | 10.8–11.9 | 4366 | — | 7.6 | 0.2–0.35 | 0.517 | 0.088 |
| ryzen-7950x3d-2t | 2 | 1.44 (1.48) | 1.87–1.97 | 3.7–3.9 | 4502 | — | 7.6 | 0.53–0.56 | 0.503 | 0.051 |

- On the GPU one CPU core stays busy while it speaks (CPU s / audio s ≈ RTF): the autoregressive
  GPT step is bound by kernel launches. Load 8–10 s, GPU memory 2.0–2.2 GB.
- On CPU it is **slower than real time** even on 8 threads of a desktop Ryzen (RTF 1.35–1.5); with
  the app's default 2 threads RTF 1.9. Per-case values are in `bench/results/{rtx3060-laptop,ryzen-7950x3d-*}.jsonl`.
- Same-language similarity 0.69–0.71 (ru), 0.64–0.65 (en); cross-lingual 0.37–0.44: as on the A4000.

**Sampling vs the babble tail** (GPU, 15 sentences per output language, before the pause cap):
temperature and top_p do not remove the tail; the differences are within the noise of one
babbling sentence. Upstream defaults kept (0.75 / 0.85).

| temperature / top_p | CER, English out | CER, Russian out | spk-sim |
|---|---|---|---|
| 0.75 / 0.85 | 0.006 | 0.059 | 0.540 |
| 0.6 / 0.85 | 0.031 | 0.043 | 0.547 |
| 0.6 / 0.7 | 0.069 | 0.035 | 0.537 |
| 0.45 / 0.85 | 0.021 | 0.036 | 0.532 |

**Pause cap and length guard.** In two runs of 60 sentences XTTS left 26 and 36 pauses ≥ 0.5 s
inside a sentence (up to 2.5 s; 5–7 % of all audio). The engine cuts silences (20 ms frames below
−40 dBFS) to 0.4 s while streaming. The length guard stops a sentence past `1 s + spoken chars / 8`
(digits count as 5 chars, abbreviations letter by letter; real speech runs 12–20 chars/s): a
safety net against runaway generation (XTTS may go on for up to ~26 s), it did not fire on any of
the 344 outputs here. It cannot catch the usual babble: those 3–4 s of invented words after the
text are no longer than a slow sentence with a pause. A simulation over the 344 outputs showed
that a budget tight enough to cut 6 of them (`/ 11` before the pause cap) also cuts the end of
slow sentences, and after the pause cap it cut only one sentence, wrongly. One-word tails after
the last word (5–8 of 60 sentences, mostly Russian output) stay as well; stage 5 can try an
attention-based stop.

| same sentences | audio s | CER, English out | CER, Russian out | mean CER | spk-sim |
|---|---|---|---|---|---|
| raw XTTS (t0.75), 60 wavs | 418.3 | 0.006 | 0.059 | 0.033 | 0.540 |
| the same 60 wavs, pause cap applied offline | 405.3 | 0.013 | 0.044 | 0.028 | 0.541 |
| fresh streaming run with the pause cap | 406.9 | 0.012 | 0.045 | 0.029 | 0.541 |

The CER changes are judge effects more than speech: with less trailing silence Whisper stops
inventing its known Russian end-of-video credit line on one sentence (−), and on one English
sentence starts transcribing a babble that the cap moved closer to the speech (+).

**Enrollment checks** (`python -m peresmeshnik.voice check`): LibriSpeech reader 44.5 s, speech
34.4 s, −22.8 dBFS, 45.6 dB above the background → OK. LibriVox reader (the "me" voice) 42.4 s,
speech 20.3 s, −25.6 dBFS, 14.4 dB above the background → OK with a noise warning (fail below
12 dB). The profile from it gives the same-language similarity above, so the thresholds do not
reject a usable home recording. Recording through PortAudio → PipeWire (`enroll --record 35`)
works on the dev server.

**Virtual microphone** (`peresmeshnik.audio.virtual_mic_linux`, PipeWire 1.0.5 + pipewire-pulse):
a tone round trip keeps frequency, level and length; play → audio at the `Peresmeshnik Mic`
source as a call app records it: **10–61 ms** (the first chunk after the stream opens 38–61 ms,
then 10–37 ms); CPU of the output path 0.01–0.017 s per second of audio (our process + pacat).
`pacat` itself spins a core if its stdin closes while it still holds audio (0.64 s CPU for a 0.7 s
tail), so the writer closes it only after the written audio has played. The stream is opened with
`node.dont-reconnect`: PipeWire otherwise moves it to a newer device, such as the call capture sink.

**End to end** (`bench/me_them.py run --remote`, re-run 2026-11-08 on the merged local pipeline):
33 s of the "me" reader in Russian (other lines than the enrollment) → `LocalDirection` with the
app's defaults (Silero VAD, GigaAM v3 for Russian, OPUS-MT ru-en, 2 threads, on the dev server)
and the clone TTS on the GPU host behind a pipe (`me_them.py serve`) → `Peresmeshnik Mic` →
recorded back from the source → Whisper-small. Per clause, from the end of its speech: `caption`
(ASR + MT), `clone audio` = first clone audio here (`synth_lag_s`: caption + clone first audio incl.
~45 ms network), `queue` = waits behind the previous translation still playing (`play_queue_s`),
`call app hears` = sound onset in the recording (`call_lag_s`); `mic link` = onset − when the audio
was due to play.

| clone on | runs × clauses | caption s | clone audio s | queue s | call app hears s, median (range) | mic link ms | WER, recording (direct TTS) | spk-sim, recording vs "me" (vs another speaker) |
|---|---|---|---|---|---|---|---|---|
| RTX 3060 Laptop | 2 × 9 | 0.34–2.9 | 0.55–3.1 | 0–6.1 | 4.4 (1.0–6.7) | 35–67 | 0.012 (0–0.012) | 0.44–0.45 (−0.06..−0.07) |
| 8 CPU threads | 1 × 9 | 0.34–2.9 | 2.0–11.6 | 0 | 8.9 (2.1–11.7) | 21–55 | 0.035 (0.012) | 0.46 (−0.06) |

- The path is lossless: the recording from the virtual mic transcribes like the TTS output itself
  (the CPU run's difference is Whisper on near-homophones: silence/silent, sweats/sweat's); the
  voice in it is the clone (0.44–0.46 vs the profile, the cross-lingual level of XTTS; the real
  Russian input 0.88).
- With the GPU clone the first translated audio is ready **0.20–0.25 s after the caption** and is
  at the call app's mic ~50 ms after it is due. What adds up is the queue: the English speech
  (29 s) nearly fills the 33 s of input, so when clauses come in bursts the next one waits up to
  6 s for the previous one to finish playing. That is the stage-2 queue policy (speed-up, then
  drop), not a link that can be made faster.
- Captions no longer wait for the voice: with the CPU clone (RTF 1.3) they stay at 0.34–2.9 s,
  as with the GPU, while the voice falls behind through the run (up to 11.6 s). The earlier run on
  a pre-merge pipeline (Parakeet for Russian, captions put out by the speak worker) had captions
  at 2.2–5.1 s with the GPU clone and growing to 16 s with the CPU one.
- Clauses cut at 5 s (`max`) get their caption at 1.3–2.9 s, clauses closed by a pause at
  0.34–0.97 s; the dev server was lightly loaded this time (load average ~1).

### Cloud: Gemini Live

Measured from the dev server (Europe), Saturday 20:14–20:45 UTC, three runs per cell for
`utterance`. Audio streamed in 100 ms chunks at real-time pace. Times are seconds, `median / p90`:
`first_out_from_start_s` = first translated audio (translate) or first text (transcribe) after
the speech starts; `done_after_end_s` = translated audio done (translate) or final text
(transcribe) after the speech ends. Modes: `utterance` — one sentence then silence;
`stream_end` — the client sends `audio_stream_end` 0.3 s after the speech ends (local VAD);
`continuous` — five sentences back to back (41–62 s). For transcribe, `direction` names the
spoken language first. `chrf_out_transcript` uses the model's own output transcript,
`chrf_back_whisper` a Whisper-small transcription of the output audio;
`voice_f0_in_out_corr` = correlation of the speaker's and the output voice's pitch across
speakers. Details: [`bench/notes/cloud_gemini.md`](../bench/notes/cloud_gemini.md).

| model | direction | mode | n | setup_s | first_out_from_start_s | done_after_end_s | wer | cer | chrf_out_transcript | chrf_back_whisper | voice_f0_in_out_corr |
|---|---|---|---|---|---|---|---|---|---|---|---|
| gemini-3.5-live-translate-preview | en-ru | continuous | 2 | 0.256 / 0.27 | 3.005 / 3.01 | 6.221 / 6.338 | — | — | 62.5 | 61.2 | — |
| gemini-3.5-live-translate-preview | en-ru | stream_end | 5 | 0.266 / 0.384 | 3.324 / 3.407 | 2.904 / 3.803 | — | — | 52.3 | 43.3 | 0.99 |
| gemini-3.5-live-translate-preview | en-ru | utterance | 30 | 0.24 / 0.326 | 3.2 / 3.417 | 3.66 / 4.699 | — | — | 61.4 | 58.4 | 0.95 |
| gemini-3.5-live-translate-preview | ru-en | continuous | 1 | 0.224 / 0.224 | 3.222 / 3.222 | 3.421 / 3.421 | — | — | 62.3 | 62.2 | — |
| gemini-3.5-live-translate-preview | ru-en | stream_end | 5 | 0.217 / 0.277 | 3.176 / 3.397 | 2.873 / 3.285 | — | — | 57.3 | 57.1 | 0.99 |
| gemini-3.5-live-translate-preview | ru-en | utterance | 30 | 0.246 / 0.283 | 3.12 / 3.272 | 3.146 / 4.045 | — | — | 59.7 | 59.9 | 0.92 |
| gemini-3.5-transcribe-live | en-ru | continuous | 1 | 0.261 / 0.261 | 0.571 / 0.571 | 3.328 / 3.328 | 0.0175 | 0.0015 | — | — | — |
| gemini-3.5-transcribe-live | en-ru | stream_end | 10 | 0.223 / 0.382 | 0.97 / 1.239 | 0.653 / 0.929 | 0.0244 | 0.0051 | — | — | — |
| gemini-3.5-transcribe-live | en-ru | utterance | 30 | 0.25 / 0.293 | 1.002 / 1.273 | 1.562 / 1.884 | 0.0146 | 0.0017 | — | — | — |
| gemini-3.5-transcribe-live | ru-en | continuous | 1 | 0.275 / 0.275 | 1.163 / 1.163 | 1.724 / 1.724 | 0.009 | 0.0083 | — | — | — |
| gemini-3.5-transcribe-live | ru-en | stream_end | 10 | 0.247 / 0.38 | 0.992 / 1.128 | 0.684 / 0.766 | 0.0255 | 0.022 | — | — | — |
| gemini-3.5-transcribe-live | ru-en | utterance | 30 | 0.246 / 0.278 | 1.028 / 1.341 | 1.698 / 2.385 | 0.0153 | 0.0083 | — | — | — |

### Idle CPU with a model loaded

Process CPU seconds during 20 s of sleep right after an inference (4 threads). Default pools
spin briefly after each call; with the flags they sleep at once.

| host | model | threads | no_spin | omp_wait_policy | busy_cpu_s | busy_wall_s | idle_s | idle_cpu_s |
|---|---|---|---|---|---|---|---|---|
| vast-a-cpu4 | ct2 | 4 | False |  | 4.359 | 1.063 | 20 | 0.0183 |
| vast-a-cpu4 | ct2 | 4 | False | PASSIVE | 4.136 | 1.117 | 20 | 0.0005 |
| vast-a-cpu4 | ort | 4 | False |  | 0.884 | 0.172 | 20 | 0.1204 |
| vast-a-cpu4 | ort | 4 | True |  | 0.846 | 0.17 | 20 | 0.0129 |
| vast-a-cpu4 | torch | 4 | False |  | 0.48 | 0.12 | 20 | 0.0187 |
| vast-a-cpu4 | torch | 4 | False | PASSIVE | 0.314 | 0.122 | 20 | 0.0 |


## Stage 1: cloud direction (`GeminiDirection`)

The app's own cloud path (`peresmeshnik cloud`, `peresmeshnik/cloud/gemini.py`), measured on
`linux-x86_64-4c` (Europe), Saturday 2026-11-07 22:36–23:30 UTC, free-tier key. Input: FLEURS dev
sentences glued with pauses and streamed at real-time pace in 20 ms frames (the direction re-blocks
to 100 ms); `--subtitles` on, so every direction holds two sessions (translate + transcribe).
Times are seconds from the per-link latency log (`peresmeshnik.latency`), `median / p90 (n)`.
Speech start / pause / end come from the direction's own energy gate (end = 1 s of silence, pause =
300 ms), so a start can sit up to ~0.3 s before the first word (FLEURS lead-in noise); this
inflates the p90 of `first_partial_after_start`.

| Link (log name) | en → ru | ru → en | stage 0 bench |
|---|---|---|---|
| connect, translate / transcribe (`*.connect`) | 0.61 / 0.27 | 0.44 / 0.40 | 0.22–0.27 |
| subtitles: first word after speech start (`transcribe.first_partial_after_start`) | 1.24 / 2.70 (12) | 1.54 / 3.60 (12) | 1.0 / 1.3 |
| subtitles: final text after a 300 ms pause, client `audio_stream_end` (`transcribe.final_after_pause`) | 0.58 / 0.62 (12) | 0.64 / 0.71 (12) | 0.65–0.68 / 0.77–0.93 |
| voice: first translated audio after the start of a turn (`translate.first_audio_after_start`) | 3.08 / 3.88 (7) | 3.86 / 4.61 (10) | 3.1–3.2 / 3.3–3.4 |
| voice: translated speech done after the speaker stops (`translate.done_after_end`) | 3.55 / 4.98 (10) | 2.93 / 3.26 (10) | 3.1–3.7 / 4.0–4.7 |

Set: 10 sentences per language (the 11th–20th of the 30-sentence bench set), 3 s pauses, 160 s (en) /
171 s (ru) of audio. `first_audio_after_start` is counted only for turns that start while the
translator is quiet (≥ 1 s): with 3 s pauses the Russian translation of an English sentence often
still plays when the next sentence starts, hence n = 7.

**CPU and memory** (`/usr/bin/time`, both sessions of one direction in one process):
5.7 / 6.2 CPU s for 160 / 171 s of audio including 0.8 s of process start and imports →
**0.03 CPU s per audio second** (3 % of one core), peak RSS 92 MB. Idle cost is zero: no
`GeminiDirection` exists while translation is off.

**Through PipeWire devices** (own PipeWire session, null sink `peresmeshnik_call` → its monitor →
`parec --latency-msec=20` → `peresmeshnik cloud --in - --out -` → `pacat` → null sink `s3_out` →
its monitor recorded): 3 sentences per language with 2 s pauses. Voiced output on the recording
21.0 s (en → ru) / 29.8 s (ru → en), the same as the direct wav output (20.9 / 30.0 s); local
back-transcription of the recording with Parakeet v3 gives the full translation, e.g. "Вскоре
офицеры с защитным снаряжением вошли во двор и загнали заключенных в угол слезоточивым газом." /
"The new environment has other resources and other competitors, therefore the new population
will need other characteristics…". Latency the same as from the file (final after pause 0.59–0.60 s,
done after end 2.9–3.7 s).

**Long sessions** (all 30 English sentences ×3 with 2 s pauses, 1125 s = 18.8 min, en → ru):

| run | reconnects | GoAway | done_after_end | final_after_pause | notes |
|---|---|---|---|---|---|
| `--reconnect-s 540` | 2 per model, at 540.3–541.4 s | — | 3.46 (54) | 0.59 (64) | translation continues past 15 min (resumption handle + context compression) |
| no timer (×2 runs) | 2 per model, at 540.3–549.4 s | sent by the server at 540 s of each connection, `time_left` 50 s; the hand-over took 0.35–0.45 s | 3.40 (54) | 0.59 (64) | at the 540 s hand-over 8 connections were open at once (2 runs × 2 models × old + new): transcribe got `1011 quota exceeded` 3 times, retried and connected after 8 s while the old connection kept working — no audio lost |
| no timer, final code | 2 per model, at 540.4 s | GoAway at 540 s of each connection, `time_left` 50 s | 3.36 / 4.56 (79) | 0.59 (99) | translation to the last sentence; 36.2 CPU s for 1125 s of audio = 0.032 per audio second; peak RSS 200 MB, most of it the CLI keeping 872 s of output audio for the wav |

Hand-over stress (the same 10 English sentences, `--reconnect-s 12`): 9 transcribe and 5 translate
hand-overs in 160 s; 42 final subtitle lines as without hand-overs, translated speech 98.9 s vs
97.1 s, final after pause 0.58 / 0.64 s, done after end 3.26 / 4.89 s — nothing lost at the switches
(the replaced transcriber gets `audio_stream_end`, so its last segment is finalised before it closes).

Seen in the 10-sentence runs: a transcribe session went silent on the server for 84 s and then
closed with `1011 internal error`; the direction now replaces a session that leaves ≥ 2 s of
speech unanswered for 6 s (translate: 10 s). The audio sent to the dead session in that window
is lost for subtitles. Music or steady noise that passes the gate gets no answer either, so the wait
before such a reconnect doubles each time (6 s → 120 s) until the session answers again.

## Stage 1: the local pipeline (them → me)

What runs is `peresmeshnik run` / `LocalDirection` as in the app: Silero VAD (1 thread), Parakeet
TDT 0.6B v3 int8 with partials every 300 ms on the growing clause, OPUS-MT CTranslate2 int8, Piper
(`john` / `dmitri`) in its own process; 2 threads per engine, BLAS pinned to one thread. A clause
ends at a VAD pause ≥ 300 ms, at sentence punctuation with a capitalised word after it, or past
`max_clause_s` (then at a comma in the second half or between words, at least 0.8 s before the
window end); after a cut the window keeps `context_s` of the previous clause in front. Scripts:
[`bench/pipeline.py`](../bench/pipeline.py), gate: [`tests/test_pipeline_cpu.py`](../tests/test_pipeline_cpu.py).
Measured 2026-11-07.

### Cost on a 30 s recording

`tests/data/call-{en,ru}.ogg` (FLEURS, −26 dBFS) fed as fast as the pipeline takes it. `rtf` =
wall time / audio; `cpu_s_per_audio_s` = CPU of the app process plus the Piper process tree per
second of audio; `*_rtf` = busy time of one link per second of audio (`vad` every frame, ASR
`partial` re-runs, ASR `final` runs, MT, TTS). Defaults: `max_clause_s` 5, `context_s` 1.

| host | model | lang | mode | clauses | rtf | cpu_s_per_audio_s | vad_rtf | asr_partial_rtf | asr_final_rtf | mt_rtf | tts_rtf | peak_rss_mb |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| vast-b-pipeline | nemo-parakeet-tdt-0.6b-v3+opus-mt+piper | en | work | 7 | 0.296 | 0.605 | 0.0025 | 0.2914 | 0.0 | 0.0078 | 0.0234 | 1503 |
| vast-b-pipeline | nemo-parakeet-tdt-0.6b-v3+opus-mt+piper | ru | work | 7 | 0.293 | 0.588 | 0.0026 | 0.2861 | 0.0 | 0.0073 | 0.0197 | 1476 |

The working-mode gate in CI (`work-gate` job, run 42455602206 on 4f9e681, English recording;
limits RTF < 1.0 and < 2.0 CPU s per audio s) and the dev box:

| host | rtf | cpu_s_per_audio_s | vad_rtf | asr_partial_rtf | asr_final_rtf | mt_rtf | tts_rtf | word recall |
|---|---|---|---|---|---|---|---|---|
| ubuntu-latest | 0.27 | 0.551 | 0.0028 | 0.2633 | 0.0 | 0.0148 | 0.0274 | 0.84 |
| windows-latest | 0.535 | 1.088 | 0.0056 | 0.5234 | 0.0 | 0.023 | 0.054 | 0.84 |
| macos-15 (M1, 3 cores) | 0.423 | 0.781 | 0.0055 | 0.4113 | 0.0 | 0.0238 | 0.0608 | 0.84 |
| linux-x86_64-4c (shared) | 0.683 | 1.167 | 0.0059 | 0.6689 | 0.0 | 0.0242 | 0.0568 | 0.84 |

In the same CI run the idle gate stayed at 0.000 s CPU in 60 s on all three OSes. The run before
(42425353588, the code before the move to the model registry) gave RTF 0.54–0.67 and 0.96–1.32 CPU
s per audio s. Both runs used the same model files (same SHA-256) and the same 2 threads per
engine, so the 2× (ubuntu 0.57 → 0.27) is the runner, not the code: one runner label covers
different machines. The gate now prints the CPU model and core count next to its numbers; its
margin rests on one run per OS so far.

- ASR partials are ~90 % of the work: re-running Parakeet every 300 ms on a window of up to 6 s
  (5 s clause + 1 s context) keeps 0.55–1.2 cores busy while someone speaks (0.6 on the 7950X3D).
  Everything else together is < 0.1 of real time; VAD costs 0.3–1 % of a core.
- `asr_final_rtf` ≈ 0: at a pause the last partial already saw the whole clause and is reused;
  a cut clause takes its text from the partial's token timestamps.
- Peak RSS of the app process 1.5–1.7 GB (Parakeet); Piper's process adds 0.3–0.4 GB (TTS table).
- When the pipeline falls more than 150 ms behind real time (a busy machine), partials are
  skipped until it catches up; clause ends are always recognised.

### Quality: streamed vs offline

The 30 FLEURS utterances per language of the ASR / MT tables, each fed alone (scaled to −26 dBFS,
TTS off). `wer` = the joined clause texts vs the reference; `wer_offline` = the same audio as one
window; `chrf` = the joined clause translations vs the FLEURS reference translation. For
comparison, MT from the gold source text: chrF 59.5 en-ru, 59.9 ru-en (MT table).
`max_clause_s` 1000 = clauses end at pauses and punctuation only.

| host | model | lang | mode | max_clause_s | context_s | clauses | wer | wer_offline | chrf | peak_rss_mb |
|---|---|---|---|---|---|---|---|---|---|---|
| vast-b-pipeline | nemo-parakeet-tdt-0.6b-v3+opus-mt | en | quality | 5.0 | 1.0 | 85 | 0.1119 | 0.0537 | 52.5 | 1685 |
| vast-b-pipeline | nemo-parakeet-tdt-0.6b-v3+opus-mt | en | quality | 8.0 | 1.0 | 66 | 0.1 | 0.0537 | 53.9 | 1729 |
| vast-b-pipeline | nemo-parakeet-tdt-0.6b-v3+opus-mt | en | quality | 5.0 | 0.0 | 83 | 0.1567 | 0.0537 | 50.4 | 1730 |
| vast-b-pipeline | nemo-parakeet-tdt-0.6b-v3+opus-mt | en | quality | 5.0 | 2.0 | 84 | 0.109 | 0.0537 | 52.6 | 1721 |
| vast-b-pipeline | nemo-parakeet-tdt-0.6b-v3+opus-mt | en | quality | 1000.0 | 1.0 | 57 | 0.0761 | 0.0537 | 55.6 | 1740 |
| vast-b-pipeline | nemo-parakeet-tdt-0.6b-v3+opus-mt | ru | quality | 5.0 | 1.0 | 93 | 0.1076 | 0.0844 | 53.1 | 1727 |
| vast-b-pipeline | nemo-parakeet-tdt-0.6b-v3+opus-mt | ru | quality | 8.0 | 1.0 | 63 | 0.096 | 0.0844 | 55.0 | 1746 |
| vast-b-pipeline | nemo-parakeet-tdt-0.6b-v3+opus-mt | ru | quality | 1000.0 | 1.0 | 48 | 0.0944 | 0.0844 | 58.1 | 1760 |

- FLEURS is read speech: ~10 s sentences with no 300 ms pause, so a 5 s limit cuts every sentence
  once or twice. Against the offline window that costs 2–6 WER points, against pause-only
  clauses 3–5 chrF points (MT translates fragments).
- Left context after a cut halves the WER cost of the cuts in English (15.7 % → 11.2 %), at no
  extra CPU; 2 s gives nothing more.
- Without the hard limit chrF is 55.6–58.1, within 2–4 points of MT on the gold text, but a long
  sentence is only translated after it ends. Conversation pauses more often than read news;
  stage 2 tunes the limit on real call recordings.
- Short windows write numbers as words ("one thousand four hundred" for "1,400"): part of the
  English WER.

After the review of stage 1 (2026-11-08, models from the registry, OPUS-MT output capped at
`2 × source tokens + 10` with `no_repeat_ngram_size=3`; Russian MT input starts with a capital
letter):

| host | model | lang | mode | max_clause_s | context_s | clauses | wer | wer_offline | chrf | peak_rss_mb |
|---|---|---|---|---|---|---|---|---|---|---|
| vast-b-pipeline | parakeet-tdt-0.6b-v3+opus-mt | en | quality | 5.0 | 1.0 | 85 | 0.1119 | 0.0537 | 52.5 | 1697 |
| vast-b-pipeline | parakeet-tdt-0.6b-v3+opus-mt | ru | quality | 5.0 | 1.0 | 93 | 0.1076 | 0.0844 | 53.9 | 1726 |
| vast-b-pipeline | parakeet-tdt-0.6b-v3+opus-mt | ru | quality | 8.0 | 1.0 | 63 | 0.096 | 0.0844 | 56.4 | 1737 |
| vast-b-pipeline | gigaam-v3-e2e-ctc+opus-mt | ru | quality | 5.0 | 1.0 | 92 | 0.0662 | 0.0579 | 53.5 | 623 |

- Russian ASR is GigaAM v3 e2e by default now: Parakeet has no language setting in onnx-asr
  (only Whisper and Canary take one) and on short windows drifts into Ukrainian ("примінив
  слізоточий газ"), which OPUS-MT ru-en then turns into invented text. Streamed GigaAM: WER 6.6 %
  against Parakeet's 10.8 % (offline 5.8 % against 8.4 %), at a third of the memory and 0.40
  against 0.59 CPU s per audio s in real time on B; chrF 53.5 against 53.9.
- The MT length cap and the repeat ban change chrF by 0–0.2 points: they stop loops ("I'm not
  sure if I'm not sure…"), not every invention. OPUS-MT ru-en also invents on a clause fragment
  that starts lowercase and ends with a full stop: "страдающим от глобального потепления для
  борьбы с его последствиями." → "The Committee notes that the Committee has not yet considered
  the report of the Secretary-General…" (UN boilerplate from its training data), within the cap.
  With the first letter capitalised it translates the fragment, and ru-en chrF rises 0.5–1.4
  (53.0 → 53.5 GigaAM, 53.3 → 53.9 and 55.0 → 56.4 Parakeet at 5 / 8 s). For en-ru the same costs
  0.6 (52.5 → 51.9: "Cope with the impacts." becomes an imperative), so only Russian input is
  capitalised. Captions keep the recognised text as it was.

### Latency per link

From the per-clause log (`peresmeshnik run --log`), the 30 s recordings in real time (ru with the
default GigaAM). Times from the end of the clause's speech: `caption_lag_s` → translated caption
on the event queue, `synth_lag_s` → first synthesized translated audio, `voice_lag_s` → estimated
start of its playback: speech is played back to back, so a clause waits until the speech before it
has played (`play_queue_s`). All include the 300 ms VAD pause for clauses that end at a pause. A
cut clause is decided ≥ 0.8 s after its end (the tail of the window is not trusted). Medians over
both recordings, measured 2026-11-08:

| host | input | clause ends at | n | asr_final_ms | mt_ms | caption_lag_s | tts_first_ms | synth_lag_s | play_queue_s | voice_lag_s | voice_lag_s max |
|---|---|---|---|---|---|---|---|---|---|---|---|
| vast-b-pipeline | file in real time | pause | 9 | 0 | 22 | 0.32 | 58 | 0.38 | 1.98 | 2.38 | 6.01 |
| vast-b-pipeline | file in real time | 5 s cut / punctuation | 5 | 0 | 44 | 1.14 | 115 | 1.25 | 0.0 | 1.31 | 1.90 |
| linux-x86_64-4c (shared) | file → `Peresmeshnik Call` sink → monitor | pause | 8 | 0 | 62 | 0.50 | 119 | 0.67 | 1.96 | 2.62 | 7.17 |
| linux-x86_64-4c (shared) | file → `Peresmeshnik Call` sink → monitor | 5 s cut / punctuation | 7 | 0 | 116 | 1.66 | 277 | 2.03 | 0.0 | 2.03 | 4.70 |

- Clause end → synthesized translation at a pause is 0.4–0.7 s, below the 0.65–0.95 s estimate
  of ARCHITECTURE §4, because the final ASR run is skipped.
- The voice itself starts later: a median 1.8 s (B) / 2.1 s (here) over all clauses, up to 6–7 s.
  These recordings are read speech with short pauses, and a translation can be longer than its
  clause (4.7 s of English → 6.2 s of Russian for the first one), so the clause after it waits in
  the play queue; the cut clauses open a long sentence and wait least. The queue has no cap and no
  speed-up or drop policy yet (stage 2). The earlier version of this table called the synthesis
  time `voice_lag_s` (0.39 s / 0.72 s at a pause); it never included the queue.
- Original subtitles (partials) are not logged per update; from the cost above a partial takes
  0.08 s (7950X3D) – 0.2 s (CI runners) every 300 ms of audio, i.e. ≈ 0.25–0.35 s from a word to
  the screen.
## Stage 2: the app shell (tray, overlay, hotkey, wizard)

Measured 2026-11-08 on `linux-x86_64-4c` (this dev server, 4 cores, shared with other work; every
run under `nice -n 19`), under Xvfb 1280×800 with `stalonetray` as the system tray and an own
PipeWire session (null sinks as headphones and microphone), and on the GitHub runners
(`macos-15`, `windows-latest`: their own desktop; `ubuntu-latest`: Xvfb + stalonetray).

### Idle CPU with the app open

App open, translation off: tray icon, overlay window (hidden), the virtual mic, the X11 hotkey grab
and, since CI is a first run, the setup wizard. On Linux the tray host is XEmbed (`stalonetray`); a
StatusNotifierItem tray over D-Bus (KDE, GNOME's AppIndicator) and Qt's wayland plugin are not
measured. CPU seconds of the process tree in 60 s after its
"idle" line (printed once the start is done and the Qt loop has nothing left to do); limit 0.13 s.

| Gate | linux-x86_64-4c | ubuntu-latest | macos-15 | windows-latest |
|---|---|---|---|---|
| `tests/test_idle_cpu.py` (`python -m peresmeshnik`) | 0.000 | 0.000 | 0.104, 0.093 | 0.000, 0.000 |
| `tests/test_ui.py`: + overlay on screen + wizard, per 15 s | 0.000: 0.000 / 0.000 / 0.000 / 0.000 | 0.000: 0.000 / 0.000 / 0.000 / 0.000 | 0.077: 0.072 / 0.002 / 0.000 / 0.002; 0.054: 0.049 / 0.002 / 0.001 / 0.001 | 0.016: 0.000 / 0.000 / 0.000 / 0.016; 0.000 |

Before stage 2 the idle process had no GUI and used 0.000 s on all three. On macOS the cost sits in
the first quarter (window server work after the first paint), not in a loop. Two things were found by
the gate and fixed: a tray balloon at start (macOS posts a notification: 0.092 → 0.045 s) and the
idle line coming before the wizard was built (0.308 → 0.104 s); the text cursor does not blink
(`QApplication.setCursorFlashTime(0)`), since a blinking cursor repaints its field twice a second.

The working-mode gate (`work-gate`, the local pipeline on the 30 s call; S3 does not touch it) stayed
green on every stage-2 push; on fe5cb0f RTF / CPU s per audio s: ubuntu-latest 0.628 / 1.25 (Xeon),
windows-latest 0.635 / 1.277, macos-15 0.381 / 0.722, against limits 1.0 / 2.0 — within the 2×
runner spread seen in stage 1 (0.27–0.67 / 0.55–1.32).

### Overlay repaint

CI numbers: two runs (78b6c1f, 9f070bc). Repaints happen only when the visible text changes (`SubtitleOverlay.updates` vs `.paints`); paint
time is the whole `paintEvent` (two rows, two lines each, software raster).

| Run | Text changes | Repaints | Paint median / p99 / max, ms |
|---|---|---|---|
| app, both sides from the 30 s test calls: them → me local, me → them cloud with local subtitles | 118 | 119 (the first show) | 0.64 / 5.24 / 5.73 |
| app, both sides cloud, subtitles from Gemini | 116 | 117 | 0.80 / 1.64 / 1.72 |
| `tests/test_ui.py`, 20 new captions (median / max): linux-x86_64-4c; ubuntu-latest; macos-15, two runs; windows-latest, two runs | 20 | 40 (each also forced) | 0.63 / 2.07; 0.29 / 1.17; 0.68 / 3.55 and 0.54 / 22.4 (one paint of 40; p90 < 16); 0.77 / 7.40 and 0.53 / 5.17 |

The first paint with Cyrillic took 23–28 ms on the macOS and Windows runners (font fallback
loading); the overlay now paints its fonts once into an image at start, so captions stay under the
16 ms budget from the first one.

### Start and switching on

| | linux-x86_64-4c |
|---|---|
| launch → "idle" (tray, overlay, virtual mic, hotkey), warm file cache, median of 5 | 0.31 s (0.27–0.34), RSS 74 MB |
| the same with the first-run wizard open | 0.39 s (0.34–0.41), RSS 88 MB |
| translation on → ready: them → me local (Parakeet en + OPUS-MT + Piper, warmed up), me → them cloud (Gemini connect) with local subtitles (GigaAM ru + OPUS-MT) | 6.2–6.3 s (2 runs) |
| me → them local through `app.Side` on `rtx3060-laptop` (server B): GigaAM + OPUS-MT + XTTS-v2 on the GPU, warmed up | 10.4 s |

### Hotkey, self-test, wizard

- **X11 hotkey** (`XGrabKey` over ctypes, read when the X socket has data): `xdotool key` of the combo
  toggles "translation ⇄ live voice" once per press (detectable auto-repeat), another combo does
  nothing, a combo already grabbed by another client is refused with a message. Ctrl+Alt+F1…F12 are
  taken by the X server (VT switch); on the `ubuntu-latest` X server xdotool also adds Alt to F-keys
  (Ctrl+Shift+F9 arrived as Ctrl+Shift+Alt+F9, state 0xd), so the tests use letters.
- **Overlay on top and click-through**: only under Xvfb: the window flags, and an `xdotool click`
  on the overlay reaches the window below it. Not checked on a real compositor (GNOME, KDE).
- **Wayland portal**: the GlobalShortcuts flow (CreateSession → BindShortcuts → Activated, answers
  through `Request.Response`) against a fake portal on a private D-Bus; only our session's
  `Activated` fires, on the GUI thread. A real portal (GNOME ≥ 48, KDE 6) is not checked here.
- **Loopback self-test** on the virtual devices: a 0.4 s chirp at −12 dBFS is found by
  cross-correlation with match 0.97–1.00 at −15 dBFS through the call capture sink (S1's capture,
  16 kHz), the virtual mic (S2's, recorded from "Peresmeshnik Mic") and the headphones sink's
  monitor; microphone level −23 dBFS (a test tone); silence, noise and a steady tone score < 0.1.
  Same on `ubuntu-latest` (match 1.00).
- **Wizard, every Linux page** (`tests/test_app_linux.py`, `PERESMESHNIK_E2E=1`): 35 s from the
  first page to "finish" with models present: devices chosen, self-test green, test phrase
  them → me local ("It is Martelli's / Fifth CEP in four years." → "Это Мартелли. / Пятый КЭП через
  четыре года.") and me → them cloud with local subtitles ("Это пятый серб Мартеи за четыре года." →
  "This is the fifth Martei Serb in four years.", Gemini voice 3.2 s after the start), settings saved.
  Downloads: silero-vad from Hugging Face through the models page; a cut download resumes with HTTP
  Range (`tests/test_ui.py`, local server). The project's GitHub release `models-1` answered 502 and
  timed out from this server and from server B for ~30 min (11:10–11:40 UTC); OPUS-MT was taken from
  the stage-1 cache (same SHA-256 pins).
- **me → them local** (clone) through the app's wiring, `rtx3060-laptop`: the 30 s Russian test call →
  7 clauses with translation in the subtitles (first at 6.5 s, i.e. right after the first sentence),
  17.2 s of cloned English speech, first audio 6.7 s after the start.

### Cloud direction with Silero VAD

`peresmeshnik cloud --subtitles` on the 30 s test calls (3 sentences each), the new Silero gate vs the
stage-1 energy gate (the fallback while silero-vad is not downloaded). Seconds, `median / p90 (n)`.
Each gate sets its own start / pause / end, so the columns compare where the gate puts them as much
as the cloud: Silero opens at the voice, not at the lead-in noise (first partial earlier), and does
not cut pauses inside a sentence (no 3.8–7.4 s outliers of the final text).

| Link | Silero en → ru | Silero ru → en | energy en → ru | energy ru → en |
|---|---|---|---|---|
| subtitles: first word after speech start | 0.55 / 1.05 (3) | 1.06 / 1.11 (3) | 1.22 / 1.91 (3) | 2.46 / 6.76 (4) |
| subtitles: final text after a 300 ms pause | 0.61 / 0.69 (3) | 0.63 / 0.66 (3) | 0.64 / 3.80 (3) | 0.66 / 7.37 (4) |
| voice: first translated audio after the start | 3.21 (1) | 3.13 (1) | 3.81 (1) | 4.66 (1) |
| voice: done after the speaker stops | 3.76 / 4.81 (3) | 3.54 / 3.89 (3) | 2.60 / 3.00 (3) | 1.96 / 3.61 (3) |

"Done after the end" is shorter with the energy gate because its end comes later (the last block above
its threshold, trailing noise included); Silero ends at the voice.

## Stage 2: me → them live (microphone, live voice, voice checks)

Measured 2026-11-08 on `linux-x86_64-4c` (own PipeWire session, Xvfb 1280×800 + `stalonetray`, shared
with other work, `nice`), the clone on `rtx3060-laptop` (as in stage 1).

### Translation ⇄ live voice

`audio/mic.py` `LiveSwitch` wraps the me → them direction (local or cloud): its output is clocked by the
microphone's frames, so the queued translation stays in the process and can be faded instead of being
cut in the device buffer. Each switch is an equal-power crossfade (cos / sin). The click measure: the
biggest sample-to-sample jump at the seam against the two signals' own biggest jumps around it, and the
energy above 4 kHz in 10 ms windows along the seam against the signals' own there.

**Crossfade length on speech** (`bench/live_switch.py`: four clone sentences as the translation, a
LibriVox reader as the live microphone, 300 seams at random moments where both talk):

| fade | jump / own: median | p95 | max | seams > 2× | energy > 4 kHz / own: p95 | seams > +6 dB |
|---|---|---|---|---|---|---|
| 0 ms (hard switch) | 0.74 | 2.77 | 5.60 | 12 % | +12.9 dB | 18 % |
| 2 ms | 0.52 | 0.98 | 1.04 | 0 % | −0.6 dB | 0 % |
| 5 ms | 0.54 | 0.98 | 1.00 | 0 % | −0.2 dB | 0 % |
| 10 ms | 0.56 | 0.98 | 1.01 | 0 % | 0.0 dB | 0 % |
| 30 ms (chosen) | 0.59 | 0.99 | 1.07 | 0 % | 0.0 dB | 0 % |
| 50 ms | 0.62 | 1.00 | 1.01 | 0 % | 0.0 dB | 0 % |

Any fade from 2 ms removes the click by both measures. 30 ms keeps a margin: two periods of a low voice
(85 Hz), so the level change is slower than the voice itself, about 1.5 frames of the 20 ms input clock,
and well under a syllable (the live voice is at full level 30 ms after the key).

**Tests** (`tests/test_virtual_mic_live.py`): tones on a DC offset, where a hard cut always jumps: 101.6×
without a crossfade (the test asserts > 5), 1.2× with 30 ms (asserts < 5). End to end under Xvfb: the app
with a fake translation (a 220 Hz tone) and a PipeWire test microphone (300 Hz), `xdotool key` toggles,
"Peresmeshnik Mic" is recorded as a call app would: translation → live voice → translation, jumps at the
seams 1.3× and 0.9× the tones' own with 30 ms (asserts < 3); with the fade at 0, 2.5–14.9× over runs (the
phases at the seam are random). It needs no models, so it runs in the `ubuntu-latest` test job: there
1.3× / 0.9× with the crossfade, 10.8× / 4.4× without (0be061c).

**Cost**: 0.012 CPU s per second of sound while translation or the live voice plays (one events thread,
20 ms frames), ~0 while the direction is silent (the frames do not wake it); the output starts each
stretch with 60 ms of silence as a cushion against scheduling jitter (on top of the device's 40 ms).

### Microphone by name

`Mic` on Linux records a PipeWire source by its `pactl` name through `parec` (20 ms frames, the same
`s.ui.mic` for the translation and for the loopback while translation is off; our own devices and sink
monitors are refused). 0.0096 CPU s per second of audio (reader thread 0.005 + parec 0.0045). While it runs,
one `pactl subscribe` watches PipeWire's events: the source's removal (or the end of the stream) makes the
app reopen on the saved source if it is back, else on the default input: 0.06–0.09 s from removal to the
new stream, one tray message; with no microphone at all the next new source is taken, no polling. The
event is needed: here (PipeWire 1.0.5, WirePlumber 0.4.17) a `node.dont-reconnect` recording ends with
"Entity killed" when its source goes, on `ubuntu-latest` (PipeWire 1.6.2, WirePlumber 0.5.13) it did not
end, and the first version of the test, which waited for that end, failed there. If the event stream itself
ends (pipewire-pulse restarted), the tray says so once and the app subscribes again after 2 s (doubling up to
30 s while the server stays away: a repeat, not polling), then reopens the microphone if its stream went down
too (`test_microphone_survives_a_sound_server_restart`).

"Peresmeshnik Mic" now has `priority.session=-10` on its sink and source: WirePlumber 0.4.17 counts a
device without a priority as −1, so the 0 asked for first won the default over such devices (measured
with null sinks; real ALSA and Bluetooth devices have ~1000–2000 and win either way).

### Through a microphone by name, end to end

`bench/me_them.py run --remote --mic test_me_mic`: the stage-1 input (33 s of the enrolled LibriVox reader,
other lines than the reference) is played into a test PipeWire microphone; the app's path takes it from
there by name (`app.Microphone` → `LiveSwitch` → `LocalDirection`: Silero VAD, GigaAM ru, OPUS-MT ru-en here;
the clone XTTS-v2 on `rtx3060-laptop` over a pipe) → "Peresmeshnik Mic", recorded the way a call app records
it, transcribed by Whisper small.

| | value |
|---|---|
| clauses | 9 (all closed by a pause) |
| WER of the recording vs the MT text | **0.024** (of the TTS output itself: 0.012) |
| clone, first audio on the GPU host / RTF | 0.17–0.21 s / 0.21 |
| virtual mic link (play → source) | 20–42 ms |
| caption after the clause end | 0.37–2.81 s (median 0.61) |
| first new sound at the call app after the caption | 0.39–3.46 s (median 1.17) |
| CPU here (VAD, ASR, MT, microphone, switch), per audio s | 1.69 (`nice 19`, the box shared) |
| after the input ended, the last speech done in | 2.0 s |

The lag after the caption is the playback queue (speech of a clause waits for the one before, as in
stage 1) plus the 60 ms cushion of the switch; S1's queue policy works on it. With `LiveSwitch` the speech
is re-chunked, so the per-clause lag is taken from the first new onset after the caption, not from the
clause's own first chunk (approximate when clauses run back to back).

### Voice checks

**The recording** (`voice.check`): length 30–60 s, speech ≥ 15 s, level, clipping, speech above the
background, and now the band: speech energy at 4–8 kHz against 0.1–4 kHz — microphones −27…−9 dB,
telephone band (300–3400 Hz) −74…−80 dB; below −45 dB the verdict warns that the clone will sound muffled.

**The clone** (`voice.check_clone`): it says two test phrases (~4 s each) in the language it will speak;
the app's ASR (GigaAM ru / Parakeet en) gives the WER; similarity = cosine of XTTS's own speaker
embeddings, clone against the recording (no extra model). Calibration (`bench/voice_check.py`): 8 voices
(4 LibriSpeech test-clean speakers, 4 chapters of the Russian LibriVox book) × 5 recordings, cloned into
the other language on `rtx3060-laptop`; ECAPA (`speechbrain/spkrec-ecapa-voxceleb`) as an independent
judge against the clean recording. "noise14" = pink noise 14 dB below speech; "babble14" = another speaker
at 14 dB; "reverb" = RT60 ~0.7 s; "phone" = 300–3400 Hz.

| recording | recording check (SNR dB) | clone WER: median / max | similarity, app (vs own recording): min / median | XTTS vs clean: min / median | ECAPA vs clean: min / median |
|---|---|---|---|---|---|
| clean | 8/8 pass (13–48) | 0.03 / 0.05 | 0.49 / 0.54 | 0.49 / 0.54 | 0.36 / 0.41 |
| noise14 | 8/8 pass (13–17) | 0.02 / 0.28 | 0.39 / 0.56 | 0.32 / 0.54 | 0.14 / 0.37 |
| babble14 | 8/8 pass (13–23) | 0.02 / 0.03 | 0.53 / 0.57 | 0.50 / 0.55 | 0.25 / 0.39 |
| reverb | 8/8 pass (14–31) | 0.00 / 0.07 | 0.46 / 0.55 | 0.34 / 0.42 | 0.20 / 0.31 |
| phone | 8/8 pass (27–41; now with the band warning) | 0.02 / 0.14 | 0.37 / 0.49 | 0.18 / 0.30 | 0.12 / 0.21 |

Another speaker's recording against the clone (English pairs): XTTS median 0.11, p90 0.28, max 0.45 (the
babble partner), ECAPA median 0.05, max 0.17. The Russian chapters are probably one reader (ECAPA between
chapters up to 0.42), so they are not used as other speakers.

Thresholds (`voice.py`): WER fail > 0.25, warn > 0.10; similarity fail < 0.25, warn < 0.35. On the 40
clones: 1 fail (noise14, WER 0.28), 3 warnings (phone, WER 0.14), 36 good; 80 % of the other-speaker
pairs fall under 0.25. XTTS copies the channel too: noise, reverb and a phone band leave the clone
understandable and close to the recording in XTTS's own space, but further from the clean voice
(ECAPA 0.12–0.21 for the phone band), so the similarity check catches a clone that sounds like someone
else, not a poor microphone; the band check and the background check cover those before cloning.

### CPU gates

| Gate | linux-x86_64-4c | ubuntu-latest | macos-15 | windows-latest |
|---|---|---|---|---|
| idle, `tests/test_idle_cpu.py` (limit 0.13 s / 60 s) | 0.000 | 0.000 | 0.008 (first 15 s after "idle", not gated: 0.120) | 0.000 |
| idle with the overlay and the wizard (`tests/test_ui.py`) | 0.000 | 0.000 | 0.080 (0.072 in the first quarter) | 0.000 |
| work-gate, RTF / CPU s per audio s (limits 1.0 / 2.0) | 0.58 / 1.13 (`nice 19`) | 0.58 / 1.20 | 0.37 / 0.69 | 0.66 / 1.31 |

On macOS the idle gate now starts its 60 s window 15 s after the idle line: on one and the same code it had
given 0.066–0.156 s against the 0.13 s limit, nearly all in the first seconds (AppKit's one-off work after
the windows are first shown, seen by the stage-2 shell too), so it flaked; that work is printed instead.

**Wizard** (`ui/wizard.py` `VoicePage`, `tests/test_clone_wizard.py` under Xvfb): record from the chosen
PipeWire source or import a file; the verdict ("Good …" / "Good enough: …" / "Record again: …") with the
numbers; a bad take is not kept and does not replace a good one (a 3 s take of a steady tone: "Record
again: 3 s recorded, need 30-60 s; only 0 s of speech …"); a good take is a draft (`voices/<name>.new`) that
replaces the saved voice only on Finish, Cancel keeps the old one; "Check the clone" runs `check_clone` (the
voice pack and the clone model are needed: on this server without the voice pack the button says so; while a
local me → them translates the button is off: a second clone would not fit next to it in 8 GB).

## Stage 2: the call's sound, both directions, the voice queue (S1)

Measured 2026-11-08 on `vast-b-stage2` (server B: Ryzen 9 7950X3D, 32 threads, RTX 3060 Laptop
12 GB; every run under `nice -n 15`, up to 6 at once) and `linux-x86_64-4c` (this dev server, 4
cores, shared; an own PipeWire session). Scripts: [`bench/conversation.py`](../bench/conversation.py)
(segmentation and the voice lag), [`bench/app_both.py`](../bench/app_both.py) (both directions).

### Moving the call's sound

Checked by `tests/test_pipeline_pipewire.py` on PipeWire 1.0.5 / WirePlumber 0.4.17 with fake
players (`pacat` streams named like Telegram, Chrome, Zoom): the chosen app's stream goes into
"Peresmeshnik Call" at switch-on, a new stream of it (a reconnect) is there 66 ms after it
appears (median of 40; p95 95 ms, max 139 ms), a browser playing next to it stays where it is;
off and exit give every stream back (to the default output when its device was unplugged
meanwhile), and the next stream of the app follows the default output again (WirePlumber's
remembered target is cleared). While
translation is off there is no `pactl subscribe` process (checked in the test): the idle gate is
unchanged. WirePlumber links ~5 % of new streams to the default output again right after our move
(the move itself succeeds): within 1 s of a move the router checks the stream's change events and
moves it again (2 of 30 stayed on the speakers before, 0 of 40 after). The call sink, like the
virtual mic, now has `priority.session=-10`: with the 0 it had, it became the default output in
place of a device without a priority (measured: the system's sound would have gone silent into it).
A call that ends while translating left the app bound to our sink (WirePlumber keeps the move after the stream
is gone): its next stream went there while translation was off. Now cleared at switch-off by a probe stream
(the test plays the app again after off: on the speakers, 10 of 10 runs).

Headphones' flush (barge-in): a 3 s chunk flushed 1.0 s after it started played 0.98–1.00 s (5
runs); before the fix (pacat left alone, stdin drained) the stream kept its last part below the
prebuffer and played ~0.2 s of the old speech in front of the next one.

### Clause segmentation on conversational speech

Twenty minutes of spontaneous speech, streamed through VAD + ASR partials + clause cuts + MT
(TTS off, as fast as the pipeline takes it): **en** — AMI meetings ES2002a and IS1009a, 5 min
each from the first word (Mix-Headset, four people: overlaps, "um", "yeah", room noise; Parakeet);
**ru** — YODAS2 ru000 YAYMb4M8-tQ and YtyM7ojCUz8, 5 min each (a vlog and an interview, long
stretches without a 300 ms pause; GigaAM). Both scaled to −26 dBFS. `wer` = joined clause texts
against the reference words (fillers such as "um" / "э" dropped on both sides); `chrf` = joined
clause translations against the translation of the reference segments (a pseudo-reference: no
human translations exist), so it measures what segmentation does to MT, not MT itself. `*_offline`
= each reference segment recognised (and translated) in one window: for ru the uploaded subtitles'
times are loose, the windows clip words, so its `wer_offline` is worse than the stream's. `wait` =
how long a word waits for its clause to end (words spread evenly over the clause): the voice can
start no earlier. First row of each language = the defaults.

| lang | VAD threshold | pause ms | max_clause_s | clauses | ≤ 2 words | clause s p50 / p95 | wait s p50 / p95 | ends: pause / punct / max | wer | wer_offline | chrf | chrf_offline |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| en | 0.5 | 300 | 5 | 206 | 0.42 | 1.59 / 4.05 | 1.09 / 3.26 | 0.74 / 0.14 / 0.12 | 23.4 % | 21.4 % | 62.2 | 68.0 |
| en | 0.5 | 300 | 4 | 219 | 0.45 | 1.6 / 3.25 | 0.97 / 2.64 | 0.70 / 0.10 / 0.20 | 26.2 % | 21.4 % | 61.4 | 68.0 |
| en | 0.35 | 300 | 5 | 207 | 0.42 | 1.6 / 4.09 | 1.09 / 3.29 | 0.70 / 0.16 / 0.14 | 22.6 % | 21.4 % | 63.3 | 68.0 |
| en | 0.5 | 200 | 5 | 218 | 0.46 | 1.48 / 3.88 | 1.01 / 3.15 | 0.86 / 0.07 / 0.06 | 24.9 % | 21.4 % | 62.2 | 68.0 |
| en | 0.5 | 500 | 5 | 182 | 0.35 | 2.09 / 4.08 | 1.19 / 3.31 | 0.54 / 0.24 / 0.21 | 24.1 % | 21.4 % | 63.3 | 68.0 |
| en | 0.65 | 300 | 5 | 209 | 0.43 | 1.57 / 3.95 | 1.02 / 3.04 | 0.79 / 0.12 / 0.09 | 23.6 % | 21.4 % | 61.7 | 68.0 |
| en | 0.5 | 300 | 6 | 199 | 0.45 | 1.6 / 4.79 | 1.13 / 3.81 | 0.77 / 0.17 / 0.06 | 23.8 % | 21.4 % | 63.0 | 68.0 |
| en | 0.5 | 300 | 8 | 191 | 0.43 | 1.58 / 5.62 | 1.21 / 4.65 | 0.81 / 0.18 / 0.01 | 22.5 % | 21.4 % | 64.3 | 68.0 |
| en | 0.5 | 300 | 10 | 190 | 0.43 | 1.57 / 5.62 | 1.21 / 4.76 | 0.81 / 0.19 / 0.00 | 22.5 % | 21.4 % | 64.8 | 68.0 |
| ru | 0.5 | 300 | 5 | 209 | 0.19 | 3.08 / 4.12 | 1.56 / 3.58 | 0.17 / 0.37 / 0.46 | 7.1 % | 15.7 % | 73.1 | 79.3 |
| ru | 0.5 | 300 | 4 | 252 | 0.18 | 2.61 / 3.14 | 1.26 / 2.75 | 0.14 / 0.27 / 0.59 | 7.4 % | 15.7 % | 71.2 | 79.3 |
| ru | 0.35 | 300 | 5 | 207 | 0.17 | 3.18 / 4.13 | 1.57 / 3.6 | 0.16 / 0.36 / 0.48 | 7.1 % | 15.7 % | 73.0 | 79.3 |
| ru | 0.5 | 200 | 5 | 211 | 0.18 | 2.88 / 4.26 | 1.51 / 3.61 | 0.40 / 0.25 / 0.35 | 7.1 % | 15.7 % | 73.9 | 79.3 |
| ru | 0.5 | 500 | 5 | 202 | 0.15 | 3.31 / 4.1 | 1.59 / 3.54 | 0.05 / 0.40 / 0.55 | 6.8 % | 15.7 % | 73.3 | 79.3 |
| ru | 0.65 | 300 | 5 | 210 | 0.19 | 3.06 / 4.12 | 1.54 / 3.56 | 0.19 / 0.35 / 0.46 | 7.1 % | 15.7 % | 73.0 | 79.3 |
| ru | 0.5 | 300 | 6 | 179 | 0.14 | 3.5 / 5.06 | 1.8 / 4.39 | 0.20 / 0.45 / 0.35 | 7.0 % | 15.7 % | 74.6 | 79.3 |
| ru | 0.5 | 300 | 8 | 153 | 0.18 | 4.1 / 6.82 | 2.21 / 5.63 | 0.23 / 0.54 / 0.23 | 6.8 % | 15.7 % | 73.1 | 79.3 |
| ru | 0.5 | 300 | 10 | 135 | 0.16 | 4.12 / 8.4 | 2.55 / 7.05 | 0.27 / 0.62 / 0.11 | 6.8 % | 15.7 % | 75.5 | 79.3 |

- **`max_clause_s`** is the one setting that matters. Longer clauses translate better (en chrF
  62.2 → 64.3, ru 73.1 → 73.1–75.5 from 5 to 8–10 s; WER −0.3…−0.9 points) and every second of it
  adds ~0.6–0.9 s to the p95 wait. In the meetings 74–81 % of the clauses end at a pause anyway
  (people take turns), so the limit acts in monologues: in the Russian videos 46 % of the clauses
  at 5 s end at the limit.
- **VAD** threshold 0.35 / 0.5 / 0.65 and the pause 200 / 300 / 500 ms move WER and chrF by ≤ 1.5
  points without a trend that holds in both languages (≈ the noise of 1100–1300 reference words):
  the defaults stay (0.5, 300 ms). A 200 ms pause cuts English more often (86 % pause ends) and
  costs 1.5 WER points; 500 ms merges back-channels into longer clauses (≤ 2 words 0.42 → 0.35).
- Short clauses are the nature of a meeting: 42–46 % have ≤ 2 words ("yeah", "okay", "mm-hmm").

### The voice queue: lag with and without the policy

Real-time runs of them → me with Piper (`bench/conversation.py lag`), back-to-back playback as in
the app. **fleurs-continuous** = 30 FLEURS sentences 0.15 s apart (5–6 min of read speech with no
300 ms pause: the worst case), **conversation** = the AMI / YODAS2 excerpts above (10 min per
language). `voice_lag` = clause end → start of its voice (spoken clauses only); `evs` (ear–voice
span) = a word → the start of its clause's voice, words spread evenly over the clause; `dropped`
= clauses / words not said (their subtitles stay); `speed` = mean TTS speed of the spoken
clauses; `speech/audio` = seconds of translated speech per second of input. `policy`: the cap
`max_lag_s`, and from which lag the speech is sped up to which speed at the cap; the default is
cap 4 s, 0.5 s → 1.4×.

| data | lang | max_clause_s | policy | clauses | voice_lag p50 / p95 / max, s | evs p50 / p95, s | dropped clauses / words | speed | speech/audio |
|---|---|---|---|---|---|---|---|---|---|
| fleurs-continuous | en | 5 | none | 85 | 1.45 / 5.95 / 7.67 | 3.4 / 6.9 | 0 % / 0.0 % | 1.0 | 0.75 |
| fleurs-continuous | en | 5 | cap 3 s, 0.5 s → 1.4× | 85 | 0.79 / 2.16 / 2.85 | 2.88 / 4.99 | 15 % / 8.2 % | 1.085 | 0.665 |
| fleurs-continuous | en | 5 | cap 3 s, 1 s → 1.25× | 85 | 0.93 / 2.5 / 2.97 | 2.96 / 5.08 | 14 % / 6.7 % | 1.04 | 0.687 |
| fleurs-continuous | en | 5 | cap 4 s, 0.5 s → 1.4× | 85 | 1.03 / 2.87 / 3.85 | 2.94 / 5.05 | 11 % / 5.4 % | 1.076 | 0.688 |
| fleurs-continuous | en | 5 | cap 4 s, 1 s → 1.25× | 85 | 1.07 / 2.57 / 3.7 | 2.97 / 5.09 | 13 % / 6.1 % | 1.029 | 0.696 |
| fleurs-continuous | en | 6 | cap 3 s, 1 s → 1.25× | 75 | 0.75 / 2.48 / 2.98 | 3.09 / 5.79 | 12 % / 10.5 % | 1.043 | 0.636 |
| fleurs-continuous | en | 8 | none | 67 | 1.36 / 5.89 / 8.02 | 4.39 / 8.36 | 0 % / 0.0 % | 1.0 | 0.729 |
| fleurs-continuous | en | 8 | cap 3 s, 1 s → 1.25× | 67 | 0.72 / 2.51 / 2.78 | 3.39 / 7.51 | 18 % / 12.2 % | 1.032 | 0.623 |
| fleurs-continuous | ru | 5 | none | 91 | 1.59 / 4.38 / 7.34 | 3.36 / 5.35 | 0 % / 0.0 % | 1.0 | 0.778 |
| fleurs-continuous | ru | 5 | cap 3 s, 0.5 s → 1.4× | 91 | 1.31 / 2.5 / 2.98 | 3.04 / 4.92 | 12 % / 5.4 % | 1.135 | 0.687 |
| fleurs-continuous | ru | 5 | cap 3 s, 1 s → 1.25× | 91 | 1.34 / 2.52 / 2.96 | 3.07 / 4.92 | 13 % / 6.4 % | 1.058 | 0.707 |
| fleurs-continuous | ru | 5 | cap 4 s, 0.5 s → 1.4× | 91 | 1.39 / 3.38 / 3.91 | 3.17 / 4.94 | 6 % / 1.8 % | 1.117 | 0.722 |
| fleurs-continuous | ru | 5 | cap 4 s, 1 s → 1.25× | 91 | 1.42 / 3.32 / 3.78 | 3.26 / 5.0 | 7 % / 2.5 % | 1.056 | 0.736 |
| fleurs-continuous | ru | 6 | cap 3 s, 1 s → 1.25× | 78 | 1.46 / 2.72 / 2.81 | 3.45 / 5.89 | 13 % / 6.7 % | 1.067 | 0.679 |
| fleurs-continuous | ru | 8 | none | 61 | 1.95 / 5.57 / 7.69 | 4.62 / 8.23 | 0 % / 0.0 % | 1.0 | 0.766 |
| fleurs-continuous | ru | 8 | cap 3 s, 1 s → 1.25× | 61 | 0.67 / 2.62 / 2.92 | 3.88 / 7.66 | 23 % / 15.5 % | 1.057 | 0.625 |
| conversation | en | 5 | none | 207 | 0.82 / 3.34 / 5.21 | 2.41 / 4.91 | 0 % / 0.0 % | 1.0 | 0.472 |
| conversation | en | 5 | cap 3 s, 0.5 s → 1.4× | 207 | 0.55 / 2.58 / 3.03 | 2.23 / 4.6 | 5 % / 3.1 % | 1.084 | 0.435 |
| conversation | en | 5 | cap 3 s, 1 s → 1.25× | 209 | 0.56 / 2.54 / 2.95 | 2.19 / 4.63 | 7 % / 5.2 % | 1.036 | 0.436 |
| conversation | en | 5 | cap 4 s, 0.5 s → 1.4× | 207 | 0.58 / 2.82 / 3.87 | 2.3 / 4.67 | 2 % / 0.8 % | 1.07 | 0.454 |
| conversation | en | 5 | cap 4 s, 1 s → 1.25× | 207 | 0.64 / 2.91 / 3.93 | 2.33 / 4.71 | 2 % / 0.8 % | 1.034 | 0.464 |
| conversation | en | 6 | cap 3 s, 1 s → 1.25× | 200 | 0.53 / 2.42 / 2.95 | 2.19 / 5.19 | 8 % / 6.7 % | 1.033 | 0.426 |
| conversation | en | 8 | none | 191 | 0.71 / 3.62 / 6.26 | 2.45 / 6.12 | 0 % / 0.0 % | 1.0 | 0.463 |
| conversation | en | 8 | cap 3 s, 1 s → 1.25× | 190 | 0.54 / 2.39 / 2.84 | 2.22 / 5.86 | 8 % / 5.9 % | 1.029 | 0.428 |
| conversation | ru | 5 | none | 210 | 4.68 / 20.06 / 23.07 | 6.32 / 22.21 | 0 % / 0.0 % | 1.0 | 0.982 |
| conversation | ru | 5 | cap 3 s, 0.5 s → 1.4× | 209 | 1.58 / 2.79 / 2.98 | 3.31 / 5.07 | 14 % / 5.8 % | 1.185 | 0.817 |
| conversation | ru | 5 | cap 3 s, 1 s → 1.25× | 209 | 1.53 / 2.74 / 2.93 | 3.36 / 5.21 | 19 % / 9.9 % | 1.088 | 0.825 |
| conversation | ru | 5 | cap 4 s, 0.5 s → 1.4× | 209 | 1.81 / 3.45 / 3.95 | 3.47 / 5.27 | 7 % / 2.9 % | 1.159 | 0.865 |
| conversation | ru | 5 | cap 4 s, 1 s → 1.25× | 210 | 1.96 / 3.7 / 3.94 | 3.66 / 5.65 | 10 % / 3.5 % | 1.09 | 0.896 |
| conversation | ru | 6 | cap 3 s, 1 s → 1.25× | 179 | 1.47 / 2.68 / 2.9 | 3.65 / 5.98 | 22 % / 13.8 % | 1.083 | 0.76 |
| conversation | ru | 8 | none | 153 | 4.25 / 7.61 / 11.06 | 6.24 / 10.42 | 0 % / 0.0 % | 1.0 | 0.911 |
| conversation | ru | 8 | cap 3 s, 1 s → 1.25× | 154 | 1.55 / 2.78 / 2.98 | 3.97 / 7.62 | 27 % / 21.4 % | 1.082 | 0.663 |

- **Without a policy the voice drifts away.** Read speech without pauses: p95 4.4–6.0 s, max
  7.3–8.0 s. The Russian videos (fast monologue; its English takes 0.98 s per second of input):
  p95 20 s, max 23 s — the voice catches up only in a pause. Meetings (people take turns): p95
  3.3–3.6 s, max 5.2–6.3 s.
- **The default** (cap 4 s, faster from 0.5 s up to 1.4×): p95 2.8–3.5 s, never above the cap;
  dropped words 0.8 % (en) and 2.9 % (ru) in conversation, 5.4 % / 1.8 % on non-stop read speech;
  mean speed 1.07–1.16×.
- A 3 s cap gains 0.3–0.7 s at the p95 and drops twice as many words. Speeding up early and
  harder (0.5 s → 1.4×) drops ~40 % fewer words than 1.0 s → 1.25× at the same lag.
- The ear–voice span is set by the clause, not the queue: its p95 moves by ≤ 0.2 s between the
  policies but by +1.2–2.4 s with `max_clause_s` 8 (longer clauses also mean longer bursts of
  speech: more is dropped, 21 % of the words in the Russian videos at cap 3 s). With the quality
  gain of 8 s being 0–2 chrF (segmentation above), `max_clause_s` stays 5; 6 s sits between.
- Barge-in (my side's partial reaches 3 words) is not in these runs: they have only one side. In
  the both-sides runs below it cuts their voice while I talk.

### Both directions at once

Their 30 s recording (`call-en.ogg`) and mine (`call-ru.ogg`), mine starting `overlap` s after
theirs (15 = half of each overlaps, 0 = both talk all the time). **B**: the app's two sides
(`app.Side`, the barge-in wiring as in the app) in one process from the files — them → me local
(Parakeet + OPUS-MT + Piper), me → them local (GigaAM + OPUS-MT + XTTS-v2 on the GPU, the voice of
S2's tests). **This dev server** (no GPU, no clone): the real app (`peresmeshnik app --on`, Qt
offscreen) through its PipeWire devices: a fake call app (`pacat` with its own
`application.process.binary`, chosen as `ui.call_app`) starts playing after translation is on and
is moved into "Peresmeshnik Call" by the app; my recording plays into a virtual microphone (the
default input); the headphones (a null sink) and "Peresmeshnik Mic" are recorded back. me → them
is the cloud (Gemini Live) with local subtitles (GigaAM), the 8 GB default. CPU = the process tree
(the app, Piper, pacat/parec, `pactl subscribe`) over the run; RSS = the tree's peak sum.

| host | sides | overlap s | CPU s / audio s | cores, mean | RSS idle → peak, MB | GPU MB | translated speech heard / into the mic, s | their clauses dropped |
|---|---|---|---|---|---|---|---|---|
| vast-b-stage2 | them local (app.Side, files) | 0 | 0.609 | 0.46 | 1611 → 1831 | 19 | 18.8 / — | 2 |
| vast-b-stage2 | me local (app.Side, files) | 0 | 0.497 | 0.37 | 2269 → 2316 | 2212 | — / 17.3 | — |
| vast-b-stage2 | them local + me local (app.Side, files) | 15 | 0.566 | 0.61 | 3409 → 3680 | 2264 | 14.5 / 17.7 | 2 |
| vast-b-stage2 | them local + me local (app.Side, files) | 0 | 0.557 | 0.83 | 3389 → 3545 | 2212 | 3.4 / 17.2 | 4 |
| linux-x86_64-4c | them local + me cloud | 15 | 0.992 | 1.12 | 1739 → 1962 | — | 7.2 / 14.9 | 1 late + 4 barge-in |

- Both sides at once cost about the sum of each: on B 0.46 + 0.37 cores alone, 0.61–0.83 together
  (CPU per second of input 0.56–0.57), 3.5–3.7 GB of RAM and 2.2 GB of VRAM; here 1.12 cores of 4
  and 2.0 GB with the cloud for my side (whose local subtitles run GigaAM).
- While I talk their translated voice is cut (barge-in): with both talking all the time (overlap
  0) little of it is heard; the subtitles keep every clause. My own voice into the call is never
  dropped (no queue policy for me → them: the other side has no subtitles).
- Found on the way: the app imported numpy before the pipeline pinned the BLAS threads, so
  OpenBLAS took every core — them → me alone cost 3.0 CPU s per audio s on B (2.24 cores) instead
  of 0.61–0.81, both sides together 2.1–2.2 instead of 0.56–0.60. The working-mode gate did not
  see it: it runs `peresmeshnik run`, which imports in the right order. The thread pins now live in the
  package's `__init__`, and a test checks them after `import peresmeshnik.app`.

## Stage 3: the macOS shell and the .app (S3)

Measured 2026-11-08 on the GitHub runner `macos-15` (a virtual M1, 3 cores, ~7 GB, no GPU or Neural Engine,
its own Aqua desktop; `tests/test_macos.py`, `tests/test_app_bundle.py` in `release.yml`, several runs). Not
measured: a real Mac (CoreML / Metal / MPS: no Apple GPU in CI), real headphones, a real call (USER-CHECKS).

### The .app

| | macos-15 |
|---|---|
| build (`packaging/macos/build.sh`: PyInstaller 6.22, the Piper runner, licenses, ad-hoc signature, `hdiutil`) | 73 s |
| `Peresmeshnik.app` / `.dmg` (UDZO) | 266 MB / 128–129 MB (no models, no torch) |
| launch of the app copied out of the mounted .dmg → "idle" (its first start; the first-run wizard and a sample subtitle on screen) | 2.04, 1.76, 1.91, 2.19 s; RSS 137–143 MB |
| the same from Python (`python -m peresmeshnik app`, no wizard) | 0.52–0.88 s; RSS 107–108 MB |
| idle CPU of the .app, 60 s after the same 15 s as `tests/test_idle_cpu.py` (limit 0.13 s) | 0.010, 0.004, 0.009, 0.005 s (the 15 s before, not gated: 0.057–0.129 s); `tests/test_idle_cpu.py` from Python: 0.007, 0.026 s |
| SIGTERM | exit code 0 |
| frozen cloud direction with a fake key | Google's refusal ("API key not valid"), exit code 1: google-genai is in the bundle |
| frozen CLI: `models fetch silero-vad` over HTTPS from the bundle; `run` on the 30 s call (Parakeet + OPUS-MT + the bundled Piper runner), RTF / CPU s per audio s | 0.67 / 1.27, 0.56 / 1.01, 0.66 / 1.17, 0.70 / 1.20, 0.68 / 1.17; from Python in the same job as the last two: 0.67 / 1.24, 0.64 / 1.16 (the frozen app costs the same; runners of this type vary: the work gate gave 0.37–0.44 / 0.69–0.83) |

The base app is torch-free (ARCHITECTURE §10): a local me → them clone needs the voice pack, which is not
built yet; the 8 GB default (me → them through Gemini) does not need it.

### Menu bar, overlay, hotkey (what the runner can show)

| Check | How | Result |
|---|---|---|
| menu-bar item | the window server's on-screen windows (`CGWindowListCopyWindowInfo`) of the app's process; a screenshot | a status window (level 25) of 38×24 at y = 0; the bird drawn as a template |
| no Dock icon | `NSApp.activationPolicy` (`LSUIElement` in the .app, `setActivationPolicy:` from Python); the screenshot | accessory; no icon in the Dock |
| a window from the menu gets the focus | `NSApp.isActive`, the window's `isKeyWindow` and its place in the window list after "Settings…" | active, key, frontmost (once in 4 runs not, right after the full-screen test's Space animation, before `orderFrontRegardless` was added) |
| overlay window | `level` 25, `collectionBehavior` CanJoinAllSpaces \| Stationary \| IgnoresCycle \| FullScreenAuxiliary, `ignoresMouseEvents`, `hidesOnDeactivate` off | as set |
| overlay over another app's full-screen window | a second process shows a full-screen window (its own Space): the overlay is on screen above it, a plain window of ours is not | on screen; only for an accessory app (as a regular app the overlay stayed on the desktop's Space) |
| click-through | a synthetic click (`CGEventPost`) on the overlay | lands in the window under it |
| hotkey | Carbon `RegisterEventHotKey`, a synthetic ⌥⇧⌘K (`CGEventPost`) | fires once per press, another combo does not; the app gets it (`mode: translation is off`) |
| hotkey refusals | ⌥⌘Space (Finder search), ⌘Space (Spotlight); a combo another process holds | "is a macOS shortcut", "already taken by another application"; the old combo stays bound |
| idle with tray, overlay and wizard (`tests/test_ui.py`) | 60 s after 15 s, as `tests/test_idle_cpu.py` | 0.013 s (the 15 s before: 0.137 s); 0.116 s before the settle was added, 0.110 of it in the first quarter: AppKit's one-off work after the first show |

The runner's enabled macOS shortcuts (`CopySymbolicHotKeys`) include ⌘Space, ⌥⌘Space, ⌃⌘Space and the fn
ones (fn+Q, fn+A…); ⇧⌘Space is not one of them, hence the new default. BlackHole 2ch and 16ch 0.7.1 install
with `brew install --cask` and show up without restarting coreaudiod.

## Stage 3: the call's sound on macOS (S1)

Measured 2026-11-08 on the GitHub Actions `macos-15` runner (macOS 15.8.2, Apple M1 Virtual, 3 cores, 7 GB, no
GPU; BlackHole 2ch / 16ch 0.7.1 from Homebrew; the runner has no real output, BlackHole 2ch stands in for the
speakers / headphones and its input side is what they would play). Tests: `tests/test_macos_audio.py`.
"System Audio Recording" was given or refused by a row in the runner VM's TCC.db (`kTCCServiceAudioCapture` for
`/opt/hca/hosted-compute-agent`, the top of the job's process chain: TCC asks the responsible process; rows for the
helper, python or bash did nothing); the real dialog on a user's Mac is a USER-CHECKS line.

### Process tap (macOS 14.4+)

| check | result |
|---|---|
| helper build (`swiftc -O -target arm64-apple-macos12`) | Swift 6.1.2, ~10 s, minos 12.0 |
| permission refused | `check` → 4; the tap exits 4 at once; the app refuses the them → me side with the path to allow it |
| permission not asked yet (no row) | `tap` asks first (TCC's own call): on the runner nobody answers, it waits until stdin closes; `--no-ask` starts and gets silence |
| tap of `afplay` (440 Hz, 0.244 peak) by pid | 48 kHz mono, RMS 0.206–0.211 (the tone: 0.173 at the tap's level), 440 Hz |
| the original on the speakers: before / while tapped / after | RMS 0.212 / **0.0000** / 0.212: muted only while the tap runs |
| by bundle id, the app started after the tap | the helper waits ("waiting for … to use sound"), taps it when it plays |
| a helper process of the app (`<id>.helper`) started later | joins the tap (Core Audio's process list changes): 440 Hz and 660 Hz equally loud |
| a mono tap of an app on a 16-channel device | 1/8 as loud (the mixdown averages every channel): call apps play stereo |

### them → me through the macOS devices

`peresmeshnik run --in tests/data/call-en.ogg --from en --play "BlackHole 2ch"` with `--blackhole` (the file is
played into BlackHole 16ch, as a call app on macOS < 14.4 would) or `--tap` (afplay plays it, a process tap of
afplay is the input), the models of the working-mode gate (Parakeet int8 with partials, OPUS-MT, Piper), real time,
the queue policy of stage 2 (4 s, up to 1.4×). Voice heard = 20 ms frames above RMS 0.01 recorded on BlackHole 2ch.
Peak RSS = the run's process (`ru_maxrss`).

Two runs each (two commits, the same code paths):

| input | word recall | CPU s / audio s | peak RSS, MB | caption lag p50 / max, s | voice lag p50 / max, s | voice heard, s | clauses spoken / dropped |
|---|---|---|---|---|---|---|---|
| BlackHole 16ch | 0.82, 0.82 | 1.06, 0.92 | 1161, 1176 | 1.31 / 2.80, 0.53 / 1.66 | 2.21 / 2.98, 2.18 / 2.39 | 17.2, 17.2 | 4 / 4, 4 / 3 |
| process tap of afplay | 0.78, 0.78 | 0.83, 1.02 | 1178, 1201 | 0.70 / 3.24, 0.93 / 1.53 | 1.51 / 1.90, 1.91 / 2.49 | 18.6, 17.1 | 4 / 4, 4 / 4 |

- Both paths give subtitles and the voice at about the same cost (0.8–1.1 CPU s per audio s, 1.2 GB); the runs
  differ more than the paths: the runner's 3 cores are noisy, the partial ASR alone takes 0.40–0.52 of real time.
- Half of the clauses' voice was dropped by the 4 s policy on both paths: a 30 s monologue read without pauses,
  Russian speech longer than the English and a 3-core runner. Their subtitles are all there. On B (stage 2) the same
  policy dropped 1.8–5.4 % of the words on continuous read speech.
- The idle gate on `macos-15` with the app's start asking the helper once (`check`): 0.008–0.016 s CPU in 60 s
  (limit 0.13); the first 15 s after "idle" 0.08–0.15 s (not gated).
- **CoreML / Metal: not measured** (no Apple GPU in CI: "paravirtual device does not support Metal 2.0"); CPU only.
  USER-CHECKS has the M1 Pro run to compare.

### macOS < 14.4: the call through the app while translation is off

`PassThrough` (BlackHole 16ch → the headphones: a PortAudio input at the device's rate into a `SoundDeviceOut`),
`PERESMESHNIK_CALL_PATH=blackhole` on the runner, afplay as the call app with BlackHole 16ch as its speaker: the
tone arrives on the "headphones" (RMS 0.212 of 0.212, 440 Hz). Cost **2.2–2.7 % of a core** (the test process over
10 s, two runs): on such a Mac the idle app is not at ≈ 0 while a call plays through it (about 1.3–1.6 s CPU per
minute against the idle gate's 0.13); on macOS 14.4+ (the tap) nothing runs while translation is off.

### Headphones back while translating: PortAudio refreshed with the streams kept

The saved headphones come back (AirPods): `Controller._refresh_devices` suspends the PortAudio streams, makes
PortAudio list the devices again and resumes them; the engines and the tap stay. On the runner (no device can be
unplugged there) it is called directly while a tone goes BlackHole 16ch → `BlackHoleIn` → `Headphones` (BlackHole
2ch), recorded by another process:

| run | suspend, ms | refresh, ms | resume, ms | longest gap heard, ms |
|---|---|---|---|---|
| streams stopped one after the other | 433–450 | 7–19 | 26–40 | (the tone did not reach the test) |
| stopped at once | 251 | 11 | 58 | 363 |
| stopped at once | 381 | 11 | 45 | 469 |

- PortAudio takes 0.1–0.25 s to stop each stream on macOS (`pa_mac_core.c` polls the audio unit with
  `Pa_Sleep(100)` until it stops): the gap is about one stop plus 0.05–0.06 s, under the 0.5 s asked for (answer
  19:40 (5)), so the refresh happens while translating. With the microphone's streams (S2) stopped at the same time it
  should stay one stop long; not measured with them.
