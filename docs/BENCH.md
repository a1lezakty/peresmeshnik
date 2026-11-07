# Benchmarks (stage 0)

Scripts: [`bench/`](../bench). Raw rows: `bench/results/<host>.jsonl`; the tables below are made
by `python bench/report.py`. Measured 2026-11-07.

## Where it was measured

| Host tag | Machine | Used for |
|---|---|---|
| `macos-15` | GitHub Actions `macos-15` runner: Apple M1 (Virtual), 3 cores, 7 GB; torch MPS unavailable ("paravirtual device does not support Metal 2.0"), onnxruntime lists CoreML | the **M1** column: lower bound for a real M1 8 GB (which has 4 performance + 4 efficiency cores) |
| `vast-a-cpu4` | AMD Ryzen 5 5600X, **4 threads**, CUDA hidden (shared Vast server) | **x86 CPU** laptop stand-in |
| `vast-a-rtx3060` | same server, NVIDIA RTX 3060 12 GB | the **NVIDIA** tier |
| `vast-c-cpu8`, `vast-c-a4000` | Intel i7-13700, 8 threads / NVIDIA RTX A4000 16 GB (shared Vast server) | voice-clone TTS (too big for the other hosts) |
| `linux-x86_64-4c` | the dev server, AMD EPYC-Genoa, 4 cores, 7.6 GB (shared, at times swapping) | Gemini Live runs (network only); Piper voices; one early XTTS run (pessimistic) |

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

