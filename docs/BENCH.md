# Benchmarks (stages 0–2)

Scripts: [`bench/`](../bench). Raw rows: `bench/results/<host>.jsonl`; the tables below are made
by `python bench/report.py`. Measured 2026-11-07.

## Where it was measured

| Host tag | Machine | Used for |
|---|---|---|
| `macos-15` | GitHub Actions `macos-15` runner: Apple M1 (Virtual), 3 cores, 7 GB; torch MPS unavailable ("paravirtual device does not support Metal 2.0"), onnxruntime lists CoreML | the **M1** column: lower bound for a real M1 8 GB (which has 4 performance + 4 efficiency cores) |
| `vast-a-cpu4` | AMD Ryzen 5 5600X, **4 threads**, CUDA hidden (a shared rented server) | **x86 CPU** laptop stand-in |
| `vast-a-rtx3060` | same server, NVIDIA RTX 3060 12 GB | the **NVIDIA** tier |
| `vast-c-cpu8`, `vast-c-a4000` | Intel i7-13700, 8 threads / NVIDIA RTX A4000 16 GB (a shared rented server) | voice-clone TTS (too big for the other hosts) |
| `linux-x86_64-4c` | a VM, AMD EPYC-Genoa, 4 cores, 7.6 GB (shared, at times swapping) | Gemini Live runs (network only); Piper voices; one early XTTS run (pessimistic); stage 1 PipeWire runs |
| `vast-b-pipeline` | AMD Ryzen 9 7950X3D (a shared rented server), the app's 2 threads per engine | stage 1 pipeline: cost, quality, real-time lags |
| `ubuntu-latest`, `windows-latest` | GitHub Actions runners, 4 vCPU | stage 1 working-mode gate (with `macos-15`) |
| `i7-13700-2t`, `rtx-a4000` | the i7-13700 server above, 2 threads per run, at most 8 at once / its RTX A4000 | stage 5 call scenarios |

All CPU runs pin every thread pool (`--threads`, `OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS`) and
run under `nice`; the rented servers host other services, so absolute times can be a few percent
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
`joe`, `dmitri`; `denis` rejected for intelligibility) were measured on the 4-core VM
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
`linux-x86_64-4c` = the 4-core VM (PipeWire, pipeline, virtual mic), which was also
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
works on the 4-core VM.

**Virtual microphone** (`peresmeshnik.audio.virtual_mic_linux`, PipeWire 1.0.5 + pipewire-pulse):
a tone round trip keeps frequency, level and length; play → audio at the `Peresmeshnik Mic`
source as a call app records it: **10–61 ms** (the first chunk after the stream opens 38–61 ms,
then 10–37 ms); CPU of the output path 0.01–0.017 s per second of audio (our process + pacat).
`pacat` itself spins a core if its stdin closes while it still holds audio (0.64 s CPU for a 0.7 s
tail), so the writer closes it only after the written audio has played. The stream is opened with
`node.dont-reconnect`: PipeWire otherwise moves it to a newer device, such as the call capture sink.

**End to end** (`bench/me_them.py run --remote`, re-run 2026-11-08 on the merged local pipeline):
33 s of the "me" reader in Russian (other lines than the enrollment) → `LocalDirection` with the
app's defaults (Silero VAD, GigaAM v3 for Russian, OPUS-MT ru-en, 2 threads, on the 4-core VM)
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
  0.34–0.97 s; the 4-core VM was lightly loaded this time (load average ~1).

### Cloud: Gemini Live

Measured from the 4-core VM (Europe), Saturday 20:14–20:45 UTC, three runs per cell for
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
  against 0.59 CPU s per audio s in real time on the 7950X3D; chrF 53.5 against 53.9.
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

Measured 2026-11-08 on `linux-x86_64-4c` (a 4-core VM, shared with other work; every
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

The working-mode gate (`work-gate`, the local pipeline on the 30 s call; not touched by this stage) stayed
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
| me → them local through `app.Side` on `rtx3060-laptop` (the 7950X3D server): GigaAM + OPUS-MT + XTTS-v2 on the GPU, warmed up | 10.4 s |

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
  cross-correlation with match 0.97–1.00 at −15 dBFS through the call capture sink (the call capture,
  16 kHz), the virtual mic (recorded from "Peresmeshnik Mic") and the headphones sink's
  monitor; microphone level −23 dBFS (a test tone); silence, noise and a steady tone score < 0.1.
  Same on `ubuntu-latest` (match 1.00).
- **Wizard, every Linux page** (`tests/test_app_linux.py`, `PERESMESHNIK_E2E=1`): 35 s from the
  first page to "finish" with models present: devices chosen, self-test green, test phrase
  them → me local ("It is Martelli's / Fifth CEP in four years." → "Это Мартелли. / Пятый КЭП через
  четыре года.") and me → them cloud with local subtitles ("Это пятый серб Мартеи за четыре года." →
  "This is the fifth Martei Serb in four years.", Gemini voice 3.2 s after the start), settings saved.
  Downloads: silero-vad from Hugging Face through the models page; a cut download resumes with HTTP
  Range (`tests/test_ui.py`, local server). The project's GitHub release `models-1` answered 502 and
  timed out from this VM and from the 7950X3D server for ~30 min (11:10–11:40 UTC); OPUS-MT was taken from
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
stage 1) plus the 60 ms cushion of the switch; the queue policy works on it. With `LiveSwitch` the speech
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

Measured 2026-11-08 on `vast-b-stage2` (Ryzen 9 7950X3D, 32 threads, RTX 3060 Laptop
12 GB; every run under `nice -n 15`, up to 6 at once) and `linux-x86_64-4c` (a 4-core VM,
shared; an own PipeWire session). Scripts: [`bench/conversation.py`](../bench/conversation.py)
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
the clone's tests). **The 4-core VM** (no GPU, no clone): the real app (`peresmeshnik app --on`, Qt
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

- Both sides at once cost about the sum of each: on the 7950X3D 0.46 + 0.37 cores alone, 0.61–0.83 together
  (CPU per second of input 0.56–0.57), 3.5–3.7 GB of RAM and 2.2 GB of VRAM; here 1.12 cores of 4
  and 2.0 GB with the cloud for my side (whose local subtitles run GigaAM).
- While I talk their translated voice is cut (barge-in): with both talking all the time (overlap
  0) little of it is heard; the subtitles keep every clause. My own voice into the call is never
  dropped (no queue policy for me → them: the other side has no subtitles).
- Found on the way: the app imported numpy before the pipeline pinned the BLAS threads, so
  OpenBLAS took every core — them → me alone cost 3.0 CPU s per audio s on the 7950X3D (2.24 cores) instead
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
| the helper killed (SIGKILL) while tapping | the original comes back (RMS 0.0000 → 0.212): Core Audio removes the private tap and its device with the process; the app is told (exit −9) |
| the app's device changes its rate (BlackHole 2ch 48 → 44.1 kHz under afplay) | the tap keeps 48 kHz (Core Audio converts), 440 Hz, RMS 0.212: no restart needed; exit 7 stays for a change of the tap's own rate |
| Safari / a WKWebView playing `<audio>` | the sound is played by `com.apple.WebKit.GPU` ("Safari Graphics and Media", "web Graphics and Media"), not by Safari: the tray lists it as "Safari (WebKit)", a tap of it takes every WebKit app |

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
  Russian speech longer than the English and a 3-core runner. Their subtitles are all there. On the 7950X3D (stage 2) the same
  policy dropped 1.8–5.4 % of the words on continuous read speech.
- The idle gate on `macos-15` with the app's start asking the helper once (`check`): 0.008–0.016 s CPU in 60 s
  (limit 0.13); the first 15 s after "idle" 0.08–0.15 s (not gated).
- The working-mode gate on `macos-15` at this branch (ci.yml `work-gate`, the file fed as fast as it goes):
  RTF 0.484, 0.876 CPU s per audio s, word recall 0.84 (limits 1.0 / 2.0 / 0.6).
- **CoreML / Metal: not measured** (no Apple GPU in CI: "paravirtual device does not support Metal 2.0"); CPU only.
  USER-CHECKS has the M1 Pro run to compare.

### macOS < 14.4: the call through the app while translation is off

`PassThrough` (BlackHole 16ch → the headphones: a PortAudio input at the device's rate into a `SoundDeviceOut`),
`PERESMESHNIK_CALL_PATH=blackhole` on the runner, afplay as the call app with BlackHole 16ch as its speaker: the
tone arrives on the "headphones" (RMS 0.212 of 0.212, 440 Hz). It opens only while a call app plays into BlackHole
16ch (Core Audio's `DeviceIsRunningSomewhere`) and closes after 5 s of exact zeros (1 s in the test):

| | CPU, % of a core (the test process, 3 s) | streams |
|---|---|---|
| no call (round 1) | **0.01** | none open |
| a call playing through (round 1; before it, two runs over 10 s: 2.2–2.7) | 2.62 | open |
| the call app stopped / the next call | — | closed, then open again (2 opens in the run) |

So the idle app is at ≈ 0 without a call on macOS < 14.4 too; while a call plays through it, it costs about 1.6 s
CPU per minute. On macOS 14.4+ (the tap) nothing runs while translation is off.

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
| round 1: every registered stream of the process (2) | 278 | 6 | 60 | 341 |

- PortAudio takes 0.1–0.25 s to stop each stream on macOS (`pa_mac_core.c` polls the audio unit with
  `Pa_Sleep(100)` until it stops): the gap is about one stop plus 0.05–0.06 s. The aim is 0.5 s (answer 19:40 (5)),
  met in every run so far (341–469 ms); the test's limit is 0.8 s, as a stop is quantised by `Pa_Sleep(100)`.
- Real Core Audio notifications (round 1): an aggregate device made and destroyed by another process stands in for
  the headphones (`tests/macos/fakedevice.swift`). The saved one comes back: `Arrival`, a refresh, the output moves to
  it in 0.38 s; unplugged: `Watch` tells in 0.08 s, the default plays again. Saved headphones Core Audio has and
  PortAudio does not see even after its refresh: the default plays meanwhile. The microphone is no PortAudio
  stream on macOS (the Core Audio helper), so a refresh does not stop it (`test_the_microphone_is_kept_by_a_refresh`).

## Stage 3: me → them on macOS (BlackHole 2ch, the microphone, the clone)

Measured 2026-11-08 on the `macos-15` runner (Apple M1 Virtual, 3 cores, 7 GB, macOS 15.8, no GPU), BlackHole
2ch and 16ch 0.7.1 from Homebrew; BlackHole 16ch stands in for the real microphone (a test process plays into
it). Tests: `tests/test_virtual_mic_macos.py`, `tests/test_wizard_macos.py`.

### The helper

The macOS side of me → them is `peresmeshnik-audio` (`native/macos/audio`, Swift, one `swiftc`): Core Audio
IOProcs in one small process per stream, as parec / pacat / module-loopback on Linux — `record` (the
microphone, mixed to mono, converted by AVAudioConverter), `play` (into BlackHole 2ch, a 40 ms cushion like
pacat's), `loop` (the microphone into BlackHole 2ch while translation is off). PortAudio was not used for it:
its device list is fixed at `Pa_Initialize` (a device plugged in later needs a re-init that closes every stream
of the process, the headphones too) and it reports no device loss.

| what | value |
|---|---|
| BlackHole 2ch round trip (play → its input as a call app reads it) | 440 Hz tone: frequency, length 1.500 s and level (rms 0.212 of 0.212) kept, played in real time (3.2–3.3 s for 3 s); a flush mid-tone leaves silence (0.0000), not the old buffer |
| play → heard at BlackHole 2ch's input | first sound 128–236 ms (the player process starts), then 16–64 ms (upper bound: the reader's 10 ms chunks included; 40 ms of it is the player's cushion since 5dfe233) |
| microphone, helper's CPU | 0.011–0.022 CPU s per s (48 kHz device → 16 kHz, 20 ms frames in real time) |
| the loop while translation is off, nobody records BlackHole 2ch | 0.0000–0.0001 CPU s per s: the microphone is not open |
| the loop while the call app records BlackHole 2ch | 0.0048–0.0058 CPU s per s with a 1024-frame IO cycle (512: 0.0068); a chirp through it found, match 0.69–0.97 |
| play's own CPU | 0.001–0.004 s for 3 s of sound |

**Idle gates with all of it** (macos-15, BlackHole 2ch/16ch installed, so the app finds BlackHole at start and
runs the loop, idle): `tests/test_idle_cpu.py` 0.030 s of CPU in 60 s (the first 15 s after "idle", not gated:
0.108); `tests/test_ui.py` with tray, overlay and wizard 0.117 s in 60 s, per quarter 0.111 / 0.003 / 0.002 / 0.000
(the first quarter is AppKit's one-off; limit 0.13 each). The loop helper is set up before the app says "idle": first
its start (AVFoundation, Core Audio listeners) landed in the gated window and that gate read 0.155.

**The loop opens the microphone only during a call.** Translation off, the call must still hear the user, so
the real microphone goes into BlackHole 2ch; a loop that is always on would keep the microphone open all day:
the orange dot in the menu bar, AirPods switched to their call profile, ~0.7 % of a core. On macOS 14+ the
helper watches Core Audio's process objects (`kAudioHardwarePropertyProcessObjectList`, each process's
`IsRunningInput` and input devices) and runs only while another process records BlackHole 2ch. Below 14 (no
process objects) it stays on.

**A microphone that goes away** (an aggregate device over BlackHole 16ch made and removed by the helper stands in
for a headset): the helper's `--follow` switches to the default input at once, says so on stderr (one tray message),
and goes back to the saved device when it returns; frames flow from it again. Events seen on the runner:
`using Test Mic → lost Test Mic → using Apple Virtual Sound Device → using Test Mic`.

**Translation ⇄ live voice through BlackHole 2ch** (the app's `Controller`, a fake translation at 220 Hz, the
stand-in microphone at 300 Hz): the call app hears translation → live → translation → (translation off) the
microphone through the loop; jumps at the hotkey's seams ≤ 1.2× the tones' own with the 30 ms crossfade,
up to 11.3× without. Turning translation off while it speaks is a hard stop (22–44×), as on Linux.

**A killed helper's last buffer keeps playing.** A helper killed (SIGKILL) while its IOProc ran sometimes left its
last IO buffer in coreaudiod's mix, repeated with no process left: the call app heard a hum at 187.5 and 281.25 Hz
(the 2nd and 3rd harmonics of 93.75 Hz = one 512-frame buffer at 48 kHz, 10.7 ms, looping), in BlackHole 2ch after a
flush and in 16ch after a test's player was killed. The helpers now stop every IOProc before they exit, on SIGTERM,
end of stdin or an error, and the app ends them with SIGTERM (SIGKILL only if one is still there after 2 s).
`tests/test_blackhole_stale.py` (a writer goes while the call app reads and another writer goes on): 0 % of the
old tone afterwards, and a reader alone afterwards hears rms 0.0000. Such a buffer, once left, stays for the
machine's session, so the suite never kills a writer into BlackHole: a SIGKILL variant of that test left its 220 Hz
writer looping on 2 of 6 runners that ran the suite three times in a row, and 11 later tests on BlackHole 2ch
failed up to the end of that runner (the round trip heard 656 Hz = 7 × 93.75).

**BlackHole 2ch changes its rate** (a call app or Audio MIDI Setup, 48 → 44.1 kHz) while the translation plays
into it or the microphone is looped into it: the call app hears 440 Hz before and after, rms 0.212 (`play`
ends with 7 and `VirtualMic` opens the next one at the new rate; `loop` restarts its output itself;
`test_blackhole_changes_its_rate_the_tone_keeps_its_pitch`). Before, the output kept the old rate: a wrong
speed and pitch until a restart.

**A Core Audio listener freed under Core Audio.** The whole macOS suite died in 2 of 6 runs (SIGSEGV / SIGBUS on
Core Audio's `HALC_ShellObject_Listener Queue`): a pass-through whose `start()` failed (a call app already playing
into BlackHole 16ch, no headphones) kept its `coreaudio.Running` registered, the garbage collector later freed it
with its ctypes callback, and the next change of BlackHole 16ch called freed memory; the app would do the same on
its next call. Now one C callback serves every listener and is never freed, the listeners are registered by key
until `close()`, and `PassThrough.start()` stops itself when opening fails. Before: 4 crashes in 15 runs of the
suite; with it: none in 18 (6 runners, 3 runs in a row each), 0 listeners left registered at the end of each
(`test_a_listener_lives_while_core_audio_holds_it`).

### The clone on the M1 runner (CPU, 2 threads as the app)

torch 2.8.0 CPU wheels, coqui-tts 0.28.0, the "me" voice of the earlier stages (a LibriVox reader, ~40 s).

| | value |
|---|---|
| a sentence alone (`python -m peresmeshnik.local.clone`, 2.8–3.3 s of English) | first audio 4.15–4.31 s, RTF 5.50–5.76, peak RSS 3.81 GB |
| in the pipeline (`bench/me_them.py run --mic BlackHole16ch_UID`: 33 s of Russian, 9 clauses) | first audio of a clause 2.3–3.4 s; synthesis 2.9–3.5 s per second of speech (28 s of English took ~85 s) |
| when the call hears each clause, from its end (`voice_lag_s`) | 3.8, 10.9, 32.4, 42.5, 47.8, 48.7, 50.6, 58.9, 64.3 s: it falls behind with every clause |
| subtitles (no waiting for the voice) | 0.37–2.75 s after the clause |
| CPU | 7.3 CPU s per second of input (~2.3 of the 3 cores over the 105 s run) |
| memory | the process 3.8 GB at the peak (getrusage; 3.4 GB sampled), the system 4.4 of 7 GB used, no swap |

**Through BlackHole 2ch, end to end**: what the call app records from BlackHole 2ch transcribes like the
clone's own output — **WER 4.9 % against the MT text, the same 4.9 % as the TTS output itself** (Whisper small
as the judge): the path loses nothing. The link play → BlackHole 2ch's input
19–22 ms (the first sound 151 ms, the player starting).

**On MPS — the runner's virtual GPU, not an M1's.** torch on the runner now reports MPS (in stage 0 it did not:
"paravirtual device does not support Metal 2.0"), so the clone was tried there too (`--device mps`,
`PYTORCH_ENABLE_MPS_FALLBACK=1`): first audio 20.0 s for the first sentence (warm-up), then 7.2–7.5 s; RTF 5.0–10.9 —
no gain over the CPU (4.2 s, 5.5) on that virtual GPU. The app stays on the CPU and the 8 GB decision below rests on
the CPU numbers; a real M1's GPU (MPS) and Neural Engine are a check on a real Mac (USER-CHECKS).

### Both directions at once on the 7 GB runner (the 8 GB question)

`bench/me_them.py run --mic BlackHole16ch_UID --with-them tests/data/call-en.ogg`: as in the app, one process holds
them → me (Parakeet, OPUS-MT en-ru, Piper) and me → them (GigaAM, OPUS-MT ru-en, the clone), both fed in real time.

| | me → them alone | both at once |
|---|---|---|
| engines loaded | 30 s | 78 s |
| process RSS, peak | 3.8 GB | 3.5–3.8 GB |
| the system, used at the peak (of 7 GB) | 4.4 GB | 5.0 GB, swap 22 MB |
| clone synthesis per second of speech | 2.9–3.5 s | 4.4–5.7 s (3 cores shared) |
| my last clause heard by the call, after its end | 64 s | 115 s |
| WER of what the call app records (of the TTS output itself) | 4.9 % (4.9 %) | 9.6 % (3.6 %) |
| CPU s per second of input | 7.3 | 10.3 |

- It fits in memory: 5 GB used of 7, next to nothing swapped. An 8 GB M1 has ~1 GB more; the clone's 3.6 GB is
  most of it.
- It is not live: the clone makes speech 3–6 times slower than it is said, so the call hears me later with every
  clause (the subtitles do not wait: 0.4–3 s). With both sides the cores are full, the playback through the mic
  clock gets gaps and the recording's WER rises to 9.6 % (the TTS output itself 3.6 %).
- **Decision for 8 GB Macs** (and any Mac until a GPU path is measured): me → them in the cloud (the default,
  `config.me_to_them = 'cloud'`), or locally in a stock voice (`[local] clone = "piper"`, the wizard's voice page:
  "Speak with a stock voice (Piper), not mine"). The wizard says so when local me → them is chosen on a Mac.

**The stock voice instead** (`bench/me_them.py run --stock`: Piper `en_US-john-medium` for me → them, the same 33 s):

| | me → them alone | both at once |
|---|---|---|
| engines loaded | 4.3 s | 7.4 s |
| process RSS, peak | 0.5–1.6 GB | 1.6–2.5 GB |
| the system, used at the peak | 3.5 GB | 4.3 GB, no swap |
| TTS: first audio of a clause / per second of speech | 0.2–0.57 s / 0.11 s | 0.21–0.56 s / 0.10 s |
| the call hears each clause, after its end | 1.1–7.0 s | 1.3–7.4 s |
| WER of what the call app records (of the TTS output itself) | 6.2 % (6.2 %) | 6.2 % (6.2 %) |
| CPU s per second of input | 1.1 | 1.7 |

Live: synthesis takes a tenth of the speech. What the call waits for is the queue: the English is as long as the
Russian it translates (34 s for 33 s), so in a monologue a clause waits for the previous ones to be said (up to
7 s here), and that does not grow with time — as with the GPU clone in stage 2.

## Stage 4: the Windows shell and the installer (S3)

Measured 2026-11-08/09 on the GitHub runner `windows-latest` (Windows Server 2025, build 26100, in a VM: 4 cores,
16 GB, no GPU, no sound devices; its own desktop at 1024×768 with a light taskbar; `tests/test_windows.py` in
ci.yml, `tests/test_app_windows.py` in release.yml, several runs). Not measured: a desktop Windows 10 / 11
(SmartScreen, a user's own theme switch, high DPI), a real call, VB-Cable, CUDA under Windows (USER-CHECKS).

### The installer and the .zip

| | windows-latest |
|---|---|
| build (`packaging/windows/build.sh`: PyInstaller 6.22, the Piper runner, licenses, `zipfile`, Inno Setup 6.7 `ISCC`, lzma2/max) | 185 s (PyInstaller ≈ 100 s) |
| installed folder / installer / `.zip` (no models, no torch) | 317 MB / 104 MB / 142 MB; before Mesa's software OpenGL (`opengl32sw.dll`, 20 MB: the app never draws with OpenGL) and PortAudio's builds for other CPUs and with ASIO left the bundle: 339 / 105 / 144 MB; the rest of the standard library (it is there for the voice pack, ARCHITECTURE §10) adds 3 / 1 / 1 MB |
| silent install (`/VERYSILENT`; for the current user only): the files, a Start menu shortcut, an uninstall entry, no start at sign-in | 13.2, 12.7, 14.1, 12.2, 11.7, 9.4 s |
| installed app → "idle" (its first start: the wizard and a sample subtitle on screen) | 0.91, 0.92, 0.90, 0.78, 0.64, 0.59 s; RSS 93–94 MB |
| idle CPU of the installed app, 60 s (limit 0.13 s) | 0.000, 0.000, 0.000, 0.000, 0.016, 0.000 s |
| taskkill (a posted WM_CLOSE) of the installed app, and of the app started from its Start menu shortcut | exit code 0 |
| update over the running app (the installer asks it to quit first) | 11.1, 11.2, 12.9, 11.3, 10.0, 8.5 s; the app exits with code 0 |
| silent uninstall with the app running | the app exits with code 0; the folder, shortcut and uninstall entry are gone |
| frozen CLI from the `.zip` in a folder with spaces: `models fetch` over HTTPS; `run` on the 30 s call (Parakeet + OPUS-MT + the bundled `piper-runner.exe`), RTF / CPU s per audio s; the first run with a complete voice pack installed whose numpy fails to import (the bundle's packages come first) | 0.633 / 1.276, 0.635 / 1.288, 0.633 / 1.281, 0.525 / 1.056, 0.274 / 0.537, 0.280 / 0.557; from Python in the same jobs 0.631 / 1.293, 0.636 / 1.284, 0.631 / 1.276, 0.525 / 1.059, 0.278 / 0.544, 0.282 / 0.560: the frozen app costs the same (runners of this type vary 2×) |
| frozen cloud direction with a fake key | Google's refusal ("API key not valid"), exit code 1: google-genai is in the bundle |
| `piper-runner.exe` (windowed) on an error (a missing voice) | its traceback, exit code 1: no dialog left waiting |
| `licenses` folder | ours, the runner's and its sources; every package's; Python's (with OpenSSL, libffi, sqlite); Qt's LGPL-3.0 / GPL-3.0 (its wheels carry only the commercial option's file); Intel OpenMP (in ctranslate2) and PortAudio (in sounddevice) |

### Tray, overlay, hotkey (what the runner can show)

| Check | How | Result |
|---|---|---|
| tray icon | `Shell_NotifyIconGetRect` for the app's tray window (Qt's hidden `QTrayIconMessageWindow`, id 0); a screenshot | in the notification area, 32×48 at the clock; the same from the Start menu shortcut |
| icon under the taskbar's theme | `SystemUsesLightTheme` flipped and `WM_SETTINGCHANGE "ImmersiveColorSet"` broadcast, as Settings does | glyph lightness 1.00 on a dark taskbar, 0.00 on a light one, switched without a restart |
| overlay window | `GetWindowLongPtr(GWL_EXSTYLE)` | `0x080800a8`: TOPMOST, TRANSPARENT, TOOLWINDOW, LAYERED, NOACTIVATE (from Qt's window flags) |
| no focus taken | `GetForegroundWindow` before and after a subtitle shows | unchanged |
| click-through | a synthetic click (`SendInput`) on the overlay | lands in the window under it |
| above full-screen windows | another process shows a full-screen window, then one that is also always on top, and takes the foreground | the overlay stays above both in the z-order (`EnumWindows`); re-raised twice per test by the foreground hook |
| hotkey | `RegisterHotKey` with `MOD_NOREPEAT`, a synthetic Ctrl+Alt+Shift+K (`SendInput`), the key held (3 key-downs) | fires once per press, another combo does not; the app gets it (`mode: translation is off`) |
| hotkey refusals | Alt+Tab, Ctrl+Space, Win+Space, a bare F9, Shift+K; a combo another process holds | "switches windows in Windows", "... input method ...", "alone would take the key", "taken by another application or by Windows"; the old combo stays bound |
| which combos the runner's Windows holds | `RegisterHotKey` | Win+E, Win+D taken (Explorer); Ctrl+Alt+Space (the default), Ctrl+Shift+Space, Win+Alt+Space, Ctrl+Alt+P free |
| closing | taskkill; the wizard's own close (`SC_CLOSE`); `WM_QUERYENDSESSION` + `WM_ENDSESSION` (sign-out); Ctrl+Break | taskkill, sign-out, Ctrl+Break: exit code 0; the wizard closes and the app stays |
| idle with tray, overlay and wizard (`tests/test_ui.py`), `tests/test_idle_cpu.py` | 60 s | 0.000 s, 0.000 s |

`taskkill` posts WM_CLOSE to the app's visible windows (the overlay, the wizard) rather than to its hidden one,
and Qt does not show the app-level filter `WM_SYSCOMMAND`: a posted WM_CLOSE is told from a window's own close
by the filter seeing it twice in a row (Qt's dispatcher, then the window procedure); any WM_CLOSE to the hidden
window (the installer's) quits. A command that fails in the
windowed `.exe` (e.g. the cloud refusing a key) printed its error and then waited forever on PyInstaller's
"Unhandled exception" dialog; `packaging/entry.py` now prints it and exits with code 1 when there is a stderr.

## Stage 4: the call's sound on Windows and CUDA (S1)

### them → me on CUDA vs CPU (ASR on onnxruntime, MT on CTranslate2)

An RTX A4000 box (Linux, 1× NVIDIA RTX A4000 16 GB, CUDA 13.2; onnxruntime-gpu 1.31.1, CTranslate2 4.8.5), the 30 s
`tests/data/call-{en,ru}.ogg` fed in real time through the them → me pipeline (VAD → streaming ASR → clauses → MT),
TTS left out (the stock Piper is CPU either way, so this isolates the GPU effect on the subtitles), 6 threads. The
device is chosen automatically (`PERESMESHNIK_DEVICE=auto`): CUDA when onnxruntime and CTranslate2 both see a GPU,
else CPU. VRAM is the peak total on the GPU over the run (the container blocks per-process accounting; nothing
else ran on it), so it includes the CUDA context and cuDNN/cuBLAS (~0.3 GB).

| pair (ASR) | device | ASR partial RTF | MT RTF | subtitle lag p50 / max, s | VRAM peak, MiB |
|---|---|---|---|---|---|
| en → ru (Parakeet-tdt-0.6b-v3 int8) | CPU | 0.49 | 0.023 | 0.48 / 1.34 | — |
| en → ru (Parakeet-tdt-0.6b-v3 int8) | **CUDA** | **0.25** | **0.0034** | **0.37 / 1.15** | **633** |
| ru → en (GigaAM-v3-e2e-ctc int8) | CPU | 0.27 | 0.011 | 0.36 / 1.67 | — |
| ru → en (GigaAM-v3-e2e-ctc int8) | **CUDA** | **0.15** | **0.0025** | **0.30 / 1.61** | **477** |

- CUDA roughly halves the ASR real-time factor (the partial ASR is the heaviest them → me link) and cuts the MT to
  near nothing; the median subtitle lag drops by ~0.07–0.11 s. The clauses and their text are the same on both.
- **It fits in 6 GB.** them → me on CUDA is 0.5–0.6 GB (models int8 + the CUDA context): on a 6 GB card it leaves
  room for the clone (me → them). Both directions together in 6 GB: measured with the clone.
- onnxruntime logs "Memcpy nodes are added … for CUDAExecutionProvider" for these int8 graphs (some ops stay on the
  CPU, so there are host↔device copies): CUDA still wins, but an fp16/CUDA-native build would win more. Not pursued —
  the int8 models are shared with the CPU path and the registry.
- A CUDA that cannot start (no driver, too little VRAM) falls back to CPU with one logged message
  (`peresmeshnik.device`), and translation goes on. The built Windows `.exe` bundles CPU-only onnxruntime / CTranslate2
  and ships no CUDA runtime, so them -> me there is always CPU; **Windows + CUDA** (a from-source install) is unproven
  (no Windows GPU in CI): USER-CHECKS.

### Windows process loopback and the mute spike (the runner)

Measured on the GitHub Actions `windows-latest` runner (Windows Server 2025, build 26100, x64, 4 cores, no real
audio device). The loopback helper `native/windows/loopback` is built with MSVC (`build.cmd`, via vswhere);
VB-Audio Virtual Cable stands in for the speakers/headphones (its "CABLE Input" is the output, "CABLE Output" the
recording side), installed silently in CI. A tone-player process (`tests/windows/tone.py`, 440 Hz) is the call app.
Tests: `tests/test_windows_audio.py`.

| check | result |
|---|---|
| helper build (`build.cmd`, MSVC via vswhere) | Visual Studio 18, cl /std:c++17 /O2, one `main.cpp`; `check` → `windows=26100 loopback=yes`, exit 0 |
| process loopback of the player by pid | the tone is captured: mono float32 48 kHz, RMS > 0.05, peak 440 Hz (±10) |
| audio sessions (`list`) across every active render endpoint | the player's `python.exe` session is listed with its pid and `playing: true` (the call app may play to a non-default device: every render endpoint is scanned, not just the default) |
| mute spike, variant A | the session volume is set to 0 (read back `< 0.01`) while the loopback still has the tone (RMS > 0.05): the loopback is taken **before** the session volume, so the translation is heard and the original is not. The device output's silence on real speakers is a USER-CHECKS line (the runner cannot record the virtual cable: WASAPI capture → PaErrorCode −9996, MME → −9999) |
| WASAPI headphones (`WasapiOut`) | opens on the virtual cable, takes audio and drains it (queue → 0) with no spurious end; a device-gone notification reports it so the owner reopens |

- The process loopback captures only the target process tree (`PROCESS_LOOPBACK_MODE_INCLUDE_TARGET_PROCESS_TREE`),
  so a browser's tab processes and a call app's helpers go with it; the exe name is resolved to the tree's root pid
  (psutil) since loopback needs a pid, not a name.
- Variant A (session volume) works, so the second virtual cable (variant B) is not needed: BENCH/ARCHITECTURE.
- The helper also serves device-change notifications (`watch`, IMMNotificationClient) for the headphones, so no
  comtypes and no polling on the Python side.
- The runner has no GPU, so **Windows + CUDA** (the voice pack's onnxruntime-gpu / CTranslate2) is a USER-CHECKS line;
  the CUDA numbers above are on Linux (an RTX A4000).

## Stage 4: me → them on Windows and the voice pack on CUDA (S2)

### me → them through VB-Cable on the runner

`windows-latest` (Windows Server 2025, x64, 4 cores, no GPU, no real audio device); VB-Audio Virtual Cable installed
silently in CI (S1's step), the microphone opened for desktop apps on that VM only. The WASAPI helper
`native/windows/audio` (MSVC, `build.cmd`) plays the translation into "CABLE Input" and records "CABLE Output" as a
call app does. `bench/me_them.py run --stock` (the CI job `bench` of the temporary workflow): 33 s of Russian, 9
clauses → VAD, GigaAM, OPUS-MT, the stock voice (Piper `en_US-john-medium`; no GPU for the clone) → CABLE Input →
recorded from CABLE Output → Whisper small.

| | value |
|---|---|
| link CABLE Input → CABLE Output (5 beeps) | 83–87 ms (the first 138 ms, the player starting) |
| WER of what the call app records (of the TTS output itself) | **5.95 %** (7.1 %) |
| TTS: first audio of a clause / per second of speech | 0.09–0.31 s / 0.06 s |
| subtitles, after the clause | 0.33–2.6 s |
| the call hears each clause, after its end (median) | 1.1–7.2 s (5.2 s) |
| engines loaded / process RSS peak / CPU s per second of input | 7.1 s / 0.47 GB / 0.89 |

The cable loses nothing (the recording transcribes as well as the TTS output: the judge's noise is ±1 %). What the
call waits for is the queue — the English is as long as the Russian (34 s for 33 s), as with the stock voice on
macOS (stage 3).

### The clone in the voice pack's runner (CUDA)

The voice pack (ARCHITECTURE §10) runs the clone in a process of its own: `voice-runner/voice_runner.py` with
XTTS-v2 (`peresmeshnik/local/xtts.py`), torch 2.8.0+cu128, coqui-tts 0.28.0; the app talks to it over a pipe
(`local/clone.py` `VoicePackClone`, the piper-runner protocol). Measured on an RTX 3060 12 GB (Linux, a box shared
with another service: ≤ 4 cores, 2 threads for the engines), the "me" voice of the earlier stages.
`bench/voice_pack.py`, 3 sentences × 3 runs per language; "cap 5.4" is a 6 GB card with them → me next to the clone
(`--vram-gb 5.4`: `torch.cuda.set_per_process_memory_fraction`).

| | en | ru |
|---|---|---|
| runner up (CUDA, the model loaded, warm) | 28.8–32.5 s | |
| first audio (median / max) | 0.41 / 0.48 s | 0.44 / 0.44 s |
| RTF (median / max) | 0.43 / 0.51 | 0.49 / 0.51 |
| stop (barge-in: `stop` → the runner's last chunk) | 345–433 ms | 428–449 ms |
| VRAM peak (allocated / reserved) | 1.89 / 1.98 GB | 1.92 / 2.10 GB |
| the same with cap 5.4 | 0.42 s, RTF 0.49, 1.89 GB | 0.38 s, RTF 0.42, 1.92 GB |

- **The pipe costs nothing measurable**: XTTS in the bench process (`bench/tts_clone.py synth --device cuda`, same
  box) has first audio 0.42–0.43 s and RTF ~0.49, the runner the same. The app's process stays torch-free: its RSS
  with the clone in the runner was 0.86 GB at the peak (me → them alone), against 4.7 GB with XTTS in the process.
- **The Windows pack's pins** (`voice-runner/requirements-windows.txt`: numpy 2.5.4, librosa 1.0.0, transformers
  5.22.2, …) installed by version on Linux (`vast-a-rtx3060-pins`): first audio 0.43–0.45 s, RTF 0.50–0.53, stop
  410–442 ms, VRAM 1.89–1.90 GB — the same as the bench venv: the pinned set runs XTTS-v2 on CUDA.

### me → them end to end with the clone on CUDA, and both directions (the 6 GB question)

`bench/me_them.py run --null-mic` (no sound server on that box: a device that plays in real time into memory; the
link through real devices is measured on the runners), 33 s of Russian, 9 clauses; `--with-them
tests/data/call-en.ogg` adds them → me (CUDA) in the same process, as in the app.

| | me → them alone | both at once (cap 5.4) |
|---|---|---|
| engines loaded | 36 s | 43 s |
| clone: first audio of a clause | 0.43–0.53 s (the first 1.1 s) | 0.43–0.62 s (the first 1.2 s) |
| clone RTF in the pipeline | 0.54 | 0.60 |
| the call hears each clause, after its end (median) | 1.4–5.4 s (2.54 s) | 1.9–5.4 s (3.55 s) |
| subtitles, after the clause | 0.38–3.0 s | 0.89–3.7 s |
| WER of what the call app records (Whisper small, against the MT text) | 5.1 % | 5.0 % |
| GPU memory, the whole card at the peak (the other service ~0.44 GB of it) | — | 3.0 GB |
| app process RSS, peak | 0.86 GB | 1.96 GB |

- **6 GB VRAM: both directions fit** — ~2.6 GB of the card for the clone and them → me together, with the clone
  capped at 5.4 GB. The wizard recommends the local clone from 5.5 GB (`ui/wizard.py` `MIN_VRAM_GB`: a 6 GB card
  reports 6144 MiB, some a little less); below that, or without an NVIDIA GPU, it says why and offers the stock voice
  or the cloud. On the Windows `.exe` them → me runs on the CPU (BENCH stage 4, S1), so there the card holds only
  the clone (~2 GB).
- Live: synthesis takes half the speech; what the call waits for is the queue, as with the stock voice (stage 3).
- **Windows + an NVIDIA GPU** (the pack installed by the wizard, CUDA under Windows) is not on any runner: USER-CHECKS.

## Stage 5: cloud fallback, crash reports, latency on screen (S3)

Measured 2026-11-09. The fallback against a fake Gemini Live server (`tests/fake_gemini.py`: a real WebSocket over TLS
for the real google-genai client; its "translator" answers each block 1 s later, as a tone for speech and zeros for
silence), with the real local path (Parakeet v3 + OPUS-MT en-ru + Piper `dmitri`) on a 4-core VM (CPU),
the 30 s test call (`tests/data/call-en.ogg`) fed in real time, the fault 8 s in (`bench/fallback.py`, 3 runs each).
Not measured: a real Gemini outage in a call, a real quota (USER-CHECKS).

### Cloud → local and back

| Fault (what the server does) | the side decides (one tray message), after the fault | first local voice, after the fault | detail |
|---|---|---|---|
| connection dropped, the server not reachable (no close frame, then HTTP 503) | 0.19, 0.19, 0.20 s | 2.93, 2.94, 3.09 s | one reconnect try is the decision; the voice waits for the clause in progress to end (VAD pause, MT, Piper) |
| quota: the connection closes and the next one is refused with 1011 "RESOURCE_EXHAUSTED" | 0.19, 0.20, 0.20 s | 2.91, 3.01, 3.07 s | the reason in the tray: the lost connection, then Google's text ("1011 … quota") |
| too slow: no translated voice (`cloud.fallback_after_s` = 6 s after a phrase began) | 6.88, 6.89, 6.88 s | 6.98, 7.01, 7.13 s | the first phrase that begins after the fault gets no voice: 6 s after its start (≥ 1 s of its speech heard); its clauses so far, muted, are said at once |
| back: the server answers again 8 s after the drop | drop as above (0.19, 0.19, 0.20 s; voice 2.94–3.17 s); back in the cloud 6.41, 6.76, 14.98 s after the server answers | — | two phrase starts: at the first one (after the pause: 0 s here, 30 s doubling to 15 min in the app) a session opens with nothing sent, at the next one the phrase goes to the cloud from its start; how long that takes is how the call's phrases fall. Two messages per outage, none at close |

The 6 s limit against the cloud's real first voice after a phrase began (`first_audio_after_start` /
`audio_from_start`, every measurement kept): stage 0 p90 3.27–3.42 s (n = 30 + 30 sentences, 3 runs) and 3.40–3.41 s
(`stream_end`, 5 + 5); stage 1 p90 3.88 / 4.61 s (7 + 10); stage 2, Silero and energy gates, 3.13–4.66 s (4); app bench runs of
2026-11-08 3.69–4.15 s (3). No phrase over 6 s among the values kept one by one (max 4.66 s); the per-sentence values of stages
0–1 are kept only as median and p90. The margin over the p90 is 1.4–2.7 s, not 2×; a real slow network (USER-CHECKS)
tells whether 6 s falls back too early.

The same faults with the pipeline's fake engines and a 0.3 s "translator" (`tests/test_fallback.py`, CI on all three
OSes): drop → decision 0.20 s, quota 0.20 s, too slow 2.08 s with a 2 s limit, back 0.61 s after the phrase's first
partial subtitle on the 4-core VM. On the CI runners (PR #13):

| runner | drop | quota | too slow (limit 2 s) | back |
|---|---|---|---|---|
| ubuntu-latest | 0.19 s | 0.19 s | 2.08 s | 0.61 s |
| macos-15 | 0.28 s | 0.31 s | 2.19 s | 0.63 s |
| windows-latest | 0.19 s | 0.19 s | 2.08 s | 0.61 s |

### Crash reports and a helper that dies

| Check | Result |
|---|---|
| what was said (whole and in part), its translation, the key, device names (microphone, headphones as PipeWire names), the call app, the voice's name, the home folder: through log arguments, the latency JSON, an exception's text and a thread's unhandled exception (`tests/test_crash.py`) | none in the report or the log file; the report keeps the version, OS, CPU, RAM, GPU, engines and models, the stack (`in worker`), the log's format strings and numbers |
| a segfault (`ctypes.string_at(0)`) in the app's process | faulthandler's stack in `fault-<pid>.txt`, a report at the next start, the fault file gone |
| a side whose direction dies 4 times in a row | 3 restarts (one message each, the report named in it), then off with "failed 4 times within a minute"; 4 reports; the app goes on |
| the Piper runner (a real helper process) killed while them → me translates the 30 s call | only the runner starts again (`crash.Respawn`), one message, one report ("the voice helper died"); the side speaks again 5.6, 5.7, 6.0, 6.2 s after the kill on the dev server (the next clause of the call; most of it is waiting for that clause), 5.8–10.0 s on the CI runners (work-gate job, 3 runs each: ubuntu 9.5, 9.7, 5.8 s; macOS 9.7, 10.0, 9.2 s; windows 8.7, 7.6, 7.6 s). Before (the whole side made again: new ASR, MT, runner and warm-up) 15.1–16.3 s there, 13.2–21.3 s on the CI runners |
| GitHub issue link | under 8000 characters; a long report is cut there and copied whole to the clipboard |
| idle CPU with the hooks and the log file (`tests/test_idle_cpu.py`, the 4-core VM under Xvfb) | 0.010 s in 60 s (limit 0.13 s) |

### taskkill right after the start (Windows)

`tests/test_app_windows.py::test_started_from_the_start_menu_shortcut` failed now and then (once in the stage-4 runs):
the app did not quit within 20 s of taskkill. Repeated (`bench/windows_close.py`, the installed app started from its
Start menu shortcut, taskkill as soon as its tray icon is there, the test's look every 0.5 s) on `windows-latest`:

| Build | runs that ended with exit code 0 within 20 s |
|---|---|
| main (bac8639) | 20 of 25, 30 of 40, 30 of 40: **80 of 105** |
| this stage | 40 of 40, 40 of 40, 40 of 40: **120 of 120** (one run hit the case below and quit) |

Every failure: no window was on screen yet (the wizard not shown), and taskkill posted its WM_CLOSE to a hidden window —
Qt's `_q_titlebar` (the wizard's title bar metrics), which is no QWindow: the filter saw that WM_CLOSE once (from Qt's
dispatcher only), took it for a window's own close, the helper window was destroyed and the app stayed, with the wizard.
A WM_CLOSE to a window without Qt's window procedure now quits (none of them has an X);
`test_wm_close_to_a_hidden_helper_of_qt_quits` posts one to `_q_titlebar`. With a look every 0.05 s taskkill comes
before the wizard is even built and goes to the tray's window, which quit on both builds (40 of 40 each).

## Stage 5: latency per link, long phrases, two people, echo (S1)

Measured 2026-11-09. Nobody uses the app live yet, so a call is rebuilt from recordings (`bench/call.py`): FLEURS dev
(CC-BY-4.0) takes of different people at −26 dBFS glued into the sound of a call: **echo** (we speak into the call,
the far end is on a loudspeaker: our speech comes back through a small speaker's band, a room of ~0.2 s and the
given delay and level; they answer, a third of the time over us), **overlap** (two people at the far end: turns that
overlap 0.3–1.5 s, and some that start together), **long** (one person, sentences back to back with every silence
cut to 0.15 s: no pause the VAD takes). A scenario is a seed; only numbers come here. Engines as in the app: Silero
VAD, Parakeet v3 int8, OPUS-MT en-ru, Piper `dmitri`, 2 threads, the app's queue policy (cap 4 s) where the voice is
measured. **i7** = a 16-core x86 server (Intel i7-13700; runs with 2 threads each, at most 8 threads at once; CPU unless said);
**runners** = the CI job work-gate (`tests/test_call.py` on the recordings of `tests/data`: 3 of our sentences, 3 of
theirs). Real calls, a real loudspeaker at the far end, real people over each other: USER-CHECKS «Этап 5 / S1».

### Latency per link (them → me)

**links** on the i7: the echo scenario (n = 4: 24 of our utterances, 24 of theirs) through the gate, in real time with
Piper and the queue policy; *runners*: `test_latency_per_link` (3 + 3 sentences, one scenario). Seconds from the end
of a clause's speech (where the clause ends in the audio) to: *VAD* — the end is known (a pause of 0.3 s, or the ASR
cuts the clause); *ASR* — its final text; *caption* — its translated subtitle; *TTS* — the first synthesized audio;
*voice* — its voice starts playing (after the voice of the clauses before it). The budget is ARCHITECTURE §4 (a short
clause, nothing queued); the stage's target: subtitle ≤ 0.5 s, voice ≤ 1.5 s from the clause end.

| where | clauses (dropped) | VAD p50 / p90 | ASR | caption | TTS first | voice |
|---|---|---|---|---|---|---|
| budget (§4, x86 / M1) | | 0.30 | ≈ 0.43 | ≈ 0.5 / 0.55–0.65 | ≈ 0.65 / 0.8–0.9 | target ≤ 1.5 |
| i7, CPU, 2 threads | 80 (3) | 0.36 / 1.42 | 0.40 / 1.42 | **0.43** / 1.47 | 0.50 / 1.59 | **1.18** / 2.54 |
| RTX A4000 (ASR and MT on CUDA, Piper on the CPU) | 82 (5) | 0.37 / 1.47 | 0.43 / 1.47 | **0.43** / 1.48 | 0.50 / 1.53 | **1.13** / 2.47 |
| runner Linux (4 vCPU) | 12 (2) | 0.41 / 1.34 | 0.48 / 1.34 | 0.55 / 1.55 | 0.53 / 2.04 | 0.53 / 2.04 |
| runner macOS (M1, 3 cores) | 12 (2) | 0.32 / 0.99 | 0.38 / 0.99 | 0.40 / 1.28 | 0.49 / 1.69 | 0.49 / 1.69 |
| runner Windows (4 vCPU) | 12 (2) | 0.44 / 1.40 | 0.56 / 1.40 | 0.63 / 1.70 | 0.66 / 2.27 | 0.66 / 2.40 |

- Medians: the subtitle is within the target on the i7 and the M1 runner (0.40–0.43 s), 0.55–0.63 s on the shared 4 vCPU
  Linux and Windows runners; once the end is known, the final ASR and MT add 0.07–0.19 s to it. *Dropped* = the
  queue policy skipped the voice (its subtitle shown); a dropped clause has no TTS or voice number.
- **The worst link is the end of a clause without a pause.** By how the clause ended (i7, CPU): a pause (58 of
  80) VAD 0.29 / 0.48 s; the 5 s limit (19) 1.27 / 1.87 s; a sentence end the ASR punctuates (3) 1.20 / 1.45 s. A
  cut takes only words that start ≥ `TAIL_S` before the window end (there the model squeezes the last words
  together; 0.8 s since stage 1), so such a clause is known ≥ `TAIL_S` after its last word.
- **The A4000 changes nothing here** (within 0.07 s of the CPU): the links are bound by when the clause end is
  known and by the voice queue, not by compute. MT RTF 0.0068 → 0.0019 on CUDA; the partial ASR (int8 graph, host ↔
  device copies, see "them → me on CUDA vs CPU") 0.20 on 2 CPU threads, 0.22 on CUDA.
- The voice p50 (1.18 s on the i7) is the queue: their sentences are 5–12 s (p10–p90, median 8 s), cut into
  clauses back to back, and a clause's voice waits for the one before it (its TTS first audio is there at 0.5 s).
  The p90 above the target (2.5 s on the i7) is that wait plus the clauses known late (above). On the runners (one
  scenario, 12 clauses) the voice p50 is its TTS first audio.

**`TAIL_S` 0.5 measured, not taken.** Both values on the i7 at the same time (2 threads each), English (Parakeet) and
Russian (GigaAM → English, Piper `john`): **links** n = 4 as above, in real time; **long** n = 6 (the monologue of
"Long phrases", 30 sentences, 27–37 clauses cut at the 5 s limit) in real time and, for the words alone, fed as fast
as it goes (`--asap`: the same on any load). *Repeated* = words a clause repeats from the end of the one before it
(a cut whose window timestamps ran ahead); `REPEAT_WORDS` = how many such words the next clause may drop (1 before
stage 5).

| | real time (links): VAD p90 / caption p90 / 5 s limit VAD p50 | real time (long): WER / chrF, run 1; run 2 | as fast as it goes (long): WER / chrF / repeated |
|---|---|---|---|
| en, `TAIL_S` 0.8, `REPEAT_WORDS` 1 | 1.53 / 1.57 / 1.38 | 0.1285 / 70.0; 0.1268 / 70.1 | 0.1447 / 70.1 / 5 |
| **en, 0.8, 3 (now)** | — | — | **0.1366** / 70.1 / **0** |
| en, 0.5, 1 | 1.13 / 1.25 / 0.99 | 0.1252 / 70.2; — | 0.1447 / 68.6 / 0 |
| en, 0.5, 3 | — | —; 0.1528 / 68.0 | 0.1447 / 68.6 / 0 |
| ru, 0.8, 1 | 1.28 / 1.36 / 1.21 | 0.0893 / 72.8; 0.0821 / 72.7 | 0.0839 / 73.0 / 0 |
| **ru, 0.8, 3 (now)** | — | — | 0.0839 / 73.0 / 0 |
| ru, 0.5, 1 | 0.86 / 0.95 / 0.85 | 0.0839 / 73.5; — | 0.0857 / 74.2 / 0 |
| ru, 0.5, 3 | — | —; 0.0875 / 72.5 | 0.0857 / 74.2 / 0 |

- `TAIL_S` 0.5 knows a cut clause 0.35–0.4 s earlier (the subtitle p90 −0.3–0.4 s, the median as before). The
  words: in real time the English WER with 0.5 moved by 0.028 between two runs (0.1252, 0.1528; `REPEAT_WORDS` makes
  no difference there when fed fast), with 0.8 by 0.002–0.007: the cuts move with the load. Fed as fast as it goes,
  0.5 is no better in English (chrF −1.5) and mixed in Russian (WER +0.002, chrF +1.2). Not taken: the gain is only
  in the p90, and the quality is not shown to hold.
- **Taken: `REPEAT_WORDS` 3** (`local/asr.py`): a clause that repeats up to 3 words of the cut before it ("…to cope
  with" | "cope with the impacts": `test_two_people_overlapping_ends` with 0.5 on a 4-core EPYC VM) drops them; with 0.8 the English WER
  0.1447 → 0.1366 (5 repeated words → 0), Russian the same.
- 2026-11-09: these "as fast as it goes" rows were measured before `00509b7` (a cut's words are dropped only when the
  next window did not hear them before the cut). Again on main `5918c0d` (rows `main-asap-*`): en 3 → 0.1366 / 70.1 / 0,
  1 → 0.1447 / 70.1 / 5; ru 0.0839 / 73.0 / 0 with both: the same numbers.

### Echo of our speech from the far end

**echo** on the i7, n = 6 (36 of our utterances and 36 of theirs per row, a third of their turns over our echo), fed as
fast as it goes, the delay searched up to 1.0 s (2.0 s now: the same within 0.2 %, below); the gate runs in the scenario's own time with the reference pushed as it is sent. *echo translated*
= our utterances that come out as a clause (≥ 2 and ≥ 25 % of their content words); *their recall* = their content
words out; *zeroed* = 16 ms blocks the gate zeroed, of their speech (false suppression) and of our echo alone.

| far end: delay, level | echo translated: no gate → gate | their recall: no gate → gate | their blocks zeroed | echo blocks zeroed |
|---|---|---|---|---|
| 50 ms, −12 dB | 32 of 36 (89 %) → 1 (2.8 %) | 0.880 → 0.892 | 0.12 % | 88.5 % |
| 150 ms, −6 dB | 34 (94 %) → 1 (2.8 %) | 0.899 → 0.899 | 0.15 % | 87.3 % |
| 150 ms, −12 dB | 32 (89 %) → 0 | 0.892 → 0.899 | 0.11 % | 88.3 % |
| 150 ms, −20 dB | 24 (67 %) → 0 | 0.879 → 0.901 | 0.09 % | 90.0 % |
| 300 ms, −12 dB | 32 (89 %) → 0 | 0.882 → 0.911 | 0.13 % | 94.0 % |
| no echo path (a headset), we talk, they talk over us | — | 0.913 → 0.913 | 0 | — |

- Their words are not lost to the gate (0.1 % of their blocks, recall the same or higher: the echo no longer mixes
  into their clauses). The 10–13 % of the echo blocks it leaves include the echo under their speech (double talk
  passes by design); in 34–36 of 36 utterances that is too little for a clause.
- **Through the devices** (`tests/test_echo_pipewire.py`, Linux: `VirtualMic` through `Referenced` → a far end that
  records "Peresmeshnik Mic", plays it back 150 ms later 12 dB quieter between its own sentences into "Peresmeshnik
  Call" → the capture → `Threaded` gate): echo zeroed 90–100 % per phrase, their speech 0 %, on a 4-core EPYC VM and
  the ubuntu runner. Before the reference kept speech queued ahead (a TTS hands out seconds at once) it was 46 %.
- Runners (`test_echo_is_not_translated`, 3 + 3 sentences, one over our echo): echo translated 2 of 3 → 0, their
  recall 0.871 → 0.871 (Linux, macOS, Windows). Without models (`tests/test_echo.py`, every OS): echo blocks zeroed
  94–100 % at 50/150/300 ms and −6/−20 dB, their blocks 0 %, double talk 0.2 %, no echo path 0.
- CPU: nothing while the reference is silent; while it sounds, per 16 ms block on one core of a 4-core EPYC VM
  (`echo_max_delay_s`, the delays searched): 1.0 s — 0.21 ms (1.3 % of a core; it was 1.35 ms before the delay
  search went to every 4th block), 1.5 s — 0.28 ms, **2.0 s (the app's) — 0.34 ms (2.1 %)**.
- **A long round trip** (two networks, two jitter buffers, device buffers): with the search up to 2 s the echo at
  0.6 / 0.9 / 1.4 / 1.9 s is zeroed 96–99 % (it was 0 beyond 1.0 s), their speech 0 %; at 2.5 s (beyond) the echo
  passes and nothing else is zeroed (`tests/test_echo.py`, 3 + 3 sentences). Frames of 100 / 160 / 320 / 512
  samples: 96.5–97.3 % (a frame shorter than a block, as BlackHole at 96 kHz gives, was 2–21 % before the fix).
  Double talk over an echo at −6 / −12 dB: 0.29 / 0.19 % of their blocks zeroed.

### Long phrases: translated partials

**long** on the i7, n = 6 (30 sentences, 6 monologues of ~45 s), real time. *subtitle* = a word → its translation on
screen (the first translated partial that has it, or its clause's final caption); *caption* = a word → its clause's
final caption (before stage 5 the only translation on screen); *first word* = a clause's first word → its first
translated subtitle; *voice* = a word → its clause's voice starts; words spread evenly over their clause. WER of the
clauses against the sentences, chrF of the clauses' translations against the MT of whole sentences
(pseudo-reference). `early` = a clause is also cut at a comma from that length (≥ 1 s before the comma). `TAIL_S`
0.8, as in the app ("Latency per link").

| variant | clauses | subtitle p50 / p90, s | caption p50 / p90, s | first word, s | voice p50 / p90, s | WER | chrF | dropped | MT RTF (+ partials) |
|---|---|---|---|---|---|---|---|---|---|
| before (stage 4) | 78 | 2.54 / 4.59 | 2.54 / 4.59 | 3.61 | 3.24 / 5.06 | 0.127 | 71.4 | 5 | 0.016 |
| **translated partials (now)** | 78 | **1.42 / 2.92** | 2.56 / 4.64 | **2.02** | 3.19 / 5.08 | 0.127 | 70.0 | 5 | 0.016 + 0.025 |
| + early 2.0 s | 86 | 1.47 / 3.05 | 2.50 / 4.57 | 2.03 | 3.22 / 4.99 | 0.153 | 68.1 | 7 | 0.017 + 0.024 |
| + early 2.5 s | 86 | 1.43 / 2.99 | 2.52 / 4.51 | 2.03 | 3.04 / 4.89 | 0.155 | 65.9 | 9 | 0.017 + 0.023 |
| + early 3.0 s | 84 | 1.48 / 3.10 | 2.57 / 4.64 | 2.04 | 3.16 / 5.11 | 0.143 | 68.3 | 6 | 0.017 + 0.022 |
| + early 2.5 s, max clause 4 s | 97 | 1.42 / 2.80 | 2.20 / 3.67 | 1.97 | 2.75 / 4.06 | 0.159 | 66.9 | 1 | 0.017 + 0.019 |

- The translated partials take a word's translation on screen from 2.5 to 1.4 s (p90 4.6 → 2.9 s) at the same WER,
  and they do not change the final text: fed as fast as it goes (`--asap`, the same on any load) with and without them
  WER 0.1366 = 0.1366, chrF 70.1 = 70.1. In real time the cuts move with the load: three runs of each, 4 at once,
  without 68.4 / 69.9 / 71.5, with 68.1 / 69.7 / 68.9 (WER 0.112–0.148 and 0.129–0.146), so the single 71.4 → 70.0
  above is within one setting's spread (3.1). They cost 2–2.5 % of a core.
- Cutting clauses earlier for the voice gains it 0.05–0.2 s (a comma from 2–3 s) or 0.5 s (a 4 s limit) and costs
  chrF: fed as fast as it goes 70.1 → 66.7 (a comma from 2.0 or 2.5 s), 67.1 (3.0 s), 65.1 (2.5 s and a 4 s limit),
  WER 0.1366 → 0.1463 / 0.1285 / 0.1675 (fragments): not taken. The voice of a monologue is bound by its clauses,
  not by a link.
- Runners (`test_long_phrases_get_earlier_subtitles`, 4 sentences, real time): subtitle p50 before → after
  3.16 → 1.41 s (Windows), 2.99 → 1.29 s (macOS M1), 3.00 → 1.37 s (Linux), WER 0.065 = 0.065 (Windows: 0.081 →
  0.065; the clauses cut differently in real time).

### Two people at once (the far end)

**overlap** on the i7, n = 6 (48 utterances: per scenario 4 pairs where one starts 0.3–1.5 s before the other ends, 4
pairs that start within 0.5 s of each other), fed as fast as it goes; *with voice* = n = 4 in real time with Piper
and the queue policy. *lost* = an utterance with < 50 % of its content words out; *merged* = a clause with ≥ 2
content words of each of two people; *empty* = clauses the ASR returned nothing for.

| run | utterances | lost | over the other's end: lost / recall | both at once: lost / recall | merged clauses | empty clauses |
|---|---|---|---|---|---|---|
| captions | 48 | 23 (48 %) | 1 of 24 / 0.88 | 22 of 24 / 0.24 | 0 of 130 | 34 |
| with voice | 32 | 14 (44 %); in the voice 14 | 1 of 16 / 0.88 | 13 of 16 / 0.26 | 0 of 65 | 6 |

- Where one only overlaps the other's end, both are out (0.88 of the words; the overlap itself is lost). Where both
  talk at once the call gives one mixed channel and the ASR returns a mix of both or nothing: lost.
- The voice queue loses nobody beyond the subtitles (9 of 65 clauses not said, their subtitles shown; no utterance
  goes silent only in the voice): the stage 2 policy stays.
- Runners (`test_two_people_overlapping_ends`, 6 turns each 0.5–1 s over the previous one's end): recall of every
  utterance 0.75 (Linux), 0.6 (macOS, Windows), the others 0.8–1.0.

**Spike: a separation model in front of the pipeline** (`bench/separation.py`). TIGER (Xu, Li et al., ICLR 2025;
Apache-2.0, `JusperLee/TIGER-speech`, 0.8 M parameters, 16 kHz, trained on EchoSet with reverberation and noise)
splits the call into two streams (10 s per call of the model, the streams may swap at a seam); each goes through the
pipeline as its own direction, the clauses of both are scored together. Same 6 scenarios, 1 thread (so the mix row
differs a little from the table above).

| input | lost | over the other's end: lost / recall | both at once: lost / recall | merged | duplicated words |
|---|---|---|---|---|---|
| the mix (now) | 21 of 48 (44 %) | 1 of 24 / 0.89 | 20 of 24 / 0.29 | 0 | — |
| TIGER, two streams | **8 of 48 (17 %)** | 0 of 24 / 0.91 | **8 of 24 / 0.62** | 0 | 0.7 % |

Cost: RTF 0.06 on the A4000; on the CPU (torch, 2 threads) RTF 2.55 and 2 GB peak RSS: not live without a GPU. Not
in the app: it would need an ONNX export, the model only on CUDA, a second ASR stream only while the second stream
carries speech at the same time as the first, and two subtitle rows (backlog "NVIDIA: two people at once").


### macOS: the call app after a killed tap helper

GitHub `macos-15`, afplay (a 440 Hz tone) as the call app on BlackHole 2ch, tapped by `peresmeshnik-tap`, the
helper SIGKILLed (it cannot remove its tap; Core Audio is meant to remove it with the process). *Heard* = the app's
tone back on its speakers.

| when the helper is killed | heard again within 2 s | then: a new tap of it, removed | the app started again | coreaudiod restarted, the same app |
|---|---|---|---|---|
| first in a job (a fresh coreaudiod) | 5 of 5 | — | — | — |
| after the whole macOS test set | 1 of 10 | 0 of 5 (a sound of rms 0.21, not the tone) | 5 of 5 | 5 of 5 (4 of them muted before) |

A new tap does not lift it and the app has no rights to restart coreaudiod: the reason it shows now says to restart the call
app or `sudo killall coreaudiod` (README). In CI the SIGKILL test now runs in its own step first and coreaudiod is
restarted after it: the rest of the macOS set passed in 10 of 10 runs; with the test in its old place 5 of 6 (one
device test after it failed; an earlier CI run had 4 such failures).

## Stage 5: the me → them voice as the call hears it, the echo in my microphone (S2)

Measured 2026-11-09. Nobody uses the app live yet: my side of a call is rebuilt from recordings. **My turns**: 30
FLEURS ru dev sentences (other people, CC-BY-4.0) at −26 dBFS with pauses of 0.6–1.2 s (367 s), fed in real time
through the app's path: Silero VAD, GigaAM v3, OPUS-MT ru-en, XTTS-v2 in the process (`XttsClone`, CUDA, 2 threads),
LiveSwitch, a device that plays into memory in real time — the stream the virtual mic gets, at 24 kHz, with where each
clause and each TTS chunk landed (`bench/me_voice.py run`). Scored by `bench/me_voice.py score`: ECAPA (SpeechBrain)
against the voice's recording, Whisper-small CER against the clause's text (and the words heard after it: a *tail*),
BS.1770 loudness (pyloudnorm). Hosts: **A4000** = an RTX A4000 (16 GB, its GPU shared with another job), the final rows;
RTX 3060 12 GB (a second host, the same picture). The voice: the "me" profile of the earlier stages (a LibriVox reader).

### The clone's voice in the stream (A4000, 90 clauses, before = main f077e58)

The model: XTTS-v2 of the registry (`models.toml`, `model.pth` sha256 `c7ea2000…`), fetched and verified. coqui-tts
0.28 rewrites `model.pth` in place at its first load (its `XttsConfig` object stored as a dict, `speakers = []`), so
every run here read a file of sha256 `3077a05b…` (the first one on the A4000 read the registry's and rewrote it; the
rows' `model_sha256`). Against the registry's file, checked on the RTX 3060: 963 of 963 weight tensors bit-identical,
the config's 94 keys the same. The app's clone rewrites it the same way at its first load.

| | before | after |
|---|---|---|
| a clause starts with a step (> −40 dB under its peak: a click) | **57 of 90** (the step −33…−10 dB) | **0** |
| a click where two clauses meet back to back | 15 of 36 | 0 of 42 |
| inside a clause, where TTS chunks meet (> 2 × the signal's own steps) | 0 of 293 | 0 |
| a hole inside a clause (its next audio late) | 0 | 1 (0.09 s) |
| pause after a sentence, clauses back to back (p50, min) | 0.15 s (0.11) | **0.36 s** (0.34) |
| pause inside a sentence (a clause cut at 5 s), back to back | 0.15 s | 0.17 s |
| loudness of a clause against the turn of mine it came from (p50, \|…\| p90) | **+7.7 LU** (8.7) | **+0.1 LU** (2.9) |
| clause loudness (p50; p10–p90) | −17.7 LUFS (1.7 LU) | −25.1 LUFS (4.3 LU: it follows my level, which moves between the 30 people of the scenario) |
| peak of the stream | 0.972 | 0.62 |
| ECAPA similarity to my voice (mean, p10) | 0.354 (0.284) | 0.356 (0.277) |
| CER (Whisper small) | 2.24 % | 2.86 % (see below) |
| clauses with a tail | 5 | 2 |
| voice after the clause's end, p50 / p90 | 1.96 / 3.21 s | 1.72 / 3.25 s |
| clone's first audio, p50 (a GPU shared with another job: it varies) | 224 ms | 166 ms |

The code after review round 1 (`Level`: power mean, noise floor, the speech's over ~3 s; the eased limiter: below), on
the RTX 3060 (row `review1b`): clause against my turn −0.13 LU p50, |…| p90 1.8 LU; clause loudness −25.8 LUFS (p10–p90
2.3 LU); peak 0.54; clicks 0 / 0 / 0; pause after a sentence back to back 0.35 s (min 0.35); 2 tails; ECAPA 0.355;
CER 3.0 %; 1 hole (0.08 s).

- The RTX 3060 (main): a click at 46 of 90 clause starts, 37 back to back with 0.16 s after a sentence (min 0.09),
  +7.6 LU against my turn, peak 0.978, ECAPA 0.363, CER 2.45 %, 3 tails, voice p50 / p90 1.70 / 3.22 s: the same picture.
- **The clicks are XTTS's own**: its audio starts mid-waveform. `xtts.shape` fades each generation in over 5 ms. The
  seams inside a clause were clean already (coqui crossfades its stream chunks).
- **Pauses**: XTTS ends a sentence with ~0.16 s of silence, so clauses queued back to back (36–42 of 89 here: a long
  turn is translated while the one before it is still said) ran sentences together. After a sentence the clone now pads
  its silence to 0.35 s; inside a sentence it stays short. It costs only queued clauses: the voice p90 +0.04 s here
  (+0.15 s in the run before the comma below; the p50 moves with the run, ±0.25 s).
- **Loudness**: the clone (and Piper, below) spoke ~8 dB louder than a microphone at a call's level with peaks near
  full scale, so the hotkey's crossfade jumped. LiveSwitch now brings the speech to my live voice's level (`mic.Level`
  of the microphone's 32 ms frames and the speech's: a power mean, pauses and room noise not counted; gain within −30…+10 dB,
  peaks ≤ 0.95). In this
  scenario each turn is another person, so the gain follows a level that jumps between turns (the |…| p90 2.9 LU is the
  first clause of a new person); one speaker on one microphone is steadier.
- **CER 2.24 → 2.86 %** is the sampling of XTTS, not the changes: the difference is in short names and fragments that
  XTTS says differently each time ("Vlas." → "class" / "where's the sky", "Canyitas."), and "8%." heard as "8 percent"
  counts as an error; the first word is wrong in 7 and 9 of 90. The same audio of the *before* run with only the 5 ms
  fade-in applied: CER 2.24 → 2.46 %, all 8 more edits in one clause and at its end (Whisper's beam on a changed input;
  with a 2 ms fade 4 clauses go the other way), the first word heard the same in 89 of 90 ("health care" /
  "healthcare"), tails 5 = 5: the fade does not eat the first sound.

### XTTS's tails: how a cut clause ends

The clause texts of the *before* run said again, 3 times each (A4000): 56 that end a sentence as they are, the 34 that
end without a sentence's punctuation (cut at 5 s or at a comma) as they are, with a comma, with a period.

| text | syntheses | with a tail (words after the text) | CER |
|---|---|---|---|
| sentences, as they are | 168 | 6 (some are the judge: "8%." → "8 percent") | 3.71 % |
| fragments, as they are ("… opposed to the") | 102 | 6 ("… the vile", "… and then") | 1.86 % |
| fragments + "," | 102 | **0** | 1.70 % |
| fragments + "." | 102 | 6 | 2.30 % |

A fragment that ends in a function word makes XTTS go on with a word of its own; a comma stops it (a period does not,
and gives a sentence's falling end). `xtts.closed`: a clause cut mid-sentence ends with a comma for the clone.

### The stock voice through the app's path (CI runners)

`tests/test_me_side.py::test_my_voice_at_the_live_level` (work-gate, every OS): my turns (`tests/data/call-en.ogg`,
30 s, English) in real time through LocalDirection (Parakeet, OPUS-MT en-ru, Piper `dmitri`) and LiveSwitch into a
device in memory; level = power mean of 20 ms frames above −45 dBFS.

| runner | my voice | Piper as it comes | out to the virtual mic | peak, as it comes → out |
|---|---|---|---|---|
| ubuntu-latest | −24.0 dB | −17.1 dB | −25.0 dB | 1.00 → 0.54 |
| macos-15 | −24.0 dB | −17.3 dB | −24.8 dB | 1.00 → 0.47 |
| windows-latest | −24.0 dB | −17.4 dB | −25.0 dB | 1.00 → 0.50 |
| the 4-core VM | −24.0 dB | −17.6 dB | −24.6 dB | 1.00 → 0.48 |
| ubuntu-latest, review round 1 (`Level`: power mean, noise floor, the speech's over ~3 s; ci on ddb4936) | −24.0 dB | −17.4 dB | −24.0 dB | 1.00 → 0.55 |
| macos-15, review round 1 | −24.0 dB | −17.7 dB | −23.5 dB | 1.00 → 0.61 |
| windows-latest, review round 1 | −24.0 dB | −17.3 dB | −23.7 dB | 1.00 → 0.54 |
| the 4-core VM, review round 1 | −24.0 dB | −17.3 dB | −23.8 dB | 1.00 → 0.56 |

### The level: room noise before my first word, the cloud's short chunks, a new voice (review, round 1)

Translation goes on and the other side speaks first, so my microphone carries only the room's noise for a minute:
`mic.Level` counted every frame above −60 dBFS, learnt the noise as my level and sent my first clauses 13–18 dB quiet
(LiveSwitch is made anew each time translation goes on). Now a microphone frame counts only 12 dB above the noise
floor (the quietest frame of the last 3 s) and LEVEL_DB (−26) stands until 1 s of such speech. The gain held down by a
peak dropped at once at the chunk's start; now it eases down over ≤ 5 ms of the speech queued before the chunk and comes
back at 20 dB/s. A voice > 15 dB quieter than the level was never followed; now 1 s of it starts the level over (the
clone's own soft runs reach 0.67 s: 0 restarts in its 6 min stream). The level is now the frames' power mean, not
their mean in dB: with the floor's cut the mean in dB of a real microphone (`call-en.ogg`, its floor −51…−58 dB) rose
1–3 dB against the speech's, and Piper went out 2.5 dB over my voice in the test above (−21.5 against −24.0). And
the speech's level now follows over ~3 s of speech, as the microphone's (it was ~1 s): Piper says a clause in one
chunk, and a level with a 1 s memory, read at the chunk's end, was that of its quieter last second (−19.7 against the
chunk's −18.0): Piper went out 1.7–1.8 dB over my voice on the CI runners; now 0.0–0.5 dB (table above). The price: a
louder new voice is followed in 2.6 s, not 1.6 s (below).
`tests/test_virtual_mic_live.py`, no models: speech = 0.6 s syllables and 0.16 s pauses down to the room's noise; the
translation is noise at −17 dB (the cloud's chunks at −30 dB); "before" is 295a7eb's `mic.py` on the same signals.

| scenario | before | after |
|---|---|---|
| a minute of room noise at −55 / −50 dB, 2 s of me at −26 dB, then the first clause | **−43.6 / −40.5 dB** | −25.7 / −25.5 dB |
| the same with me at −18 dB | −40.6 / −37.5 dB | −17.9 / −17.8 dB |
| the cloud's 20–40 ms chunks at the +10 dB gain, a peak 2 samples into every third: the gain's biggest step from one sample to the next | **6.0 dB** (at a seam) | 0.07 dB |
| … the gain's swing after the start | 6.0 dB | 1.4 dB |
| a new voice in the same LiveSwitch (the cloud's fallback), 12 dB louder: back within 2 dB of my level after | 1.6 s | 2.6 s |
| … 16 dB quieter | **never** (−41.8 dB after 4 s) | 1.7 s |

The cloud's own level through LiveSwitch is not measured (this machine has no Gemini key): USER-CHECKS.

### The echo of the translation in my microphone (speakers, no headphones)

`bench/mic_echo.py` on the 4-core VM, n = 6 (per row 36 utterances from the speakers and 36 of mine; I answer 0.6–1.2 s
after the speakers stop, every third time 0.5–1.5 s before they stop). The speakers say their translation (Piper
`dmitri` reading FLEURS ru sentences); the microphone hears it through laptop speakers (200–8000 Hz), a room (a few
reflections and a tail of `rt60`), `delay` after `play()` and at `gain` against my own voice; my turns are FLEURS ru
recordings at −26 dBFS. Through the me → them side's subtitles (GigaAM, OPUS-MT ru-en): *echo translated* = speaker
utterances that come out as a clause (≥ 2 and ≥ 25 % of their content words); *false barge-ins* = clauses of ≥ 3 words
made of the echo alone (my partial subtitle that long cuts the translation I hear); *my recall* = my content words out.
The gate is S1's (`audio/echo.py`, the app's `EchoGate(mic_echo)` on the 'me' side), first at its threshold
`rho_on` 0.5:

| speakers → microphone | echo translated: no gate → gate | false barge-ins | my recall | my blocks zeroed | echo blocks zeroed |
|---|---|---|---|---|---|
| 0 dB, 80 ms, RT60 0.4 s | 36 of 36 (100 %) → **0** | 59 → **0** | 0.929 → 0.922 | 0.94 % | 96 % |
| −6 dB | 36 → 0 | 56 → 0 | 0.929 → 0.925 | 0.68 % | 96 % |
| −12 dB | 36 → 0 | 53 → 0 | 0.927 → 0.927 | 0.57 % | 96 % |
| +6 dB (louder than me) | 36 → 0 | 57 → 0 | 0.927 → 0.916 | 1.17 % | 96 % |
| 250 ms (a slow output) | 36 → 0 | 59 → 0 | 0.929 → 0.920 | 0.95 % | 96 % |
| a reverberant room, RT60 0.8 s | 36 → **2** | 57 → **6** | 0.928 → 0.924 | 0.53 % | 73 % |
| headphones (nothing comes back) | — | 0 → 0 | 0.929 → 0.929 | 0 | — |

The reverberant room smears the echo's spectrum and the correlation of the band envelopes falls under 0.5; the
reverb's tail, the margin and the excess share did not help (`tail_s` 0.15–0.45: 81 % of the echo blocks either way,
without models). **The microphone's gate now has `[local] mic_echo_rho_on` = 0.4** (the call side keeps 0.5):

| speakers → microphone, `rho_on` 0.4 | echo translated: no gate → gate | false barge-ins | my recall | my blocks zeroed | echo blocks zeroed |
|---|---|---|---|---|---|
| 0 dB, 80 ms, RT60 0.4 s | 36 → 0 | 58 → 0 | 0.924 → 0.916 | 0.95 % | 96 % |
| a reverberant room, RT60 0.8 s | 36 → **0** (was 2) | 55 → **0** (was 6) | 0.928 → 0.920 | 0.72 % | 92 % (was 73 %) |
| +6 dB | 36 → 0 | 60 → 0 | 0.929 → 0.913 | 1.29 % | 97 % |
| headphones | — | 0 → 0 | 0.927 → 0.927 | 0.16 % | — |

My recall moves by 0.005–0.016 with the gate; the same scenario without the gate gave 0.929 and 0.924 in two runs
(the pipeline's own spread). Louder speakers than my voice (+6 dB) cost the most: a block of my speech over the echo
passes only where it is clearly above the echo. For the call side (S1's, unchanged; FLEURS en, 150 ms, −6 dB, n = 6)
0.4 would give: echo translated 2.8 % → 0, their recall 0.907 → 0.909, their blocks zeroed 0.13 → 0.25 %, no echo path
the same.

On the runners (`tests/test_me_side.py`, work-gate, the recordings of tests/data: 3 played, 3 of mine, one over the
end of the played one; 0 dB, RT60 0.4 s): echo translated 3 of 3 → 0, false barge-ins 7 → 0, my recall 0.871 = 0.871
on Linux, macOS and Windows; without models (every OS, `rho_on` 0.4): echo blocks zeroed 91 % (0 dB), 89 % (RT60
0.8 s; 81 % at 0.5), mine ≤ 0.4 %, with headphones 0. A real laptop's speakers and microphone, its own processing and a
real room: USER-CHECKS «Этап 5 / S2».

## Stage 6: the LuxTTS spike on the M1 runner, languages in the shell (S3)

### LuxTTS for me → them on an 8 GB Mac: not taken

The question (a research run of 2026-11-08 on an NVIDIA GPU machine, not published): LuxTTS int8 ONNX, faster than
real time on 3 fast x86 cores (RTF 0.23), was estimated at RTF 0.9–1.2 on the `macos-15` runner. Taken further only if
the runner shows **RTF ≤ 0.7 and first sound ≤ 2.5 s**. Measured 2026-11-09 on `macos-15` (Apple M1 Virtual, 3 cores,
7 GB) by that run's method (one temporary workflow, removed): LuxTTS git `28ae6a6`, weights `527f245`, torch 2.8.0,
onnxruntime CPU; the same 6 shortest FLEURS dev sentences (4.9–7.1 s), the same references cut at a pause (en 3.54 s,
ru 3.80 s; "r10": 9.96 / 9.46 s), Russian references as espeak `ru` phonemes (the transcript from the book text), one
untimed warm-up; judges ECAPA (cosine to the full 40 s reference) and faster-whisper `small` int8. LuxTTS does not
stream: first sound = the whole sentence (≈ 4 s of its fast speech). n = 6 sentences per row, one run.

| setting | threads | ru → en: first sound s, mean (max) | RTF | sim (min) | WER | en → en: first sound s (max) | RTF | sim | WER | peak RSS GB |
|---|---|---|---|---|---|---|---|---|---|---|
| int8 ONNX + duration fix, 3.8 s ref | 3 | 2.44 (3.10) | 0.62 | 0.47 (0.41) | 11.7 % | 2.60 (3.59) | 0.63 | 0.54 | 5.3 % | 1.24 |
| int8 ONNX + duration fix, 3.8 s ref | 2 (the app's) | 2.59 (3.20) | 0.66 | 0.49 (0.39) | 7.5 % | 1.88 (2.33) | 0.46 | 0.53 | 3.2 % | 1.15 |
| int8 ONNX + duration fix, 9.5 s ref | 3 | 5.51 (6.11) | 1.38 | 0.53 (0.49) | 12.8 % | 4.11 (6.39) | 1.12 | 0.53 | 3.2 % | 1.40 |
| torch on MPS (upstream), 3.8 s ref | 3 | 3.94 (4.97) | 1.00 | 0.46 (0.42) | 5.3 % | 2.76 (3.23) | 0.67 | 0.59 | 3.2 % | 1.70 (+1.77 GPU) |
| torch on MPS (upstream), 9.5 s ref | 3 | 3.07 (3.68) | 0.77 | 0.50 (0.40) | 10.6 % | 3.19 (4.08) | 0.87 | 0.55 | 4.3 % | 1.83 (+1.74 GPU) |
| int8 ONNX, upstream (no fix: speech squeezed) | 3 | 1.64 (2.34) | 0.53 | 0.42 (0.33) | 16.0 % | 1.46 (1.81) | 0.44 | 0.54 | 9.6 % | 1.02 |

- **RTF passes** (0.62–0.66 with the short reference), **the first sound does not**: 2.4–2.6 s on average and 3.1–3.6 s
  for the longest sentences. It is RTF × the sentence: the threshold means RTF ≤ ~0.6 for these. The 9.5 s reference
  (better likeness on the NVIDIA machine) doubles the cost; the runner's virtual GPU (MPS) is slower than its CPU. Against XTTS-v2 on the
  same runner (stage 3: first audio 4.2–4.3 s, RTF 5.5–5.8, 3.8 GB) its RTF is 8–9× lower and it needs a third of the
  memory.
- **ru → en drops the first word** in 4–5 of the 6 sentences on every path ("Soon officers …" → "officers …", "Some
  festivals have …" → "festivals have …"; en → en keeps them): most of its WER. The Russian prompt is a phoneme hack (the
  model has no Russian), and its end seems to swallow the start of the target. The NVIDIA GPU run of 2026-11-08 had WER 3.2 % there.
- Likeness ru → en 0.46–0.53, as on the NVIDIA GPU machine (0.46–0.53) and above XTTS on CPU (0.43). Nobody listened to it.
- Load 8–15 s; the voice prompt from the reference 0.8–12 s (the first call in a process initialises librosa).

**Decision: me → them on a Mac stays in the cloud** (or a stock voice); the wizard says so on a Mac. LuxTTS is not
taken: borderline speed on the runner, the first-word loss in the product's direction, and its license and dependency
questions (an NC-trained model under Apache, GPL espeak in-process, an unlicensed vocoder). A real M1 (4 performance
cores, not virtual) may be faster: a check on a real Mac, if it is ever taken up again.

### Subtitles in Chinese, Japanese, Korean and Arabic

`tests/test_languages_ui.py::test_overlay_draws_chinese_japanese_korean_and_arabic` on the CI runners (2026-11-09,
snapshots of zh ⇄ ja and ko ⇄ ar per OS as the `overlay-<os>` artifact of the ci run):

| runner | fonts per script (first of the font database) | font of a zh / ja / ko row | first paint of a zh, ja row; ko, ar row — cold → pair warmed | later rows |
|---|---|---|---|---|
| macos-15 | zh 10, ja 10, ko 7, ar 38 | PingFang SC / Hiragino Sans / Apple SD Gothic Neo | 84.5, 41.2; 27.6, 16.2 ms → 74.7, 30.1; 6.2, 2.8 ms | 0.8–4.3 ms |
| windows-latest (Server 2025) | zh 10, ja 12, ko 6, ar 22 | Microsoft YaHei / Yu Gothic UI / Malgun Gothic | 120.9, 98.2; 78.2, 65.5 ms → 88.3, 74.0; 39.7, 52.5 ms | 1.4–2.3 ms |
| ubuntu-latest | zh 0, ja 0, ko 0, ar 3 (DejaVu) | none: the wizard, the tray and the start of translation say what to install | 2.1, 1.0; 6.9, 1.5 ms (empty boxes, no CJK font) | 0.9–1.2 ms |
| dev server (Linux, Noto/WQY/IPA fonts) | zh 7, ja 9, ko 6, ar 7 | WenQuanYi Zen Hei / IPAGothic / fallback | 2.5–6.3 ms cold, 0.9–2.9 ms warmed | ~1 ms |

- Before the own-language font a Chinese row on Windows came out half in Korean glyph forms (Malgun Gothic, the
  system's fallback for Han); now each CJK row is drawn in a font of its language. `QFont.setFamilies` as a fallback
  list does not work for this on Linux (fontconfig picks its own fallback), so the row's main font is set.
- The first paint of a row costs what its new glyphs cost to rasterise: tens of ms on the macOS and Windows runners
  (no GPU) for a sentence of Han characters never shown before, ~1 ms once they were. Loading the pair's fonts at
  translation on (`overlay.set_pair`) takes 10–30 ms off the first row, not the rest: in a call, new characters keep
  coming, so a CJK subtitle may show up to ~0.1 s later on such a machine. Arabic: right to left by Qt's own layout.

## Stage 6: them → me in 14 languages (S1)

Measured 2026-11-09 on FLEURS (CC-BY-4.0; the same sentences in every language): ASR on dev (the first recording of
the first 50 sentences), MT on test (= FLoRes devtest, the first 100 sentences), the call scenarios of stage 5 on dev.
**CPU** = an i7-13700 with each model pinned to 2 physical cores (the app's 2 threads per runtime; `taskset`, 4 such
lanes at a time), onnxruntime with the app's options (no spinning: three sessions' pools on 2 cores fight otherwise).
**GPU** = an RTX A4000 (16 GB). Rows: `bench/results/i7-13700-2t.jsonl`, `rtx-a4000.jsonl`.

### Speech recognition per language

`bench/asr.py --fleurs`: WER (CER for zh / ja: no spaces), the cost of one partial on a 1 / 2 / 5 s window (what the
stream pays every 0.3 s of speech), *other* = results that read as another of the 14 languages (letter trigrams
against FLEURS test text; a model without a language setting drifting).

| language | model (device) | WER / CER | other: whole / first 1.5 s | window 1 / 2 / 5 s, ms | RSS |
|---|---|---|---|---|---|
| uk | Parakeet TDT 0.6B v3 (CPU) | 0.095 / 0.026 | 0 / 0.14 | 155 / 209 / 451 | 1.45 GB |
| de | Parakeet v3 (CPU) | 0.078 / 0.029 | 0 / 0.18 | 142 / 221 / 476 | 1.49 GB |
| fr | Parakeet v3 (CPU) | 0.083 / 0.022 | 0.02 / 0.04 | 145 / 220 / 450 | 1.41 GB |
| fr | FastConformer French (CPU; the default: Parakeet drifts into English in a call, "Calls" below; `linux-x86_64-4c`, Parakeet there 0.080 / 0.021, 127 ms) | 0.148 / 0.067 | 0.02 / 0.10 | 46 / 67 / 176 (EPYC VM) | |
| es | Parakeet v3 (CPU) | 0.035 / 0.013 | 0.02 / 0.06 | 127 / 200 / 466 | 1.43 GB |
| it | Parakeet v3 (CPU) | 0.040 / 0.013 | 0 / 0.12 | 118 / 173 / 329 | 1.55 GB |
| pt | Parakeet v3 (CPU) | 0.053 / 0.023 | 0 / 0.18 | 110 / 189 / 345 | 1.45 GB |
| pl | Parakeet v3 (CPU) | 0.099 / 0.031 | 0 / 0.16 | 188 / 301 / 882 | 1.43 GB |
| (en, ru) | Parakeet v3 (CPU) | 0.051, 0.077 | 0, 0 / 0.02, **0.22** | 101 / 175 / 360 | 1.44 GB |
| (ru) | GigaAM v3 e2e CTC (CPU, the default) | 0.051 / 0.013 | 0 / 0.06 | 63 / 109 / 225 | 0.53 GB |
| ja | **Parakeet TDT-CTC 0.6B ja** (CPU) | CER 0.052 | 0 / 0.22 | 116 / 173 / 347 | 1.36 GB |
| ja | Whisper large-v3-turbo (GPU) | CER 0.024 | 0 / 0.28 | 104 / 103 / 141 | 1.89 GB |
| ar | **FastConformer hybrid large pcd** (CPU) | 0.151 / 0.050 | 0 / 0 | **39 / 72 / 115** | 0.62 GB |
| ar | Whisper large-v3-turbo (GPU) | 0.137 / 0.038 | 0 / 0 | 115 / 123 / 168 | 1.87 GB |
| tr | **Whisper large-v3-turbo** (GPU) | 0.078 / 0.023 | 0 / 0.06 | 104 / 108 / 160 | 1.89 GB |
| tr | Whisper large-v3-turbo int8 (CPU, n = 10) | 0.053 / 0.008 | 0 / 0 | **4647** / 4704 / 5989 | 2.85 GB |
| tr | Whisper small (GPU / CPU, n = 20) | 0.176 / 0.172 | | 31 / 57 / 130; CPU 1075 / 1708 / 1682 | |
| tr | Whisper base int8 (CPU) | 0.347 / 0.108 | | 372 / 538 / 919 | 0.55 GB |
| tr | Parakeet v3 (CPU: not its language) | 0.976 | 0.70 | | |
| ko | **Whisper large-v3-turbo** (GPU) | CER 0.070 | 0 / 0.08 | 107 / 112 / 151 | 1.89 GB |
| ko | Citrinet 1024 (call-center data, CPU) | CER 0.281 | | 96 / 145 / 292 | 0.73 GB |
| zh | **Whisper large-v3-turbo** (GPU) | CER 0.106 | 0.10 / 0.22 | 102 / 103 / 121 | 1.87 GB |
| zh | Whisper small (GPU) | CER 0.254 (traditional characters) | 0.26 | 28 / 51 / 57 | 1.45 GB |
| zh | Citrinet 1024 gamma 0.25 (CPU) | CER 0.156 | 0.12 | 283 / 511 / 1074 | 0.75 GB |

- **Taken:** Parakeet v3 for uk, de, fr, es, it, pt, pl; Parakeet TDT-CTC ja and FastConformer ar — live on 2 CPU
  cores. tr, zh, ko: Whisper large-v3-turbo, **live only on an NVIDIA GPU** (registry `needs = "gpu"`): on the CPU a
  1 s window costs 4.6 s, small 1.1 s, base is poor (tr WER 0.35); no permissive CTC/TDT model with an onnx-asr export
  was found for them (Citrinet ko is call-center data: CER 0.28). Without a GPU those sides go to the cloud.
- *First 1.5 s*: counted only where the reference's as many first words read as the language; 3–4 words still mislead
  the trigrams (German "read" as Turkish or Italian). The known Parakeet drift shows in Russian (0.22, half of it
  Ukrainian); what reaches the subtitles is in the call scenarios below (clauses in another language).
- Whisper through onnx-asr gives text without timestamps: the stream re-runs it on the clause like Parakeet, a long
  clause is cut at the quietest 0.1 s of its second half (one more run for the words before it), decoding is capped
  at 128 tokens (base on a CPU once looped to 448 tokens: a 3 s window took 7.2 s).
- Whisper small writes Chinese in traditional characters (CER against simplified FLEURS 0.25), turbo mostly
  simplified.

### Stock voices

The app's path (`PiperTts`, the piper-runner process, 2 threads), the 10 shortest of the first 60 FLEURS dev sentences;
CER = their wavs back through Whisper large-v3-turbo (GPU) in the voice's language.

| language | voice (license) | first audio, ms | RTF | CER / WER |
|---|---|---|---|---|
| uk | `ukrainian_tts` medium (CC0) | 192 | 0.043 | 0.067 / 0.26 |
| uk | `tetiana` high (Apache-2.0 data) — not taken | 1676 | 0.322 | 0.102 / 0.18 |
| de | `thorsten` medium (CC0) | 219 | 0.051 | 0.018 / 0.07 |
| fr | `siwis` medium (CC-BY-4.0) | 189 | 0.043 | 0.025 / 0.06 |
| es | `es_MX-ald` medium (Unlicense; Latin American as FLEURS es-419) | 247 | 0.048 | 0.013 / 0.05 |
| es | `es_ES-davefx` medium (CC0) — not taken | 214 | 0.045 | 0.009 / 0.06 |
| it | `paola` medium (CC0 data) | 213 | 0.048 | 0.016 / 0.07 |
| pt | `pt_BR-faber` medium (CC0) | 349 | 0.082 | 0.017 / 0.05 |
| pl | `gosia` medium (CC0) | 206 | 0.044 | 0.011 / 0.04 |
| tr | `dfki` medium (**CC-BY-NC-SA-4.0**, consent) | 199 | 0.044 | 0.029 / 0.08 |
| ko | `kss` medium (**CC-BY-NC-SA-4.0**, consent) | 207 | 0.041 | 0.077 / 0.26 |
| ja | `hi_fi_captain` medium (**CC-BY-NC-SA-4.0**, consent; Piper's `[ja]` extra) | 166 | 0.040 | 0.046 (CER) |

zh and ar: no Piper voice with a usable license (MODELS.md) — no local voice, those sides go to the cloud
(`languages.direction`). Kokoro-82M (Apache-2.0, ja and zh without espeak) is in the backlog.

### Translation (MT)

Method: CTranslate2 int8 on 2 physical cores of the i7-13700 (one model per process, 2 threads; Hy-MT2 Q4_K_M through
llama-server on the A4000), each clause sentence by sentence in one batch, beam 1, as the app. Cells: **Tatoeba chrF++ /
FLEURS chrF++ / COMET** · clause ms median (p90). Tatoeba: n = 200 short everyday sentences (median 6 words) of the
Tatoeba-Challenge test sets (commit 6faf2da, CC-BY 2.0 FR), the call's register; FLEURS: n = 100 test sentences
(FLoRes devtest, long news-like); COMET: wmt22-comet-da on the FLEURS hypotheses (`bench/mt_comet.py`, the A4000). Clause
= the first 10 words (zh / ja: 15 characters) of 30 FLEURS sentences. The OPUS-MT rows are our int8 conversions (tc-big
`*-zle` / `zle-*` for de / fr / es / it / pt / pl, Tatoeba-MT `opus-2021-02-19` for ru ↔ uk); quickmt: its own CTranslate2
models (float16 → int8 at load). RSS of one model: OPUS-MT base 175 MB, tc-big 460 MB, quickmt 450–610 MB.

X ↔ ru, direct against through English (the default in the last column):

| direction | direct OPUS-MT | through English, quickmt both | through English, OPUS-MT then quickmt | Hy-MT2 1.8B on the A4000 | default |
|---|---|---|---|---|---|
| uk → ru | 82.4 / 48.0 / 0.866 · 63 (80) ms | 69.8 / 50.7 / 0.899 · 109 (128) ms | — | 77.5 / 50.0 / 0.910 · 141 (172) ms | direct |
| ru → uk | 79.6 / 46.1 / 0.888 · 69 (85) ms | 65.2 / 50.1 / 0.915 · 97 (119) ms | — | 74.0 / 47.3 / 0.915 · 154 (182) ms | direct |
| de → ru | 62.6 / 50.5 / 0.867 · 126 (176) ms | 60.0 / 50.0 / 0.845 · 128 (200) ms | — | 60.8 / 47.6 / 0.879 · 134 (163) ms | direct |
| ru → de | 68.0 / 52.5 / 0.840 · 136 (168) ms | 60.8 / 51.9 / 0.821 · 140 (167) ms | — | 64.7 / 48.3 / 0.830 · 131 (157) ms | direct |
| fr → ru | 63.0 / 49.2 / 0.869 · 81 (126) ms | 58.9 / 49.0 / 0.863 · 76 (94) ms | — | 57.9 / 46.3 / 0.870 · 125 (149) ms | direct |
| ru → fr | 68.1 / 56.6 / 0.825 · 106 (128) ms | 63.6 / 55.9 / 0.817 · 239 (289) ms | — | 60.0 / 52.6 / 0.829 · 128 (143) ms | direct |
| es → ru | 66.7 / 46.5 / 0.871 · 79 (93) ms | 60.4 / 45.1 / 0.845 · 127 (220) ms | — | 62.4 / 44.3 / 0.875 · 125 (157) ms | direct |
| ru → es | 70.9 / 47.9 / 0.827 · 97 (126) ms | 66.6 / 45.5 / 0.814 · 131 (176) ms | — | 65.0 / 47.1 / 0.829 · 123 (148) ms | direct |
| it → ru | 63.7 / 46.5 / 0.861 · 142 (189) ms | 58.5 / 46.8 / 0.863 · 126 (181) ms | — | 57.7 / 45.0 / 0.878 · 126 (155) ms | direct |
| ru → it | 61.8 / 47.5 / 0.841 · 167 (215) ms | 60.0 / 47.8 / 0.830 · 144 (181) ms | — | 62.4 / 45.9 / 0.853 · 131 (153) ms | direct |
| pt → ru | 66.3 / 52.4 / 0.875 · 153 (197) ms | 62.8 / 50.4 / 0.864 · 104 (126) ms | — | 64.4 / 48.1 / 0.878 · 126 (148) ms | direct |
| ru → pt | 64.0 / 57.1 / 0.855 · 143 (179) ms | 63.8 / 56.6 / 0.846 · 178 (226) ms | — | 64.3 / 53.1 / 0.857 · 118 (143) ms | direct |
| pl → ru | 72.4 / 45.4 / 0.873 · 150 (225) ms | 61.0 / 46.0 / 0.844 · 143 (174) ms | — | 69.1 / 43.9 / 0.877 · 148 (185) ms | direct |
| ru → pl | 69.2 / 45.2 / 0.882 · 186 (262) ms | 63.0 / 44.4 / 0.870 · 153 (196) ms | — | 68.4 / 44.4 / 0.883 · 152 (175) ms | direct |
| tr → ru | — | 54.9 / 46.0 / 0.838 · 136 (170) ms | 54.5 / 44.1 / 0.810 · 73 (90) ms | 52.8 / 41.3 / 0.851 · 176 (217) ms | OPUS-MT then quickmt |
| ru → tr | — | 56.6 / 47.8 / 0.822 · 134 (200) ms | 55.9 / 46.0 / 0.796 · 67 (84) ms | 55.4 / 44.1 / 0.840 · 147 (176) ms | OPUS-MT then quickmt |
| zh → ru | — | 44.6 / 43.7 / 0.842 · 85 (115) ms | — | 50.1 / 41.6 / 0.878 · 129 (145) ms | quickmt both |
| ru → zh | — | 24.4 / 20.9 / 0.812 · 184 (239) ms | 23.5 / 19.6 / 0.788 · 71 (86) ms | 27.9 / 21.4 / 0.854 · 74 (95) ms | OPUS-MT then quickmt |
| ja → ru | 41.8 / 31.1 / 0.715 · 44 (66) ms | 50.0 / 42.1 / 0.845 · 85 (110) ms | — | 50.7 / 37.8 / 0.858 · 103 (154) ms | quickmt both |
| ru → ja | — | 26.5 / 22.4 / 0.806 · 127 (168) ms | 25.4 / 21.4 / 0.786 · 63 (74) ms | 32.2 / 23.0 / 0.871 · 132 (160) ms | OPUS-MT then quickmt |
| ko → ru | 49.1 / 34.3 / 0.741 · 104 (156) ms | 53.6 / 42.7 / 0.830 · 153 (180) ms | — | 51.1 / 42.0 / 0.861 · 196 (229) ms | quickmt both |
| ru → ko | — | 29.2 / 29.6 / 0.832 · 145 (184) ms | 28.7 / 27.5 / 0.808 · 66 (78) ms | 29.6 / 27.4 / 0.859 · 135 (155) ms | OPUS-MT then quickmt |
| ar → ru | 57.9 / 43.1 / 0.802 · 88 (120) ms | 61.2 / 49.7 / 0.854 · 124 (167) ms | — | 61.1 / 43.8 / 0.854 · 158 (191) ms | quickmt both |
| ru → ar | 44.5 / 39.5 / 0.778 · 78 (103) ms | 49.8 / 46.8 / 0.817 · 122 (159) ms | 51.2 / 44.3 / 0.800 · 63 (75) ms | 50.3 / 40.8 / 0.833 · 149 (191) ms | OPUS-MT then quickmt |

X ↔ en:

| direction | quickmt | OPUS-MT | default |
|---|---|---|---|
| ru → en | 71.9 / 59.5 / 0.859 · 87 (116) ms | 72.1 / 55.3 / 0.832 · 48 (63) ms | OPUS-MT |
| en → ru | 64.6 / 55.0 / 0.863 · 58 (71) ms | 63.6 / 51.6 / 0.849 · 40 (57) ms | quickmt |
| uk → en | 73.2 / 62.4 / 0.861 · 44 (51) ms | 75.4 / 51.8 / 0.793 · 56 (75) ms | OPUS-MT |
| en → uk | 65.2 / 57.1 / 0.883 · 50 (64) ms | 67.6 / 45.0 / 0.800 · 43 (60) ms | quickmt |
| de → en | 69.5 / 66.3 / 0.880 · 44 (60) ms | 70.9 / 65.0 / 0.877 · 68 (82) ms | OPUS-MT |
| en → de | 60.9 / 62.2 / 0.856 · 42 (53) ms | 61.3 / 59.4 / 0.833 · 58 (70) ms | quickmt |
| fr → en | 72.6 / 68.0 / 0.889 · 45 (57) ms | 69.9 / 63.2 / 0.873 · 37 (51) ms | OPUS-MT |
| en → fr | 70.3 / 68.2 / 0.865 · 64 (94) ms | 68.9 / 68.0 / 0.849 · 73 (103) ms | quickmt |
| es → en | 75.2 / 57.4 / 0.856 · 36 (48) ms | 74.7 / 57.8 / 0.862 · 44 (55) ms | OPUS-MT |
| en → es | 71.7 / 53.3 / 0.829 · 52 (67) ms | 74.4 / 54.3 / 0.846 · 51 (66) ms | quickmt |
| it → en | 81.2 / 59.3 / 0.873 · 35 (45) ms | 81.4 / 57.4 / 0.864 · 73 (93) ms | OPUS-MT |
| en → it | 75.6 / 55.7 / 0.862 · 43 (56) ms | 73.1 / 53.7 / 0.845 · 58 (75) ms | quickmt |
| pt → en | 75.3 / 70.9 / 0.886 · 53 (60) ms | 72.6 / 69.6 / 0.884 · 50 (56) ms | OPUS-MT |
| en → pt | 70.6 / 71.3 / 0.885 · 46 (59) ms | 70.0 / 70.8 / 0.885 · 146 (198) ms | quickmt |
| pl → en | 64.2 / 54.2 / 0.835 · 55 (69) ms | 65.3 / 51.0 / 0.823 · 55 (71) ms | OPUS-MT |
| en → pl | 57.6 / 47.7 / 0.834 · 72 (97) ms | 57.2 / 43.4 / 0.801 · 49 (61) ms | quickmt |
| tr → en | 71.8 / 59.3 / 0.877 · 48 (64) ms | 72.8 / 55.0 / 0.858 · 89 (101) ms | OPUS-MT |
| en → tr | 65.4 / 59.5 / 0.886 · 39 (53) ms | 63.4 / 58.6 / 0.880 · 101 (134) ms | quickmt |
| zh → en | 57.2 / 55.3 / 0.858 · 46 (86) ms | 54.5 / 50.1 / 0.837 · 51 (63) ms | quickmt |
| en → zh | 27.0 / 26.7 / 0.838 · 51 (73) ms | 22.8 / 21.8 / 0.812 · 53 (77) ms | quickmt |
| ja → en | 53.6 / 55.5 / 0.873 · 31 (40) ms | 52.3 / 44.6 / 0.807 · 40 (51) ms | quickmt |
| en → ja | 25.2 / 28.2 / 0.850 · 49 (71) ms | 17.6 / 18.5 / 0.805 · 52 (72) ms | quickmt |
| ko → en | 58.7 / 52.3 / 0.853 · 60 (108) ms | 57.2 / 47.4 / 0.818 · 90 (136) ms | quickmt |
| en → ko | 30.1 / 37.6 / 0.875 · 42 (64) ms | 2.6 / 2.8 / 0.290 · 104 (134) ms | quickmt |
| ar → en | 66.7 / 66.0 / 0.876 · 57 (68) ms | 63.9 / 58.9 / 0.838 · 54 (73) ms | quickmt |
| en → ar | 48.1 / 58.2 / 0.858 · 40 (48) ms | 43.5 / 53.0 / 0.845 · 52 (66) ms | quickmt |

What the numbers above do not show, and what decided the defaults:

- **Through English the speaker's gender and ты / вы get lost.** Conversational phrases written for the check (feminine
  first person in the past tense or with an adjective, informal and formal address): X → ru, 10 per language — direct
  OPUS-MT 8–10 / 10 (de 8, fr 9 with tc-big, es 9 with tc-big, it 8, pt 10, pl 10), quickmt through English 3–5 / 10;
  ru ↔ uk, 20 each way — direct keeps the gender 9 / 9 and 8 / 8, through English 0 / 9 and 1 / 8 («Я готова» → «Я
  готовий»), and mixes ты / вы in 3–4 of 12.
- **quickmt X → en makes up a continuation of a one- or two-word reply**: 39 of 41 replies of ru / uk / de / fr / es /
  it / pt / pl («Да.» → "Yes, I am very", «Ja.» → "I think we're going to get a job.", «Sí.» → "I'm not sure I'll be
  there.", pl → en adds "✔"); OPUS-MT X → en and the direct X → ru models translate all 41 as replies. Beam 2 or 4,
  `length_penalty` 0 do not help (pl → en then gives "Skip to content"); cutting a translation to its source's number of
  sentences costs 0.2–1.6 chrF++ on FLEURS. On Tatoeba a sentence too many in sources of ≤ 4 words: quickmt es → en
  24 %, it 14.5 %, de 14.3 %, pt 9.1 %, through English pl → ru 16.7 %, es → ru 14.1 %, de → ru 12.8 %; OPUS-MT 0–2 %.
  ja / ar → en and en → X of quickmt are clean on them.
- Hence the defaults (`config.Local.mt`, the registry's `pivot` rows): X ↔ ru direct for uk, de, fr, es, it, pt, pl
  (release `models-2`); X → en OPUS-MT for ru, uk, de, fr, es, it, pt, pl, tr, quickmt for ja, ko, zh, ar; en → X quickmt
  (en → ru too: 64.6 / 55.0 / 0.863 against OPUS-MT 63.6 / 51.6 / 0.849); every other pair through English from those
  two. The price where English stays in between: OPUS-MT as the first half costs 1–2 chrF++ on FLEURS and 0.02–0.03
  COMET (tr → ru 0.810 against 0.838), and is equal on Tatoeba. A user pair with ru on CPU: 156–470 MB for both
  directions instead of ~1.6 GB of four quickmt models.
- Hy-MT2 1.8B on an NVIDIA GPU has the best COMET on most pairs (ru → ja 0.871, ru → zh 0.854 against 0.806 / 0.812 through
  English) at 75–200 ms per clause, but needs llama-server with CUDA in the builds: backlog. NLLB-200 600M (noncommercial)
  and M2M100 418M: 420–760 ms per clause on the CPU and below the better of direct and through English on FLEURS (de → ru
  46.5 / 44.1, ru → uk 46.3 / 48.6, uk → ru 47.9 / 49.1); stopped after uk / de / fr / es of 12.

### Calls in the 14 languages

`bench/call.py` (stage 5 scenarios) with `--lang X --to ru|en`, the defaults of this stage (ASR, MT, stock voice per
language; tr / zh / ko: Whisper turbo, NVIDIA only). `links`: the echo scenario through the gate in real time with the
app's queue policy, n = 2 (12 turns of theirs), seconds from the end of a clause's speech to its subtitle and to the
start of its voice. `long --asap`: FLEURS dev sentences of one language back to back with no pause the VAD would take
(a monologue, n = 3, 5 sentences each; clauses cut at 5 s), and `--turns`: the same with 0.6–1.2 s pauses between
sentences (a talk, n = 4); WER / CER of the clauses against the read text, chrF of their translations against the
translation of the read sentences, "another language": clauses of ≥ 4 words (zh / ja: 6 characters) that the letter-
trigram LID of `bench/asr.py` puts in another language — on short clauses mostly false alarms (es 9.8 %: Spanish
clauses), a real drift only where the text says so. CPU: 2 lanes of 4 physical cores of the i7-13700 (a side: ASR, MT,
TTS with 2 threads each); "A4000, two scenarios at once": the two lanes shared the one GPU (ASR and MT on CUDA) — slower
than the CPU for some languages (pl → ru 1.45 s against 0.57 s), so tr / zh / ko / de / fr ran again one at a time
(tr → ru subtitle p50 0.39 s against 0.71 s, de → ru 0.48 s against 0.73 s).

| them speak | host | → ru: subtitle s p50 / p90 | → ru: voice s p50 / p90 | → en: subtitle | → en: voice | clauses (dropped ru / en) | RSS MB ru / en |
|---|---|---|---|---|---|---|---|
| uk | i7-13700, 2 cores | 0.52 / 1.83 | 1.37 / 2.79 | 0.54 / 1.98 | 1.74 / 2.82 | 34 (0 / 0) | 1608 / 1614 |
| uk | A4000, two scenarios at once | 0.58 / 2.08 | 1.34 / 2.8 | 0.55 / 2.21 | 1.8 / 3.03 | 34 (0 / 0) | 2271 / 2274 |
| de | i7-13700, 2 cores | 0.62 / 1.55 | 0.87 / 2.17 | 0.51 / 1.41 | 0.96 / 2.35 | 36 (1 / 1) | 1771 / 1609 |
| de | A4000, two scenarios at once | 0.73 / 1.34 | 0.95 / 2.51 | 0.7 / 1.55 | 1.19 / 2.53 | 35 (1 / 1) | 2502 / 2281 |
| de | A4000, one at a time | 0.48 / 1.45 | 0.64 / 2.34 | — | — | 35 (0 / —) | 2504 / — |
| fr | i7-13700, 2 cores | 0.49 / 1.65 | 1.33 / 3.39 | 0.36 / 1.54 | 1.32 / 2.95 | 37 (1 / 7) | 837 / 626 |
| fr | A4000, two scenarios at once | 0.32 / 1.32 | 1.16 / 2.91 | 0.31 / 1.31 | 1.2 / 2.92 | 37 (1 / 6) | 1696 / 1430 |
| fr | A4000, one at a time | 0.32 / 1.32 | 1.16 / 2.97 | — | — | 37 (1 / —) | 1696 / — |
| es | i7-13700, 2 cores | 0.66 / 1.59 | 1.24 / 2.56 | 0.56 / 1.62 | 1.24 / 2.94 | 48 (0 / 2) | 1798 / 1611 |
| es | A4000, two scenarios at once | 0.64 / 1.6 | 1.37 / 2.73 | 0.73 / 1.64 | 1.56 / 3.25 | 48 (0 / 2) | 2520 / 2292 |
| it | i7-13700, 2 cores | 0.51 / 1.63 | 0.88 / 2.18 | 0.53 / 1.56 | 1.0 / 2.62 | 54 (0 / 2) | 1795 / 1625 |
| it | A4000, two scenarios at once | 0.63 / 1.75 | 0.82 / 2.09 | 0.62 / 1.71 | 1.23 / 2.9 | 54 (1 / 2) | 2512 / 2278 |
| pt | i7-13700, 2 cores | 0.6 / 1.89 | 0.95 / 2.5 | 0.49 / 1.36 | 0.89 / 2.63 | 54 (2 / 2) | 1777 / 1617 |
| pt | A4000, two scenarios at once | 1.17 / 2.0 | 1.43 / 2.38 | 1.22 / 1.81 | 1.47 / 2.59 | 55 (1 / 2) | 2510 / 2284 |
| pl | i7-13700, 2 cores | 0.57 / 2.07 | 1.31 / 2.61 | 0.57 / 1.94 | 1.6 / 3.09 | 38 (3 / 4) | 1751 / 1606 |
| pl | A4000, two scenarios at once | 1.45 / 2.43 | 2.18 / 2.99 | 1.49 / 2.3 | 2.03 / 2.94 | 38 (3 / 7) | 2505 / 2272 |
| tr | A4000, two scenarios at once | 0.71 / 2.56 | 2.09 / 3.23 | 0.81 / 2.44 | 2.11 / 3.5 | 41 (1 / 3) | 2357 / 2039 |
| tr | A4000, one at a time | 0.39 / 2.02 | 1.48 / 2.88 | 0.39 / 2.01 | 1.87 / 3.34 | 41 (0 / 1) | 2356 / 2039 |
| zh | A4000, two scenarios at once | 0.37 / 0.6 | 0.54 / 2.25 | 0.3 / 0.48 | 0.48 / 2.33 | 40 (0 / 0) | 2255 / 2256 |
| zh | A4000, one at a time | 0.3 / 0.43 | 0.46 / 2.12 | 0.29 / 0.42 | 0.46 / 2.18 | 40 (0 / 0) | 2255 / 2255 |
| ja | i7-13700, 2 cores | 0.62 / 1.6 | 1.55 / 3.02 | 0.51 / 1.36 | 1.32 / 2.84 | 41 (2 / 3) | 2008 / 1724 |
| ja | A4000, two scenarios at once | 0.56 / 1.34 | 1.29 / 2.76 | 0.49 / 1.32 | 1.3 / 2.78 | 41 (2 / 2) | 2492 / 2471 |
| ko | A4000, two scenarios at once | 0.41 / 2.65 | 0.71 / 2.91 | 0.38 / 2.63 | 0.79 / 2.72 | 43 (1 / 2) | 2466 / 2164 |
| ko | A4000, one at a time | 0.35 / 2.44 | 0.52 / 2.6 | 0.34 / 2.4 | 0.57 / 2.58 | 43 (1 / 1) | 2466 / 2164 |
| ar | i7-13700, 2 cores | 1.01 / 1.84 | 1.41 / 3.27 | 0.95 / 1.82 | 1.5 / 3.25 | 58 (3 / 1) | 1264 / 943 |
| ar | A4000, two scenarios at once | 0.9 / 1.73 | 1.27 / 3.21 | 0.88 / 1.72 | 1.34 / 2.98 | 58 (2 / 2) | 1807 / 1799 |

| them speak → ru | host | monologue: WER / CER / chrF | another language (LID) | cuts pause / punct / max | talk with pauses: WER / CER / chrF |
|---|---|---|---|---|---|
| uk | CPU | 0.1564 / 0.0616 / 71.7 | 0.0 | 18 / 4 / 16 | 0.1441 / 0.0574 / 74.5 |
| de | CPU | 0.1569 / 0.0585 / 62.3 | 0.059 | 19 / 8 / 18 | 0.1017 / 0.0283 / 64.5 |
| fr | CPU | 0.1558 / 0.0544 / 67.3 | 0.032 | 7 / 8 / 19 | 0.1679 / 0.0665 / 68.7 |
| es | CPU | 0.0574 / 0.0326 / 74.2 | 0.098 | 21 / 2 / 22 | 0.0352 / 0.0178 / 72.4 |
| it | CPU | 0.0472 / 0.008 / 58.2 | 0.039 | 21 / 9 / 28 | 0.0625 / 0.0259 / 59.1 |
| pt | CPU | 0.0875 / 0.0337 / 65.2 | 0.028 | 29 / 2 / 16 | 0.0652 / 0.0257 / 65.6 |
| pl | CPU | 0.1434 / 0.0483 / 62.5 | 0.0 | 13 / 4 / 19 | 0.1178 / 0.0465 / 67.4 |
| tr | A4000 | 0.1161 / 0.026 / 59.6 | 0.088 | 23 / 0 / 13 | — |
| zh | A4000 | — / 0.1113 / 55.9 | 0.241 | 15 / 0 / 16 | — |
| ja | CPU | — / 0.0682 / 57.1 | 0.028 | 31 / 0 / 11 | — |
| ko | A4000 | 0.0877 / 0.0238 / 62.7 | 0.027 | 15 / 0 / 25 | — |
| ar | CPU | 0.2261 / 0.1038 / 57.0 | 0.0 | 14 / 6 / 29 | — |

- **Parakeet drifts into English — in French.** In the monologue Parakeet v3 (no language setting) wrote clauses after a
  5 s cut in English ("The repas romains à l'antiquité…", "and republicanism, the nationalism and the conviction that
  the pay", "The force of Lord never present…"): fr WER 0.32, CER 0.164, chrF 57.8 on the i7-13700 (0.31 / 0.176 / 59.0
  on `linux-x86_64-4c`; the rows say their ASR in `asr`). NVIDIA's FastConformer French (`fastconformer-fr-pc`) does
  not: 0.156 / 0.054 / 67.3, the fr default now. Its price: on whole FLEURS dev utterances Parakeet is better (WER 0.080
  against 0.148, CER 0.021 against 0.067, `linux-x86_64-4c`, n = 50), and in the talk with pauses too (0.126 / 0.038 /
  72.2 against 0.156 / 0.064 / 68.5, `linux-x86_64-4c`) — but it drifted there once as well; a drifted clause is nonsense in Russian, not a word wrong. FastConformer's
  1 s window: 46 ms against 127 ms. In the other languages the clauses show one such drift in German per run ("Your
  primary leaves raum is in Flusmündungen") and none in uk / es / it / pt / pl: they stay on Parakeet; their higher WER
  in the monologue than offline (de 0.157 against 0.078) comes from the 5 s cuts (names and numbers split).
- **Whisper writes Chinese partly in traditional characters** ("我們對待你他們"; 24 % of the zh clauses taken for another
  language): the text is right, the script mixed. Converting needs OpenCC (a new dependency): backlog.
- Peak RSS of a side: 1.6–2.0 GB on the CPU (ASR + MT + the Piper runner); fr with FastConformer 0.6–0.8 GB.

## Stage 6: the clone in 13 languages (S3)

Measured 2026-11-09 on one RTX A4000 of the C server (CUDA memory cap 0.3, 2 CPU threads); the CPU columns on its
i7-13700 with 2 threads on 2 physical cores (`taskset`), CUDA hidden. "The engine" is the app's (`XttsClone`: streaming,
text split, pause cap, length guard and, from this stage, `xtts.normalize`; `bench/tts_clone.py --model xtts-engine`),
"raw XTTS" the upstream call without them (`--model xtts`).

- **Sentences:** the same 6 FLEURS dev sentences in each language (ids 1552, 1572, 1578, 1582, 1645, 1656: the shortest
  that all 13 have, 3.9–10 s as FLEURS speakers read them; `bench/tts_clone.py set`); the CPU columns: the 3 shortest.
- **Voices:** "me" is the LibriVox reader recorded in Russian (42 s); the second voice the LibriSpeech reader in
  English. So each language is said by a voice recorded in another language, as in the app.
- **Judge:** SpeechBrain ECAPA cosine against the whole recording; Whisper large-v3-turbo (fp16) for CER and WER (zh, ja:
  CER over characters without spaces, no WER). Baselines: two halves of one recording 0.90 / 0.93, the two readers
  0.085, the FLEURS speakers against "me" −0.02–0.17; Whisper turbo on the real FLEURS recordings of these sentences:
  CER 0.2–5.4 % (ru 0.5, pt 0.2, es 1.0, ar 1.2, fr 1.3, en 1.5, de 1.6, it 1.6, ko 2.7, pl 3.1, tr 4.8, zh 4.8, ja 5.4).
- `first call s`: the first synthesis in a language (an untimed warm-up sentence), with loading its text tools
  (pinyin, MeCab and UniDic for Japanese). The app does that at its warm-up, in the other side's language.

**The engine as the app runs it now** (Italian and Spanish from a second run, German, French, Portuguese and Korean
from a third, after the change at the end of a clause; their speed columns from the first run, the GPU to itself:
in the third someone else's work shared it):

| language | sim ru → X (min) | CER / WER ru → X | sim en → X | CER / WER en → X | GPU first audio s ru → X: mean (max) | GPU RTF | CPU 2 thr. first audio s ru → X | CPU RTF | first call s |
|---|---|---|---|---|---|---|---|---|---|
| English | 0.37 (0.28) | 1.3 % / 4.2 % | 0.58 | 5.4 % / 6.9 % | 0.15 (0.15) | 0.17 | 1.76 | 2.36 | 0.96 |
| Russian | 0.65 (0.55) | 3.3 % / 5.3 % | 0.35 | 7.9 % / 14.0 % | 0.15 (0.15) | 0.18 | 2.07 | 2.53 | 1.33 |
| German | 0.44 (0.39) | 13.7 % / 15.9 % | 0.41 | 24.3 % / 30.4 % | 0.15 (0.15) | 0.18 | 1.81 | 2.80 | 1.22 |
| French | 0.42 (0.37) | 2.2 % / 3.6 % | 0.34 | 1.1 % / 4.8 % | 0.15 (0.15) | 0.18 | 1.64 | 2.62 | 0.79 |
| Spanish | 0.49 (0.44) | 0.6 % / 2.3 % | 0.31 | 1.8 % / 3.5 % | 0.17 (0.21) | 0.20 | 1.88 | 2.76 | 0.77 |
| Italian | 0.57 (0.49) | 0.9 % / 2.7 % | 0.35 | 1.6 % / 5.3 % | 0.15 (0.15) | 0.18 | 1.52 | 2.75 | 1.15 |
| Portuguese | 0.63 (0.57) | 1.2 % / 4.0 % | 0.29 | 1.5 % / 4.0 % | 0.15 (0.15) | 0.18 | 1.53 | 2.53 | 0.59 |
| Polish | 0.63 (0.57) | 0.2 % / 1.6 % | 0.36 | 1.3 % / 6.2 % | 0.15 (0.15) | 0.18 | 2.03 | 2.61 | 1.07 |
| Turkish | 0.55 (0.46) | 4.0 % / 13.0 % | 0.33 | 4.5 % / 14.8 % | 0.15 (0.15) | 0.18 | 1.94 | 3.08 | 0.73 |
| Chinese | 0.51 (0.44) | 5.8 % / — | 0.37 | 8.6 % / — | 0.15 (0.15) | 0.18 | 1.65 | 2.33 | 0.83 |
| Japanese | 0.58 (0.55) | 3.6 % / — | 0.34 | 4.2 % / — | 0.15 (0.16) | 0.20 | 1.87 | 2.48 | 0.77 |
| Korean | 0.56 (0.46) | 5.4 % / 21.7 % | 0.33 | 3.8 % / 21.7 % | 0.16 (0.19) | 0.19 | 2.04 | 3.34 | 0.95 |
| Arabic | 0.60 (0.51) | 1.5 % / 8.2 % | 0.36 | 1.5 % / 4.9 % | 0.17 (0.21) | 0.19 | 1.70 | 2.71 | 0.57 |

- **My voice in every language.** Recorded in Russian, the clone keeps 0.42–0.63 similarity in the 12 other languages
  (Russian itself 0.65). Recorded in English, 0.29–0.41 (English itself 0.58). That matches the cross-lingual
  ru → en of stage 1 (0.37–0.44) and is far above another speaker (0.085).
- **Understood.** From the Russian recording CER is ≤ 5.8 % in 12 of 13 languages, ≤ 1.5 % in 6. German is the
  exception: in every run XTTS sometimes went on after the last word ("…bis zum einfachen Bürger popfing per den
  mund…", "…dass Felix so friedlich gewählt wird"). That gave 13.7 % from the Russian recording and 24.3 % from the
  English one in the last run, against 2.0 % in another run of the same sentences: 1 to 3 of 6 sentences, with a
  period or a comma at the end. A stop for these tails is in the backlog. Korean WER (13–22 % at 3–5 % CER) is mostly
  Whisper's spacing ("정부청사" / "정부 청사").
- **Speed.** On the A4000 every language starts in 0.15–0.17 s on average (max 0.21 from the Russian recording, 0.26
  from the English one) and runs at RTF 0.17–0.20. On
  2 CPU threads, RTF is 2.3–3.3 and the first audio comes after 1.5–2.1 s, in every language. Without an NVIDIA GPU the
  clone falls behind the conversation in each language: the wizard says so, and the app measures the clone at its
  start and gives way to the stock voice.

**Before → after** (CER from the Russian recording; the audio of the 6 sentences: shorter means cut):

| language | CER ru → X: raw XTTS | engine before | engine after | audio s of the 6, ru → X: raw / before / after |
|---|---|---|---|---|
| English | 1.0 % | 1.5 % | 1.3 % | 21.9 / 20.9 / 21.4 |
| Russian | 2.0 % | 6.6 % | 3.3 % | 31.8 / 30.2 / 29.0 |
| German | 4.9 % | 2.0 % | 13.7 %¹ | 32.6 / 31.2 / 33.3 |
| French | 2.6 % | 3.3 % | 2.2 % | 27.8 / 29.4 / 25.9 |
| Spanish | 1.2 % | 1.6 % | 0.6 % | 35.4 / 37.0 / 35.1 |
| Italian | 9.5 % | 8.1 % | 0.9 % | 32.0 / 35.7 / 29.8 |
| Portuguese | 3.7 % | 2.0 % | 1.2 % | 31.9 / 31.4 / 28.0 |
| Polish | 0.9 % | 0.7 % | 0.2 % | 31.9 / 36.2 / 32.0 |
| Turkish | 5.0 % | 4.5 % | 4.0 % | 26.0 / 25.9 / 25.9 |
| Chinese | 6.7 % | 17.3 % | 5.8 % | 26.2 / 20.4 / 26.5 |
| Japanese | 6.6 % | 7.8 % | 3.6 % | 31.9 / 27.6 / 29.5 |
| Korean | 2.7 % | 6.5 % | 5.4 % | 33.8 / 30.7 / 34.6 |
| Arabic | 3.2 % | 2.3 % | 1.5 % | 31.7 / 30.2 / 31.3 |

- **Chinese 17.3 % → 5.8 %.** The length guard (1 s + letters / 8) cut Chinese speech. XTTS says 4–5 Han characters a
  second (FLEURS speakers 2.6), so the guard allowed about a third of the time a sentence takes. The 6 sentences had
  20.4 s of audio against 26.2 s raw. Now a Han / kana / Hangul character counts as 3 letters: 26.5 s. Japanese 7.8
  → 3.6 % and Korean 6.5 → 5.4 % for the same reason.
- **Italian 8.1 → 0.9 %, Spanish 1.6 → 0.6 %, French 3.3 → 2.2 %, Portuguese 2.0 → 1.2 %:** the comma at the end of a
  clause (below). ¹ German 2.0 → 13.7 %: 6 sentences, one run each, so the noise of one run: the tails above come in
  1 to 3 of the 6 sentences in any run, with either ending.

### Numbers, money and clock times

Three sentences per language with a price, a clock time, a decimal, a percentage, a date and counts (e.g. "At 15:30 it
costs 250.50 € (12.5% more).", "The meeting is on 3 March 2024 at 9:00, room 214.", "We sold 1,250 tickets, about 40% of
the 3,000 seats."). Each is said twice by the engine from the "me" voice: with XTTS's own cleaning only ("raw"), and
with `xtts.normalize` first. Whisper turbo is the judge.

| | raw | normalize |
|---|---|---|
| the currency named (the price sentence, 13 languages × 2; pt from a second run with the final code) | 0 / 26 | 26 / 26 |
| "9:00" said without a stray zero ("9-0", "9.00", "nine zero") | 15 / 26 | 26 / 26 |
| Japanese: CER, numbers heard as said | 78 %, 0 / 30 | 9.3 %, 26 / 30 |
| Arabic: CER, numbers heard | 59 %, 11 / 28 | 7.1 %, 26 / 28 |
| Portuguese: numbers heard | 15 / 24 | 18 / 24 |
| Polish / Turkish / Korean: numbers heard | 15 / 20 / 23 of 24, 24, 30 | 20 / 22 / 25 |

- **Raw XTTS loses the price everywhere:** "250,5 Freaking", "250. Punto C punto", Turkish "250-50E", Arabic "riyal"
  for every sign. Normalized: "250 Euro und 50 Cent", "250 евро 50 центов", "250,50 avro", "250.5 ユーロ".
- **Japanese digits were noise:** XTTS's tokenizer leaves digits as they are for Japanese ("2時間" → "u時間",
  "3つ" → "破裂", "15:30 … 250.50" → "ファイアングに1周図…"). Normalized, they are read as Japanese numbers ("2024年3月3日の9時から214号室").
- CER against the written digits grows in de, en, es, fr, it, ru, zh (3–10 % → 9–21 %) for a reason that is not the
  speech: the price is now said in words, and Whisper writes "250 euro, 50 cents" where the text has "250.50 €".
- What is left to XTTS: other numbers, abbreviations (read as words, lower-cased: "USA" → "usa"), symbols. Its
  num2words reads Spanish decimals with "punto" and Turkish numbers as one word ("ikiyüzelli"). Not changed: nobody
  listened to them.
- After these runs (review of the stage, tests only, not said by XTTS yet): R$ as reais and E£ / £ in Arabic as the
  (Egyptian) pound, thousands with several groups (tr 1.250.000, ar / ja 1,250,000), a decimal with a leading zero
  (3,05: "üç virgül sıfır beş", not "beş"), tr 15.30, de 9.30 Uhr, fr / pt 9h and "une heure" / "uma hora", the
  comma ending of every part of a long text, the period of an abbreviation kept (p.m., etc.).

### The final period

In Italian, raw XTTS said the full stop as a word after the sentence ("…da un giorno circa punto tonto"). It did so in
3 of 6 sentences raw, in 2 + 2 of 12 through the engine, and in 6 of 36 more in two repeats. Each FLEURS sentence was
said again with its period, without it, and with a comma instead ("me" voice, the engine):

| | with the period | no period | a comma |
|---|---|---|---|
| Italian: "punto" said / 2+ words of tail / CER (24 each) | 5 / 6 / 10.3 % | 0 / 0 / 1.5 % | 0 / 0 / 1.5 % |
| Spanish: the same | 1 / 1 / 3.2 % | 0 / 0 / 1.4 % | 0 / 0 / 1.2 % |

The other 11 languages, 3 repeats of the 6 sentences each (18), with the period or a comma (zh, ja: ，for 。); a tail =
the transcript runs 2+ words (zh, ja: 3+ characters) longer than the text:

| tails / CER | en | ru | de | fr | pt | pl | tr | zh | ja | ko | ar |
|---|---|---|---|---|---|---|---|---|---|---|---|
| with the period | 0 / 1.5 % | 4 / 8.5 % | 2 / 5.5 % | 2 / 4.3 % | 1 / 3.4 % | 0 / 0.9 % | 0 / 4.4 % | 0 / 2.6 % | 0 / 5.4 % | 3 / 4.0 % | 0 / 1.7 % |
| a comma | 0 / 2.0 % | 0 / 0.9 % | 1 / 4.9 % | 0 / 2.1 % | 0 / 1.1 % | 0 / 0.8 % | 0 / 4.1 % | 0 / 2.9 % | 0 / 5.4 % | 0 / 2.9 % | 0 / 1.5 % |

So the engine ends a clause with a comma in Italian, Spanish, German, French, Portuguese and Korean
(`xtts.COMMA_END`): the languages where XTTS went on after the period; with a comma 1 tail in 120 against 15 in 120.
German is the one where it does not hold up: in the engine run above it went on with a comma too.
Polish, Turkish, Chinese, Japanese, Arabic: no tail either way, the period stays. English and Russian are left as they
are here (stage 5 measures the clone's tails in them: Russian had 4 of 18 with the period, 0 with a comma). Whether a
comma at the end sounds unfinished (rising) is a check by ear (USER-CHECKS).

## Stage 7: light clone engines for both directions (research)

Measured 2026-11-10. Research only: nothing here is in the app yet. The question: which engine can speak in a cloned
voice **without a GPU** (and on a Mac), from a few seconds of a voice picked up in the call, with the first sound
right after the translated clause. `bench/clone_light.py` (engines, short references, judge),
`bench/clone_light_ci.sh` (the macOS runner), `bench/me_them.py --remote` with `$CLONE_SERVER` (the whole pipeline).

- **Method** as "Stage 6: the clone" above: the 6 shortest FLEURS sentences (macOS: 4), the LibriVox reader (ru) and
  the LibriSpeech reader (en) as voices, ECAPA cosine against the **whole** ~40 s recording, CER / WER by Whisper
  small. New: the engine gets only the first **3 / 5 / 10 s** of the recording (cut at a pause; the 3 s transcript by
  the judge), as a voice picked up in a call would be. Baselines: another speaker 0.085, the same reader 0.90–0.93,
  XTTS in the app with the whole recording ru → en 0.37–0.40, en → ru 0.35–0.38.
- **Hosts:** `i7-13700-2t` / `-4t`: 2 / 4 P-cores by `taskset`, CUDA hidden; `rtx-a4000`: one A4000, 2 CPU threads;
  `macos-15`: the GitHub runner (virtual M1, 3 cores, 7 GB, no usable GPU). The judge runs after the synthesis.
- **Speed** columns are from the quiet pass (rows `clone_light_quiet`: nothing else on the machine); the similarity
  rows (`clone_light`) were made while other work shared the CPU, their own speed fields are up to 1.5× slower.
- `first audio`: from the text to the first chunk of sound; `RTF`: synthesis time per second of sound; `voice s`:
  preparing the voice from the reference, once.

**Engines, 5 s reference unless said** (sim / WER: ru → en, then en → ru; "—": the engine has no Russian):

| engine | weights | 2 thr.: first audio s / RTF | 4 thr.: first audio s / RTF | RAM GB | voice s | sim ru → en | WER | sim en → ru | WER |
|---|---|---|---|---|---|---|---|---|---|
| Piper, no clone (the floor) | 60 MB per voice | 0.11 / 0.02 | 0.11 / 0.02 | 0.45 | 0 | 0.12 | 7 % | 0.19 | 11 % |
| **Pocket TTS** (int8 ONNX) | 146 MB per language | **0.09 / 0.28** | 0.09 / 0.28 | 0.9 (10 s ref.: 1.4) | 0.2–0.9 | 0.32–0.36 | 4–7 % | — | — |
| MOSS-TTS-Nano (ONNX) | 0.76 GB | 0.23 / 0.53 | 0.18 / 0.43 | 1.9 (10 s ref.: 2.5–3.2) | 0.2–0.9 | 0.39 | 6 % | 0.36 | **39–40 %** |
| ZONOS3-Mini (GGUF, CPU) | 1.2 GB | 0.37 / 0.72 | 0.22 / 0.52 | 2.2 | 2–8 | **0.61** | 7 % | — | — |
| Piper → MeanVC2 (int8 ONNX) | 56 MB + Piper | 0.13 / 0.10 | 0.12 / 0.09 | 1.1 | 0.2–0.9 | 0.37–0.43 | 7–18 % | 0.48 | **63 %** |
| Piper → OpenVoice v2 converter (fp32 ONNX) | 128 MB + Piper | 1.6 / 0.35 | 1.2 / 0.26 | 0.9 | 0.5 | 0.26 | 10–12 % | 0.22–0.24 | 21–36 % |
| Qwen3-TTS 0.6B, C engine, int8 | 0.6B, int8 | — | 1.0–1.2 / 1.4–2.0 | 3.6 | 4–5 | 0.52 (macOS) | 0 % | 0.43–0.45 (macOS) | 4 % |
| Piper → kNN-VC, 10 s ref. | 0.4 GB | — | 1.6 / 0.44 | 1.8 | 0 | 0.45 | 41 % | 0.37 | 56 % |
| Piper → Chatterbox VC (S3Gen), 10 s ref. | 1 GB | — | 15 / 3.1–3.5 | 2.4 | 0 | 0.45 | 10 % | 0.45 | 30 % |

| GPU, RTX A4000 | first audio s | RTF | VRAM GB | sim ru → en | WER | sim en → ru | WER | sim ru → ru | sim en → en |
|---|---|---|---|---|---|---|---|---|---|
| XTTS-v2 in the app (stage 6, whole recording) | 0.15 | 0.18 | — | 0.37 | 4 % | 0.35 | 14 % | 0.65 | 0.58 |
| **Qwen3-TTS 0.6B, `faster-qwen3-tts`, speaker vector only** | **0.08–0.12** | 0.42–0.54 | 2.4–2.5 | **0.52** | 3 % | **0.41** | 10 % | 0.69 | 0.63 |
| the same, the reference with its text as a prompt, 5 s | 0.13 | 0.52 | 2.6 | 0.58 | 4 % | 0.45 | 10 % | 0.70 | 0.65 |
| the same, 10 s | 0.13–0.15 | 0.46–0.61 | 2.8 | 0.55 | **264 %** | 0.63 | 29 % | 0.76 | 0.68 |
| Qwen3-TTS 0.6B, the official package (no streaming) | 6.0–8.0 | 1.2–1.3 | 2.5 | 0.61 | 5 % | 0.45 | 4 % | 0.71 | 0.61 |
| ZONOS3-Mini (GGUF, CUDA) | 0.03–0.12 | 0.06–0.26 | — | 0.57–0.68 | 28–29 % | — | — | — | 0.67–0.69 |

**How many seconds of a voice** (sim ru → en / en → en; Qwen3: ru → en / en → ru):

| reference | Pocket TTS | MOSS-TTS-Nano | ZONOS3-Mini | Piper → MeanVC2 | Piper → OpenVoice | Qwen3, speaker vector |
|---|---|---|---|---|---|---|
| 3 s | 0.30 / 0.52 | 0.30 / 0.47 | 0.58 / 0.63 | 0.42 / 0.54 | 0.27 / 0.36 | 0.52 / 0.38 |
| 5 s | 0.32–0.36 / 0.54–0.55 | 0.39 / 0.48 | 0.61 / 0.68 | 0.37 / 0.60 | 0.26 / 0.40 | 0.52 / 0.41 |
| 10 s | 0.35–0.36 / 0.59–0.60 | 0.50 / 0.52 | 0.69 / 0.69 | 0.43 / 0.60 | 0.29 / 0.39 | 0.51 / 0.42 |
| whole (~40 s) | breaks (noise, 14 GB) | — | 0.71 / 0.74 | — | 0.30 / 0.35 | — |
| 5 s through Opus 24 kbit/s | 0.41 / 0.54 | 0.37 / — | 0.59 / 0.67 | — | — | — |

**The whole pipeline, live pace** (`me_them.py --remote --null-mic`: the app's ASR and MT **and** the engine on the same
4 cores, rows `e2e_me_them` on `i7-13700-4c` / `-2c`; 9–10 clauses; `voice`: from the end of a clause to the start of
its sound; `tail`: how long the voice goes on after the speech ended):

| engine | direction | engine first audio ms, median | voice s: median (max) | tail s | WER of what was said |
|---|---|---|---|---|---|
| **Pocket TTS**, 10 s ref. | ru → en | 84 | **1.8** (4.2) | 1.7 | 1 % |
| the same on 2 cores | ru → en | 88 | 2.2 (4.8) | 1.7 | 8 % |
| Piper, no clone | ru → en | 96 | 3.1 (5.2) | 2.6 | 9 % |
| Piper → MeanVC2 | ru → en | 106 | 3.7 (5.6) | 3.3 | 9 % |
| Piper → OpenVoice | ru → en | 1234 | 5.1 (7.2) | 4.6 | 9 % |
| MOSS-TTS-Nano | ru → en | 321 | 7.3 (14.6) | 17.1 | 12 % |
| ZONOS3-Mini | ru → en | 269 | 8.5 (11.4) | 12.9 | 29 % |
| Piper, no clone | en → ru | 65 | 1.5 (4.2) | 3.6 | 21 % |
| Piper → OpenVoice, 5 s ref. | en → ru | 765 | 2.9 (5.8) | 5.1 | 37 % |
| MOSS-TTS-Nano, 5 s ref. | en → ru | 267 | 2.8 (31.0) | 33.9 | 57 % |

**The macOS runner** (5 s reference, 4 sentences, 2 threads; 3 threads: close to these):

| engine | first audio s | RTF | RAM GB | sim ru → en | WER ru → en | sim en → ru | WER en → ru |
|---|---|---|---|---|---|---|---|
| Piper, no clone | 0.57–0.70 | 0.13–0.16 | 0.43 | 0.11 | 2 % | 0.19 | 11 % |
| **Pocket TTS** | 0.11–0.20 | 0.28–0.39 | 1.0 | 0.30 | 2 % | — | — |
| MOSS-TTS-Nano | 0.50–0.70 | 0.79–0.91 | 1.8 | 0.41 | 2 % | 0.32 | 20 % |
| Piper → MeanVC2 | 0.82–0.85 | 0.29–0.31 | 1.2 | 0.37 | 3 % | — | — |
| Piper → OpenVoice | 3.5–4.1 | 0.80–0.86 | 0.9 | 0.23 | 11 % | 0.26 | 22 % |
| Qwen3-TTS 0.6B, C engine, int8 (3 threads) | 1.4–3.3 | 1.4–2.3 | 3.7 | 0.52 | 0 % | 0.45 | 4 % |
| ZONOS3-Mini, CPU | 7.0 | 37–39 | 2.0 | 0.63 | 2 % | — | — |

- **Pocket TTS is the one engine that holds a call on a CPU.** First sound in 0.09 s, 3.5× faster than real time on 2
  threads (4 threads add nothing), under 1 GB with a 5 s reference, and next to the app's own ASR and MT on 4 cores
  its voice starts 1.8 s after a clause ends (the stock Piper voice in the same run: 3.1 s; the runs are single, so
  read the difference between the two as noise, not as "Pocket is faster than Piper"). Its similarity is XTTS's
  (ru → en 0.32–0.36 against 0.37) at a tenth of XTTS's CPU time. It has no Russian, Ukrainian or Arabic:
  English, French, German, Spanish, Portuguese, Italian (and community weights for Polish, Turkish, Japanese,
  Chinese, Korean, not measured).
- **A reference must be short for it:** 3 s already gives 0.30 / 0.52, 10 s 0.35 / 0.60; the whole 40 s recording
  breaks Pocket TTS (noise instead of speech, 14 GB), so the app would cut the reference to 10 s.
- **Cloned Russian on a CPU: see "Russian" below** (community weights for Pocket TTS, found after this table).
  Of the engines above, MOSS-TTS-Nano is the only one with Russian that is faster than
  real time, and the judge cannot read 33–56 % of its Russian words (its English: 6–12 %); in the pipeline it falls
  behind as well (tail 17–34 s). Piper → MeanVC2 destroys Russian (WER 63–89 %: the converter was trained on
  Chinese), Piper → OpenVoice keeps the words (21–37 %) but its similarity (0.22–0.26) is next to the floor
  (0.12–0.19) and it waits for the whole clause (first sound 1.2–1.6 s).
- **ZONOS3-Mini sounds most like the speaker** (0.58 from 3 s, 0.69 from 10 s; its voice is one 192-number vector), but
  it is one day old, about 1 clause in 6 from the Russian voice comes out broken (WER 12–29 % in the worse runs), it
  needs 4 cores to itself (RTF 0.52; with the pipeline on the same cores it falls behind) and computing the vector
  takes 2–8 s. On CUDA: first sound 0.03–0.12 s. Its Metal build aborted on the runner's virtual GPU, and its CPU
  build there is 37× slower than real time: a real Mac has to say.
- **With an NVIDIA card, Qwen3-TTS 0.6B through `faster-qwen3-tts` beats XTTS in the app** on similarity (ru → en
  0.52 against 0.37, en → ru 0.41 against 0.35) with the same first sound (0.08–0.12 s) and readable Russian (WER
  3–13 %), under Apache-2.0 instead of the non-commercial CPML. In the speaker-vector mode: with the reference as a
  prompt a 10 s cross-lingual reference made it go on and on (WER 264 %). It lacks Polish, Turkish, Arabic and
  Ukrainian, so XTTS stays for those.
- **The C engine of Qwen3-TTS** (no Python, int8) is 1.4–2.3× slower than real time on 4 x86 cores and on the 3-core
  runner; its author reports 0.7–0.9 on a real M1 and 0.3–0.4 with Metal. Not measured here.
- **Opus does not hurt:** a 5 s reference through Opus 24 kbit/s and back gives the same similarity within the noise
  of 6 sentences (Pocket 0.41 / 0.54, MOSS 0.37, ZONOS3 0.59 / 0.67).
- Nobody has listened to any of this: the similarity and the words are the judges' numbers.

**Russian on a CPU** (added the same day). Pocket TTS has community fine-tunes for Russian; measured:
Shtorm-PocketTTS-RU (v2 "fast", CC-BY-4.0, 6 layers, 438 MB, the voice path kept; its training data is not named) and
xVibePocketTTS (1885 h of Russian; no license on the weights), both on the stock `pocket_tts` package (torch, CPU;
`--model pocketru`, `$POCKET_RU_CONFIG`). The text gets stress marks from `silero-stress` (the models were trained
with them), numbers in words and Latin letters spelled out (the Shtorm vocabulary has neither). i7-13700, 5 s
reference unless said; en → ru is an English voice speaking Russian ("them → me"):

| engine | first audio s | RTF | RAM GB | sim en → ru | WER | sim ru → ru | WER |
|---|---|---|---|---|---|---|---|
| **Shtorm, 2 threads, int8** (torch dynamic quantization) | **0.10** | **0.20** | 1.2 | **0.38** | 6 % | 0.49 | 4 % |
| Shtorm, 2 threads, fp32 | 0.15 | 0.33 | 1.3 | 0.36 | 6 % | 0.47 | 6 % |
| Shtorm, 4 threads, fp32 | 0.13–0.15 | 0.31–0.32 | 1.3 | 0.39 | 14 % | 0.48 | 9 % |
| Shtorm, 2 threads, fp32, 3 s reference | 0.15 | 0.32 | 1.2 | 0.36 | 7 % | 0.52 | 10 % |
| Shtorm, 2 threads, fp32, 10 s reference | 0.15 | 0.33 | 1.5 | 0.39 | 4 % | 0.54 | 7 % |
| xVibe, 2 threads, fp32 | 0.17 | 0.33 | 1.3 | 0.32 | 30 % | 0.49 | 11 % |
| Piper → FreeVC24 (coqui-tts), 4 threads | 2.2–2.8 | 0.49–0.64 | 3.3 | 0.16 | 29 % | 0.40 | 34 % |

| the whole pipeline, en → ru, live pace (fp32) | cores | engine first audio ms, median | voice s: median (max) | tail s | WER of what was said |
|---|---|---|---|---|---|
| **Pocket TTS, Shtorm weights** | 4 | 120 | **1.6** (5.6) | 5.4 | 9 % |
| the same | 2 | 142 | 2.2 (6.2) | 5.5 | 4 % |
| Piper, no clone (from the table above) | 4 | 65 | 1.5 (4.2) | 3.6 | 21 % |

- **So there is a cloned Russian voice on a CPU:** an English voice speaking Russian at XTTS's similarity on a GPU
  (0.36–0.39 against 0.35), with words the judge reads better than the stock Piper voice's in the pipeline (9 % against
  21 %), first sound in 0.10–0.15 s, and it holds the live pace on 4 and on 2 cores. From 3 s of a voice: 0.36.
- xVibe is worse from an English reference (0.32, WER 30 %); the text here was not fitted to its vocabulary.
- No voice converter after Piper gives Russian a cloned voice: OpenVoice 0.22–0.24 next to the floor of 0.19, FreeVC24
  below the floor, MeanVC2 and kNN-VC lose the words (56–63 %), Chatterbox VC is 3× slower than real time.
- **Not every voice.** Two FLEURS dev speakers as voices (the listening set; 5 s references, the same 6 sentences,
  similarity against the speaker's whole utterance): the male Russian one: Pocket en 0.30, Pocket ru 0.39, Qwen3 0.37 /
  0.64, XTTS 0.34 / 0.59 (to English / to Russian); the female English one: **Pocket 0.16 in English and in Russian**,
  Qwen3 0.51 / 0.31, XTTS 0.40 / 0.33. Four voices are too few to say whether it is female voices, phone-grade
  recordings or chance.
- Not heard by anyone: the accent of Russian from an English reference, the stress marks' mistakes.

**Eight FLEURS voices** (rows in `bench/results/voices.jsonl`; dev speakers, one 9–12 s utterance each, the engine
gets its first 5 s, similarity against the whole utterance, the same 6 sentences; Pocket en: int8 ONNX, Pocket ru: the
Shtorm weights, int8; Qwen3 on the A4000 beside them):

| voice | Pocket → en | Qwen3 → en | Pocket ru → ru | Qwen3 → ru | WER ru: Pocket / Qwen3 |
|---|---|---|---|---|---|
| English, male | 0.42 | 0.70 | 0.18 | 0.42 | 4 % / 4 % |
| English, female | 0.19 | 0.54 | 0.19 | 0.32 | 11 % / 74 % |
| English, female | 0.12 | 0.53 | 0.15 | 0.29 | 4 % / 129 % |
| Spanish, male | 0.36 | 0.54 | 0.23 | 0.39 | 7 % / 3 % |
| Russian, male | 0.22 | 0.38 | 0.44 | 0.62 | 7 % / 3 % |
| Russian, male | 0.29 | 0.52 | 0.34 | 0.66 | 7 % / 7 % |
| Russian, female | 0.34 | 0.61 | 0.45 | 0.75 | 3 % / 13 % |
| Russian, female | 0.40 | 0.56 | 0.46 | 0.69 | 7 % / 4 % |

- It is not the speaker's sex: Russian female voices clone as well as male ones. **The Russian Pocket weights carry a
  voice across languages badly:** an English or Spanish voice in Russian is at 0.15–0.23, next to another speaker
  (0.09–0.19), a Russian voice at 0.34–0.46. The words stay (WER 3–11 %). The studio English reader above had 0.36–0.39.
- Qwen3 loses similarity across languages too (0.29–0.42 against 0.62–0.75) and, in the speaker-vector mode, runs away
  on Russian with the two English female voices (WER 74 % and 129 %).

**A wider search for a cloned Russian voice on a CPU** (rows `vNN-chatterbox`, `vNN-voxtream` in
`bench/results/voices.jsonl`). Of the candidates found (F5-TTS Russian fine-tunes, OuteTTS, Audio8, Fish Speech,
CosyVoice3, Russian XTTS fine-tunes, GPT-SoVITS with Russian weights, Supertonic 3 style fitting, voice converters on
top of a stock voice) only two claim a clone in Russian with a CPU runtime; both measured on the same eight voices,
voice → Russian, 5 s reference, the same 6 sentences. Chatterbox Multilingual v3 as GGUF q8 through `tts-cli` of
codec.cpp (lower case + NFKD, `[ru]` prefix, 4 threads); VoXtream2-RU (`simba9/voxtream2-ru`, its own Russian text
front end, torch on the CPU, 4 threads, `VOXTREAM_BEST_OF=1`), on 5 of the 8 voices.

| voice → Russian (sim / WER) | Pocket ru (CPU) | Qwen3 (A4000) | Chatterbox GGUF (CPU) | VoXtream2-RU (CPU) |
|---|---|---|---|---|
| English, male | 0.18 / 4 % | 0.42 / 4 % | 0.50 / 14 % | 0.43 / 14 % |
| English, female | 0.19 / 11 % | 0.32 / 74 % | 0.39 / 9 % | 0.39 / 13 % |
| English, female | 0.15 / 4 % | 0.29 / 129 % | 0.60 / 14 % | 0.53 / 13 % |
| Spanish, male | 0.23 / 7 % | 0.39 / 3 % | 0.57 / 10 % | — |
| Russian, male | 0.44 / 7 % | 0.62 / 3 % | 0.72 / 9 % | 0.61 / 11 % |
| Russian, male | 0.34 / 7 % | 0.66 / 7 % | 0.74 / 13 % | — |
| Russian, female | 0.45 / 3 % | 0.75 / 13 % | 0.75 / 4 % | 0.70 / 7 % |
| Russian, female | 0.46 / 7 % | 0.69 / 4 % | 0.70 / 7 % | — |

| speed, i7-13700, 4 physical cores, nothing else on them (3 sentences) | first audio | RTF | memory | streaming |
|---|---|---|---|---|
| Chatterbox GGUF q8 | the whole clause: 6.4–11.6 s for 2.6–5.2 s of speech | 2.2 | 1.6 GB | no |
| VoXtream2-RU | 2.9–3.5 s | 9.7 | 6.2 GB | yes, 80 ms frames |
| Pocket TTS, Shtorm weights, 2 threads (above) | 0.10 s | 0.20 | 1.2 GB | yes |

- **Both carry a foreign voice into Russian far better than the Russian Pocket weights** (0.39–0.60 against 0.15–0.23)
  and better than Qwen3 on the GPU (0.29–0.42), without Qwen3's runaway on the two English female voices. A cloned
  Russian that keeps an English speaker recognisable is possible; the Pocket weights are what loses it.
- **Neither holds a live pace on a CPU:** Chatterbox is 2.2× slower than real time and speaks only after the whole
  clause (the S3Gen decoder takes about two thirds of it; `--timesteps 4/2` does not help); VoXtream streams, but
  10× slower than real time and in 6.2 GB. The whole pipeline was not run: with an RTF above 1 the voice falls behind
  every clause.
- Chatterbox GGUF ran away once in 48 sentences (a short one with Latin letters: 20 s of repeats instead of 2.5 s; the
  GGUF port has no alignment analyser) — a length check as for XTTS would be needed. The similarity rows were made with
  the two engines sharing the same physical cores (hyper-threads), their own speed fields are slower than the table.

**Chatterbox Multilingual on the GPU, as a fallback to Qwen3** (rows `vNN-chatterbox-gpu`, `vNN-chatterbox-gpu-cfg5`):
`chatterbox-tts` 0.1.7 (torch 2.6.0, CUDA 12.4), one A4000, 2 CPU threads, the same eight voices → Russian.

| voice → Russian (sim / WER) | Qwen3 (A4000) | Chatterbox torch (A4000), `cfg_weight` 0.5 | Chatterbox GGUF (CPU) |
|---|---|---|---|
| English, male | 0.42 / 4 % | 0.51 / 7 % | 0.50 / 14 % |
| English, female | 0.32 / 74 % | 0.43 / 7 % | 0.39 / 9 % |
| English, female | 0.29 / 129 % | 0.60 / 19 % | 0.60 / 14 % |
| Spanish, male | 0.39 / 3 % | 0.60 / 9 % | 0.57 / 10 % |
| Russian, male | 0.62 / 3 % | 0.71 / 10 % | 0.72 / 9 % |
| Russian, male | 0.66 / 7 % | 0.72 / 6 % | 0.74 / 13 % |
| Russian, female | 0.75 / 13 % | 0.75 / 9 % | 0.75 / 4 % |
| Russian, female | 0.69 / 4 % | 0.75 / 11 % | 0.70 / 7 % |

- Chatterbox on the A4000: the first audio is the whole clause, 2.0–2.7 s (median per voice), up to 3.2 s; RTF
  0.34–0.40; 3.4 GB of GPU memory at the peak (Qwen3 2.4 GB); 5.0–5.6 GB of process memory; load 6.8 s from a warm
  cache, preparing a voice 0.03 s. No streaming.
- A foreign voice in Russian: 0.54 on average against Qwen3's 0.35, and no runaways in 48 sentences, where Qwen3 ran
  away on 3 of 24 (163 s of repeats instead of 5–6 s: the 74 % and 129 % above). A Russian voice: 0.73 against 0.68.
- `cfg_weight` 0, the upstream advice for a reference in another language, is worse here: WER 9–27 % (an accent),
  the same similarity (rows `vNN-chatterbox-gpu` for the foreign voices); the table uses 0.5 everywhere.
- Code and weights MIT (`ResembleAI/chatterbox` at `5bb1f6ee`, not gated); every output carries an inaudible Perth
  watermark. It pins transformers 5.2.0, `faster-qwen3-tts` needs 5.16.1: not one environment. The optional
  `russian_text_stresser` was not installed (no stress marks).
