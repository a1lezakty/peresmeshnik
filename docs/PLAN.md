# Plan: stages 1–6

Engines and numbers: [ARCHITECTURE.md](ARCHITECTURE.md) §6 and [BENCH.md](BENCH.md). Every stage
is one branch (`stage-N`) with a few large commits, reviewed before merge.

## Three parallel streams

| Stream | Owns | Files (only this stream edits them) |
|---|---|---|
| **S1 — them → me** | capture of the call sound, VAD, streaming ASR, MT, stock-voice TTS, headphone output, caption events, the local pipeline | `peresmeshnik/audio/capture_*.py`, `peresmeshnik/audio/out.py`, `peresmeshnik/local/{vad,asr,mt,tts}.py`, `peresmeshnik/pipeline.py`, `tests/test_pipeline*.py` |
| **S2 — me → them** | mic capture, voice enrollment, cloned TTS, virtual microphone per OS, live ⇄ translation switch in the direction | `peresmeshnik/audio/mic.py`, `peresmeshnik/audio/virtual_mic_*.py`, `peresmeshnik/local/clone.py`, `peresmeshnik/voice.py`, `tests/test_clone*.py`, `tests/test_virtual_mic*.py` |
| **S3 — cloud and shell** | Gemini Live direction, settings, model registry and downloads, UI (tray, overlay, hotkey, setup wizard), app wiring, packaging and releases, more languages | `peresmeshnik/cloud/`, `peresmeshnik/ui/`, `peresmeshnik/app.py`, `peresmeshnik/__main__.py`, `packaging/`, `.github/workflows/release.yml` |

**Shared, frozen at the start of a stage** (a change is a small separate commit agreed by all
three): `peresmeshnik/engines.py` (interfaces), `peresmeshnik/config.py` (settings dataclass,
TOML), `peresmeshnik/models.py` + `models.toml` (registry and fetch; S3 owns, a minimal version
lands on day one of stage 1), `tests/test_idle_cpu.py`.

Contract between streams = `engines.py`: S1 and S2 deliver engines and `LocalDirection`; S3 only
talks to `Direction`, `Overlay`, `Hotkey`. S2 reuses S1's `pipeline.py` (S1 makes the clause
segmentation and threading generic over `Tts`, S2 plugs in the clone TTS and the virtual mic).

## Stage 1 — first end-to-end probe on Linux (files and PipeWire) — ≈ 40 h

Goal: as early as possible, a full round trip on a Linux dev box without a real call.

| Stream | Work | Hours |
|---|---|---|
| S1 | Silero VAD, clause segmentation, streaming ASR (§6 picks), OPUS-MT CTranslate2, stock TTS; `LocalDirection`; PipeWire null sink + monitor capture; `peresmeshnik run --in file.wav` printing captions and writing translated speech; per-link latency log | 16 |
| S2 | enrollment CLI (record/import 30–60 s, quality checks), clone engine behind `Tts`/`VoiceCloner`, PipeWire virtual source `Peresmeshnik Mic`; file → me→them direction → virtual mic → recorded back and checked by ASR | 14 |
| S3 | `config.py`, model registry + downloads with SHA-256 and license screen (CLI), `GeminiDirection` for both directions, cloud latency log; publish our OPUS-MT CT2 conversions | 10 |

Exit: both directions run from files through PipeWire devices on Linux; latency per link
logged; CI gets the **working-mode CPU gate** (pipeline on a 30 s recording: RTF and CPU seconds
per audio second under a threshold) next to the idle gate; numbers added to BENCH.md.

## Stage 2 — the app on Linux — ≈ 44 h

| Stream | Work | Hours |
|---|---|---|
| S1 | live call capture (move the call app's stream to our sink), both directions concurrently, barge-in and queue policy, segmentation tuning on real speech | 12 |
| S2 | live mic path, hotkey live ⇄ translation (no clicks, crossfade), voice quality checks, enrollment UI hooks | 12 |
| S3 | Qt tray, subtitle overlay (click-through, repaint on change only), settings, setup wizard (Linux pages, loopback self-test), hotkey on X11 / portal, per-direction local/cloud choice | 20 |

Exit: a real Meet/Telegram call on Linux works in both directions; idle and working CPU gates green.

## Stage 3 — macOS and release 0.1.0 — ≈ 36 h

| Stream | Work | Hours |
|---|---|---|
| S1 | Core Audio process tap (muted while translating), fallback via BlackHole 16ch for macOS < 14.4; CoreML / Metal where BENCH shows a gain | 12 |
| S2 | BlackHole 2ch routing, wizard pages, clone engine on M1 (MPS/CPU per BENCH), memory fit on 8 GB | 12 |
| S3 | hotkey (`RegisterEventHotKey`), menu-bar icon (template image), `.app` + `.dmg` ad-hoc signed, `release.yml` on tags, CPU gates on macos-15 | 12 |

Exit: **0.1.0** for macOS; a live check on a real M1 Pro (subtitles, voice both ways, hotkey,
CPU in Activity Monitor).

## Stage 4 — Windows and NVIDIA — ≈ 32 h

| Stream | Work | Hours |
|---|---|---|
| S1 | WASAPI process loopback; spike: muting the call app's original (session volume vs loopback), fallback second cable | 10 |
| S2 | VB-Cable routing and wizard pages; CUDA voice pack (onnxruntime-gpu / CTranslate2 CUDA / torch CUDA) | 12 |
| S3 | hotkey (`RegisterHotKey`), tray, installer (Inno Setup) + zip, release on `windows-latest` | 10 |

Exit: **0.2.0**; Windows and the NVIDIA path checked by a tester on Windows/NVIDIA.

## Stage 5 — quality and latency — ≈ 30 h

Fixes from live use: latency per link vs budget, partial translation of long clauses, two people
talking at once, echo cases, voice quality, automatic cloud ⇄ local fallback, crash reports
without user data. Separate research track (not in the three streams): our own timbre converter
on FocalCodec (Apache-2.0) to replace a noncommercial clone model.

## Stage 6 — more languages — ≈ 24 h

Registry by language pair: ASR (Parakeet v3 covers 25 European languages, Whisper the rest), MT
(HY-MT1.5 33 languages / OPUS-MT per pair / NLLB-200), TTS and clone per language (XTTS 17,
Chatterbox 23); language picker; one BENCH row set per added language; **0.3.0**.

Total ≈ 206 h of work; S1/S2/S3 run in parallel inside each stage, so wall time is about
a third of that plus reviews.
