# Architecture

Status: stage 0 design. Numbers come from [BENCH.md](BENCH.md); where a decision rests on a
measurement the table row is named. Engine interfaces live in
[`peresmeshnik/engines.py`](../peresmeshnik/engines.py).

## 1. What the app does

Two independent directions run at the same time during a call:

| Direction | Input | Output | Voice |
|---|---|---|---|
| **them → me** | the call app's sound (system/app loopback) | translated speech in my headphones **instead of** the original + subtitles (translation and original) | a stock TTS voice |
| **me → them** | my microphone | translated speech into a **virtual microphone** that the call app uses | **my cloned voice** |

A global hotkey switches "translation ⇄ live voice": in live mode my microphone goes to the
virtual mic untouched and the original call sound plays in my headphones.

## 2. Pipeline

```
them → me
call app ─▶ capture ─▶ queue ─▶ VAD ─▶ streaming ASR ──partial──▶ overlay: original (grey)
 (tap/loopback/monitor)               │
                                      └─clause end─▶ MT ──▶ overlay: translation (white)
                                                     └──▶ TTS (stock voice) ─▶ headphones
me → them
mic ─▶ capture ─▶ queue ─▶ VAD ─▶ ASR ─clause end─▶ MT ─▶ TTS (my voice) ─▶ virtual mic ─▶ call app
                                                     (hotkey "live": mic ─────────────▶ virtual mic)
```

- **Frames.** Capture delivers 16 kHz mono float32 in 20–32 ms frames. The device callback only
  copies the frame into a queue; everything else runs on worker threads (per direction one for
  VAD+ASR, one for MT, one for TTS, so captions never wait for speech; one playback thread per
  output device).
- **VAD first.** Silero VAD (32 ms frames, 0.1–0.3 ms per frame measured in stage 1: 0.3–1 % of a core)
  gates everything: on silence ASR, MT and TTS do not run at all.
- **Clauses, not sentences.** A segment ends at a VAD pause ≥ 300 ms, at punctuation from the
  ASR (GigaAM e2e and Parakeet emit it), or after 5 s of continuous speech (then at a comma in
  the clause's second half or between words; the next window keeps 1 s of the previous clause in
  front, since Parakeet loses words on a window that starts mid-speech). MT translates each
  clause on its own (OPUS-MT has no context input; a context-aware LLM MT is a stage-5 option);
  TTS starts on the first clause while the speaker is still talking.
- **Subtitles** show ASR partials of the current clause immediately and replace them with the
  final text + translation at clause end. The overlay repaints only when the text changes.
- **Queue and barge-in** (`pipeline.py`; numbers: BENCH, stage 2). Translated speech plays back to
  back and is often longer than its clause, so on continuous speech the voice would fall ever
  further behind. The speak worker knows when a clause's voice would start: from 0.5 s behind the
  clause end it asks the TTS for faster speech (linearly up to 1.4× at the cap; Piper's length
  scale), and a clause whose voice would start more than `max_lag_s` = 4 s after its end is not
  said at all — its subtitle stays. Them → me only: the other side has no subtitles of what me → them
  would drop (`make_local(max_lag_s=None)`, the default). Barge-in: once my own side's partial caption has 3 words (not
  a "yeah" / "ага" while listening), their side drops what is not heard yet (the queue, the clause
  being synthesized, the headphones' buffer: `PulseOut.flush` ends pacat within ~50 ms).

## 3. Engine interfaces

`peresmeshnik/engines.py` (stage 0 contract; streams work against it in parallel):

| Interface | Methods | Implementations (stage 1–3) |
|---|---|---|
| `AudioIn` | `start(on_frames)`, `stop()` | mic (all OS), macOS process tap, WASAPI (process) loopback, PipeWire monitor |
| `AudioOut` | `play(pcm, sr)`, `flush()`, `stop()` | headphones, virtual mic (BlackHole / VB-Cable / PipeWire virtual source) |
| `Vad` | `feed(frame) → events` | Silero VAD (ONNX) |
| `Asr` / `AsrStream` | `stream(lang)`, `feed(pcm) → partial`, `finish() → text` | see §6 |
| `Translator` | `translate(text, src, tgt, context)` | see §6 |
| `Tts` | `synth(text, lang, voice) → chunks` | see §6 |
| `VoiceCloner` | `enroll(pcm, sr, transcript) → Voice` | see §6 |
| `Direction` | `feed(pcm)`, `events()`, `close()`; `set_live(on)` where `supports_live` | `LocalDirection` (VAD+ASR+MT+TTS), `GeminiDirection` (one Live session) |
| `Overlay` | `show(caption)`, `hide()` | Qt (PySide6) frameless, click-through, always-on-top window (`ui/overlay.py`) |
| `Hotkey` | `register(combo, cb)`, `unregister(combo)` | macOS `RegisterEventHotKey`, Windows `RegisterHotKey`, Linux X11 `XGrabKey` (ctypes) / XDG GlobalShortcuts portal (D-Bus, jeepney) (`ui/hotkey.py`) |

**Local and cloud behind one interface.** The UI and audio layers only see `Direction`: it eats
16 kHz PCM and yields `Caption` and `Speech` events. `LocalDirection` composes the engines above;
`GeminiDirection` streams the same PCM to a Gemini Live session and maps its transcripts and audio
to the same events. Settings choose per direction. Typical mixes: on an 8 GB Mac *them → me*
local (cheap) and *me → them* in the cloud, because no local clone runs in real time there (§6);
with an NVIDIA GPU both local, since only the local path speaks in a true clone of my voice.

## 4. Latency budget

Definitions: **subtitle latency** = a word is spoken → its original text is on screen.
**Voice latency** (ear–voice span) = a word is spoken → its translation is heard; for the local
pipeline that is *word → clause end* plus *clause end → translated audio*. Numbers: M1 = GitHub
`macos-15` runner (virtual M1, 3 cores, CPU only), x86 = Ryzen 5600X with 4 threads, GPU =
RTX 3060 / A4000. Row names refer to BENCH.md tables.

**Them → me, local**

| Link | M1 | x86 | Source |
|---|---|---|---|
| capture buffer | 20 ms | 20 ms | design |
| wait for the next partial (half of a 300 ms refresh) | 150 ms | 150 ms | design |
| ASR partial: Parakeet v3 on a 2 s window (EN) / T-one 300 ms chunk (RU) | 106 / 19 ms | 169 / 12 ms | asr `win2s_latency_ms`, `chunk_latency_ms` |
| overlay repaint | ≤ 16 ms | ≤ 16 ms | |
| **subtitle, original** | **≈ 0.30 s EN / 0.21 s RU** | **≈ 0.36 / 0.20 s** | target 0.3–0.5 s ✓ |
| end of clause (VAD silence) | 300 ms | 300 ms | design |
| final ASR on a ~3 s clause | ~0.12 s | ~0.13 s | asr `utt_latency_ms` scaled from 10 s |
| MT, OPUS-MT | 112–226 ms | 57–67 ms | mt |
| **subtitle, translation (after clause end)** | **≈ 0.55–0.65 s** | **≈ 0.5 s** | |
| TTS first audio, Piper (`lessac`/`irina`; the chosen `john`/`dmitri` are within ~30 % of them on one host, §6) | 232–265 ms | 125–172 ms | tts `ttfa_ms_median` |
| output buffer | 40 ms | 40 ms | design |
| **clause end → voice** | **≈ 0.8–0.95 s** | **≈ 0.65–0.71 s** | |
| **voice, ear–voice span** (clauses of 2–3 s: a word is on average 1–1.5 s before the clause end) | **≈ 1.8–2.5 s** | **≈ 1.7–2.2 s** | target 1.5–2.5 s ✓ |

Clause length drives the voice latency: the segmenter cuts at pauses ≥ 300 ms, at ASR
punctuation and hard at 5 s, so the worst case (first word of a 5 s clause) is ≈ 5.8–6 s.
Measured on conversational speech (BENCH, stage 2; B, real time): ear–voice span p50 2.3 s (en
meetings) – 3.5 s (ru monologue), p95 4.7–5.3 s, with the queue policy (§2) holding the voice
within 4 s of its clause end; without it the queue alone let the voice fall 20 s behind a fast
Russian speaker. An 8 s limit translates 0–2 chrF better and adds 1.2–2.4 s to the p95, so 5 s
stays.

**Me → them, local clone** — the same chain with the clone TTS instead of Piper:

| Clone TTS first audio / RTF | M1 | x86 CPU | NVIDIA |
|---|---|---|---|
| XTTS-v2 streaming | not live: M1 runner CPU TTFA 4.5–5.3 s, RTF 6.5–8.2 | TTFA 1.2–1.4 s, RTF 1.8 (8 threads) → falls behind | **TTFA 0.14 s, RTF 0.17** → voice ≈ 1.6–2.1 s ✓ |
| Chatterbox Multilingual | — (7 GB on CPU) | RTF 3.3–3.5 | RTF 0.4, ~2.5 s per sentence (no streaming) → ≈ 3.5–4 s |

**Cloud (Gemini Live), measured from the dev server in Europe, Saturday evening**

| | median | p90 |
|---|---|---|
| translate: first translated audio after speech **start** | 3.1–3.2 s | 3.3–3.4 s |
| translate: translated audio done after speech **end** | 3.1–3.7 s | 4.0–4.7 s |
| translate, 41–62 s of continuous speech: done after end | 3.4 s ru→en, **6.2 s en→ru** (Russian is longer, lag grows) | |
| transcribe-live: first word after speech start | 1.0 s | 1.3 s |
| transcribe-live: final text after speech end, local VAD sends `audio_stream_end` | 0.65–0.68 s | 0.77–0.93 s |

Gemini's voice runs ~1 s behind the local estimate and its subtitles miss the 0.3–0.5 s target,
so even in cloud mode the subtitles come from the local ASR (Parakeet costs ~0.1 CPU s per audio s
on M1, 0.17–0.18 on x86).

## 5. CPU and memory

Requirement from day one (Zagon 0.1.2 burned 104 % CPU on a redraw loop):

**Idle (app open, translation off):** ~0 CPU. No timers, no polling, no redraws. Audio streams
are closed, `Direction.close()` has unloaded the engines or left them without busy threads. CI
gate: `tests/test_idle_cpu.py` starts the app (tray, hidden overlay, virtual mic, and the setup
wizard, since CI is a first run; Linux under Xvfb with a tray host), waits 60 s and fails above
**0.13 s CPU** (on macOS the 60 s start 15 s after the idle line; AppKit's one-off work after the first
show, 0.07–0.09 s, is printed, not gated) (Linux, macOS, Windows on pushes to main and on PRs); `tests/test_ui.py` repeats it
with the overlay on screen. The working-mode gate (`tests/test_pipeline_cpu.py`,
CI job `work-gate`, three OSes) runs `peresmeshnik run` on a 30 s recording and fails at RTF ≥ 1.0
or ≥ 2.0 CPU s per audio s (0.27–0.54 and 0.55–1.09 on the runners, which vary up to 2× between runs; BENCH.md, stage 1).

**Working:**
- device callbacks only enqueue; heavy work on worker threads;
- VAD cuts silence before ASR; on silence ASR/MT/TTS do not run;
- every runtime gets an explicit thread count: onnxruntime `intra_op_num_threads`,
  CTranslate2 `intra_threads`, torch `set_num_threads`, llama.cpp `-t`; the process sets
  `OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, `VECLIB_MAXIMUM_THREADS` before importing numpy
  (the bench caught numpy/OpenBLAS in onnx-asr preprocessing using every core);
- onnxruntime sessions use `session.intra_op.allow_spinning=0`, OpenMP `OMP_WAIT_POLICY=PASSIVE`,
  so pools sleep between calls (`idle_loaded` rows in BENCH.md);
- the overlay repaints only on new text.

**Per link** (threads are fixed; CPU = CPU seconds per second of speech for a full pass;
RAM = peak RSS of the model's own process):

| Link | Threads | CPU s / audio s, M1 / x86 | RAM | Notes |
|---|---|---|---|---|
| Silero VAD | 1 | 0.003–0.01 (stage 1, BENCH `pipeline` vad_rtf) | ~50 MB | runs on every frame while a direction is on |
| Microphone (`parec`) + live switch (me → them) | 1 reader + 1 events thread | 0.0096 + 0.012 while sound goes out (BENCH stage 2) | — | the switch is idle while the direction is silent |
| ASR Parakeet v3 int8 | 2 | 0.10 / 0.18 | 1.4–1.8 GB | partials: re-running every 300 ms on a window growing to 3 s costs ~0.5 (M1) – 0.9 (x86) core while someone speaks; a 500 ms refresh halves it for +100 ms subtitle latency |
| ASR GigaAM v3 e2e (RU, ≥ 16 GB) | 2 | 0.16 / 0.15 | 1.7–1.8 GB | fp32; int8 to try in stage 1 |
| ASR T-one (RU partials, ≥ 16 GB) | 1 | 0.21 / 0.19 | 0.7 GB | 12–19 ms per 300 ms chunk |
| MT OPUS-MT (per direction) | 2 | a burst of ~60 ms per clause | 0.17 GB | load 0.3 s |
| TTS Piper (own process, §10) | 2 | RTF 0.03–0.06 | 0.3–0.5 GB | |
| Clone XTTS-v2 | GPU (+1 CPU) | CPU RTF 1.8 → not live | 3.6–4.5 GB RAM, 2.1 GB VRAM | |
| Clone Chatterbox | GPU | CPU RTF 3.3–3.5 → not live | 5–7 GB RAM, 3.7 GB VRAM | |
| Gemini Live direction (translate + transcribe) | 1 asyncio thread | 0.03 (x86, measured in stage 1) | ~90 MB process | network only; nothing exists while translation is off |

Concurrently busy threads stay ≤ the number of performance cores (4 on M1): ASR, MT and TTS of
one direction mostly alternate, and the two directions rarely talk at the same time.

**Per tier:**

| Tier | Idle | Working, them → me | Working, both directions |
|---|---|---|---|
| M1 8 GB | ~0 CPU; engines unloaded when translation is turned off ("free memory" default on), UI ~150 MB | ≈ 1 core while they speak, 0 in silence; ~2.5 GB | me → them through Gemini: +~0 CPU, +0.1 GB → ~2.6 GB |
| M1 Pro 16 GB | ~0 CPU; engines may stay loaded (pools sleep: `idle_loaded` 0.0005–0.013 CPU s per 20 s with the flags above) | ~1 core, ~4.5 GB (GigaAM + T-one for RU) | local XTTS only if the machine's own check gives RTF < 0.7, else Gemini |
| NVIDIA ≥ 6 GB VRAM | same | ~1 core | + XTTS on GPU: ~0.5 core, +4.5 GB RAM, 2.1 GB VRAM |

Measured with both directions at once (BENCH, stage 2): on the 7950X3D with an RTX 3060 Laptop
(both local, XTTS on the GPU) 0.61–0.83 cores, 3.5–3.7 GB RAM, 2.2 GB VRAM; on a 4-core x86 box
(them → me local, me → them Gemini with local subtitles) 1.1 cores, 2.0 GB.

Engines load when translation turns on (cached models: OPUS-MT 0.3 s, Parakeet 2–7 s, XTTS ~10 s),
so the first switch-on shows "warming up".

## 6. Engine choice per tier

| Link | Choice | Why (BENCH.md) | Not chosen |
|---|---|---|---|
| VAD | **Silero VAD** (ONNX) | tiny, MIT, 32 ms frames | — |
| ASR, English | **Parakeet TDT 0.6B v3**, int8, via onnx-asr | WER 4.6–4.8 %, the same as Whisper large-v3-turbo, at 106–169 ms per 2 s window vs turbo's 5.4–10.4 s; RTF 0.04; also does Russian | Whisper: pads every window to 30 s, so a 2 s refresh costs 1.1 s (small) to 10 s (turbo) on CPU; on M1 CPU turbo is slower than real time (RTF 1.2). Whisper turbo stays for other languages on GPU (0.3 s per utterance). |
| ASR, Russian | **GigaAM v3 e2e CTC** (WER 5.6 %, punctuation; streamed 6.6 % vs Parakeet's 10.8 %, 0.6 GB vs 1.7 GB; stage 1 default, since Parakeet has no language setting and drifts into Ukrainian on short windows — BENCH stage 1). Both directions on 8 GB: Parakeet (en) + GigaAM (ru) ≈ 2.3 GB. ≥ 16 GB: + **T-one** for partials (12–19 ms per 300 ms chunk) | memory vs accuracy | T-one alone: WER 13 % on FLEURS (it is an 8 kHz telephony model; real call audio may favour it — re-check in stage 2); Whisper medium/turbo: 3–10 s per window on CPU |
| MT | **OPUS-MT** en-ru / ru-en, CTranslate2 int8 | chrF 59.5 / 59.9 at 57–67 ms per sentence, 0.17 GB, CC-BY | NLLB-600M: chrF 57.7 / 63.6 but 0.6–0.75 s and noncommercial. HY-MT1.5-1.8B: 54.8 / 62.2, 1.5–2 s on CPU. HY-MT1.5-7B: best ru-en (65.7) but 6–10 s on CPU, 0.5–1.1 s on GPU, 4.7 GB — optional "quality" MT on the NVIDIA tier for ru → en only. Gemini translate: 58–62. |
| TTS, stock voice (them → me) | **Piper** in its own process (§10); voices **`en_US-john`** (public domain) and **`ru_RU-dmitri`** (CC0), `en_US-joe` (CC0) as a second English voice | chosen voices on the dev server (4 cores): TTFA 0.21–0.27 s, RTF 0.05–0.06, back-transcription CER 1.8 % (john), 3.0 % (joe), 5.9 % (dmitri), 0.3–0.37 GB; on that host they are within ~30 % of `lessac`/`irina`, the voices measured on M1 (0.23–0.27 s) and on the x86 host (0.13–0.17 s) | `ru_RU-denis` (CC0): CER 10.3 %; `lessac`/`irina`: research-only or unknown data licenses; Silero: TTFA 0.2–0.4 s, 1–1.3 GB with torch, CC-BY-NC-SA |
| Clone TTS (me → them) | NVIDIA: **XTTS-v2** streaming with a length/tail guard (TTFA 0.14 s, RTF 0.17); **Chatterbox** as the "better likeness" option (speaker similarity 0.52–0.78 vs XTTS 0.35–0.68, cross-lingual ru → en 0.68–0.70 vs 0.40–0.44, but RTF 0.4 without streaming). CPU / M1: no live local clone — Gemini translate (keeps pitch and gender, F0 correlation 0.92–0.95, not a true clone) until stage 3 measures MPS on a real M1 Pro | similarity baseline: two different speakers 0.085, same speaker 0.90–0.93 | CosyVoice3: 8.4–8.8 GB RAM, RTF 1.2–2 even on the A4000, truncates Russian |
| Cloud | **Gemini Live**: `gemini-3.5-live-translate-preview` per direction for voice; subtitles local (or `gemini-3.5-transcribe-live` when the machine is too weak) | see §4 | |

Recommended default on M1 8 GB: **them → me local, me → them cloud**.

## 7. Audio routing per OS and setup guidance

| | Capture "them" (call sound) | Mute original while translating | Virtual microphone "me → them" |
|---|---|---|---|
| **macOS ≥ 14.4** | Core Audio **process tap** of the call app (`CATapDescription`), no driver | tap with `muteBehavior = mutedWhenTapped` | **BlackHole 2ch** (user installs: `brew install blackhole-2ch` or the pkg); call app mic = BlackHole |
| macOS < 14.4 | call app speaker = BlackHole 16ch, the app captures it and plays to headphones | implicit (the app is the only path to the headphones) | BlackHole 2ch |
| **Windows 10 2004+ / 11** | WASAPI **process loopback** of the call app (`AUDIOCLIENT_ACTIVATION_TYPE_PROCESS_LOOPBACK`); fallback: endpoint loopback | lower the call app's audio session volume (`ISimpleAudioVolume`); stage-4 spike checks that process loopback is taken before the session volume, fallback: call app speaker = second virtual cable | **VB-Audio Virtual Cable** (user installs); call app mic = "CABLE Output" |
| **Linux (PipeWire / PulseAudio)** | the app creates a null sink `Peresmeshnik Call`, moves the call app's stream there (`pactl move-sink-input`) and records its monitor | implicit (only the app plays to the headphones) | the app creates a virtual source `Peresmeshnik Mic` (`module-null-sink` + `module-remap-source`); call app mic = Peresmeshnik Mic; removed on exit |

Python audio: `sounddevice` (PortAudio) for mic, headphones and virtual devices on all OS;
process taps (macOS) and process loopback (Windows) through small native helpers (PyObjC /
`comtypes`, or a tiny Swift/C++ binary streaming PCM over a pipe if bindings fall short).

**Echo safety.** Translated speech goes to headphones only; on first run the app asks to use
headphones and warns when the output device is built-in speakers (the mic would pick up the
translation and translate it back).

**Setup wizard (first run, re-runnable from the tray):**
1. Language pair and mode (local / cloud), tier detected from the hardware (§9), model download
   with sizes and licenses (§8).
2. Virtual mic: detect BlackHole / VB-Cable / PipeWire. Missing → one screen per OS with the
   install link (macOS: Homebrew command or pkg; Windows: VB-Cable page, "run as administrator",
   reboot note), then re-check.
3. Loopback self-test: the app plays a short chirp into the virtual mic, records it back from
   the device's input side and shows "OK" — proves the routing without a call.
4. "In your call app choose microphone **X**" with screenshots for Meet, Zoom, Telegram.
5. Voice enrollment: read a prompted text for 30–60 s; the app checks level, clipping and noise
   and makes the voice profile (only the user's own voice; the profile stays local).
6. Hotkey choice and a test phrase in both directions.

**Stage 2 on Linux** (`ui/wizard.py`): pages 1 (languages, local / cloud per direction with the
recommendation from RAM and GPU, Gemini key, then the models the choice needs with size, license
and a consent box for restricted ones; download resumes), 2 (headphones and microphone from
PipeWire; "Peresmeshnik Mic" is made by the app at start and removed at exit; a warning when the
output looks like loudspeakers), 3 (`ui/selftest.py`: a 0.4 s chirp through the call capture sink,
the virtual mic and the headphones sink, each recorded back and found by cross-correlation; the
microphone's level while the user speaks), 6 (hotkey; a recorded phrase played through each
direction into the headphones, subtitles in the overlay) and a last page for step 4 in words.
Voice enrollment (5, `VoicePage`): record from the chosen microphone or import 30–60 s; the verdict of
`voice.check` (length, speech time, level, clipping, background, a telephone band) and, for a local
me → them, of `voice.check_clone` (the clone says two test phrases: the app's ASR must understand them,
and XTTS's speaker encoder must find the recorded voice; thresholds in BENCH; not while a local
me → them runs, a second clone would not fit). A bad take is not kept; a good one replaces the saved voice
on Finish only (a draft `voices/<name>.new` until then, Cancel drops it).

**App shell** (`app.py`, `ui/`). The tray menu: translation on/off, live voice (the hotkey's
fallback), local / cloud per direction, settings, wizard, quit; without a system tray (GNOME without
the AppIndicator extension) the same menu sits on a button in a small window. Translation on loads
the engines on one control thread ("warming up"), then opens capture; off closes every stream and
engine again. A cloud direction speaks with Gemini and takes its subtitles from the local ASR + MT
(`cloud.subtitles = false`, faster, §4) or from Gemini. The hotkey calls `Direction.set_live` where a
direction supports it; the me → them direction, local or cloud, is wrapped in `LiveSwitch`
(`audio/mic.py`): its output is clocked by the microphone's frames, so the queued translation stays in
the process, and each switch is a 30 ms equal-power crossfade, no click (BENCH). While live, the side's
other directions (the local subtitles of a cloud me → them) get no input, only a pause that closes their
open clause: nothing is translated. For any other direction
(them → me for now) the app sends the input straight to the output and skips it (a hard switch). The overlay keeps the last two clauses (rows keyed by clause
number; cloud captions without one attach to the side's open row) and repaints only when the visible
text changes. On Wayland the windows go through XWayland (`QT_QPA_PLATFORM=xcb` unless it is set):
a Wayland client cannot keep itself on top or let clicks through. Hotkey: X11 grab on X11; on
Wayland the GlobalShortcuts portal (bound off the GUI thread, a new portal session for each change
of the key, since a session binds once), whose desktop dialog may change the key. When there is no
global hotkey or the key is taken, the reason shows once as a tray message and on the wizard's
hotkey page, and the tray item "Live voice" does the same job; a failed change keeps the old key.

Devices live with the app (Linux). At start it removes `peresmeshnik_*` modules left by a killed
run, then makes the call capture sink "Peresmeshnik Call" and the virtual mic; it removes them at
exit, also on SIGTERM, SIGINT and SIGHUP. While translation is off a `module-loopback` sends the
real microphone into the virtual mic, so the other side still hears the user; it is unloaded while
the me → them side runs, and "Live voice" is disabled while translation is off. Our own devices are
refused as microphone or headphones, the system defaults included (the app would hear itself).
Headphones that go away (a Bluetooth profile switch, unplugging) are reopened on the saved device
or the default one, with one tray message per outage; so is the microphone (`app.Microphone`: a
`pactl subscribe` sees its source go, since with WirePlumber 0.5 the recording may not end by itself; with
no microphone at all it takes the next new source; when the event stream ends, as on a pipewire-pulse restart,
it subscribes again after a pause and reopens). `Mic` records the PipeWire source by its name, the same
one for the translation and for the loopback (`s.ui.mic`, else the default input; never ours, never a
sink's monitor). The virtual mic's sink and source have `priority.session=-10`, below any real device and
below a device without a priority (WirePlumber counts that as −1), so they never become the default. One instance per user (`QLockFile` in
`XDG_RUNTIME_DIR`): a second start shows a message and exits without touching the first one's
devices.

The call's sound (`audio/capture_pipewire.py` `CallRouter`). While translation is on, every playback
stream of the call app goes into "Peresmeshnik Call" (`pactl move-sink-input`): the ones playing at
switch-on and each new one (a reconnect, the next call), seen from `pactl subscribe` — a process
blocked on the sound server's events, which exists only while translating. Translation off and exit
give each stream back to the device it came from, or to the default output when that one is gone.
The call app is remembered as `ui.call_app` (`application.process.binary`, else
`application.name`: Electron apps all call themselves "Chromium"); with none chosen, the one known
call app playing (Telegram, Zoom, Discord, …) is taken and remembered; otherwise the tray menu
"Call app" lists what plays now (read when the menu opens). A browser is never taken by itself:
its sound is one stream for all tabs (Chrome names every stream "Playback"), so YouTube in another
tab would be translated too — the menu, the wizard and a tray message say so. WirePlumber
remembers every move per application name (`restore-stream`): a stream left in our sink would go
there again next time, so streams go back by a move as well (a move to the default device makes
it forget the target), and at start the app sends other apps' streams out of its sink. After a
SIGKILL the remembered target stays: a new stream of the call app then lands in our sink while
translation is off (silence) until translation goes on and off once.
Output and capture report their end: `PulseOut` waits on pacat's exit and calls `on_end` (the
headphones' wrapper reopens on the next chunk), `PipeWireCapture` on parec's (a tray error).

Checked so far only in a test desktop: always-on-top and click-through under Xvfb (window flags,
and an `xdotool click` through the overlay reaches the window below); the idle gate with an XEmbed
tray (`stalonetray`), not a StatusNotifierItem tray over D-Bus (KDE, GNOME's AppIndicator) nor Qt's
wayland plugin; the portal against a fake one.

## 8. Model download and cache

- Registry: [`peresmeshnik/models.toml`](../peresmeshnik/models.toml) lists every model: id, role,
  languages, license, a URL prefix pinned to a revision (Hugging Face `resolve/<sha>/` or a release
  of this project), files with SHA-256 and size. The tier is picked by the setup wizard (stage 2).
- Cache: `platformdirs.user_cache_dir("Peresmeshnik")/models` (`~/Library/Caches/Peresmeshnik`,
  `%LOCALAPPDATA%\Peresmeshnik\Cache`, `~/.cache/Peresmeshnik`); overridable in settings.
- Download over plain HTTPS with `urllib` (HTTP Range resume, no credentials sent), then SHA-256
  check and a "verified" marker with a fingerprint of the pins; engines only call
  `models.path(id)`, nothing is fetched at run time after setup — the app works offline in local
  mode. CLI: `peresmeshnik models list | fetch <id>...`.
- First download shows each license; noncommercial and territory-limited ones (CPML, CC-BY-NC,
  Tencent HY; `consent = true` in the registry) need an explicit "yes" (`--accept-licenses` without
  a terminal).
- Conversions we need (OPUS-MT → CTranslate2 int8, made from the original Marian models without
  torch) are published once as assets of the project's GitHub release
  [`models-1`](https://github.com/a1lezakty/peresmeshnik/releases/tag/models-1) with the original
  license, README and attribution.
- Voice profiles: `platformdirs.user_data_dir`, never uploaded.

## 9. Hardware tiers

| Tier | Detected by | Them → me | Me → them | RAM while working | Expected |
|---|---|---|---|---|---|
| **1. Apple M1 8 GB** (or any 8 GB laptop) | RAM < 12 GB, no CUDA | local: Parakeet v3 + OPUS-MT + Piper | Gemini translate (cloud); local clone off | ~2.6 GB | subtitles 0.2–0.3 s, voice ≈ 2 s; me → them ≈ 3.2 s |
| **2. M1 Pro 16 GB** (or 16 GB laptop) | RAM ≥ 12 GB | + GigaAM e2e + T-one for Russian | local XTTS if the self-check RTF < 0.7, else cloud | 4.5–9 GB | same; voice ≈ 2 s both ways if the clone is fast enough |
| **3. NVIDIA ≥ 6 GB VRAM** | CUDA device | as tier 2 (optional HY-MT 7B for ru → en on GPU) | XTTS-v2 or Chatterbox on GPU | +4.5 GB RAM, 2–4 GB VRAM | voice ≈ 1.6–2.1 s both ways |
| **Cloud** | user's choice, any machine | Gemini translate + local subtitles | Gemini translate | ~0.5 GB + subtitles ASR | voice ≈ 3.2 s (up to 6 s in long en → ru monologues), ~$2.2/h per direction on the paid tier |

Hardware alone does not decide: on first run (and at voice enrollment) the app times a short
self-test of ASR, TTS and the clone on the actual machine and picks the tier from the measured
RTF; the user can override it in settings.

## 10. Packaging

| OS | Build | Signing | Artifact |
|---|---|---|---|
| macOS (arm64) | PyInstaller one-folder `.app` on `macos-15` runner | ad-hoc (`codesign -s -`), like Zagon: first launch → System Settings → Privacy & Security → "Open Anyway" | `.dmg` |
| Windows (x64) | PyInstaller one-folder on `windows-latest` | unsigned (SmartScreen "More info → Run anyway") | `.zip` + Inno Setup installer |
| Linux (x64) | PyInstaller one-folder on the oldest supported Ubuntu runner (older glibc) | — | `.tar.gz`, AppImage later |

- The base app is **torch-free**: onnxruntime, CTranslate2, llama.cpp and sounddevice keep it
  ~150–250 MB. Engines that need torch (voice cloning) ship as an optional "voice pack" that the
  app downloads into its cache with its own runtime, so users who only need subtitles never get
  a 1–3 GB bundle.
- GPU: macOS uses CoreML/Metal where it pays (llama.cpp Metal, onnxruntime CoreML EP);
  NVIDIA builds use the CUDA variants of onnxruntime/CTranslate2/llama.cpp from the same voice-
  pack mechanism.
- Piper (GPL, with espeak-ng) is never imported: it runs as its own process. Its upstream CLI has
  no thread option and onnxruntime pins its default pool to every physical core (`taskset` does
  not bound it — found in the bench), so the app ships a tiny Piper runner as a separate GPL-3.0
  package that sets `intra_op_num_threads` and `allow_spinning=0`; the app talks to it over
  stdin/stdout.
- CI builds artifacts only on tags (`release.yml`), tests on push/PR as now.

## 11. Open questions and risks

1. **Voice clone on M1.** No local clone model is live on CPU (XTTS RTF 1.8 on 8 i7 threads,
   ~4 GB RAM; Chatterbox 7 GB). GitHub's M1 runner has no Metal for torch, so MPS is unmeasured:
   stage 3 measures XTTS/Chatterbox on a real M1 Pro. Fallback: cloud for me → them.
   Research (stage 5): fast TTS + our own timbre converter on FocalCodec.
2. **Clone quality.** XTTS adds garbage tails after the sentence (most of its CER) → length and
   energy guard; cross-lingual similarity is modest (0.35–0.44); CosyVoice3 truncates Russian.
3. **Gemini Live.** Preview models; the voice lags ~3.2 s and up to 6 s in long en → ru speech;
   `audio_stream_end` cuts the end of a translation (never send it to translate); connections
   last ~10 min (server GoAway at ~540 s), sessions 15 min → handled in stage 1: hand-over at a
   quiet moment with the resumption handle and context compression, dead sessions replaced (BENCH,
   stage 1); the free tier refuses connections beyond a few at once (seen at 8); paid tier bills
   open time incl. silence (~$0.037/min translate); on the free tier Google uses the data — the app says so and suggests
   a paid key for real calls.
4. **Real calls ≠ FLEURS.** Read speech here; calls bring codecs, noise, overlap and accents —
   WER will be higher. Stage 2 adds consented recordings of real calls to the bench.
5. **Capture APIs.** macOS process taps from Python (PyObjC coverage) and muting the original
   on Windows with process loopback are unproven → spikes in stages 3 and 4, with virtual-cable
   fallbacks.
6. **Memory.** onnxruntime keeps Parakeet at 1.4–1.8 GB RSS for a 0.6B int8 model; try the arena
   off and one shared session for both directions.
7. **Licenses.** Piper is GPL → separate process only; XTTS (CPML), NLLB (CC-BY-NC) and Silero
   (CC-BY-NC-SA) are noncommercial; HY-MT is not licensed in the EU/UK/South Korea. Defaults
   (Parakeet, GigaAM, T-one, OPUS-MT, Piper CC0 voices) are permissive.
8. **Echo.** With speakers instead of headphones the mic hears the translation → headphone check
   in the wizard, warning on built-in speakers.
9. **M1 numbers are a lower bound.** The runner is a 3-core virtual M1 without GPU/ANE; a real M1
   has 4 performance cores and CoreML. Stage 3 re-measures on real hardware.
