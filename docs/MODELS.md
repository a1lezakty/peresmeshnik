# Models and licenses

Peresmeshnik itself is [PolyForm Noncommercial 1.0.0](../LICENSE). Models are **not** bundled
with the app: it downloads them on first use into the model cache (see
[ARCHITECTURE.md](ARCHITECTURE.md#model-download-and-cache)) and shows each license before the
download. Because the app is noncommercial, noncommercial model licenses (CPML, CC-BY-NC) are
usable; they still bind the user, so the app says so.

Checked 2026-11-07 from the model cards and license files linked below. Which model serves which language is the
registry's `langs` ([`models.toml`](../peresmeshnik/models.toml)); what runs locally per language is the table in the
[README](../README.md#languages), made from it. A language without a local model is translated in the cloud.

## Speech recognition (ASR) and voice activity (VAD)

| Model | Role | Code | Weights | Notes |
|---|---|---|---|---|
| [Whisper](https://github.com/openai/whisper) small / medium / large-v3-turbo via [faster-whisper](https://github.com/SYSTRAN/faster-whisper) | ASR, 99 languages | MIT | MIT | CTranslate2 int8 conversions by Systran (MIT) |
| [GigaAM v3](https://github.com/salute-developers/GigaAM) (Sber) | ASR ru | MIT | MIT | e2e variants add punctuation |
| [T-one](https://huggingface.co/t-tech/T-one) (T-Bank) | streaming ASR ru, 300 ms chunks | Apache-2.0 | Apache-2.0 | trained on 8 kHz telephony |
| [Parakeet TDT 0.6B v3](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3) (NVIDIA) | ASR, 25 European languages incl. en, ru | — | CC-BY-4.0 | attribution required |
| [onnx-asr](https://github.com/istupakov/onnx-asr) | ONNX runtime wrapper for the three above | MIT | — | ONNX conversions keep the original licenses |
| [Silero VAD](https://github.com/snakers4/silero-vad) | VAD | MIT | MIT | |

## Machine translation (MT)

| Model | Role | Code | Weights | Notes |
|---|---|---|---|---|
| [OPUS-MT](https://github.com/Helsinki-NLP/OPUS-MT-train) en-ru, ru-en (Helsinki-NLP, `opus-2020-02-11`) | MT | MIT (Marian), [CTranslate2](https://github.com/OpenNMT/CTranslate2) MIT | CC-BY-4.0 | one model per direction; our CTranslate2 int8 conversion with the original LICENSE/README: release [`models-1`](https://github.com/a1lezakty/peresmeshnik/releases/tag/models-1) |
| [NLLB-200 distilled 600M](https://huggingface.co/facebook/nllb-200-distilled-600M) (Meta) | MT, 200 languages | MIT | **CC-BY-NC-4.0** | noncommercial |
| [HY-MT1.5 1.8B / 7B](https://huggingface.co/tencent/HY-MT1.5-1.8B-GGUF) (Tencent) | LLM MT, 33 languages | — | **Tencent HY Community License** | not licensed in the EU, UK and South Korea; acceptable-use policy; must be shown to the user and be opt-in |
| [llama.cpp](https://github.com/ggml-org/llama.cpp) | GGUF runtime | MIT | — | |

## Speech synthesis (TTS)

| Model | Role | Code | Weights / data | Notes |
|---|---|---|---|---|
| [Piper](https://github.com/OHF-Voice/piper1-gpl) (`piper-tts` 1.x) | fast TTS without cloning | **GPL-3.0** (and espeak-ng, GPL-3.0) | per voice (dataset license): chosen en `john` public domain, ru `dmitri` CC0, en `joe` CC0; ru `denis` CC0 (worse intelligibility); `irina` unknown; en `lessac` Blizzard research license; `ryan`, ru `ruslan` CC-BY-NC-SA | GPL code cannot be linked into a PolyForm app: Piper runs as a **separate process**, never imported. [`piper-runner/`](../piper-runner) is that process: a separate program under GPL-3.0 (its own LICENSE) that the app starts with `uv run --no-project --with piper-tts==1.8.0` and talks to over stdin/stdout; the bench uses the upstream CLI the same way |
| [Silero TTS](https://github.com/snakers4/silero-models) v5 ru, v3 en | TTS without cloning | — | **CC-BY-NC-SA-4.0** | noncommercial, share-alike; torch |
| [XTTS-v2](https://huggingface.co/coqui/XTTS-v2) via [coqui-tts](https://github.com/idiap/coqui-ai-TTS) | voice cloning, 17 languages incl. ru | MPL-2.0 | **Coqui Public Model License 1.0** (noncommercial) | |
| [Chatterbox Multilingual](https://github.com/resemble-ai/chatterbox) (Resemble AI) | voice cloning, 23 languages | MIT | MIT | adds an inaudible watermark |
| [CosyVoice 2 / Fun-CosyVoice3](https://github.com/FunAudioLLM/CosyVoice) | voice cloning | Apache-2.0 | Apache-2.0 | |
| [FocalCodec](https://github.com/lucadellalib/focalcodec) | planned own timbre converter (later research) | Apache-2.0 | Apache-2.0 | |

## Cloud

| Service | Role | Terms |
|---|---|---|
| [Gemini Live API](https://ai.google.dev/gemini-api/docs/live) (`gemini-3.5-live-translate-preview`, `gemini-3.5-transcribe-live`) | cloud mode: speech-to-speech translation, subtitles; all 14 languages of the app ([translate](https://ai.google.dev/gemini-api/docs/live-api/live-translate), [transcribe](https://ai.google.dev/gemini-api/docs/live-api/live-transcribe) language tables, checked 2026-11-09) | Google APIs Terms and Gemini API Additional Terms; the user's own API key; on the free tier Google may use the data to improve its products |

## Not bundled, installed by the user

| Component | OS | License |
|---|---|---|
| [BlackHole](https://github.com/ExistentialAudio/BlackHole) 2ch | macOS virtual microphone | GPL-3.0 (separate driver) |
| [VB-Audio Virtual Cable](https://vb-audio.com/Cable/) | Windows virtual microphone | donationware, no redistribution |
| PipeWire / PulseAudio | Linux, devices created by the app at run time | MIT / LGPL |

## Libraries the app ships with (UI)

| Library | Role | License |
|---|---|---|
| [Qt 6](https://www.qt.io/) via [PySide6-Essentials](https://pypi.org/project/PySide6-Essentials/) | tray, subtitle overlay, settings, setup wizard | LGPL-3.0 (Qt and PySide6 as shared libraries, unmodified; the user may swap them) |
| [jeepney](https://gitlab.com/takluyver/jeepney) | Linux: D-Bus for the XDG GlobalShortcuts portal (hotkey on Wayland) | MIT |
| libX11 (system) | Linux: hotkey on X11 (`XGrabKey` through ctypes) | MIT/X11 |

## Benchmark data (not shipped)

| Data | Use | License |
|---|---|---|
| [FLEURS](https://huggingface.co/datasets/google/fleurs) en_us, ru_ru dev | ASR WER, MT references (parallel sentences) | CC-BY-4.0 |
| FLEURS, 3 + 3 sentences in [`tests/data/`](../tests/data) | 30 s test recordings for the pipeline and the working-mode CPU gate | CC-BY-4.0, attribution in its README |
| FLEURS, sentence 1515 in each of the 14 languages in [`peresmeshnik/ui/samples/`](../peresmeshnik/ui/samples) (shipped) | the setup wizard's test phrase | CC-BY-4.0, attribution in its README |
| [LibriSpeech](https://www.openslr.org/12) test-clean | English voice reference for cloning | CC-BY-4.0 |
| [Russian LibriSpeech](https://huggingface.co/datasets/bond005/rulibrispeech) test (LibriVox) | Russian voice reference for cloning | public domain recordings |
| [AMI Meeting Corpus](https://groups.inf.ed.ac.uk/ami/corpus/) ES2002a, IS1009a (Mix-Headset, manual words) | English conversational speech: clause segmentation, voice lag (`bench/conversation.py`) | CC-BY-4.0 |
| [YODAS2](https://huggingface.co/datasets/espnet/yodas2) ru000 shard 0, two videos with uploaded subtitles | Russian conversational speech: the same | CC-BY-3.0 (videos under Creative Commons on YouTube) |
