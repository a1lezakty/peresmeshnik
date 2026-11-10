# Models and licenses

Peresmeshnik itself is [PolyForm Noncommercial 1.0.0](../LICENSE). Models are **not** bundled
with the app: it downloads them on first use into the model cache (see
[ARCHITECTURE.md](ARCHITECTURE.md#8-model-download-and-cache)) and shows each license before the
download. Because the app is noncommercial, noncommercial model licenses (CPML, CC-BY-NC) are
usable; they still bind the user, so the app says so.

Checked 2026-11-07 (stage 6 rows: 2026-11-09) from the model cards and license files linked below. Which model serves which language is the
registry's `langs` ([`models.toml`](../peresmeshnik/models.toml)); what runs locally per language is the table in the
[README](../README.md#languages), made from it. A language without a local model is translated in the cloud.

## Speech recognition (ASR) and voice activity (VAD)

| Model | Role | Code | Weights | Notes |
|---|---|---|---|---|
| [Whisper](https://github.com/openai/whisper) small / medium / large-v3-turbo via [faster-whisper](https://github.com/SYSTRAN/faster-whisper) | ASR, 99 languages | MIT | MIT | CTranslate2 int8 conversions by Systran (MIT) |
| [GigaAM v3](https://github.com/salute-developers/GigaAM) (Sber) | ASR ru | MIT | MIT | e2e variants add punctuation |
| [T-one](https://huggingface.co/t-tech/T-one) (T-Bank) | streaming ASR ru, 300 ms chunks | Apache-2.0 | Apache-2.0 | trained on 8 kHz telephony |
| [Parakeet TDT 0.6B v3](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3) (NVIDIA) | ASR, 25 European languages incl. en, ru | — | CC-BY-4.0 | attribution required |
| [Parakeet TDT-CTC 0.6B ja](https://huggingface.co/nvidia/parakeet-tdt_ctc-0.6b-ja) (NVIDIA) | ASR ja (stage 6) | — | CC-BY-4.0 | ONNX export by [OpenVoiceOS](https://huggingface.co/OpenVoiceOS/nvidia-parakeet-tdt_ctc-0.6b-ja-onnx) (no license file there; the original's applies) |
| [FastConformer hybrid large pc, French](https://huggingface.co/nvidia/stt_fr_fastconformer_hybrid_large_pc) (NVIDIA) | ASR fr with punctuation and case (stage 6): Parakeet v3 drifts into English mid-monologue | — | CC-BY-4.0 | ONNX export by [OpenVoiceOS](https://huggingface.co/OpenVoiceOS/stt_fr_fastconformer_hybrid_large_pc_onnx) |
| [FastConformer hybrid large pcd v1.0, Arabic](https://huggingface.co/nvidia/stt_ar_fastconformer_hybrid_large_pcd_v1.0) (NVIDIA) | ASR ar with punctuation and diacritics (stage 6) | — | CC-BY-4.0 | ONNX export by [OpenVoiceOS](https://huggingface.co/OpenVoiceOS/stt_ar_fastconformer_hybrid_large_pcd_v1.0_onnx) |
| [Whisper large-v3-turbo](https://huggingface.co/openai/whisper-large-v3-turbo) (OpenAI), ONNX by [onnx-community](https://huggingface.co/onnx-community/whisper-large-v3-turbo) | ASR tr, zh, ko (stage 6), NVIDIA GPU only | — | MIT | fp16 for onnx-asr; the export's card states no license, the weights are MIT |
| [onnx-asr](https://github.com/istupakov/onnx-asr) | ONNX runtime wrapper for the three above | MIT | — | ONNX conversions keep the original licenses |
| [Silero VAD](https://github.com/snakers4/silero-vad) | VAD | MIT | MIT | |
| [WeSpeaker ResNet34](https://huggingface.co/Wespeaker/wespeaker-voxceleb-resnet34) (WeSpeaker team) | speaker vector (stage 8): your voice is confirmed when a live reading at the microphone matches the recording; run through onnx-asr's Kaldi features | Apache-2.0 | Apache-2.0 | trained on VoxCeleb (CC-BY-4.0); 26.5 MB, downloaded for the check |

## Machine translation (MT)

| Model | Role | Code | Weights | Notes |
|---|---|---|---|---|
| [OPUS-MT](https://github.com/Helsinki-NLP/OPUS-MT-train) en-ru, ru-en (Helsinki-NLP, `opus-2020-02-11`) | MT: ru→en; en-ru only when chosen (since stage 6 en→ru is quickmt's) | MIT (Marian), [CTranslate2](https://github.com/OpenNMT/CTranslate2) MIT | CC-BY-4.0 | one model per direction; our CTranslate2 int8 conversion with the original LICENSE/README: release [`models-1`](https://github.com/a1lezakty/peresmeshnik/releases/tag/models-1) |
| [quickmt](https://huggingface.co/quickmt) X↔en for the 13 languages (stage 6) | MT: en→X for all 13 (en→ru too: 404 MB to download against OPUS-MT's 86, 450–610 MB of RAM against 175 — on an 8 GB Mac about 0.3 GB more of each), ja / ko / zh / ar→en; a pair without a model of its own goes through English (two models) | [CTranslate2](https://github.com/OpenNMT/CTranslate2) MIT | CC-BY-4.0 | ready CTranslate2 models (float16, int8 at load on a CPU), pinned by revision; the license is in each model card, the repos have no LICENSE file |
| OPUS-MT ru↔uk ([Tatoeba-Challenge](https://github.com/Helsinki-NLP/Tatoeba-Challenge/tree/master/models/rus-ukr) `opus-2021-02-19`) | MT ru↔uk direct (stage 6) | MIT (Marian) | CC-BY-4.0 | our CTranslate2 int8 conversion with the archive's LICENSE/README: release `models-2` |
| OPUS-MT tc-big X↔ru for de / fr / es / it / pt / pl ([Helsinki-NLP](https://huggingface.co/Helsinki-NLP) `opus-mt-tc-big-{de,fr,es,it,pt,zlw}-zle`, `-zle-{de,fr,es,it,pt,zlw}`) | MT X↔ru direct (stage 6): through English the speaker's gender and ты / вы get lost | MIT (Marian) | CC-BY-4.0 | our CTranslate2 int8 conversion of the Hugging Face weights at pinned commits (`>>rus<<` / `>>pol<<` in source_prefix.txt), with the model card and the CC-BY-4.0 text: release `models-2` |
| OPUS-MT X→en for uk / de / fr / es / it / pt / pl / tr ([Helsinki-NLP](https://huggingface.co/Helsinki-NLP) `opus-mt-{uk,de,fr,es,it,pl,tr}-en`, `opus-mt-ROMANCE-en` for pt) | MT X→en and the first half of a pair through English (stage 6): quickmt makes up a continuation of a one-word reply | MIT (Marian) | Apache-2.0 | our CTranslate2 int8 conversion of the Hugging Face weights at pinned commits, with the model card and the Apache-2.0 text: release `models-2` |
| [NLLB-200 distilled 600M](https://huggingface.co/facebook/nllb-200-distilled-600M) (Meta) | MT, 200 languages | MIT | **CC-BY-NC-4.0** | noncommercial |
| [M2M100 418M](https://huggingface.co/facebook/m2m100_418M) (Meta) | MT, 100 languages (measured, stage 6) | MIT | MIT | |
| [Hy-MT2 1.8B](https://huggingface.co/tencent/Hy-MT2-1.8B-GGUF) (Tencent) | LLM MT, 33 languages (measured on a GPU, stage 6) | — | Apache-2.0 | its LICENSE.txt is in the GGUF repo |
| [HY-MT1.5 1.8B / 7B](https://huggingface.co/tencent/HY-MT1.5-1.8B-GGUF) (Tencent) | LLM MT, 33 languages | — | **Tencent HY Community License** | not licensed in the EU, UK and South Korea; acceptable-use policy; must be shown to the user and be opt-in |
| [llama.cpp](https://github.com/ggml-org/llama.cpp) | GGUF runtime | MIT | — | |

## Speech synthesis (TTS)

| Model | Role | Code | Weights / data | Notes |
|---|---|---|---|---|
| [Piper](https://github.com/OHF-Voice/piper1-gpl) (`piper-tts` 1.x) | fast TTS without cloning | **GPL-3.0** (and espeak-ng, GPL-3.0) | per voice (dataset license): chosen en `john` public domain, ru `dmitri` CC0, en `joe` CC0; ru `denis` CC0 (worse intelligibility); `irina` unknown; en `lessac` Blizzard research license; `ryan`, ru `ruslan` CC-BY-NC-SA | GPL code cannot be linked into a PolyForm app: Piper runs as a **separate process**, never imported. [`piper-runner/`](../piper-runner) is that process: a separate program under GPL-3.0 (its own LICENSE) that the app starts with `uv run --no-project --with piper-tts==1.8.0` and talks to over stdin/stdout; the bench uses the upstream CLI the same way |
| Piper voices of stage 6 ([rhasspy/piper-voices](https://huggingface.co/rhasspy/piper-voices), each voice's MODEL_CARD) | stock voice per language | as Piper | uk `ukrainian_tts` CC0, de `thorsten` CC0, fr `siwis` CC-BY-4.0, es `es_MX-ald` Unlicense, it `paola` CC0 (its card says "See URL": the dataset is cc0-1.0), pt `pt_BR-faber` CC0, pl `gosia` CC0; **noncommercial (consent):** tr `dfki`, ko `kss`, ja `hi_fi_captain` (CC-BY-NC-SA-4.0); zh, ar: none usable (zh `chaowen` is fine-tuned from the noncommercial `xiao_ya`, `huayan` unknown; ar `kareem` unknown). **Fine-tuned from `lessac`:** en `joe`, ru `dmitri`, de `thorsten`, fr `siwis`, it `paola`, pt `faber`, pl `gosia`, tr `dfki` (their MODEL_CARDs: "Finetuned from U.S. English lessac voice"), es `ald` through `davefx`: their weights come from a model trained on lessac's data, licensed for research purposes only (CSTR Blizzard 2013, §3.2). The license above is that of each voice's own recordings; whether the research-only terms carry over to fine-tuned weights is open (rhasspy/piper-voices says MIT). The app does not ask for consent for them (it does for tr `dfki`, for its own license); for commercial use check this first | the Japanese voice needs Piper's `[ja]` extra: the runner starts with `--with piper-tts[ja]==1.8.0` ([pyopenjtalk-plus](https://github.com/tsukumijima/pyopenjtalk-plus) 0.4.x, MIT; its bundled dictionary: mecab-naist-jdic, BSD-3-Clause, with UniDic-CSJ words, BSD-3-Clause among UniDic's three licenses); the builds carry it (packaging/piper-requirements.txt) with sudachipy (Apache-2.0) but not Sudachi's 192 MB dictionary: the runner reads kanji by OpenJTalk's dictionary alone (FLEURS ja test: 6 of 321 sentences, 0.05 % of the phonemes differ) |
| [Silero TTS](https://github.com/snakers4/silero-models) v5 ru, v3 en | TTS without cloning | — | **CC-BY-NC-SA-4.0** | noncommercial, share-alike; torch |
| [XTTS-v2](https://huggingface.co/coqui/XTTS-v2) via [coqui-tts](https://github.com/idiap/coqui-ai-TTS) | voice cloning, 17 languages: 13 of the app's 14 (not uk) | MPL-2.0 | **Coqui Public Model License 1.0** (noncommercial) | its text tools: the next table |
| [Chatterbox Multilingual](https://github.com/resemble-ai/chatterbox) (Resemble AI) | voice cloning, 23 languages | MIT | MIT | adds an inaudible watermark |
| [CosyVoice 2 / Fun-CosyVoice3](https://github.com/FunAudioLLM/CosyVoice) | voice cloning | Apache-2.0 | Apache-2.0 | |
| [FocalCodec](https://github.com/lucadellalib/focalcodec) | planned own timbre converter (later research) | Apache-2.0 | Apache-2.0 | |

### The clone's text tools (in the Windows voice pack and the Linux clone venv, never in the app)

XTTS-v2's tokenizer cleans the text of each language with these before the model reads it (coqui-tts imports them;
`voice-runner/requirements-*.in`). Checked on PyPI and in the packages' own license files, 2026-11-09.

| Package | For | License |
|---|---|---|
| [num2words](https://github.com/savoirfairelinux/num2words) 0.5.14 | numbers as words in every language (a coqui-tts dependency) | LGPL-2.1 (in the clone's own process) |
| [ko-speech-tools](https://pypi.org/project/ko-speech-tools/) 0.1.1 | Korean: Hangul to the romanization XTTS reads (a coqui-tts dependency) | Apache-2.0 |
| [pypinyin](https://github.com/mozillazg/python-pinyin) 0.55.0 | Chinese: characters to pinyin with tones | MIT |
| [cutlet](https://github.com/polm/cutlet) 0.5.2, with [jaconv](https://github.com/ikegami-yukino/jaconv) 0.5.0 and [mojimoji](https://github.com/studio-ousia/mojimoji) 0.0.13 | Japanese: kanji and kana to romaji | MIT; MIT; Apache-2.0 |
| [fugashi](https://github.com/polm/fugashi) 1.5.2 | Japanese: word splitting (MeCab, bundled in its wheel) | MIT; MeCab BSD-3-Clause (of its BSD / LGPL / GPL choice) |
| [unidic-lite](https://github.com/polm/unidic-lite) 1.0.8 | Japanese: the MeCab dictionary (UniDic 2.1.2, 260 MB unpacked); no wheel: the voice pack unpacks its folder from the source archive, with its license files | MIT (packaging); UniDic BSD-3-Clause, © The UniDic Consortium |

## Texts to record your voice

The wizard shows a text to read aloud in your language (`voice.PROMPTS`): for ru and en one written for the app; for
the other 12 public-domain prose, verbatim from these sources (every author died before 1950).

| Language | Author (died) | Work, year | Source |
|---|---|---|---|
| uk | Михайло Коцюбинський (1913) | Intermezzo, 1909 | [uk.wikisource.org](https://uk.wikisource.org/w/index.php?title=Твори_%28Коцюбинський,_1955%29/2/Intermezzo&action=render) |
| de | Franz Kafka (1924) | Die Verwandlung, 1915 | [www.gutenberg.org](https://www.gutenberg.org/cache/epub/22367/pg22367.txt) |
| fr | Alphonse Daudet (1897) | Lettres de mon moulin (Installation), 1869 | [www.gutenberg.org](https://www.gutenberg.org/cache/epub/36780/pg36780.txt) |
| es | Horacio Quiroga (1937) | Cuentos de la selva: La tortuga gigante, 1918 | [es.wikisource.org](https://es.wikisource.org/w/index.php?title=La_tortuga_gigante&action=render) |
| it | Carlo Collodi (1890) | Le avventure di Pinocchio, 1883 | [www.gutenberg.org](https://www.gutenberg.org/cache/epub/52484/pg52484.txt) |
| pt | Machado de Assis (1908) | Conto de Escola (Várias Histórias), 1884 | [pt.wikisource.org](https://pt.wikisource.org/w/index.php?title=Conto_de_Escola&action=render) |
| pl | Henryk Sienkiewicz (1916) | Latarnik, 1881 | [wolnelektury.pl](https://wolnelektury.pl/media/book/txt/latarnik.txt) |
| tr | Ömer Seyfettin (1920) | Kaşağı, 1919 | [tr.wikisource.org](https://tr.wikisource.org/w/index.php?title=Kaşağı&action=raw) |
| zh | 鲁迅 (1936) | 从百草园到三味书屋（朝花夕拾）, 1926 | [www.marxists.org](https://www.marxists.org/chinese/reference-books/luxun/06/008.htm) |
| ja | 芥川龍之介 (1927) | トロッコ, 1922 | [www.aozora.gr.jp](https://www.aozora.gr.jp/cards/000879/files/43016_16836.html) |
| ko | 이효석 (1942) | 메밀꽃 필 무렵, 1936 | [ko.wikisource.org](https://ko.wikisource.org/w/index.php?title=메밀꽃_필_무렵&action=raw) |
| ar | جبران خليل جبران (1931) | الأجنحة المتكسرة — في باب الهيكل, 1912 | [ar.wikisource.org](https://ar.wikisource.org/w/index.php?title=الأجنحة_المتكسرة/في_باب_الهيكل&action=raw) |

To confirm a voice as yours (stage 8), the app shows two of eight short everyday sentences in your language (`voice.LIVE_PHRASES`), written for the app, the same eight in every language.

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
