# Cloud candidate: Gemini Live (translate + transcribe)

Bench: `bench/cloud_gemini.py` (google-genai 3.0.0, Python 3.12). Measured 2026-11-07 from a 4-core VPS, free-tier key.

## API facts (docs read 2026-11-07)

- Models: `gemini-3.5-live-translate-preview` (speech-to-speech, preview, "June 2026") and
  `gemini-3.5-transcribe-live` (streaming STT, Live endpoint of `gemini-3.5-transcribe`).
  Neither shows up in `client.models.list()` (that list only has generateContent/embed models), but both connect.
  [models](https://ai.google.dev/gemini-api/docs/models),
  [live-translate](https://ai.google.dev/gemini-api/docs/live-api/live-translate),
  [live-transcribe](https://ai.google.dev/gemini-api/docs/live-api/live-transcribe).
- **Gotcha:** google-genai 3.x defaults to `api_version='v1'`; Live is `v1beta` only, otherwise the WebSocket
  handshake fails with HTTP 404. Use `genai.Client(http_options=types.HttpOptions(api_version='v1beta'))`.
- Audio: in = raw PCM s16le 16 kHz mono (`audio/pcm;rate=16000`), 100 ms chunks recommended;
  translate out = PCM s16le 24 kHz mono.
- Translate config: `response_modalities=['AUDIO']`, `translation_config={target_language_code, echo_target_language}`,
  optional `input_audio_transcription` / `output_audio_transcription` (source and translated text).
  No system prompt, no tools, no voice choice; audio input only. `echo_target_language=false` (default) =
  stay silent when the speaker already talks the target language. 70+ languages incl. en, ru.
  It is a continuous interpreter, not turn-based: there are no turns, it translates while you talk.
  Docs limitation: "Voice replication can be inconsistent ... might shift after long pauses, assign the wrong
  gender based on how the speech starts, or get stuck on one voice during rapid multi-speaker conversations"
  - i.e. it does try to imitate the speaker's voice.
- Transcribe config: `response_modalities=['TEXT']`, `input_audio_transcription={language_codes:[...] or [] = auto,
  custom_vocabulary (<=1000, best <=100), mode: VERBATIM|SMART}`. Server emits `interim_input_transcription`
  (cumulative hypothesis of the current segment, ~every 0.5 s) and `input_transcription` (final per segment),
  plus `voice_activity` START/END events with audio offsets. VAD: automatic (default), hybrid
  (server start + client `audio_stream_end` for instant finalisation), manual (`activity_start/end`).
  No diarization, no word timestamps in Live. Max 10 min audio per session.
- Sessions: audio-only sessions 15 min without context compression; a single connection lives ~10 min
  (GoAway before close) -> use session resumption (handles valid 2 h; translate sends a new handle every ~1 s)
  [session-management](https://ai.google.dev/gemini-api/docs/live-api/session-management).
- Pricing (paid tier, [pricing](https://ai.google.dev/gemini-api/docs/pricing)); free tier: free of charge,
  but data "used to improve our products" = Yes on free, No on paid:
  - Live Translate: in $3.50 / 1M tok (~$0.0053/min audio), out $21 / 1M (~$0.0315/min audio),
    25 tok/s of audio -> **~$0.037 per minute of translated speech** (~$2.2/h per direction).
  - Transcribe Live: in $3.50 / 1M (~$0.005/min), out $21 / 1M text (~$0.004/min) -> **~$0.009/min** (~$0.54/h).
- Rate limits: per project, shown only in AI Studio ([rate-limits](https://ai.google.dev/gemini-api/docs/rate-limits));
  preview models are "more restricted"; free tier has no spend cap, Tier 1 = linked billing account.
  No public number for concurrent Live sessions.

## Method

FLEURS dev, 10 parallel EN/RU sentences (5-17 s each). Each utterance = one Live session: 100 ms PCM chunks sent
at mic pace (a chunk leaves when its last sample would have been captured), then 6 s of silence. Speech start/end
= first/last >=100 ms run within 20 dB of the loudest 20 ms frame (may be ~0.1 s early at the end, so
"after end" numbers are slightly pessimistic). Translated audio arrives at playback pace, so "translated speech done"
= arrival of the last voiced 250 ms chunk + its length. Three runs ~20:14, ~20:24, ~20:30 UTC (n=30 per cell),
plus `--continuous 5` (5 sentences glued with 0.15 s gaps, 41-62 s of non-stop speech) and `--stream-end`
(hybrid VAD: client sends `audio_stream_end` ~0.3 s after speech end). ~180 sessions, up to 3 concurrent:
**no 429, no errors, no GoAway** (sessions were < 1 min).

## Latency (s, median / p90; n=30, 3 runs)

| model | dir | setup | first output from speech start | output from speech end | per-run median of last column |
|---|---|---|---|---|---|
| translate | en→ru | 0.24 / 0.33 | first voiced audio 3.20 / 3.42 | translated speech done 3.66 / 4.70 | 3.84, 3.50, 3.65 |
| translate | ru→en | 0.25 / 0.28 | first voiced audio 3.12 / 3.27 | translated speech done 3.15 / 4.05 | 3.08, 3.15, 3.25 |
| transcribe | en | 0.25 / 0.29 | first interim word 1.00 / 1.27 | final text 1.56 / 1.88 (server VAD) | 1.56, 1.56, 1.57 |
| transcribe | ru | 0.25 / 0.28 | first interim word 1.03 / 1.34 | final text 1.70 / 2.39 (server VAD) | 1.68, 1.70, 1.67 |
| transcribe + hybrid VAD | en | 0.22 / 0.38 | 0.97 / 1.24 | final text **0.65 / 0.93** (signal at 0.28 after end) | n=10 |
| transcribe + hybrid VAD | ru | 0.25 / 0.38 | 0.99 / 1.13 | final text **0.68 / 0.77** (signal at 0.30 after end) | n=10 |

Continuous speech (no pauses): translate en→ru translated speech done **6.2-6.3 s** after the speaker stops
(2 runs, 41 s monologue; Russian is longer, so the voice falls behind and the lag grows); ru→en 3.4 s (no growth).
Transcribe: finals only at the speaker's breath pauses (4-5 finals per 40-60 s, 7-17 s apart) - subtitles must
be driven by the interim text; the last final came 3.3 s (en) / 1.7 s (ru) after the end.

Free-tier latency was stable across the three runs (spread of run medians ≤ 0.35 s); the forum complaint did not
reproduce today (Saturday evening UTC). Repeat on a weekday before deciding.

## Quality

| model | dir | metric |
|---|---|---|
| transcribe | en | WER 1.5 %, CER 0.2 % (hybrid VAD: WER 2.4 %) |
| transcribe | ru | WER 1.5 %, CER 0.8 % (hybrid VAD: WER 2.6 %, CER 2.2 % - final words sometimes clipped) |
| translate | en→ru | chrF 61.4 (on the API's output transcript); source-side transcript WER 5.2 % |
| translate | ru→en | chrF 59.7 (output transcript); source transcript WER 3.7 % |

The translation is free and fluent (e.g. "Вскоре надзиратели ... применив слезоточивый газ" → "Soon, the guard in
protective gear went into the courtyard and drove the prisoners into a corner, using tear gas"); names/acronyms
suffer ("CEP Мартейи" → "the fifth sickle for Marte"). Suspected: the output transcript sometimes ends before the audio does
(1510 en→ru: transcript stops at "...от глобального потепления," while voiced audio continues ~2 s), so chrF of the
transcript may underestimate. Back-transcription of the audio is not done here (no local models): index of all
73 translated wavs with references in `audio/out/gemini/back_index.json`; put `{wav: text}` into
`audio/out/gemini/back.json` and rerun `score` to get `chrf_back_whisper`.

**Voice imitation: yes.** Output pitch follows the speaker: F0 correlation in vs out 0.95 (en→ru) / 0.92 (ru→en)
across 10 different FLEURS speakers, median |ΔF0| 1.1-1.2 semitones, gender kept 30/30 and 29/30. Timbre similarity
was not measured (no speaker encoder here) - listen to `audio/out/gemini/translate/*/`.

## Behaviour and gotchas

- Translate is a continuous 1x stream: ~1 s processing blocks (source text block, then target text ~0.2 s later,
  then 250 ms 24 kHz audio chunks at real-time pace). Silence keeps flowing as zero chunks, and usage keeps
  counting: output tokens = input tokens, ~20-25 tok/s of **open session time**, speech or not. Billing is per
  connected minute, not per spoken minute.
- Do **not** send `audio_stream_end` to translate: it ends the output early (median translated speech 5.25 vs 6.25 s
  for the same en→ru sentences; output-transcript chrF on the same ids 52.3 vs 54.6 en→ru, 57.3 vs 58.3 ru→en). Keep streaming silence instead.
- Translate's own input transcript has the same ~3 s lag in 1 s blocks - useless for fast subtitles; run
  `gemini-3.5-transcribe-live` in parallel for those.
- Transcribe interim text is cumulative per segment (replace, not append), ~2 updates/s, so the first word shows
  ~1 s after onset - above the 0.3-0.5 s target. Server VAD waits ~1.2 s of silence (VAD END audio_offset ≈ end + 1.2 s)
  before the final; hybrid VAD cuts that to ~0.35 s after the client signal.
- The transcribe session reports no usage_metadata.
- Long sessions (not tested here): connection ~10 min, session 15 min (transcribe: 10 min audio) → needs
  session resumption + reconnect on GoAway, and a gap-free hand-off of audio.

## Recommendation for the app's cloud mode

1. Per direction run two persistent sessions: `gemini-3.5-transcribe-live` (subtitles from interim text, local VAD →
   `audio_stream_end` for ~0.65 s finals) and `gemini-3.5-live-translate-preview` (voice); stream silence to the
   translator, never `audio_stream_end`; reconnect with resumption handles before the ~10 min limit.
2. Expect the translated voice ~3.2 s behind the speaker and done ~3.2-3.7 s after they stop (6+ s in long en→ru
   monologues): that misses the 1.5-2.5 s target, so present it as "interpreter mode". For lower voice latency, test
   transcribe-live final/interim → fast text MT → local TTS.
3. Cost on paid tier: ~$2.2/h per translated direction (billed for silence too) + ~$0.54/h per subtitle stream →
   ~$5.5/h for a two-way call with subtitles. Free tier costs nothing, but Google uses the data to improve products -
   do not use it for real calls with third parties; free tier is fine for development.
4. Client-side apps should use ephemeral tokens with a locked `translation_config`, not the raw key.
