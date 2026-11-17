# Changelog

Versions are tagged and published by hand after a check by a person (no release is published by CI). The section of a
version is the top of its GitHub release notes (`.github/workflows/release.yml`).

## 0.3.2 — 2026-11-17

Your voice reaches the call sooner and sounds smoother, and can speak their language from a sample of yours in it;
their subtitles come sooner; Russian is recognized with fewer mistakes. Measured on GitHub's runners (a virtual M1 of 3
cores, CPU only, and a 4-core x86), not yet on a real Mac: `docs/BENCH.md`, the "0.3.2" sections.

- **Me → them runs on this Mac by default** on a Mac with 16 GB or more, once your cloned voice can speak their
  language here: Hugging Face token given, your voice recorded and confirmed, the pair's models downloaded. Until then
  the cloud as before; the start after it is ready runs locally. A choice you made in the wizard or the tray stays.
  Windows and Linux as before.
- **Your voice keeps up and drops nothing:** from 0.5 s behind it speaks faster at the same pitch, up to 1.4× at 4 s
  behind. Its start after your phrase, median: 1.97 → 1.45 s on the virtual M1, 2.19 → 1.86 s on x86. The same for the
  local voice that takes over when the cloud fails. A voice that speeds up no longer cuts the last ~50 ms of a clause
  (both directions).
- **Their subtitles sooner:** a phrase of theirs ends after 200 ms of silence instead of 300 (the subtitles of a side
  in the cloud or on your server too): median 0.78 → 0.64 s, p90 1.97 → 1.62 s on the virtual M1. Your own voice to the
  call keeps 0.3.1's clause end, so that its clauses stay whole.
- **The cloud says what it is:** "cloud (Gemini): simultaneous interpreter, lags ~3–4 s, the voice is not yours" (not
  theirs, for them → me) in the tray and the wizard.
- **Clause joins in your cloned voice:** between clauses of one sentence ~0.1 s of pause (English: 0.74 → 0.20 s), the
  next clause starts near the loudness of the one before (a jump of +8.0 → +5.1 dB in English, +5.0 → +2.4 dB in
  Russian), no clicks; a clause cut for its length no longer ends on a falling full stop. The next clause is held up
  to 0.4 s for it, mostly while the one before still plays.
- **Your voice in their language** (the wizard's "Your voice in English", or another language they speak; Voices…):
  45 s read in that language, checked for noise, room echo and missing high frequencies (one line says why when it
  fails). When you confirm it, the clones of both recordings say test phrases against one live reading of yours, and
  the sample speaks only if its clone sounds more like you; otherwise your main recording speaks, as before. Your own
  server still clones the main recording.
- **Whose voice speaks:** a tray line "Voice: me → them: your cloned voice, from your English recording; them → me:
  their cloned voice" (or a stock voice, Gemini's voice, … on your server).
- **Russian recognition on GigaAM v3 e2e RNN-T** (was CTC): far-field speech 5.0 % of words wrong instead of 7.8 %
  (Golos), Common Voice 1.8 instead of 2.1 %; its final text comes ~0.1 s later, your voice in the call not later.
  After the update download it in Components… (226 MB); until then the old model recognizes as in 0.3.1, and once the
  new one is there the old one (225 MB) can be removed in Components….
- **Headphones:** README (macOS) and the wizard's Sound devices page advise wired ones, not the AirPods microphone
  (macOS switches AirPods to their phone-call mode, slower and with poor sound both ways).

## 0.3.1 — 2026-11-16

The first published version: a real-time translator for calls on macOS 14+ (Apple Silicon) and Windows 10 2004+ / 11,
from source on Linux. You hear the other side in your language, as a voice in your headphones and as subtitles; they
hear you in theirs, in your cloned voice (on any computer in 7 languages, in 13 on an NVIDIA GPU), else in a stock voice
or the cloud's. Each direction runs on this computer or in the cloud (Google Gemini Live, with your own key; its sound
goes to Google).

Fixed after a check on a real Mac, before the release:

- Model downloads work on every Mac: the app carries its own certificates for HTTPS (on a Mac without them every download
  failed with "certificate verify failed"). Your own `SSL_CERT_FILE` / `SSL_CERT_DIR` still win.
- The setup wizard shows the whole BlackHole hint: `brew install --cask blackhole-2ch`, or the installer from
  existential.audio without Homebrew (the site e-mails the link), and that the Mac may need a restart. On macOS 14.4+
  and Windows it says that in the call app BlackHole 2ch (CABLE Output) is only the microphone and the speaker stays
  your headphones; below 14.4 the next page sets the call app's speaker.
- The microphone list shows only real microphones; the headphones list has no BlackHole, VB-Cable, Zoom's or Teams'
  virtual devices (nobody hears them; eqMac, Loopback, VoiceMeeter and the like stay: they play on to your device).
  "System default" is offered only while the system's default is a real device (installing BlackHole can make it the
  default input). A virtual device saved earlier is refused with what to do, instead of silence.
- With translation off the call hears your microphone again as soon as you choose a real one, set the system's
  default back, or connect one; no need to turn translation on and off.
- The self-test says what silence from the microphone means (no permission, a MacBook with its lid closed, a wrong
  input) instead of "say something".

Languages: any pair of 14 — English, Russian, Ukrainian, German, French, Spanish, Italian, Portuguese, Polish, Turkish,
Chinese, Japanese, Korean, Arabic — both ways.

- Your cloned voice speaks into the call only once confirmed: read two random phrases at the microphone; the app hears
  the words and matches the voice of every part of the recording; the start of the recording (what the clone takes) must
  sound like the rest of it, and the clone made from it like your live voice: someone else's 3 s or more cut into the
  start are refused (record again alone; 2 s may pass, the clone still speaks in your voice). The quiet room before your
  first word is cut from the recording; one with too little speech after its first 12 s is refused when you make it. The
  live reading's speaker vector is kept in `voice.json` and goes with the recording to your own server. A voice recorded
  or imported by an earlier build speaks in a stock voice until confirmed once (tray → Voices…). Tray → **Voices…**
  lists, confirms and deletes the voices.
- The installer puts only the app on the computer. The wizard's **Components** page (and tray → **Components…**) lists
  what can be downloaded — the pair's models, the clone engines, the NVIDIA voice pack — with size, license and what
  it is for, ticked where it suits this computer and the pair; nothing downloads before Download, an unticked one says
  what the pair does without it, and anything downloaded can be removed there. A pair or an engine chosen in the tray
  downloads only after Yes to its size (or Choose…); your cloned voice is downloaded only when you tick it.
- Back to an earlier build after this version: its `python -m peresmeshnik.voice list` stops at a kept voice of someone
  you talk to (`them-…`, it has no `quality`).
- **Your cloned voice without an NVIDIA GPU, on any computer, a Mac too:** Kyutai's Pocket TTS on the CPU (onnxruntime,
  no torch) speaks English, French, German, Spanish, Portuguese and Italian in your voice, and Russian
  (Shtorm-PocketTTS-RU, with stress marks). Your voice starts about 2 s after you finish a phrase; both directions
  cloned at once kept up on 4 cores of a desktop CPU (P-cores of an i7-13700; a weaker laptop is not measured yet). Its
  English to Italian weights download with your Hugging Face token, of an account that accepted Kyutai's conditions
  (without it: a stock voice, said why). Polish, Turkish, Chinese, Japanese, Korean and Arabic without an NVIDIA GPU: a
  stock voice, and the wizard says why.
- On an NVIDIA GPU XTTS-v2 as before; with the voice pack's addon **Qwen3 + Chatterbox** (Components…, only when ticked:
  75 MB of packages and 5.7 GB of models, a card of 8 GB) Qwen3-TTS speaks en ru de fr es it pt zh ja ko (streaming, the
  first sound in ~0.1 s), Chatterbox Multilingual saying again a clause it does not stop on.
- **The other person in their own voice:** "them → me" clones the voice of the person talking from their first 3 s of
  clear speech (made again at 5, 10 and 20 s, between phrases), only into your headphones; who is speaking and their
  clone stay in memory for the call, on disk only when you keep it (tray: Keep the voice of the person in this call…,
  with their agreement, one person on their side, after 10 s of their speech). With two people on their side each
  phrase is in the voice of whoever said it, one not known yet in a stock voice, and a kept voice speaks from the
  first phrase. Phrases with our own speech around them (its echo from their side, my microphone going into the call
  as it is while my side loads) never go into their voice. When the voice falls behind it speaks faster at the same
  pitch; a computer too slow for it gets the stock voice, said why. Its download: the clone engine of your language
  (as for your voice; Russian on the CPU 0.44 GB) and the 27 MB speaker model. With the NVIDIA voice pack keeping it
  saves up to 10 s of their clear speech on this computer (the pack makes their voice from it in the next call; Pocket
  TTS keeps its state, not a recording); delete it in Voices…. Off: tray → Their own voice (them → me), or
  `[local] their_voice = false`: the stock voice as before, nothing more downloaded.
- **Before anything is downloaded**, the setup wizard says for each direction of the pair whether it runs on this
  computer, why not (no speech recognition, translation or voice for a language), how much it downloads and under
  which licenses. A direction that cannot run here is translated in the cloud (Gemini Live has all 14), with the
  cloud's subtitles too. In the apps Turkish, Chinese and Korean are recognized in the cloud (their local model needs
  an NVIDIA GPU with a CUDA onnxruntime: from source only).
- **Another pair from the tray** (Languages → I speak / They speak) without restarting the app: the new pair's models
  are downloaded first (restricted licenses are asked) while translation goes on with the old pair, then both
  directions start again in the new one (meanwhile the call and you hear each other untranslated). A direction you
  keep on this computer goes to the cloud for a new pair only after you say yes, and comes back with a pair that runs
  here. "Local" chosen in the tray or the settings downloads its models first.
- Where your cloned voice does not speak the language, or no NVIDIA GPU here runs it (too little memory; on Windows also
  an old driver or GPU), a stock voice speaks instead, and the app says so before the download: no clone download and no
  recording asked for nothing (and none for a cloud side whose clone could not take over). Recording or importing a
  voice asks you to confirm it is your own.
- A direction in the cloud gets its stock voice too, so it goes on locally when the cloud fails (unless its subtitles
  come from the cloud; a noncommercial voice — Turkish, Korean, Japanese — only when you accept its license).
- **Your cloned voice in 13 of the 14 languages** (all but Ukrainian) with an NVIDIA GPU: first audio in about 0.2 s and
  3–5 times faster than real time in each of them on an RTX A4000 (other cards not measured); money and clock times are
  said right, and most other numbers (Japanese digits, Turkish and Arabic decimals, a currency sign apart from its
  number); Chinese and Japanese sentences are no longer cut short. The voice pack for Windows brings the text tools for
  Chinese and Japanese (it is a new download).
- A text to read for your voice recording in each of the 14 languages: public-domain prose (Kafka, Daudet, Collodi,
  Lu Xun, Akutagawa and others).
- Subtitles in Chinese, Japanese, Korean and Arabic (right to left). Without a font for the script (some Linux
  installs) the wizard and the tray say what to install.
- A test phrase in each of the 14 languages for the wizard's "Test" buttons (FLEURS, CC-BY-4.0).
- README: a table of what runs locally and what in the cloud per language, made from the model registry; what the
  cloud costs and sends, the download sizes, and what stays on disk when you remove the app.
- A direction in the cloud goes on locally when the cloud fails; a crash leaves a report without your data, to read
  and send yourself; the tray can show each phrase's latency.
- **Windows: the call app is no longer left silent.** While translating it is turned down to near silence instead of
  muted (muted, its sound was lost to the translation too), and its volume comes back when translation stops, also
  when the call app was closed first, after a crash, or on a headset unplugged meanwhile (at the app's next sound
  there). A restart of Windows Audio no longer silences the translation of the call; closing the call app is no error.
- **macOS:** after `sudo killall coreaudiod` the sound helpers start again by themselves; Ctrl+C in the Terminal the
  app was started from no longer leaves the call app muted; a busy microphone is taken once it is free; the helper
  no longer grows with every call.
- **Licenses and builds:** the source offer of the LGPL libraries (Qt, libsndfile) and the notices of the code inside
  Qt, of hts_engine and of the Unicode data are in the licenses folder; the macOS app runs with the hardened runtime;
  the Windows uninstaller says what stays on disk (your Gemini key, the models) and offers to remove your settings.
  Releases are built by pinned actions and tools, only from a commit whose tests and voice-clone checks passed.

## 0.3.0 — built, not published

0.3.1 without the fixes above: found on a real Mac before it went out.

## 0.2.0 — built, not published

Windows 10 2004+ / 11 (x64): installer and `.zip`, the tray, the hotkey, the subtitles, the call's sound by process
loopback with the call app muted while translating, VB-Audio Virtual Cable as the microphone of the call; the NVIDIA
path for the voice clone through a separate voice pack.

## 0.1.0 — built, not published

macOS 14+ on Apple Silicon: `.dmg`, the menu-bar app, the hotkey, the subtitles, the call's sound by a process tap
(BlackHole 16ch below 14.4), BlackHole 2ch as the microphone of the call; English ↔ Russian on Linux, macOS, locally or
in the cloud.
