# Changelog

Versions are tagged and published by hand after a check by a person (no release is published by CI). The section of a
version is the top of its GitHub release notes (`.github/workflows/release.yml`).

## 0.3.0

The first published version: a real-time translator for calls on macOS 14+ (Apple Silicon) and Windows 10 2004+ / 11,
from source on Linux. You hear the other side in your language, as a voice in your headphones and as subtitles; they
hear you in theirs, in your cloned voice on an NVIDIA GPU, else in a stock voice or the cloud's. Each direction runs on
this computer or in the cloud (Google Gemini Live, with your own key; its sound goes to Google). Everything of 0.1.0 and
0.2.0 below is in it.

Languages: any pair of 14 — English, Russian, Ukrainian, German, French, Spanish, Italian, Portuguese, Polish, Turkish,
Chinese, Japanese, Korean, Arabic — both ways.

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
- Where your cloned voice does not speak the language, or no NVIDIA GPU here runs it (too little memory; on Windows
  also an old driver or GPU), a stock voice speaks instead, and the app says so before the download: no clone download
  and no recording asked for nothing (and none for a cloud side whose clone could not take over). Recording or importing a voice asks you
  to confirm it is your own.
- A direction in the cloud gets its stock voice too, so it goes on locally when the cloud fails (unless its subtitles
  come from the cloud; a noncommercial voice — Turkish, Korean, Japanese — only when you accept its license).
- **Your cloned voice in 13 of the 14 languages** (all but Ukrainian) with an NVIDIA GPU: first audio in about 0.2 s and
  3–5 times faster than real time in each of them on an RTX A4000 (other cards not measured); money and clock times
  are said right, and most other numbers (Japanese digits, Turkish and Arabic decimals, a currency sign apart from its
  number); Chinese and Japanese sentences are no longer cut short. The voice pack for Windows brings the text tools for Chinese and Japanese (it is a new download).
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

## 0.2.0 — built, not published

Windows 10 2004+ / 11 (x64): installer and `.zip`, the tray, the hotkey, the subtitles, the call's sound by process
loopback with the call app muted while translating, VB-Audio Virtual Cable as the microphone of the call; the NVIDIA
path for the voice clone through a separate voice pack.

## 0.1.0 — built, not published

macOS 14+ on Apple Silicon: `.dmg`, the menu-bar app, the hotkey, the subtitles, the call's sound by a process tap
(BlackHole 16ch below 14.4), BlackHole 2ch as the microphone of the call; English ↔ Russian on Linux, macOS, locally or
in the cloud.
