# Changelog

Versions are tagged and published by hand after a check by a person (no release is published by CI). The section of a
version is the top of its GitHub release notes (`.github/workflows/release.yml`).

## 0.3.0 — not released yet

Languages: any pair of 14 — English, Russian, Ukrainian, German, French, Spanish, Italian, Portuguese, Polish, Turkish,
Chinese, Japanese, Korean, Arabic — both ways.

- **Before anything is downloaded**, the setup wizard says for each direction of the pair whether it runs on this
  computer, why not (no speech recognition, translation or voice for a language, or your cloned voice falls behind
  without an NVIDIA GPU), how much it downloads and under which licenses. A direction that cannot run here is
  translated in the cloud (Gemini Live has all 14), with the cloud's subtitles too.
- **Another pair from the tray** (Languages → I speak / They speak) without restarting the app: the new pair's models
  are downloaded first (restricted licenses are asked) while translation goes on with the old pair, then both
  directions start again in the new one (meanwhile the call and you hear each other untranslated).
- Where your cloned voice does not speak the language, a stock voice speaks instead, and the app says so.
- **Your cloned voice in 13 of the 14 languages** (all but Ukrainian) with an NVIDIA GPU: first audio in ~0.2 s, about
  5 times faster than real time, in each of them; numbers, money and clock times are said right (Japanese digits,
  Turkish and Arabic decimals, a currency sign apart from its number), Chinese and Japanese sentences are no longer
  cut short. The voice pack for Windows brings the text tools for Chinese and Japanese (it is a new download).
- A text to read for your voice recording in each of the 14 languages: public-domain prose (Kafka, Daudet, Collodi,
  Lu Xun, Akutagawa and others).
- Subtitles in Chinese, Japanese, Korean and Arabic (right to left). Without a font for the script (some Linux
  installs) the wizard and the tray say what to install.
- A test phrase in each of the 14 languages for the wizard's "Test" buttons (FLEURS, CC-BY-4.0).
- README: a table of what runs locally and what in the cloud per language, made from the model registry.
- Cloud fallback, crash reports without your data and the latency on screen (stage 5) are in this version too.
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
