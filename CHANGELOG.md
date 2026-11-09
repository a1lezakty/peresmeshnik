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
- Subtitles in Chinese, Japanese, Korean and Arabic (right to left). Without a font for the script (some Linux
  installs) the wizard and the tray say what to install.
- A test phrase in each of the 14 languages for the wizard's "Test" buttons (FLEURS, CC-BY-4.0).
- README: a table of what runs locally and what in the cloud per language, made from the model registry.
- Cloud fallback, crash reports without your data and the latency on screen (stage 5) are in this version too.

## 0.2.0 — built, not published

Windows 10 2004+ / 11 (x64): installer and `.zip`, the tray, the hotkey, the subtitles, the call's sound by process
loopback with the call app muted while translating, VB-Audio Virtual Cable as the microphone of the call; the NVIDIA
path for the voice clone through a separate voice pack.

## 0.1.0 — built, not published

macOS 14+ on Apple Silicon: `.dmg`, the menu-bar app, the hotkey, the subtitles, the call's sound by a process tap
(BlackHole 16ch below 14.4), BlackHole 2ch as the microphone of the call; English ↔ Russian on Linux, macOS, locally or
in the cloud.
