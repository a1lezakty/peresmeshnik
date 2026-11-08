<p align="center"><img src="assets/icons/app-256.png" width="128" alt="Peresmeshnik icon"></p>

# Peresmeshnik

Real-time translator for calls. You hear the other side in your language — as a voice in your
headphones and as subtitles — and they hear you in theirs, spoken in **your own cloned voice**.
Works with any call app (Google Meet, Zoom, Telegram, …) through system audio devices.

**Status: in development.** Version 0.1.0 for macOS is the first build to try; on Linux it runs
from source. Not yet tried in a real call (see below).

- English ↔ Russian first, more languages later.
- macOS, Windows and Linux. Local models (CPU, Apple Silicon, optional NVIDIA GPU) or cloud
  (Gemini Live), switchable in settings.
- Voice cloning of **your own voice only**, from a 30–60 s recording.
- A hotkey switches between translation and your live voice.

Design and numbers: [architecture](docs/ARCHITECTURE.md), [benchmarks](docs/BENCH.md),
[plan](docs/PLAN.md), [models and licenses](docs/MODELS.md).

## Installation on macOS

macOS 14 Sonoma or later on Apple Silicon (M1 and newer).

1. Download `Peresmeshnik-<version>-macos-arm64.dmg` from
   [Releases](https://github.com/a1lezakty/peresmeshnik/releases) (its SHA-256 is in the release notes),
   open it and drag Peresmeshnik to Applications.
2. The app is not signed by Apple (ad-hoc signature only), so macOS blocks its first start ("Apple could not
   verify…"). Click Done, then System Settings → Privacy & Security → "Open Anyway" next to Peresmeshnik, and
   confirm (on macOS 14 right-click the app → Open also works). The same from the Terminal:
   `xattr -dr com.apple.quarantine /Applications/Peresmeshnik.app`.
3. Install [BlackHole 2ch](https://existential.audio/blackhole/) (`brew install --cask blackhole-2ch`, or
   its pkg): the virtual microphone that the other side hears. In your call app choose
   **BlackHole 2ch** as the microphone. On macOS older than 14.4 also BlackHole 16ch (`blackhole-16ch`) as
   the call app's speaker: the app takes the call's sound from there.
4. Start Peresmeshnik: its icon is in the menu bar (no Dock icon). The setup wizard downloads the models
   (they are not in the app, 0.5–2 GB depending on the choice).
5. macOS asks once for two permissions, both in System Settings → Privacy & Security: **Microphone**
   (what you say is translated) and, when translation is first turned on, **Screen & System Audio Recording →
   System Audio Recording Only** (the call app's sound is recorded while translation is on and muted for you,
   so you hear only the translation). Use headphones: through speakers the microphone hears the translation.
   FaceTime cannot be translated this way: its sound is played by a macOS system process, not by FaceTime.
   The app is signed ad hoc, so after an update to a new version macOS asks for both permissions again.

The hotkey ⇧⌘Space switches between translation and your live voice (Settings to change it; a combo
that is a macOS shortcut or another app's global hotkey is refused with the reason; a shortcut inside
another app's own menus is not detected and the hotkey takes it over everywhere).

Checked so far only on GitHub's `macos-15` runners (a virtual M1 without GPU, without a real call or real
headphones): the app starts from the .dmg, its menu-bar icon, overlay and hotkey work there, it idles at
~0 CPU and translates a recording with the bundled engines. A live check on a real Mac comes before the
release is published.

### Trying it from source (Linux)

Not ready for calls yet. With PipeWire and Python 3.11+: `uv run peresmeshnik app`. Qt needs
`libxcb-cursor0` (`sudo apt install libxcb-cursor0` on Debian / Ubuntu); on Wayland the app runs
through XWayland so that its subtitles stay on top. While translation is on, the app moves the call
app's sound to "Peresmeshnik Call" and gives it back when it is off: Telegram, Zoom, Discord and the like
are found by themselves, otherwise choose the app in the tray menu "Call app" while the call plays. A
browser (Meet) moves with all of its sound, other tabs too. One instance runs per user:
while yours is open, the app instances that the tests start exit with code 3 ("already running").

## License

[PolyForm Noncommercial 1.0.0](LICENSE): anyone may use, change and share it for any
noncommercial purpose; selling it or using it commercially is not allowed. Models downloaded by
the app have their own licenses, listed in [docs/MODELS.md](docs/MODELS.md).

Required Notice: Copyright (c) 2026 Akty (https://github.com/a1lezakty)

The macOS app bundles a Piper TTS runner ([`piper-runner/`](piper-runner/), GPL-3.0, with
[piper-tts](https://github.com/OHF-Voice/piper1-gpl) and espeak-ng): a separate program the app talks
to over a pipe. Its license and those of the other bundled packages are in
`Peresmeshnik.app/Contents/Resources/licenses/`.

---

**По-русски.** Пересмешник — переводчик звонков в реальном времени. Собеседника вы слышите
на своём языке — голосом в наушниках и субтитрами, а он вас — на своём, вашим же клонированным
голосом. Работает с любой программой созвона через системные аудиоустройства; macOS, Windows и
Linux, локальные модели или облако. Сейчас в разработке; первая сборка для macOS — 0.1.0 (установка — выше).
Лицензия некоммерческая: пользоваться можно всем, продавать нельзя.
