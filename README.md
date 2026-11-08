<p align="center"><img src="assets/icons/app-256.png" width="128" alt="Peresmeshnik icon"></p>

# Peresmeshnik

Real-time translator for calls. You hear the other side in your language — as a voice in your
headphones and as subtitles — and they hear you in theirs, spoken in **your own cloned voice**.
Works with any call app (Google Meet, Zoom, Telegram, …) through system audio devices.

**Status: in development** — stage 0: architecture and model benchmarks. Nothing to install yet.

- English ↔ Russian first, more languages later.
- macOS, Windows and Linux. Local models (CPU, Apple Silicon, optional NVIDIA GPU) or cloud
  (Gemini Live), switchable in settings.
- Voice cloning of **your own voice only**, from a 30–60 s recording.
- A hotkey switches between translation and your live voice.

Design and numbers: [architecture](docs/ARCHITECTURE.md), [benchmarks](docs/BENCH.md),
[plan](docs/PLAN.md), [models and licenses](docs/MODELS.md).

## Installation

Later, with release 0.1.0 (macOS first). Builds are ad-hoc signed: on first launch macOS asks you
to allow the app in System Settings → Privacy & Security → "Open Anyway".

### Trying it from source (Linux)

Not ready for calls yet. With PipeWire and Python 3.11+: `uv run peresmeshnik app`. Qt needs
`libxcb-cursor0` (`sudo apt install libxcb-cursor0` on Debian / Ubuntu); on Wayland the app runs
through XWayland so that its subtitles stay on top. The call's sound is not routed into the app
yet: until that lands, move the call app's output to "Peresmeshnik Call" yourself (pavucontrol or
the sound settings), otherwise you hear silence instead of the translation. One instance runs per user:
while yours is open, the app instances that the tests start exit with code 3 ("already running").

## License

[PolyForm Noncommercial 1.0.0](LICENSE): anyone may use, change and share it for any
noncommercial purpose; selling it or using it commercially is not allowed. Models downloaded by
the app have their own licenses, listed in [docs/MODELS.md](docs/MODELS.md).

Required Notice: Copyright (c) 2026 Akty (https://github.com/a1lezakty)

---

**По-русски.** Пересмешник — переводчик звонков в реальном времени. Собеседника вы слышите
на своём языке — голосом в наушниках и субтитрами, а он вас — на своём, вашим же клонированным
голосом. Работает с любой программой созвона через системные аудиоустройства; macOS, Windows и
Linux, локальные модели или облако. Сейчас в разработке, установка появится с версии 0.1.0.
Лицензия некоммерческая: пользоваться можно всем, продавать нельзя.
