<p align="center"><img src="assets/icons/app-256.png" width="128" alt="Peresmeshnik icon"></p>

# Peresmeshnik

Real-time translator for calls. You hear the other side in your language — as a voice in your
headphones and as subtitles — and they hear you in theirs, spoken in **your own cloned voice**.
Works with any call app (Google Meet, Zoom, Telegram, …) through system audio devices.

**Status: in development.** Version 0.2.0 adds Windows to macOS; on Linux it runs from source. Not yet
tried in a real call (see below).

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

The call hears a hum or an old tone while nothing plays: a program that played into BlackHole was killed hard
(Force Quit, a crash), and the BlackHole driver may keep repeating its last sound buffer. That is the driver's known
behaviour; restarting the sound system clears it: `sudo killall coreaudiod` in the Terminal (or restart the Mac).
Peresmeshnik's own audio helpers close their streams themselves when the app goes, even after a Force Quit.
The call app stays silent after its sound helper `peresmeshnik-tap` was killed (Force Quit, a crash): macOS may keep
the app muted as it was while translated, and Peresmeshnik says so. Restart the call app, or run
`sudo killall coreaudiod` (then the call app's sound comes back without restarting it).

The hotkey ⇧⌘Space switches between translation and your live voice (Settings to change it; a combo
that is a macOS shortcut or another app's global hotkey is refused with the reason; a shortcut inside
another app's own menus is not detected and the hotkey takes it over everywhere).

Checked so far only on GitHub's `macos-15` runners (a virtual M1 without GPU, without a real call or real
headphones): the app starts from the .dmg, its menu-bar icon, overlay and hotkey work there, it idles at
~0 CPU and translates a recording with the bundled engines. A live check on a real Mac comes before the
release is published.

## Installation on Windows

Windows 10 version 2004 or later, or Windows 11 (x64).

1. Download `Peresmeshnik-<version>-windows-x64-setup.exe` from
   [Releases](https://github.com/a1lezakty/peresmeshnik/releases) (its SHA-256 is in the release notes) and run
   it. It installs for your user only, without administrator rights, into `%LOCALAPPDATA%\Programs\Peresmeshnik`,
   with a Start menu shortcut; starting at sign-in is a checkbox, off unless you tick it. The `.zip` of the same
   release is the same folder without an installer: unpack it anywhere and run `Peresmeshnik.exe`.
2. The app is not code-signed, so SmartScreen stops the first start of the installer ("Windows protected your
   PC"): click **More info → Run anyway**. Some antivirus programs distrust unsigned programs too.
3. Install [VB-Audio Virtual Cable](https://vb-audio.com/Cable/) yourself (unpack it, run its setup as
   administrator, restart Windows): the virtual microphone that the other side hears. In your call app choose
   **CABLE Output** as the microphone.
4. Start Peresmeshnik from the Start menu: its icon is in the notification area (Windows may put it behind the
   ^ arrow at first; drag it out to keep it in view). The setup wizard downloads the models (they are not in the
   app, 0.5–2 GB depending on the choice). Use headphones: through speakers the microphone hears the translation.
5. While translation is on, the app records the call app's own sound and mutes it on your speakers (you hear the
   translation instead), and gives it back when translation is off. Telegram, Zoom, Discord and the like are found
   by themselves; otherwise choose the app in the tray menu **Call app** while the call plays. A browser (Meet) is
   taken with all of its sound, other tabs too.
6. Without an NVIDIA GPU, what you say is translated and spoken to the other side by the cloud (Gemini); a local
   clone of your own voice needs an NVIDIA GPU.

The hotkey Ctrl+Alt+Space switches between translation and your live voice (Settings to change it; a combo that
Windows keeps for itself, such as Alt+Tab or Win+Space, or that another app has taken is refused with the reason).
The uninstaller (Settings → Apps → Peresmeshnik, or `unins000.exe` in its folder) closes the app and removes its files;
your settings, voice and models stay in `%LOCALAPPDATA%\Peresmeshnik` until you delete that folder. If the voice pack
or its downloads are there (up to about 12 GB), it asks whether to remove them too ("No" by default); a silent
uninstall (`/VERYSILENT`) keeps them.

Checked so far only on GitHub's `windows-latest` runners (Windows Server 2025 in a VM, without a GPU, real sound
devices or a real call): the installer runs silently, the installed app starts, its tray icon, overlay and hotkey
work there, it idles at ~0 CPU, closes cleanly when asked, and the frozen app translates a recording with the
bundled engines. SmartScreen on a fresh PC, VB-Cable, a real call and the NVIDIA voice pack on Windows are
checked by hand before the release is published.

### Trying it from source (Linux)

Not ready for calls yet. With PipeWire and Python 3.11+: `uv run peresmeshnik app`. Qt needs
`libxcb-cursor0` (`sudo apt install libxcb-cursor0` on Debian / Ubuntu); on Wayland the app runs
through XWayland so that its subtitles stay on top. While translation is on, the app moves the call
app's sound to "Peresmeshnik Call" and gives it back when it is off: Telegram, Zoom, Discord and the like
are found by themselves, otherwise choose the app in the tray menu "Call app" while the call plays. A
browser (Meet) moves with all of its sound, other tabs too. One instance runs per user:
while yours is open, the app instances that the tests start exit with code 3 ("already running").

## When something goes wrong

- **The cloud fails.** A side set to the cloud (Gemini) goes local by itself when the connection drops, Google answers
  with an error or the quota is used up, or no translated voice comes `cloud.fallback_after_s` (6 s) after a phrase
  began. What the cloud had not said yet is said by the local voice, what it had said is not repeated; the tray says
  so once. Later, when someone starts a phrase, the app tries the cloud again (after 30 s, then less and less often, at
  most every 15 min while it stays down: nothing runs on a timer) and goes back to it at the next phrase. Going local
  needs the local models (the wizard downloads them) and, for your own voice, the voice pack and an NVIDIA GPU; without
  them the tray says what is missing, and the hotkey gives the call your own voice meanwhile.
- **A crash.** A report goes to the `reports` folder in the app's data folder (macOS `~/Library/Application
  Support/Peresmeshnik`, Windows `%LOCALAPPDATA%\Peresmeshnik`, Linux `~/.local/share/Peresmeshnik`), and the tray menu
  shows **Crash report…**: read it, copy it, or open a GitHub issue with it and send it yourself — the app sends
  nothing. A report has the app's version, the OS, CPU, RAM, GPU, the engines and models of the settings, where it
  failed and the last lines of the log. It has no subtitles or translations, no audio, no names of devices, of the
  call app or of your voice, no key and not your user name. When the Piper voice helper dies, only it starts again and
  the voice goes on a few seconds later; a side whose translation dies otherwise starts again by itself (its models
  load again: up to ~20 s); after 4 failures within a minute it stays off and says why. One tray message each time.
- **Latency.** Tray → **Show latency**: each subtitle ends with its clause's times from the end of the speech to the
  subtitle and to the voice (a cloud phrase: to the end of its voice). Off by default.

## License

[PolyForm Noncommercial 1.0.0](LICENSE): anyone may use, change and share it for any
noncommercial purpose; selling it or using it commercially is not allowed. Models downloaded by
the app have their own licenses, listed in [docs/MODELS.md](docs/MODELS.md).

Required Notice: Copyright (c) 2026 Akty (https://github.com/a1lezakty)

The macOS and Windows apps bundle a Piper TTS runner ([`piper-runner/`](piper-runner/), GPL-3.0, with
[piper-tts](https://github.com/OHF-Voice/piper1-gpl) and espeak-ng): a separate program the app talks
to over a pipe. Its license and those of the other bundled packages are in
`Peresmeshnik.app/Contents/Resources/licenses/` and in the `licenses` folder of the Windows install.

The Windows voice pack (your own voice on an NVIDIA GPU, downloaded by the wizard after you accept its licenses) is a
Python of its own with PyTorch (BSD-3-Clause) and coqui-tts (MPL-2.0); the XTTS-v2 model it runs is under the
[Coqui Public Model License](https://huggingface.co/coqui/XTTS-v2/blob/main/LICENSE.txt) (noncommercial only). It needs an NVIDIA GPU with 6 GB of memory
or more, compute capability 7.0 or newer (GeForce GTX 16xx, RTX 20xx and later: the pack's PyTorch
is not built for GTX 10xx) and driver 572.30 or newer; the wizard checks before the download.

---

**По-русски.** Пересмешник — переводчик звонков в реальном времени. Собеседника вы слышите
на своём языке — голосом в наушниках и субтитрами, а он вас — на своём, вашим же клонированным
голосом. Работает с любой программой созвона через системные аудиоустройства; macOS, Windows и
Linux, локальные модели или облако. Сейчас в разработке; сборки для macOS и Windows — 0.2.0 (установка — выше).
Лицензия некоммерческая: пользоваться можно всем, продавать нельзя.
