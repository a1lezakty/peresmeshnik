<p align="center"><img src="assets/icons/app-256.png" width="128" alt="Peresmeshnik icon"></p>

# Peresmeshnik

Real-time translator for calls. You hear the other side in your language — as a voice in your
headphones and as subtitles — and they hear you in theirs, spoken in **your own cloned voice** (on an NVIDIA GPU;
elsewhere in a stock voice or the cloud's). Works with any call app (Google Meet, Zoom, Telegram, …) through system
audio devices.

**Status: in development.** Version 0.3.0 adds languages: any pair of 14. macOS and Windows have builds; on Linux it
runs from source. Not yet tried in a real call (see below).

- 14 languages, any pair both ways: English, Russian, Ukrainian, German, French, Spanish, Italian, Portuguese, Polish,
  Turkish, Chinese, Japanese, Korean, Arabic — on this computer where there are models for it, else in the cloud
  ([Languages](#languages)).
- macOS, Windows and Linux. Local models (CPU, Apple Silicon, optional NVIDIA GPU) or cloud
  (Gemini Live), switchable in settings.
- Voice cloning of **your own voice only** (the wizard asks you to confirm the recording is yours), from a 30–60 s
  recording, on an NVIDIA GPU with 6 GB or more (Windows, Linux). On a Mac and without such a GPU no clone keeps up
  live: a stock voice or the cloud speaks for you.
- A hotkey switches between translation and your live voice.

**The cloud is Google's, with your own key.** A direction set to the cloud sends its sound to Google Gemini Live: the
other person's voice for "them → me", yours for "me → them". It needs your own Gemini API key (aistudio.google.com →
Get API key). On the free tier Google may use that audio to improve its models; a paid key is billed by Google, about
$2.2 an hour per direction ([ARCHITECTURE §9](docs/ARCHITECTURE.md#9-hardware-tiers)). Without an NVIDIA GPU the wizard
suggests the cloud for what you say; a direction whose languages have local models (the table below) can stay on
this computer instead (a stock voice then speaks for you). With both directions here no key is needed and nothing is
sent.

Design and numbers: [architecture](docs/ARCHITECTURE.md), [benchmarks](docs/BENCH.md),
[plan](docs/PLAN.md), [models and licenses](docs/MODELS.md).

## Languages

Choose "I speak" and "They speak" in the setup wizard, or later in the tray menu **Languages** (no restart: the new
pair's models are downloaded first while translation goes on, then both directions start again in it; for the seconds
that takes, the call and you hear each other as with translation off). For each
direction the wizard says before anything is downloaded whether it can run on this computer, why not (no speech
recognition, translation or voice for a language), what it downloads and under which licenses; a direction that
cannot run here is translated in the cloud (Gemini Live has all 14 languages). When a new pair from the tray would
move a direction you keep on this computer to the cloud, the app asks first (its sound would go to Google); a later pair
that runs here brings it back. Where your cloned voice does not speak the language, or there is no NVIDIA GPU for it,
a stock voice speaks instead, and the wizard says so.

Speech recognition of Turkish, Chinese and Korean on this computer is Whisper on an NVIDIA GPU, through a CUDA build of
onnxruntime: the macOS and Windows apps have none, so there these three are recognized in the cloud. From source on
Linux with an NVIDIA GPU they can run here ([below](#trying-it-from-source-linux)).

<!-- languages:begin (generated: python -m peresmeshnik.languages) -->
| Language | Speech recognition | Stock voice | Your cloned voice | Cloud (Gemini Live) |
|---|---|---|---|---|
| English | parakeet-tdt-0.6b-v3 | piper-en_US-john-medium | xtts-v2 (NVIDIA GPU) | ✔ |
| Русский (Russian) | gigaam-v3-e2e-ctc | piper-ru_RU-dmitri-medium | xtts-v2 (NVIDIA GPU) | ✔ |
| Українська (Ukrainian) | parakeet-tdt-0.6b-v3 | piper-uk_UA-ukrainian_tts-medium | — (cloud or a stock voice) | ✔ |
| Deutsch (German) | parakeet-tdt-0.6b-v3 | piper-de_DE-thorsten-medium | xtts-v2 (NVIDIA GPU) | ✔ |
| Français (French) | fastconformer-fr-pc | piper-fr_FR-siwis-medium | xtts-v2 (NVIDIA GPU) | ✔ |
| Español (Spanish) | parakeet-tdt-0.6b-v3 | piper-es_MX-ald-medium | xtts-v2 (NVIDIA GPU) | ✔ |
| Italiano (Italian) | parakeet-tdt-0.6b-v3 | piper-it_IT-paola-medium | xtts-v2 (NVIDIA GPU) | ✔ |
| Português (Portuguese) | parakeet-tdt-0.6b-v3 | piper-pt_BR-faber-medium | xtts-v2 (NVIDIA GPU) | ✔ |
| Polski (Polish) | parakeet-tdt-0.6b-v3 | piper-pl_PL-gosia-medium | xtts-v2 (NVIDIA GPU) | ✔ |
| Türkçe (Turkish) | — (cloud; whisper-large-v3-turbo from source with onnxruntime-gpu on an NVIDIA GPU) | piper-tr_TR-dfki-medium | xtts-v2 (NVIDIA GPU) | ✔ |
| 中文 (Chinese) | — (cloud; whisper-large-v3-turbo from source with onnxruntime-gpu on an NVIDIA GPU) | — (cloud) | xtts-v2 (NVIDIA GPU) | ✔ |
| 日本語 (Japanese) | parakeet-tdt_ctc-0.6b-ja | piper-ja_JP-hi_fi_captain-medium | xtts-v2 (NVIDIA GPU) | ✔ |
| 한국어 (Korean) | — (cloud; whisper-large-v3-turbo from source with onnxruntime-gpu on an NVIDIA GPU) | piper-ko_KR-kss-medium | xtts-v2 (NVIDIA GPU) | ✔ |
| العربية (Arabic) | fastconformer-ar-pcd | — (cloud) | xtts-v2 (NVIDIA GPU) | ✔ |

Translation on this computer: any pair of the 14 (142 of the 182 through English).
<!-- languages:end -->

Subtitles in Chinese, Japanese, Korean and Arabic need a font for that script. macOS and Windows have them; on Linux
install e.g. `fonts-noto-cjk` and `fonts-noto-core` (Debian / Ubuntu) — without one the subtitles show empty boxes,
and the wizard and the tray say so. Arabic is shown right to left.

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
   (they are not in the app): 0.6–2.8 GB for a pair, about 1.5 GB for most; the wizard shows the exact size first.
   A direction in the cloud needs your Gemini key ([above](#peresmeshnik)).
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

Removing it: quit it from its menu, drag Peresmeshnik from Applications to the Bin, then delete two folders it
leaves: `~/Library/Application Support/Peresmeshnik` (settings with your Gemini key, your voice recording, logs, crash
reports) and `~/Library/Caches/Peresmeshnik` (the models). The two permissions go with
`tccutil reset All io.github.a1lezakty.peresmeshnik`. The key alone: Settings… → Forget the key.

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
   app): 0.6–2.8 GB for a pair, about 1.5 GB for most; the wizard shows the exact size first. A direction in the
   cloud needs your Gemini key ([above](#peresmeshnik)). Use headphones: through speakers the microphone hears the
   translation.
5. While translation is on, the app records the call app's own sound and turns it almost all the way down on your
   speakers (you hear the translation instead), and gives it back when translation is off. If the call app closes
   first, or its headset is off, it gets its volume back the next time it plays there, also after you quit
   Peresmeshnik (a small helper, `peresmeshnik-loopback.exe`, waits for that and then ends; the tray says which app).
   A sign-out or an update of Peresmeshnik ends that helper: start Peresmeshnik once. After a removal (a silent one
   does not say which app it was), raise the app in the Windows Volume mixer. Telegram, Zoom, Discord and the like are found
   by themselves; otherwise choose the app in the tray menu **Call app** while the call plays. A browser (Meet) is
   taken with all of its sound, other tabs too.
6. Without an NVIDIA GPU, what you say is spoken to the other side by the cloud (Gemini) or, on this computer, by a
   stock voice. Your own cloned voice needs an NVIDIA GPU with 6 GB or more; it adds the clone model (1.9 GB, up to
   4.7 GB with the pair's models) and the voice pack the wizard installs (3.7 GB to download, 8.2 GB on disk).

The hotkey Ctrl+Alt+Space switches between translation and your live voice (Settings to change it; a combo that
Windows keeps for itself, such as Alt+Tab or Win+Space, or that another app has taken is refused with the reason).
The uninstaller (Settings → Apps → Peresmeshnik, or `unins000.exe` in its folder) closes the app and removes its files.
It asks whether to remove the voice pack and its downloads (up to about 12 GB) and your settings with your Gemini key
("No" by default for both); your voice recording, logs and the models stay in `%LOCALAPPDATA%\Peresmeshnik` (the models
in its `Cache` folder) until you delete that folder. A silent uninstall (`/VERYSILENT`) asks nothing and keeps it all.
The key alone: Settings… → Forget the key.

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
Your cloned voice runs in a venv of its own (PyTorch and coqui-tts, ~7 GB with CUDA): `uv venv -p 3.12 ~/.local/share/Peresmeshnik/clone-venv`,
`uv pip install -p ~/.local/share/Peresmeshnik/clone-venv/bin/python --require-hashes -r voice-runner/requirements-linux-cuda.txt --build-constraints voice-runner/requirements-linux-cuda.txt`
(every package pinned and checked by its SHA-256, and so is the setuptools that builds the two packages that come as source,
docopt and unidic-lite; with CPU-only PyTorch: `requirements-linux.txt` in both places and `--index-strategy unsafe-best-match`), then start the
app with `PERESMESHNIK_VOICE_CMD="$HOME/.local/share/Peresmeshnik/clone-venv/bin/python voice-runner/voice_runner.py"`.
Turkish, Chinese and Korean speech recognition on an NVIDIA GPU (Whisper, 1.6 GB more) needs onnxruntime with CUDA in the app's own
environment, in place of the CPU one (CUDA 12 and cuDNN 9 installed): `uv pip uninstall onnxruntime`,
`uv pip install onnxruntime-gpu==1.31.1`, then start with `uv run --no-sync peresmeshnik app` (a plain `uv run` puts
the CPU one back). These steps are not checked by the project; its benchmarks ran Whisper with onnxruntime-gpu
1.31.1.
What stays when you stop using it: `~/.config/Peresmeshnik` (settings with your Gemini key), `~/.local/share/Peresmeshnik`
(your voice recording, logs, crash reports, the clone's venv if you made it there) and `~/.cache/Peresmeshnik` (the
models).

**Models pile up.** The app does not delete models: every pair you try keeps its own in the `models` folder of the
cache folder above (macOS `~/Library/Caches/Peresmeshnik/models`, Windows `%LOCALAPPDATA%\Peresmeshnik\Cache\models`,
Linux `~/.cache/Peresmeshnik/models`), one folder per model. Delete the ones you no longer need; choosing a pair again
downloads them again.

## When something goes wrong

- **The cloud fails.** A side set to the cloud (Gemini) goes local by itself when the connection drops, Google answers
  with an error or the quota is used up, or no translated voice comes `cloud.fallback_after_s` (6 s) after a phrase
  began. What the cloud had not said yet is said by the local voice, what it had said is not repeated; the tray says
  so once. Later, when someone starts a phrase, the app tries the cloud again (after 30 s, then less and less often, at
  most every 15 min while it stays down: nothing runs on a timer) and goes back to it at the next phrase. Going local
  needs that direction's speech recognition, translation and stock voice on this computer: the wizard downloads them
  for a cloud direction too, unless its subtitles come from the cloud ("Subtitles from the cloud") or its languages
  have no local models; a noncommercial stock voice (Turkish, Korean, Japanese) only if you accept its license. Your cloned voice takes over where it is ready, else the stock voice. Without a local path
  the wizard says so beforehand, the tray says what is missing, and the hotkey gives the call your own voice
  meanwhile.
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
Python of its own with PyTorch (BSD-3-Clause; most of its 3.7 GB are the NVIDIA CUDA and cuDNN libraries inside it,
under [NVIDIA's license terms](https://docs.nvidia.com/cuda/eula/)) and coqui-tts (MPL-2.0), with the text tools for Chinese and Japanese
(pypinyin, cutlet with jaconv and mojimoji, fugashi with MeCab, the UniDic dictionary: MIT, Apache-2.0, BSD-3-Clause;
[docs/MODELS.md](docs/MODELS.md)); the XTTS-v2 model it runs is under the
[Coqui Public Model License](https://huggingface.co/coqui/XTTS-v2/blob/main/LICENSE.txt) (noncommercial only). It needs an NVIDIA GPU with 6 GB of memory
or more, compute capability 6.1 or newer (GeForce GTX 10xx and later; a GTX 10xx is not measured: the app
measures the clone at start and uses the stock voice if the card is slower than real time) and driver 570.65 or
newer (528.33 to 570.64: not checked, may work); the wizard checks before the download.

---

**По-русски.** Пересмешник — переводчик звонков в реальном времени. Собеседника вы слышите
на своём языке — голосом в наушниках и субтитрами, а он вас — на своём, вашим же клонированным
голосом (с видеокартой NVIDIA; без неё — стоковым голосом или голосом облака). Облако — Google Gemini с вашим ключом:
звук стороны в облаке уходит в Google. Работает с любой программой созвона через системные аудиоустройства; macOS, Windows и
Linux, локальные модели или облако; 14 языков в любой паре (0.3.0). Сейчас в разработке; сборки для macOS и Windows
— установка выше.
Лицензия некоммерческая: пользоваться можно всем, продавать нельзя.
