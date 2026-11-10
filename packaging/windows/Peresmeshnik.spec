# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0 OR GPL-3.0-or-later
# (GPL too: it builds the GPL program piper-runner, and GPL-3.0 counts its build scripts in its source)
# PyInstaller spec: Peresmeshnik for Windows (x64), one folder, with the Piper runner as a second program beside it.
# Build: packaging/windows/build.sh (it also adds the licenses, the .zip and the Inno Setup installer).
import re
import shutil
from pathlib import Path

from PyInstaller.config import CONF
from PyInstaller.utils.hooks import collect_data_files, collect_submodules, copy_metadata
from PyInstaller.utils.win32 import versioninfo as vi

ROOT = Path(SPECPATH).parents[1]
VERSION = re.search(r"__version__ = '([^']+)'", (ROOT / 'peresmeshnik' / '__init__.py').read_text())[1]
EXCLUDES = ['torch', 'torchaudio', 'TTS', 'tkinter', 'matplotlib', 'pytest', 'IPython', 'jeepney']  # no torch: §10

app = Analysis(
    [str(ROOT / 'packaging' / 'entry.py')],
    pathex=[str(ROOT)],
    hiddenimports=collect_submodules('peresmeshnik'),  # commands and engines are imported by name, when used
    datas=[(str(ROOT / 'peresmeshnik' / 'models.toml'), 'peresmeshnik'),
           (str(ROOT / 'peresmeshnik' / 'ui' / 'samples'), 'peresmeshnik/ui/samples'),
           (str(ROOT / 'assets' / 'icons'), 'assets/icons'),  # ui/tray.py finds them two levels above itself
           # the voice pack's runner, run by the pack's own Python (local/voicepack.py runner_dir): it matches this app
           *((str(ROOT / f), 'voice-runner') for f in ('voice-runner/voice_runner.py', 'voice-runner/pack-windows.json',
                                                      'peresmeshnik/local/xtts.py')),
           *collect_data_files('onnx_asr'), *copy_metadata('onnx-asr'), *copy_metadata('google-genai')],
    excludes=EXCLUDES + ['piper'],  # GPL: only in piper-runner's own archive (licenses.py fails if the app imports it)
)
shutil.copyfile(CONF['warnfile'], Path(CONF['warnfile']).with_name('warn-app.txt'))  # the next Analysis overwrites it
# piper's Hebrew and Arabic diacritizers (nakdimon.onnx 21 MB, tashkeel 5 MB) are left out: no he/ar Piper voice in models.toml
assert not re.search(r'^\["?piper-(he|ar)_', (ROOT / 'peresmeshnik' / 'models.toml').read_text(), re.M), 'a he/ar voice: ship them'
piper = Analysis(
    [str(Path(SPECPATH) / 'piper_entry.py')],  # piper-runner/piper_runner.py's main(); an error exits, no dialog
    pathex=[str(ROOT / 'piper-runner')],
    # pyopenjtalk: its compiled OpenJTalk module imports pure-Python ones the analysis cannot see
    hiddenimports=collect_submodules('piper', filter=lambda n: '.train' not in n) + collect_submodules('pyopenjtalk'),
    datas=[*collect_data_files('piper', excludes=['train/**', 'img/**', 'templates/**', 'hebrew/**', 'tashkeel/**']),
           *collect_data_files('pyopenjtalk', excludes=['htsvoice/**'])],  # the Japanese voice: OpenJTalk's dictionary
    excludes=EXCLUDES + ['PySide6', 'peresmeshnik'],
)
# not shipped: Mesa's software OpenGL (20 MB; the app draws with QWidget's raster engine, never OpenGL); PortAudio for other
# CPUs and with ASIO (the Steinberg SDK's own terms): sounddevice loads libportaudio64bit.dll unless SD_ENABLE_ASIO is set;
# Qt's JPEG, TIFF and WebP image plugins (libjpeg-turbo, libtiff, libwebp inside): the app loads only PNG (ui/tray.py);
# Qt's OpenSSL TLS backend and the second OpenSSL it pulls from the build machine's PATH (Git's MinGW libssl-3-x64):
# the app has no QtNetwork (Python's own ssl and its OpenSSL stay, with their license: licenses/OpenSSL-LICENSE.txt)
DROP = ('opengl32sw.dll', 'libportaudio32bit', 'libportaudioarm64', 'libportaudio64bit-asio', 'libportaudio.dylib',
        'qjpeg', 'qtiff', 'qwebp', 'qopensslbackend', 'libssl-3-x64', 'libcrypto-3-x64')
app.binaries, app.datas = ([e for e in toc if not Path(e[0]).name.lower().startswith(DROP)] for toc in (app.binaries, app.datas))
icon = str(ROOT / 'assets' / 'icons' / 'app.ico')


def version(name: str, description: str, legal: str) -> vi.VSVersionInfo:
    """The .exe's version resource (Properties > Details, Task Manager), from peresmeshnik.__version__."""
    nums = (*map(int, re.match(r'(\d+)\.(\d+)\.(\d+)', VERSION).groups()), 0)
    strings = {'CompanyName': 'Akty', 'FileDescription': description, 'FileVersion': VERSION, 'InternalName': name,
               'LegalCopyright': legal, 'OriginalFilename': f'{name}.exe',
               'ProductName': 'Peresmeshnik', 'ProductVersion': VERSION}
    return vi.VSVersionInfo(ffi=vi.FixedFileInfo(filevers=nums, prodvers=nums),
                            kids=[vi.StringFileInfo([vi.StringTable('040904B0', [vi.StringStruct(k, v) for k, v in strings.items()])]),
                                  vi.VarFileInfo([vi.VarStruct('Translation', [0x0409, 1200])])])  # en-US, Unicode


# both windowed: no console window opens with the app, nor when it starts the runner (its pipes are the runner's stdio)
app_exe = EXE(PYZ(app.pure), app.scripts, [], exclude_binaries=True, name='Peresmeshnik', console=False, icon=icon,
              version=version('Peresmeshnik', 'Peresmeshnik', 'Copyright (c) 2026 Akty'))
piper_exe = EXE(PYZ(piper.pure), piper.scripts, [], exclude_binaries=True, name='piper-runner', console=False, icon=icon,
                version=version('piper-runner', 'Piper runner of Peresmeshnik', 'GPL-3.0-or-later: licenses\\piper-runner-*.txt'))
COLLECT(app_exe, app.binaries, app.datas, piper_exe, piper.binaries, piper.datas, name='Peresmeshnik')
