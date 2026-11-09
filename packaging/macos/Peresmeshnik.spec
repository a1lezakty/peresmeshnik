# PyInstaller spec: Peresmeshnik.app, one folder, with the Piper runner as a second program in Contents/MacOS.
# PyInstaller: one tool for the .app now and the Windows / Linux folders of stages 4+ (ARCHITECTURE §10).
# Build: packaging/macos/build.sh (it also adds the Swift tap helper, the licenses, the ad-hoc signature, the .dmg).
import re
import shutil
from pathlib import Path

from PyInstaller.config import CONF
from PyInstaller.utils.hooks import collect_data_files, collect_submodules, copy_metadata

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
           *collect_data_files('onnx_asr'), *copy_metadata('onnx-asr'), *copy_metadata('google-genai')],
    excludes=EXCLUDES + ['piper'],  # GPL: only in piper-runner's own archive (build.sh fails if the app imports it)
)
shutil.copyfile(CONF['warnfile'], Path(CONF['warnfile']).with_name('warn-app.txt'))  # the next Analysis overwrites it
piper = Analysis(
    [str(ROOT / 'piper-runner' / 'piper_runner.py')],
    # pyopenjtalk: its compiled OpenJTalk module imports pure-Python ones the analysis cannot see
    hiddenimports=collect_submodules('piper', filter=lambda n: '.train' not in n) + collect_submodules('pyopenjtalk'),
    datas=[*collect_data_files('piper', excludes=['train/**', 'img/**', 'templates/**']),
           *collect_data_files('pyopenjtalk', excludes=['htsvoice/**'])],  # the Japanese voice: OpenJTalk's dictionary
    excludes=EXCLUDES + ['PySide6', 'peresmeshnik'],
)
app_exe = EXE(PYZ(app.pure), app.scripts, [], exclude_binaries=True, name='Peresmeshnik', console=False,
              target_arch='arm64', codesign_identity=None)
piper_exe = EXE(PYZ(piper.pure), piper.scripts, [], exclude_binaries=True, name='piper-runner', console=True,
                target_arch='arm64', codesign_identity=None)
coll = COLLECT(app_exe, app.binaries, app.datas, piper_exe, piper.binaries, piper.datas, name='Peresmeshnik')
BUNDLE(
    coll,
    name='Peresmeshnik.app',
    icon=str(ROOT / 'assets' / 'icons' / 'app.icns'),
    bundle_identifier='io.github.a1lezakty.peresmeshnik',
    version=VERSION,
    info_plist={
        'CFBundleDisplayName': 'Peresmeshnik',
        'LSUIElement': True,  # a menu-bar app: no Dock icon
        'LSMinimumSystemVersion': '14.0',  # Qt 6.12 and onnxruntime wheels need macOS 14
        'NSHighResolutionCapable': True,
        'NSMicrophoneUsageDescription': 'Peresmeshnik translates what you say and speaks it into the call: '
                                        'it listens to your microphone while translation is on.',
        'NSAudioCaptureUsageDescription': 'Peresmeshnik records the sound of your call app to translate it; '
                                          'while it does, macOS mutes that app on your speakers.',
    },
)
