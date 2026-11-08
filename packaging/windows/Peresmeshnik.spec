# PyInstaller spec: Peresmeshnik for Windows (x64), one folder, with the Piper runner as a second program beside it.
# Build: packaging/windows/build.sh (it also adds the licenses, the .zip and the Inno Setup installer).
import importlib.util
import shutil
import sys
from pathlib import Path

from PyInstaller.config import CONF
from PyInstaller.utils.hooks import collect_data_files, collect_submodules, copy_metadata

ROOT = Path(SPECPATH).parents[1]
EXCLUDES = ['torch', 'torchaudio', 'TTS', 'tkinter', 'matplotlib', 'pytest', 'IPython', 'jeepney']  # no torch: §10

# the whole standard library: the voice pack (torch, TTS) runs on this interpreter and imports stdlib modules our own
# code never does (unittest.mock, cProfile, ...)
SKIP = {'tkinter', '_tkinter', 'turtle', 'turtledemo', 'idlelib', 'test', 'ensurepip', 'lib2to3', 'pydoc_data', 'venv', 'this', 'antigravity'}
STDLIB = []
for name in sorted(sys.stdlib_module_names - SKIP):
    spec = importlib.util.find_spec(name)
    if spec is None:  # POSIX-only modules
        continue
    STDLIB += collect_submodules(name, filter=lambda n: '.test' not in n) if spec.submodule_search_locations else [name]

app = Analysis(
    [str(ROOT / 'packaging' / 'entry.py')],
    pathex=[str(ROOT)],
    hiddenimports=collect_submodules('peresmeshnik') + STDLIB,  # commands and engines are imported by name, when used
    datas=[(str(ROOT / 'peresmeshnik' / 'models.toml'), 'peresmeshnik'),
           (str(ROOT / 'peresmeshnik' / 'ui' / 'samples'), 'peresmeshnik/ui/samples'),
           (str(ROOT / 'assets' / 'icons'), 'assets/icons'),  # ui/tray.py finds them two levels above itself
           *collect_data_files('onnx_asr'), *copy_metadata('onnx-asr'), *copy_metadata('google-genai')],
    excludes=EXCLUDES + ['piper'],  # GPL: only in piper-runner's own archive (licenses.py fails if the app imports it)
)
shutil.copyfile(CONF['warnfile'], Path(CONF['warnfile']).with_name('warn-app.txt'))  # the next Analysis overwrites it
piper = Analysis(
    [str(Path(SPECPATH) / 'piper_entry.py')],  # piper-runner/piper_runner.py's main(); an error exits, no dialog
    pathex=[str(ROOT / 'piper-runner')],
    hiddenimports=collect_submodules('piper', filter=lambda n: '.train' not in n),
    datas=collect_data_files('piper', excludes=['train/**', 'img/**', 'templates/**']),
    excludes=EXCLUDES + ['PySide6', 'peresmeshnik'],
)
icon = str(ROOT / 'assets' / 'icons' / 'app.ico')
# both windowed: no console window opens with the app, nor when it starts the runner (its pipes are the runner's stdio)
app_exe = EXE(PYZ(app.pure), app.scripts, [], exclude_binaries=True, name='Peresmeshnik', console=False, icon=icon)
piper_exe = EXE(PYZ(piper.pure), piper.scripts, [], exclude_binaries=True, name='piper-runner', console=False, icon=icon)
COLLECT(app_exe, app.binaries, app.datas, piper_exe, piper.binaries, piper.datas, name='Peresmeshnik')
