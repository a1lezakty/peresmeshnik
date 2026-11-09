"""The licenses of a frozen build, after two checks. Shared by packaging/macos/build.sh and packaging/windows/build.sh:

    python packaging/licenses.py OUT_DIR WARN_FILE RUNNER

- WARN_FILE (the spec's copy of the app's PyInstaller warnings): the app must not import piper (GPL-3.0); it only
  talks to the Piper runner over a pipe. The build fails otherwise, or when the file is missing (the check could
  not run);
- OUT_DIR gets the license files of every package in this environment that the app ships, ours, the runner's, and
  where the runner's sources are (RUNNER: its path in the build, e.g. Contents/MacOS/piper-runner). A package that
  only names its license gets the full text from packaging/licenses/; an unknown one fails the build. So do the
  libraries a wheel carries under their own terms (BUNDLED) and Python's own license (with OpenSSL, libffi, sqlite...).
"""
import re
import shutil
import sys
import sysconfig
from importlib import metadata
from pathlib import Path

TOOLS = {'pyinstaller', 'pyinstaller-hooks-contrib', 'pytest', 'ruff', 'pluggy', 'iniconfig', 'altgraph', 'macholib',
         'pefile', 'pywin32-ctypes', 'pip', 'colorama'}  # build and test tools, never in the app (a license too many does no harm)
LICENSE_WORDS = ('license', 'licence', 'copying', 'notice', 'authors', 'thirdparty')
ROOT = Path(__file__).resolve().parents[1]
TEXTS = ROOT / 'packaging' / 'licenses'  # full texts for packages that only name their license
BUNDLED = {  # file name start -> its texts: libraries inside a wheel that its own license file does not cover
    'libiomp5md': ['libiomp5md-LICENSE.txt', 'libiomp5md-third-party-programs.txt'],  # Intel OpenMP in ctranslate2 (Windows;
    # from the intel-openmp 2025.3.0 wheel, the DLL's build of 2025-09-10)
    'libportaudio': ['PortAudio-LICENSE.txt'],  # in sounddevice (PortAudio v19.7.0 LICENSE.txt)
    'sys.dic': ['OpenJTalk-dictionary-COPYING.txt'],  # pyopenjtalk-plus's dictionary: mecab-naist-jdic and UniDic words
    # (BSD-3-Clause; pyopenjtalk/dictionary/COPYING of the 0.4.1.post10 wheel)
}

out, warn, runner = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
if not warn.is_file() or not warn.stat().st_size:
    sys.exit(f'no {warn}: the GPL check cannot run')
if re.search(r'excluded module named piper( |\.)', warn.read_text('utf-8', 'replace')):  # piper and piper.voice, not piper_x
    sys.exit('the app imports piper (GPL): it must only talk to piper-runner over the pipe')
version = re.search(r"__version__ = '([^']+)'", (ROOT / 'peresmeshnik' / '__init__.py').read_text())[1]

out.mkdir(parents=True, exist_ok=True)
missing = []
for dist in metadata.distributions():
    name = dist.metadata['Name']
    if name.lower() in TOOLS or name == 'peresmeshnik':  # ours: below
        continue
    files = [f for f in dist.files or [] if any(w in f.name.lower() for w in LICENSE_WORDS)
             and (len(f.parts) <= 2 or f.parts[1] == 'licenses')  # <pkg>/LICENSE, *.dist-info/licenses/* (PEP 639)
             and not f.name.startswith('LicenseRef-')]  # a commercial option we do not take (Qt): its open license below
    (out / name).mkdir(exist_ok=True)
    for f in files:
        shutil.copyfile(f.locate(), out / name / '-'.join(f.parts[2:] or [f.name]))
    for text in {t for f in dist.files or [] for start, ts in BUNDLED.items() if f.name.lower().startswith(start) for t in ts}:
        shutil.copyfile(TEXTS / text, out / name / text)
    if files:
        continue
    # the wheel names its license only: that, where the source is, and the full text
    said = dist.metadata.get('License-Expression') or dist.metadata.get('License') or ''
    urls = [dist.metadata.get('Home-page') or '', *(dist.metadata.get_all('Project-URL') or [])]
    (out / name / 'LICENSE-declared.txt').write_text(f'{name} {dist.version}: {said}\n' + '\n'.join(u for u in urls if u) + '\n')
    if (TEXTS / f'{name}.txt').exists():  # the upstream file of that version, with its copyright (MIT, BSD)
        texts = [f'{name}.txt']
    elif 'LGPL-3.0' in said:  # the LGPL is a set of permissions on top of the GPL: both texts
        texts = ['LGPL-3.0.txt', 'GPL-3.0.txt']
    elif 'Apache' in said:
        texts = ['Apache-2.0.txt']
    else:  # MIT / BSD need the upstream copyright: add packaging/licenses/<name>.txt from its repository
        missing.append(f'{name} {dist.version} ({said or "no license named"})')
        continue
    for text in texts:
        shutil.copyfile(TEXTS / text, out / name / text.replace(f'{name}.txt', 'LICENSE'))
    print(f'licenses: {name} {dist.version} ships no license file ({said}): added {", ".join(texts)}')
# Python's, with the libraries of its build: in the stdlib (macOS, Linux), in the installation's root (Windows)
python_license = next((p for p in (Path(sysconfig.get_paths()['stdlib']) / 'LICENSE.txt', Path(sys.base_prefix) / 'LICENSE.txt')
                       if p.is_file()), None)
if not python_license:
    sys.exit(f"licenses: no LICENSE.txt of Python in {sysconfig.get_paths()['stdlib']} or {sys.base_prefix}")
shutil.copyfile(python_license, out / 'Python-LICENSE.txt')
if missing:
    sys.exit('licenses: no license text for ' + '; '.join(missing))

shutil.copyfile(ROOT / 'LICENSE', out / 'Peresmeshnik-LICENSE.txt')
shutil.copyfile(ROOT / 'piper-runner' / 'LICENSE', out / 'piper-runner-LICENSE.txt')
(out / 'piper-runner-SOURCE.txt').write_text(f'''\
{runner} is a separate program under GPL-3.0-or-later, built with PyInstaller from:
- piper-runner/piper_runner.py (on Windows with its entry packaging/windows/piper_entry.py) of
  https://github.com/a1lezakty/peresmeshnik (version {version}, tag v{version});
- piper-tts 1.8.0: https://github.com/OHF-Voice/piper1-gpl, tag v1.8.0 (its PyPI sdist has no C sources);
- espeak-ng, built statically into piper's espeakbridge module: https://github.com/espeak-ng/espeak-ng, commit 724808c
  (the GIT_TAG in piper1-gpl v1.8.0's CMakeLists.txt);
- for the Japanese voice, pyopenjtalk-plus 0.4.1.post10 (MIT, with OpenJTalk and its dictionary) and sudachipy 0.7.0;
- onnxruntime and numpy, whose licenses are in this folder (and those of the packages above).
Peresmeshnik itself never imports it; it talks to it over stdin/stdout.
''')
