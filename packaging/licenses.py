"""The licenses of a frozen build, after two checks. Shared by packaging/macos/build.sh and packaging/windows/build.sh:

    python packaging/licenses.py OUT_DIR WARN_FILE RUNNER

- WARN_FILE (the spec's copy of the app's PyInstaller warnings): the app must not import piper (GPL-3.0); it only
  talks to the Piper runner over a pipe. The build fails otherwise, or when the file is missing (the check could
  not run);
- OUT_DIR gets the license files of every package in this environment that the app ships, ours, the runner's, and
  where the runner's sources are (RUNNER: its path in the build, e.g. Contents/MacOS/piper-runner). A package that
  only names its license gets the full text from packaging/licenses/; an unknown one fails the build. So do the
  libraries a wheel carries under their own terms (BUNDLED) and Python's own license (with OpenSSL, libffi, sqlite...);
- and LGPL-SOURCE.txt: where the source of each LGPL library in the build is (Qt, PySide6, libsndfile with mpg123 and
  LAME), at the versions in this build, with the written offer of that source.
"""
import platform
import re
import shutil
import ssl
import sys
import sysconfig
from importlib import metadata
from pathlib import Path

TOOLS = {'pyinstaller', 'pyinstaller-hooks-contrib', 'pytest', 'ruff', 'pluggy', 'iniconfig', 'altgraph', 'macholib',
         'pefile', 'pywin32-ctypes', 'pip', 'colorama'}  # build and test tools, never in the app (a license too many does no harm)
LICENSE_WORDS = ('license', 'licence', 'copying', 'notice', 'authors', 'thirdparty')
MPG123 = {  # Windows: vcpkg's source folder of mpg123 in libsndfile (buildtrees\\mpg123\\src\\<base>-<hash>.clean) -> the version
    # and vcpkg's commit of the port. The hash: the first 10 hex of SHA-512 over the hex SHA-512 of the archive (the
    # portfile's SHA512) and of each of its PATCHES, which only that commit of ports/mpg123 gives
    b'-66150af195': ('1.32.9', 'b7601ed37e1e1ce11a4f61b4b94a23e39d465f1d'),
}  # packaging/sources.sh attaches that source to a release: change both together
ROOT = Path(__file__).resolve().parents[1]
TEXTS = ROOT / 'packaging' / 'licenses'  # full texts for packages that only name their license
BUNDLED = {  # file name start -> its texts: libraries inside a wheel that its own license file does not cover
    'libiomp5md': ['libiomp5md-LICENSE.txt', 'libiomp5md-third-party-programs.txt'],  # Intel OpenMP in ctranslate2 (Windows;
    # from the intel-openmp 2025.3.0 wheel, the DLL's build of 2025-09-10)
    'libportaudio': ['PortAudio-LICENSE.txt'],  # in sounddevice (PortAudio v19.7.0 LICENSE.txt)
    'sys.dic': ['OpenJTalk-dictionary-COPYING.txt'],  # pyopenjtalk-plus's dictionary: mecab-naist-jdic and UniDic words
    # (BSD-3-Clause; pyopenjtalk/dictionary/COPYING of the 0.4.1.post10 wheel)
    'openjtalk': ['OpenJTalk-COPYING.txt'],  # Open JTalk (BSD-3-Clause) with its changed MeCab, compiled into pyopenjtalk-plus:
    # src/COPYING and src/mecab/COPYING of github.com/tsukumijima/open_jtalk at 1e4f83b, the submodule of v0.4.1-post10
    'htsengine': ['hts_engine-COPYING.txt'],  # hts_engine API (BSD-3-Clause) compiled into pyopenjtalk-plus: src/COPYING of
    # github.com/syoyo/hts_engine_API at 2821c7a, the submodule of pyopenjtalk-plus v0.4.1-post10
    'espeakbridge': ['espeak-ng-COPYING.UCD.txt'],  # piper's espeak-ng links ucd-tools, whose tables are generated from the
    # Unicode Character Database: src/ucd-tools/COPYING.UCD of github.com/espeak-ng/espeak-ng at 724808c (piper1-gpl v1.8.0)
    'qtcore': ['Qt-third-party.txt'],  # the third-party code inside Qt (PCRE2, FreeType, HarfBuzz...): packaging/qt_notices.py
}
if sys.platform == 'win32':  # wheels built with MinGW-w64 GCC carry its runtime: protobuf's upb module (the CRT, gdtoa) and
    # numpy's OpenBLAS (with winpthreads; numpy's LICENSE.txt names the GCC runtime only)
    BUNDLED |= {'_message.pyd': ['mingw-w64-runtime-COPYING.txt'], 'libscipy_openblas64_': ['mingw-w64-runtime-COPYING.txt']}

out, warn, runner = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
if not warn.is_file() or not warn.stat().st_size:
    sys.exit(f'no {warn}: the GPL check cannot run')
if re.search(r'excluded module named piper( |\.)', warn.read_text('utf-8', 'replace')):  # piper and piper.voice, not piper_x
    sys.exit('the app imports piper (GPL): it must only talk to piper-runner over the pipe')
version = re.search(r"__version__ = '([^']+)'", (ROOT / 'peresmeshnik' / '__init__.py').read_text())[1]
pyside = metadata.version('PySide6_Essentials')
qt = '.'.join(pyside.split('.')[:3])  # PySide6 6.12.0 (or a repackaging, 6.12.0.1) ships Qt 6.12.0
if not (TEXTS / 'Qt-third-party.txt').read_text('utf-8').startswith(f'Third-party code in Qt {qt}\n'):
    sys.exit(f'licenses: packaging/licenses/Qt-third-party.txt is not of Qt {qt}: python packaging/qt_notices.py {qt}')

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
# Python's ssl module ships OpenSSL (libssl / libcrypto): Apache-2.0 since 3.0, and the LICENSE.txt of a uv Python does
# not name it, so its own text goes along. cryptography's wheels build another OpenSSL into its _rust module (statically)
try:
    from cryptography.hazmat.bindings._rust import openssl as rust_openssl
    also = (f'; so is {rust_openssl.openssl_version_text()}, built statically into the _rust module of cryptography '
            f'{metadata.version("cryptography")}')
except ModuleNotFoundError:  # not in this build
    also = ''
(out / 'OpenSSL-LICENSE.txt').write_text(f'{ssl.OPENSSL_VERSION} (libssl and libcrypto of Python\'s ssl module, '
                                         f'https://www.openssl.org) is under the Apache License 2.0{also}:\n\n'
                                         + (TEXTS / 'Apache-2.0.txt').read_text())
if missing:
    sys.exit('licenses: no license text for ' + '; '.join(missing))

OFFER = '''for at least three years after we last distribute this version, Akty (https://github.com/a1lezakty)
will give anyone who asks a complete machine-readable copy of the source code of {what},
at the exact versions in this build, by download at no charge or on a physical medium for no more than the cost of
making and sending it. Ask at https://github.com/a1lezakty/peresmeshnik/issues.
'''
ort, np = metadata.version('onnxruntime'), metadata.version('numpy')
py = platform.python_version()
tarball = f'https://www.python.org/ftp/python/{py}/Python-{py}.tar.xz'
# which build of CPython this is: a uv-managed one is python-build-standalone, whose release (its date tag) uv writes
# to BUILD in the installation's root; any other is python.org's (the runner's hosted tool cache, macOS and Windows)
pbs = Path(sys.base_prefix) / 'BUILD'
if pbs.is_file():
    tag = pbs.read_text().strip()
    if not re.fullmatch(r'\d{8}', tag):
        sys.exit(f'licenses: {pbs} names no python-build-standalone release: {tag!r}')
    cpython = (f'built by python-build-standalone, release {tag}\n  (https://github.com/astral-sh/python-build-standalone/'
               f'releases/tag/{tag}: its build recipe and patches),\n  from {tarball}')
elif Path(sys.base_prefix).name.startswith('cpython-'):  # uv's, installed before uv wrote BUILD: the release is unknown
    sys.exit(f'licenses: {sys.base_prefix} (uv\'s Python) has no BUILD: uv python install --reinstall {py}')
else:
    cpython = f'https://www.python.org/downloads/source/\n  ({tarball})'
shutil.copyfile(ROOT / 'LICENSE', out / 'Peresmeshnik-LICENSE.txt')
shutil.copyfile(ROOT / 'piper-runner' / 'LICENSE', out / 'piper-runner-LICENSE.txt')
(out / 'piper-runner-SOURCE.txt').write_text(f'''\
{runner} is a separate program under GPL-3.0-or-later, built with PyInstaller from:
- piper-runner/piper_runner.py (on Windows with its entry packaging/windows/piper_entry.py) of
  https://github.com/a1lezakty/peresmeshnik (version {version}, tag v{version}), by the build scripts there, which are
  under GPL-3.0-or-later too (their SPDX line): packaging/{{macos,windows}}/Peresmeshnik.spec, packaging/{{macos,windows}}/build.sh,
  packaging/piper-requirements.txt and the installer's packaging/windows/Peresmeshnik.iss;
- piper-tts 1.8.0: https://github.com/OHF-Voice/piper1-gpl, tag v1.8.0 (its PyPI sdist has no C sources);
- espeak-ng, built statically into piper's espeakbridge module: https://github.com/espeak-ng/espeak-ng, commit 724808c
  (the GIT_TAG in piper1-gpl v1.8.0's CMakeLists.txt);
- for the Japanese voice, pyopenjtalk-plus 0.4.1.post10 (MIT, with OpenJTalk and its dictionary; its tag v0.4.1-post10
  has them as the submodules https://github.com/tsukumijima/open_jtalk at 1e4f83b and
  https://github.com/syoyo/hts_engine_API at 2821c7a) and sudachipy 0.7.0;
- onnxruntime {ort}: https://github.com/microsoft/onnxruntime, tag v{ort};
- numpy {np}: https://github.com/numpy/numpy, tag v{np} (with its submodules), or its sdist on PyPI;
- CPython {py} (this build: {sys.version}): {cpython};
the licenses of all of these are in this folder.
Peresmeshnik itself never imports it; it talks to it over stdin/stdout. The runner's source with its build scripts,
piper1-gpl and espeak-ng are attached to the GitHub release of this version, Peresmeshnik-{version}-sources.tar
(https://github.com/a1lezakty/peresmeshnik/releases/tag/v{version}; packaging/sources.sh).

Written offer: {OFFER.format(what=f'{runner} (its Corresponding Source: all of the above)')}''')

# LGPL: the source of each LGPL library, at the version this build ships, and the written offer of it
sf = metadata.distribution('soundfile')
lib = next((f.locate() for f in sf.files if f.name.startswith('libsndfile')), None)  # _soundfile_data/libsndfile_x64.dll...
data = Path(lib).read_bytes() if lib else b''
sndfile, lame = (re.search(p, data) for p in (rb'libsndfile-(\d+\.\d+\.\d+)', rb'LAME(\d+\.\d+)'))
if not sndfile:
    sys.exit(f'licenses: no libsndfile version in {lib}')
vcpkg = re.search(rb'buildtrees\\mpg123\\src\\([^\\]*)\.clean', data)  # Windows (vcpkg); macOS: mac_build.sh names it
if vcpkg and vcpkg[1] not in MPG123:
    sys.exit(f'licenses: mpg123 of vcpkg {vcpkg[1].decode()} in {lib} is unknown: add its version to MPG123')
mpg123, port = MPG123[vcpkg[1]] if vcpkg else ('', '')
mpg123, port = mpg123 and f' {mpg123}', port and (f': mpg123 by https://github.com/microsoft/vcpkg/tree/{port}/ports/mpg123;'
                                                  f' its source and that port are attached to the GitHub release of this'
                                                  f' version, Peresmeshnik-{version}-sources.tar')
(out / 'LGPL-SOURCE.txt').write_text(f'''\
Libraries in Peresmeshnik {version} under the GNU Lesser General Public License, and where their source is.
Their license texts are in this folder: PySide6_Essentials and shiboken6 (LGPL-3.0, with the GPL-3.0 it builds on),
soundfile (COPYING: LGPL-2.1; license_notes.md: the copyrights of the codecs). Each is a separate library file in the
app (the _internal folder on Windows, Contents/Frameworks on macOS): you may replace it with your own build of the same
version and interface (on macOS, sign the app again after that, as packaging/macos/build.sh does). Nothing in
Peresmeshnik's license (PolyForm Noncommercial) restricts modifying these libraries for your own use, or reverse
engineering the app to debug such modifications, whatever the purpose (LGPL-2.1 section 6, LGPL-3.0 section 4).

- Qt {qt} (Qt Core, Gui, Widgets, Network, Svg and their plugins; LGPL-3.0-only):
  https://download.qt.io/archive/qt/{qt.rsplit('.', 1)[0]}/{qt}/single/qt-everywhere-src-{qt}.tar.xz
  (also the tags v{qt} of https://code.qt.io/cgit/qt/qtbase.git, qtsvg.git and qtimageformats.git there);
  its third-party parts: PySide6_Essentials/Qt-third-party.txt.
- PySide6 {pyside} and shiboken6 {metadata.version('shiboken6')} (LGPL-3.0-only):
  https://download.qt.io/official_releases/QtForPython/pyside6/PySide6-{pyside}-src/pyside-setup-everywhere-src-{pyside}.tar.xz
  (also the tag v{pyside} of https://code.qt.io/cgit/pyside/pyside-setup.git).
- libsndfile {sndfile[1].decode()} (LGPL-2.1-or-later) in the soundfile {sf.version} package, built with libmpg123{mpg123} (LGPL-2.1)
  and libmp3lame{f" {lame[1].decode()}" if lame else ""} (LAME, LGPL-2.0-or-later), and libFLAC, libogg, libvorbis, libopus (BSD-3-Clause):
  https://github.com/libsndfile/libsndfile/releases/tag/{sndfile[1].decode()}, https://www.mpg123.de/download/,
  https://sourceforge.net/projects/lame/files/lame/. The build recipe with the exact versions of each:
  https://github.com/bastibe/libsndfile-binaries at the commit that https://github.com/bastibe/python-soundfile
  tag {sf.version} pins as its _soundfile_data submodule (mac_build.sh; on Windows, vcpkg's libsndfile{port}).

Written offer: {OFFER.format(what='the libraries above')}''')
