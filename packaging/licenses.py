"""Copy the license files of every package in this environment that the app ships into one folder.

    python packaging/licenses.py OUT_DIR     (build.sh: Peresmeshnik.app/Contents/Resources/licenses)
"""
import shutil
import sys
import sysconfig
from importlib import metadata
from pathlib import Path

TOOLS = {'pyinstaller', 'pyinstaller-hooks-contrib', 'pytest', 'ruff', 'pluggy', 'iniconfig', 'altgraph', 'macholib',
         'pefile', 'pywin32-ctypes', 'pip'}  # build and test tools, never in the app (a license too many does no harm)
LICENSE_WORDS = ('license', 'licence', 'copying', 'notice', 'authors', 'thirdparty')
TEXTS = Path(__file__).parent / 'licenses'  # full texts for packages that only name their license

out = Path(sys.argv[1])
out.mkdir(parents=True, exist_ok=True)
for dist in metadata.distributions():
    name = dist.metadata['Name']
    if name.lower() in TOOLS:
        continue
    files = [f for f in dist.files or [] if any(w in f.name.lower() for w in LICENSE_WORDS)
             and (len(f.parts) <= 2 or f.parts[1] == 'licenses')]  # <pkg>/LICENSE, *.dist-info/licenses/* (PEP 639)
    (out / name).mkdir(exist_ok=True)
    for f in files:
        shutil.copyfile(f.locate(), out / name / '-'.join(f.parts[2:] or [f.name]))
    if not files:  # the wheel names its license only: that, where the source is, and the license's text
        said = dist.metadata.get('License-Expression') or dist.metadata.get('License') or ''
        urls = [dist.metadata.get('Home-page') or '', *(dist.metadata.get_all('Project-URL') or [])]
        (out / name / 'LICENSE-declared.txt').write_text(f'{name} {dist.version}: {said}\n' + '\n'.join(u for u in urls if u) + '\n')
        for key, text in (('LGPL-3.0', 'LGPL-3.0.txt'), ('Apache', 'Apache-2.0.txt')):
            if key in said:
                shutil.copyfile(TEXTS / text, out / name / text)
        print(f'licenses: {name} {dist.version} has no license file; its metadata says {said!r}')
python_license = Path(sysconfig.get_paths()['stdlib']) / 'LICENSE.txt'
if python_license.exists():
    shutil.copyfile(python_license, out / 'Python-LICENSE.txt')
