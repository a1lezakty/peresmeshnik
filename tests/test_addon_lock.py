"""The Windows addon's section of pack-windows.json is the lock it came from (voice-runner/lock.py --addon windows):
its wheels and source archives are exactly the pins of requirements-windows-addon.txt that requirements-windows.txt
does not have, each with one of that pin's hashes. A new pack lock keeps the old section (lock.py main): this catches it
going stale. No network."""
import importlib.util
import json
from pathlib import Path

RUNNER = Path(__file__).resolve().parents[1] / 'voice-runner'
_spec = importlib.util.spec_from_file_location('lock', RUNNER / 'lock.py')
lock = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(lock)


def test_the_addon_section_is_the_addon_lock_minus_the_pack_lock():
    base = {(n, v) for n, v, _ in lock.pins_of(RUNNER / 'requirements-windows.txt')}
    want = {n: h for n, v, h in lock.pins_of(RUNNER / 'requirements-windows-addon.txt') if (n, v) not in base}
    addon = json.loads((RUNNER / 'pack-windows.json').read_text())['addon']
    got = {f['name']: f['sha256'] for f in addon['wheels'] + addon['archives']}
    assert sorted(got) == sorted(want)
    assert [n for n, sha in got.items() if sha not in want[n]] == []  # the hash pins the version too
    assert {a['name'] for a in addon['archives']} <= set(lock.DATA)
