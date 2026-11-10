# Test recordings

`call-en.ogg` and `call-ru.ogg`: 30 s each, 16 kHz mono Ogg Opus. Three sentences per language
from the FLEURS dev split (the stage-0 bench set) (ids 1510, 1515, 1520 for English; 1510, 1515, 1516 for Russian),
each scaled to −26 dBFS RMS (a usual call level; some FLEURS takes are at −50 dBFS) and
followed by 0.8 s of silence. Reference texts in `call-*.txt`, one sentence per line.

FLEURS: Conneau et al., 2022, https://huggingface.co/datasets/google/fleurs, licensed CC-BY-4.0.

`ours-en.ogg`: 22 s, made the same way from FLEURS dev ids 1511, 1513, 1521 (other people than in
`call-en.ogg`): "our" speech in the call scenarios (tests/test_echo.py, tests/test_call.py), texts in
`ours-en.txt`.

`pack-tiny.json` (from `requirements-tiny.in` by `voice-runner/lock.py --dir tests/data tiny`): a real voice pack of
numpy and soundfile on the embeddable CPython 3.12, for the Windows CI test `test_voice_pack_windows.py`.

`reader-ru.ogg`: 42 s, 16 kHz mono Ogg Vorbis, the voice reference `ref_ru.wav` of `bench/data.py`: the first six
utterances of chapter `poemi_01_pushkin` of the Russian LibriSpeech test split (OpenSLR 96, via
https://huggingface.co/datasets/bond005/rulibrispeech) with 0.3 s of silence between them, written with soundfile
(`compression_level=0.9`). A real reader for `voice.check` (tests/test_clone.py). Vorbis, not Opus: Opus drops the
recording's DC offset (-40 dBFS, its floor in the check), and the check's numbers would no longer be the recording's.
The recording: A. S. Pushkin, "Ruslan and Ludmila" (section 1: the dedication and the prologue), read by Foreign Girl
for LibriVox (https://librivox.org/poemi-alexander-pushkin/), public domain.
