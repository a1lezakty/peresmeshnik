# Test recordings

`call-en.ogg` and `call-ru.ogg`: 30 s each, 16 kHz mono Ogg Opus. Three sentences per language
from the FLEURS dev split (the stage-0 bench set) (ids 1510, 1515, 1520 for English; 1510, 1515, 1516 for Russian),
each scaled to −26 dBFS RMS (a usual call level; some FLEURS takes are at −50 dBFS) and
followed by 0.8 s of silence. Reference texts in `call-*.txt`, one sentence per line.

FLEURS: Conneau et al., 2022, https://huggingface.co/datasets/google/fleurs, licensed CC-BY-4.0.

`ours-en.ogg`: 22 s, made the same way from FLEURS dev ids 1511, 1513, 1521 (other people than in
`call-en.ogg`): "our" speech in the call scenarios (tests/test_echo.py, tests/test_call.py), texts in
`ours-en.txt`.
