#!/usr/bin/env bash
# Voice-clone bench on the macOS runner: XTTS-v2 on CPU (GitHub's virtual M1 has no MPS), judged in a separate venv.
# Chatterbox needs ~7 GB on CPU and CosyVoice3 ~9 GB: not run here (see bench/notes/tts_clone.md).
set -e
cd "$(dirname "$0")/.."
export TTS_HOME=$PWD/models/tts BENCH_THREADS=3 OMP_NUM_THREADS=3 PYTORCH_ENABLE_MPS_FALLBACK=1
uv run -q --no-project -p 3.11 --with numpy --with soundfile --with pyarrow python bench/data.py
for m in xtts eval; do
  uv venv -q -p 3.11 .venv-$m && uv pip install -q -p .venv-$m/bin/python -r bench/requirements-tts-clone-$m.txt
done
.venv-xtts/bin/python bench/tts_clone.py synth --model xtts --device cpu --n ${CLONE_N:-4}
.venv-eval/bin/python bench/tts_clone.py eval --model xtts
