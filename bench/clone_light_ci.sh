#!/usr/bin/env bash
# The light clone engines on the macOS runner (CPU; GitHub's virtual M1 has 3 cores, 7 GB and no usable GPU):
# MOSS-TTS-Nano and Pocket TTS on onnxruntime, ZONOS3-Mini on ggml, Piper -> OpenVoice. See bench/clone_light.py.
set -e
cd "$(dirname "$0")/.."
W=$PWD/clone-light
mkdir -p "$W/models" "$W/logs"
export BENCH_OUT=$W/out HF_HUB_DISABLE_PROGRESS_BARS=1 N=${CLONE_N:-4}
uv run -q --no-project -p 3.11 --with numpy --with soundfile --with pyarrow python bench/data.py
uv venv -q -p 3.11 "$W/venv-eval" && uv pip install -q -p "$W/venv-eval/bin/python" -r bench/requirements-tts-clone-eval.txt
uv venv -q -p 3.11 "$W/venv-onnx" && uv pip install -q -p "$W/venv-onnx/bin/python" onnxruntime==1.23.2 numpy soundfile sentencepiece \
  scipy huggingface_hub safetensors psutil voiceclonnx piper-tts librosa
git clone -q https://github.com/OpenMOSS/MOSS-TTS-Nano "$W/MOSS-TTS-Nano" && git -C "$W/MOSS-TTS-Nano" checkout -q 8b7bcc9
git clone -q https://github.com/Zyphra/ZONOS3-Mini "$W/ZONOS3-Mini" && git -C "$W/ZONOS3-Mini" checkout -q b820f6d
git -C "$W/ZONOS3-Mini" submodule update -q --init servers/ggml/ggml
Z=$W/ZONOS3-Mini/servers/ggml
(cd "$Z" && cmake -B build -DCMAKE_BUILD_TYPE=Release > "$W/logs/zonos-build.log" 2>&1 && cmake --build build -j3 >> "$W/logs/zonos-build.log" 2>&1) \
  || { tail -30 "$W/logs/zonos-build.log"; echo "ZONOS3 build failed"; }
"$W/venv-onnx/bin/python" - <<PY
from huggingface_hub import snapshot_download as d, hf_hub_download as h
M = '$W/models'
d('OpenMOSS-Team/MOSS-TTS-Nano-100M-ONNX', local_dir=M + '/MOSS-TTS-Nano-100M-ONNX')
d('OpenMOSS-Team/MOSS-Audio-Tokenizer-Nano-ONNX', local_dir=M + '/MOSS-Audio-Tokenizer-Nano-ONNX')
d('KevinAHM/pocket-tts-onnx', local_dir=M + '/pocket-tts-onnx', allow_patterns=['*.py', 'onnx/english_2026-04/*'])
for f in ('zonos3-ptq.gguf', 'zac-keep2.gguf', 'spk-encoder.gguf'):
    h('Zyphra/ZONOS3-Mini', 'gguf/' + f, local_dir='$Z')
for v in ('en/en_US/joe/medium/en_US-joe-medium', 'ru/ru_RU/dmitri/medium/ru_RU-dmitri-medium'):
    for e in ('.onnx', '.onnx.json'):
        h('rhasspy/piper-voices', v + e, revision='e7229f7d33a10e145e109b886d661984201ac3c4', local_dir=M + '/piper')
PY
export MOSS_SRC=$W/MOSS-TTS-Nano MOSS_MODELS=$W/models POCKET_DIR=$W/models/pocket-tts-onnx PIPER_DIR=$W/models/piper
export ZONOS_BIN=$Z/build ZONOS_GGUF=$Z/gguf
BENCH_THREADS=3 "$W/venv-eval/bin/python" bench/clone_light.py refs
run() {  # label threads model args...
  local L=$1 T=$2 M=$3; shift 3
  echo "== $L"
  if BENCH_THREADS=$T OMP_NUM_THREADS=$T "$W/venv-onnx/bin/python" bench/clone_light.py synth --model "$M" --label "$L" --n "$N" "$@" > "$W/logs/$L.log" 2>&1; then
    BENCH_THREADS=3 "$W/venv-eval/bin/python" bench/clone_light.py eval --label "$L" 2>&1 | grep '^{' || echo "eval failed: $L"
  else
    tail -15 "$W/logs/$L.log"; echo "synth failed: $L"
  fi
}
run pocket-2t-r5 2 pocket --ref-s 5
run pocket-3t-r5 3 pocket --ref-s 5
run moss-2t-r5 2 moss --ref-s 5
run moss-3t-r5 3 moss --ref-s 5
run piper-2t 2 pipervc --vc none --ref-s 0
run openvoice-2t-r5 2 pipervc --vc openvoice --precision fp32 --ref-s 5
run zonos-cpu-r5 3 zonos --ref-s 5 --cases ru2en,en2en
run zonos-metal-r5 3 zonos --device mps --ref-s 5 --cases ru2en,en2en
cp "$W"/out/tts_clone/*/server.log "$W/logs/" 2>/dev/null || true
