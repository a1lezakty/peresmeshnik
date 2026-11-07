#!/usr/bin/env bash
# Run a benchmark suite one model at a time:  bench/suite.sh asr|mt|tts [extra args for every run]
# Each run is a separate process (clean peak RSS). LOCK=<file> serialises heavy runs with other jobs.

cd "$(dirname "$0")"
PY=${PY:-python}
suite=$1; shift
run() { echo "== $*"; ${LOCK:+flock "$LOCK"} nice -n 10 "$PY" "$@" || echo "FAILED: $*"; }
case $suite in
  asr)
    for lang in en ru; do
      for m in small medium large-v3-turbo; do run asr.py fw $m $lang "$@"; done
      run asr.py onnx nemo-parakeet-tdt-0.6b-v3 $lang "$@"
    done
    for m in gigaam-v3-e2e-ctc gigaam-v3-ctc t-tech/t-one; do run asr.py onnx $m ru "$@"; done ;;
  mt)
    for d in en-ru ru-en; do
      run mt.py opus $d "$@"; run mt.py nllb $d "$@"
      for g in ${GGUF:-}; do run mt.py llama $d --gguf "$g" "$@" --device "${LLAMA_DEVICE:-cpu}"; done
    done ;;
  tts)
    for lang in en ru; do
      for e in piper silero; do run tts.py $e $lang "$@"; run tts.py judge $e $lang; done
    done ;;
esac
