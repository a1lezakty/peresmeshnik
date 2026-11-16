#!/usr/bin/env bash
# Run a benchmark suite one model at a time:  bench/suite.sh asr|mt|tts [extra args for every run]
# Each run is a separate process (clean peak RSS). LOCK=<file> serialises heavy runs with other jobs.
# Exit code: 0 all runs passed, 1 some run failed, 2 unknown suite.

cd "$(dirname "$0")"
PY=${PY:-python}
suite=$1; shift
# numpy/BLAS/OpenMP pools default to all cores: pin them to --threads too
T=4; prev=; for x in "$@"; do [ "$prev" = --threads ] && T=$x; prev=$x; done
export OMP_NUM_THREADS=$T OPENBLAS_NUM_THREADS=$T MKL_NUM_THREADS=$T VECLIB_MAXIMUM_THREADS=$T
failed=0
run() { echo "== $*"; ${LOCK:+flock "$LOCK"} nice -n 10 "$PY" "$@" || { echo "FAILED: $*"; failed=$((failed + 1)); }; }
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
    for v in ${VOICES:-joe john denis dmitri silero-en silero-ru}; do run tts.py run $v "$@"; run tts.py judge $v "$@"; done ;;
  *)
    echo "unknown suite: $suite"; exit 2 ;;
esac
[ $failed = 0 ] || { echo "$failed run(s) failed"; exit 1; }
