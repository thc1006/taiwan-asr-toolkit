#!/usr/bin/env bash
# Convenience wrapper around the asr-* CLI commands installed by `pip install -e .`.
# Falls back to `python -m taiwan_asr.X` if the CLI scripts are not on PATH.
set -euo pipefail
cd "$(dirname "$0")"

# Pick a runner: prefer the installed CLI; otherwise use the package directly.
if command -v asr-breeze >/dev/null 2>&1; then
  BREEZE=asr-breeze
  QWEN3=asr-qwen3
  BENCH=asr-bench
else
  echo "(asr-* commands not found on PATH; using 'python -m taiwan_asr.X')" >&2
  BREEZE="python -m taiwan_asr.breeze"
  QWEN3="python -m taiwan_asr.qwen3"
  BENCH="python -m taiwan_asr.benchmark"
fi

usage() {
  cat <<EOF
Usage:
  ./run.sh qwen3 <audio...>     Run Qwen3-ASR transcription
  ./run.sh breeze <audio...>    Run Breeze-ASR-25 transcription
  ./run.sh both <audio...>      Run both models on the same input
  ./run.sh report               Generate docs/BENCHMARK.md from existing outputs
  ./run.sh all                  Transcribe everything in music/ with both models + report

For full flag documentation see:
  asr-breeze --help
  asr-qwen3 --help
EOF
}

case "${1:-}" in
  qwen3)  shift; $QWEN3 "$@" ;;
  breeze) shift; $BREEZE "$@" ;;
  both)
    shift
    $BREEZE "$@"
    $QWEN3 "$@"
    ;;
  report) $BENCH ;;
  all)
    files=( music/* )
    $BREEZE "${files[@]}"
    $QWEN3  "${files[@]}"
    $BENCH
    ;;
  *) usage; exit 1 ;;
esac
