#!/usr/bin/env bash
# 統一執行入口 — 優先使用 uv (若安裝),否則 fallback 到 python3
set -euo pipefail
cd "$(dirname "$0")"

if command -v uv >/dev/null 2>&1; then
  PY="uv run --no-project --script python3"
else
  PY="python3"
fi

usage() {
  cat <<EOF
用法:
  ./run.sh qwen3 <音訊檔...>      Qwen3-ASR 轉錄
  ./run.sh breeze <音訊檔...>     Breeze-ASR 轉錄
  ./run.sh both <音訊檔...>       兩個都跑 (模型分開載入)
  ./run.sh report                 產生比對報告 (transcripts/REPORT.md)
  ./run.sh all                    跑 music/ 內所有音檔 + 產報告
EOF
}

case "${1:-}" in
  qwen3)  shift; $PY qwen3_asr.py "$@" ;;
  breeze) shift; $PY breeze_asr.py "$@" ;;
  both)
    shift
    $PY breeze_asr.py "$@"
    $PY qwen3_asr.py "$@"
    ;;
  report) $PY benchmark.py ;;
  all)
    files=( music/* )
    $PY breeze_asr.py "${files[@]}"
    $PY qwen3_asr.py  "${files[@]}"
    $PY benchmark.py
    ;;
  *) usage; exit 1 ;;
esac
