#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
if [[ $# -lt 5 || $# -gt 7 ]]; then
  echo "usage: $0 {dace|accr} GPU_IDS AUDIO_DIR ASR_CSV_OR_XLSX WORK_DIR [BATCH_SIZE_PER_GPU] [MAX_AUDIO_SECONDS]" >&2
  echo "example GPU_IDS: 0,3,6; defaults: BATCH_SIZE_PER_GPU=2, MAX_AUDIO_SECONDS=16" >&2
  exit 2
fi

METHOD=${1,,}
GPU_IDS=$2
AUDIO_DIR=$3
ASR_FILE=$4
WORK_DIR=$(mkdir -p "$5" && cd "$5" && pwd)
BATCH_SIZE=${6:-2}
MAX_AUDIO_SECONDS=${7:-16}
export SER_RUN_TIMESTAMP=${SER_RUN_TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}
INPUT_DIR="$WORK_DIR/input_$SER_RUN_TIMESTAMP"

python "$ROOT/scripts/prepare_input.py" \
  --audio-dir "$AUDIO_DIR" \
  --asr-file "$ASR_FILE" \
  --output-dir "$INPUT_DIR"

exec bash "$ROOT/scripts/run_multigpu.sh" \
  "$METHOD" "$GPU_IDS" "$INPUT_DIR/manifest.json" "$WORK_DIR/output" \
  "$BATCH_SIZE" "$MAX_AUDIO_SECONDS"
