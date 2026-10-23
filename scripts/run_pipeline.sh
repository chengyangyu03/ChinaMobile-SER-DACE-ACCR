#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
if [[ $# -lt 4 || $# -gt 7 ]]; then
  echo "usage: $0 {dace|accr} [GPU_ID] AUDIO_DIR ASR_CSV_OR_XLSX WORK_DIR [BATCH_SIZE] [MAX_AUDIO_SECONDS]" >&2
  echo "defaults: GPU_ID=0, BATCH_SIZE=2, MAX_AUDIO_SECONDS=16" >&2
  exit 2
fi

METHOD=${1,,}
shift

# Keep the original `METHOD GPU_ID AUDIO ...` interface, while also allowing
# GPU_ID to be omitted: `METHOD AUDIO ...`.  Audio directories whose names are
# integers can still use the explicit legacy form.
if [[ $# -ge 4 && $1 =~ ^[0-9]+$ ]]; then
  GPU_ID=$1
  shift
else
  GPU_ID=0
fi

if [[ $# -lt 3 || $# -gt 5 ]]; then
  echo "usage: $0 {dace|accr} [GPU_ID] AUDIO_DIR ASR_CSV_OR_XLSX WORK_DIR [BATCH_SIZE] [MAX_AUDIO_SECONDS]" >&2
  echo "defaults: GPU_ID=0, BATCH_SIZE=2, MAX_AUDIO_SECONDS=16" >&2
  exit 2
fi

AUDIO_DIR=$1
ASR_FILE=$2
WORK_DIR=$(mkdir -p "$3" && cd "$3" && pwd)
BATCH_SIZE=${4:-2}
MAX_AUDIO_SECONDS=${5:-16}
export SER_RUN_TIMESTAMP=${SER_RUN_TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}
INPUT_DIR="$WORK_DIR/input_$SER_RUN_TIMESTAMP"

python "$ROOT/scripts/prepare_input.py" \
  --audio-dir "$AUDIO_DIR" \
  --asr-file "$ASR_FILE" \
  --output-dir "$INPUT_DIR"

case "$METHOD" in
  dace)
    exec bash "$ROOT/scripts/run_dace.sh" "$GPU_ID" \
      "$INPUT_DIR/manifest.json" "$WORK_DIR/output" \
      "$BATCH_SIZE" "$MAX_AUDIO_SECONDS"
    ;;
  accr)
    exec bash "$ROOT/scripts/run_accr.sh" "$GPU_ID" \
      "$INPUT_DIR/manifest.json" "$WORK_DIR/output" \
      "$BATCH_SIZE" "$MAX_AUDIO_SECONDS"
    ;;
  *)
    echo "METHOD must be dace or accr" >&2
    exit 2
    ;;
esac
