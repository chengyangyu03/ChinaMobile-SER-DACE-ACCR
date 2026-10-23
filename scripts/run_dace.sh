#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
if [[ $# -lt 3 || $# -gt 5 ]]; then
  echo "usage: $0 GPU_ID MANIFEST_JSON OUTPUT_ROOT [BATCH_SIZE] [MAX_AUDIO_SECONDS]" >&2
  exit 2
fi

GPU_ID=$1
MANIFEST=$2
OUTPUT_ROOT=$3
BATCH_SIZE=${4:-2}
MAX_AUDIO_SECONDS=${5:-16}
TIMESTAMP=${SER_RUN_TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}
OUTPUT_DIR="$OUTPUT_ROOT/dace_$TIMESTAMP"

exec env CUDA_VISIBLE_DEVICES="$GPU_ID" python "$ROOT/src/dace_inference.py" \
  --manifest "$MANIFEST" \
  --base-model "$ROOT/models/base/Qwen2-Audio-7B-Instruct" \
  --primary-adapter "$ROOT/models/shared/primary_adapter" \
  --secondary-adapter "$ROOT/models/shared/secondary_adapter" \
  --selection "$ROOT/models/dace/dace_frozen_parameters.json" \
  --expected-selection-sha256 5ed53c1a1455fa52e699f2ceffa210ee983a75a929e2e4dadea778da8f2e7836 \
  --output-dir "$OUTPUT_DIR" \
  --run-timestamp "$TIMESTAMP" \
  --batch-size "$BATCH_SIZE" \
  --max-audio-seconds "$MAX_AUDIO_SECONDS"
