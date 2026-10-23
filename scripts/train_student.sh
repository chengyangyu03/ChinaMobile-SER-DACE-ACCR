#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PYTHON_BIN=${PYTHON_BIN:-python}
NPROC_PER_NODE=${NPROC_PER_NODE:-4}
if [[ $# -lt 5 ]]; then
  echo "Usage: $0 BASE_MODEL TRAIN_CSV VALIDATION_CSV TEACHER_OOF_CSV OUTPUT_DIR" >&2
  exit 2
fi

"$PYTHON_BIN" -m torch.distributed.run \
  --standalone --nproc_per_node="$NPROC_PER_NODE" \
  "$ROOT/training/train_qwen2audio_directhead_student_valonly_abc.py" \
  --model "$1" --train-csv "$2" --val-csv "$3" \
  --teacher-train "$4" --output-dir "$5" \
  --steps 1400 --save-steps 350,700,1050,1400 \
  --batch-size 2 --gradient-accumulation 4 \
  --learning-rate 7.5e-5 --kd-alpha 0.40 --temperature 2.0 \
  --lora-rank 16 --lora-alpha 32 \
  --max-audio-seconds 16 --include-context --seed 20260722
