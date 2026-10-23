#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
if [[ $# -lt 4 || $# -gt 6 ]]; then
  echo "usage: $0 {dace|accr} GPU_IDS MANIFEST_JSON OUTPUT_ROOT [BATCH_SIZE_PER_GPU] [MAX_AUDIO_SECONDS]" >&2
  echo "example GPU_IDS: 0,3,6; defaults: BATCH_SIZE_PER_GPU=2, MAX_AUDIO_SECONDS=16" >&2
  exit 2
fi

METHOD=${1,,}
GPU_IDS_RAW=$2
MANIFEST=$(cd "$(dirname "$3")" && pwd)/$(basename "$3")
OUTPUT_ROOT=$(mkdir -p "$4" && cd "$4" && pwd)
BATCH_SIZE=${5:-2}
MAX_AUDIO_SECONDS=${6:-16}
TIMESTAMP=${SER_RUN_TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}

if [[ "$METHOD" != "dace" && "$METHOD" != "accr" ]]; then
  echo "METHOD must be dace or accr" >&2
  exit 2
fi
IFS=',' read -r -a GPU_IDS <<< "$GPU_IDS_RAW"
if [[ ${#GPU_IDS[@]} -lt 1 ]]; then
  echo "GPU_IDS must contain at least one GPU" >&2
  exit 2
fi
declare -A SEEN_GPUS=()
for index in "${!GPU_IDS[@]}"; do
  GPU_IDS[$index]=${GPU_IDS[$index]//[[:space:]]/}
  if [[ ! ${GPU_IDS[$index]} =~ ^[0-9]+$ ]]; then
    echo "invalid GPU ID: ${GPU_IDS[$index]}" >&2
    exit 2
  fi
  if [[ -n ${SEEN_GPUS[${GPU_IDS[$index]}]:-} ]]; then
    echo "duplicate GPU ID: ${GPU_IDS[$index]}" >&2
    exit 2
  fi
  SEEN_GPUS[${GPU_IDS[$index]}]=1
done
AVAILABLE_CPU_THREADS=$(nproc)
CPU_THREADS_PER_WORKER=${SER_CPU_THREADS_PER_WORKER:-$((AVAILABLE_CPU_THREADS / ${#GPU_IDS[@]}))}
if [[ $CPU_THREADS_PER_WORKER -lt 1 ]]; then
  CPU_THREADS_PER_WORKER=1
fi

FINAL_DIR="$OUTPUT_ROOT/${METHOD}_$TIMESTAMP"
WORK_DIR="$OUTPUT_ROOT/.${METHOD}_${TIMESTAMP}_multigpu"
if [[ -e "$FINAL_DIR" || -e "$WORK_DIR" ]]; then
  echo "refusing to overwrite an existing run: $FINAL_DIR or $WORK_DIR" >&2
  exit 1
fi

WORKFLOW_START=$(date +%s.%N)
SPLIT_START=$(date +%s.%N)
python "$ROOT/scripts/split_manifest.py" \
  --manifest "$MANIFEST" \
  --output-dir "$WORK_DIR" \
  --num-shards "${#GPU_IDS[@]}" \
  --max-audio-seconds "$MAX_AUDIO_SECONDS" \
  > "$OUTPUT_ROOT/.${METHOD}_${TIMESTAMP}_split.log"
SPLIT_END=$(date +%s.%N)
SPLIT_SECONDS=$(python -c "print($SPLIT_END - $SPLIT_START)")

PIDS=()
PARALLEL_START=$(date +%s.%N)
for index in "${!GPU_IDS[@]}"; do
  shard=$(printf "shard_%02d" "$index")
  shard_dir="$WORK_DIR/$shard"
  mkdir -p "$shard_dir/output"
  (
    export SER_RUN_TIMESTAMP="${TIMESTAMP}_${shard}"
    export SER_CPU_THREADS="$CPU_THREADS_PER_WORKER"
    export OMP_NUM_THREADS="$CPU_THREADS_PER_WORKER"
    export MKL_NUM_THREADS="$CPU_THREADS_PER_WORKER"
    exec bash "$ROOT/scripts/run_${METHOD}.sh" \
      "${GPU_IDS[$index]}" \
      "$shard_dir/manifest.json" \
      "$shard_dir/output" \
      "$BATCH_SIZE" \
      "$MAX_AUDIO_SECONDS"
  ) > "$shard_dir/process.log" 2>&1 &
  pid=$!
  PIDS+=("$pid")
  echo "started $METHOD $shard on GPU ${GPU_IDS[$index]} (PID $pid)"
done

STATUS=0
for index in "${!PIDS[@]}"; do
  if ! wait "${PIDS[$index]}"; then
    echo "shard $(printf '%02d' "$index") failed; see $WORK_DIR/shard_$(printf '%02d' "$index")/process.log" >&2
    STATUS=1
  fi
done
PARALLEL_END=$(date +%s.%N)
if [[ $STATUS -ne 0 ]]; then
  exit "$STATUS"
fi
PARALLEL_SECONDS=$(python -c "print($PARALLEL_END - $PARALLEL_START)")

MERGE_ARGS=()
for index in "${!GPU_IDS[@]}"; do
  shard=$(printf "shard_%02d" "$index")
  summary=$(find "$WORK_DIR/$shard/output" -type f -name "${METHOD}_*_summary.json" -print)
  if [[ $(printf '%s\n' "$summary" | sed '/^$/d' | wc -l) -ne 1 ]]; then
    echo "expected exactly one summary for $shard" >&2
    exit 1
  fi
  MERGE_ARGS+=(--shard-summary "$summary")
done

python "$ROOT/scripts/merge_multigpu.py" \
  --method "$METHOD" \
  --manifest "$MANIFEST" \
  --split-summary "$WORK_DIR/split_summary.json" \
  "${MERGE_ARGS[@]}" \
  --gpu-ids "$(IFS=,; echo "${GPU_IDS[*]}")" \
  --output-dir "$FINAL_DIR" \
  --run-timestamp "$TIMESTAMP" \
  --batch-size "$BATCH_SIZE" \
  --max-audio-seconds "$MAX_AUDIO_SECONDS" \
  --cpu-threads-per-worker "$CPU_THREADS_PER_WORKER" \
  --workflow-start-epoch "$WORKFLOW_START" \
  --split-seconds "$SPLIT_SECONDS" \
  --parallel-wall-seconds "$PARALLEL_SECONDS"
