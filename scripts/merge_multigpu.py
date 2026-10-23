#!/usr/bin/env python3
"""Merge dialogue-complete DACE/ACCR GPU shards in original input order."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
from pathlib import Path
from typing import Any


LABELS = ("A", "B", "C")
LABEL_TO_ID = {label: index for index, label in enumerate(LABELS)}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {path}")
        return list(reader.fieldnames), list(reader)


def metrics(rows: list[dict[str, str]]) -> dict[str, Any] | None:
    labeled = [row for row in rows if row["label"] in LABEL_TO_ID]
    if not labeled:
        return None
    confusion = [[0, 0, 0] for _ in LABELS]
    for row in labeled:
        truth = LABEL_TO_ID[row["label"]]
        prediction = LABEL_TO_ID[row["pred"]]
        confusion[truth][prediction] += 1
    correct = sum(confusion[index][index] for index in range(3))
    f1_values: list[float] = []
    for index in range(3):
        true_positive = confusion[index][index]
        actual = sum(confusion[index])
        predicted = sum(confusion[row][index] for row in range(3))
        recall = true_positive / max(1, actual)
        precision = true_positive / max(1, predicted)
        f1_values.append(
            0.0
            if precision + recall == 0
            else 2.0 * precision * recall / (precision + recall)
        )
    return {
        "rows": len(labeled),
        "correct": correct,
        "accuracy": correct / len(labeled),
        "macro_f1": sum(f1_values) / len(f1_values),
        "confusion_matrix_A_B_C": confusion,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=("dace", "accr"), required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--split-summary", type=Path, required=True)
    parser.add_argument("--shard-summary", type=Path, action="append", required=True)
    parser.add_argument("--gpu-ids", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-timestamp", required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--max-audio-seconds", type=float, required=True)
    parser.add_argument("--cpu-threads-per-worker", type=int, required=True)
    parser.add_argument("--workflow-start-epoch", type=float, required=True)
    parser.add_argument("--split-seconds", type=float, required=True)
    parser.add_argument("--parallel-wall-seconds", type=float, required=True)
    args = parser.parse_args()

    merge_started = time.time()
    method = args.method.upper()
    full_manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    full_csv = Path(str(full_manifest["input_csv"]))
    _, source_rows = read_csv(full_csv)
    source_names = [str(row["wav_name"]) for row in source_rows]
    if len(source_names) != len(set(source_names)):
        raise ValueError("source manifest contains duplicate wav_name values")

    gpu_ids = [value.strip() for value in args.gpu_ids.split(",") if value.strip()]
    summaries = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in args.shard_summary
    ]
    if len(summaries) != len(gpu_ids):
        raise ValueError("number of shard summaries does not match GPU IDs")
    if any(summary["method"] != method for summary in summaries):
        raise ValueError("method mismatch in shard summaries")

    prediction_by_name: dict[str, dict[str, str]] = {}
    fields: list[str] | None = None
    for summary in summaries:
        prediction_path = Path(summary["outputs"]["predictions"])
        current_fields, rows = read_csv(prediction_path)
        if fields is None:
            fields = current_fields
        elif fields != current_fields:
            raise ValueError("prediction schemas differ between shards")
        if sha256(prediction_path) != summary["outputs"]["predictions_sha256"]:
            raise ValueError(f"prediction SHA256 mismatch: {prediction_path}")
        for row in rows:
            name = str(row["wav_name"])
            if name in prediction_by_name:
                raise ValueError(f"duplicate prediction across shards: {name}")
            prediction_by_name[name] = row

    if set(prediction_by_name) != set(source_names):
        missing = sorted(set(source_names).difference(prediction_by_name))[:5]
        extra = sorted(set(prediction_by_name).difference(source_names))[:5]
        raise ValueError(f"merged row set mismatch: missing={missing}, extra={extra}")
    ordered = [prediction_by_name[name] for name in source_names]
    if fields is None:
        raise RuntimeError("no prediction fields found")

    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True)
    stem = f"{args.method}_{args.run_timestamp}"
    prediction_path = args.output_dir / f"{stem}_predictions.csv"
    with prediction_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(ordered)

    called_second = sum(int(row["called_second"]) for row in ordered)
    split_payload = json.loads(args.split_summary.read_text(encoding="utf-8"))
    peak_by_gpu = {
        gpu_id: int(summary["peak_cuda_memory_bytes"])
        for gpu_id, summary in zip(gpu_ids, summaries)
    }
    common_hash_fields = (
        "base_config_sha256",
        "primary_adapter_sha256",
        "primary_head_sha256",
        "secondary_adapter_sha256",
        "secondary_head_sha256",
        "selection_sha256",
    )
    common_hashes: dict[str, str] = {}
    for field in common_hash_fields:
        values = {str(summary[field]) for summary in summaries}
        if len(values) != 1:
            raise ValueError(f"shards used different {field}")
        common_hashes[field] = values.pop()

    merge_seconds = time.time() - merge_started
    total_seconds = time.time() - args.workflow_start_epoch
    result = {
        "method": method,
        "method_full_name": summaries[0]["method_full_name"],
        "execution": "multi-GPU complete-dialogue sharding",
        "inputs": ["raw_audio", "ASR"],
        "run_timestamp": args.run_timestamp,
        "rows": len(ordered),
        "labeled_rows": sum(row["label"] in LABEL_TO_ID for row in ordered),
        "metrics": metrics(ordered),
        "gpu_ids": gpu_ids,
        "num_gpus": len(gpu_ids),
        "batch_size_per_gpu": int(args.batch_size),
        "max_audio_seconds": float(args.max_audio_seconds),
        "cpu_threads_per_worker": int(args.cpu_threads_per_worker),
        "base_model_load_count": len(gpu_ids),
        "minimum_model_forwards": 2 if method == "DACE" else 1,
        "maximum_model_forwards": 2,
        "called_second": called_second,
        "call_rate": called_second / len(ordered),
        "average_model_forwards": 1.0 + called_second / len(ordered),
        "manifest": str(args.manifest.resolve()),
        "manifest_sha256": sha256(args.manifest),
        "input_csv": str(full_csv.resolve()),
        "input_csv_sha256": sha256(full_csv),
        **common_hashes,
        "sharding": {
            "strategy": split_payload["strategy"],
            "dialogues_kept_complete": True,
            "shards": split_payload["shards"],
        },
        "peak_cuda_memory_bytes_per_gpu": peak_by_gpu,
        "peak_cuda_memory_bytes_max": max(peak_by_gpu.values()),
        "timing_seconds": {
            "split_manifest": float(args.split_seconds),
            "parallel_inference_wall": float(args.parallel_wall_seconds),
            "merge_and_validation": merge_seconds,
            "total_to_summary_write": total_seconds,
            "slowest_shard_internal_total": max(
                float(summary["timing_seconds"]["total"])
                for summary in summaries
            ),
        },
        "shard_summaries": [str(path.resolve()) for path in args.shard_summary],
        "outputs": {
            "predictions": str(prediction_path.resolve()),
            "predictions_sha256": sha256(prediction_path),
            "summary": str((args.output_dir / f"{stem}_summary.json").resolve()),
        },
    }
    summary_path = args.output_dir / f"{stem}_summary.json"
    summary_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
