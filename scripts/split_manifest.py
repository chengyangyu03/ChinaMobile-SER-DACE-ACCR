#!/usr/bin/env python3
"""Split a prepared manifest without breaking dialogue context."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import soundfile as sf


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
        rows = list(reader)
        return list(reader.fieldnames), rows


def effective_seconds(path: str, maximum: float) -> float:
    info = sf.info(path)
    if info.samplerate <= 0:
        raise ValueError(f"invalid sample rate: {path}")
    return min(float(maximum), float(info.frames) / float(info.samplerate))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Balance whole dialogues across independent GPU shards"
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-shards", type=int, required=True)
    parser.add_argument("--max-audio-seconds", type=float, default=16.0)
    args = parser.parse_args()

    if args.num_shards <= 0:
        raise ValueError("num-shards must be positive")
    if args.max_audio_seconds <= 0:
        raise ValueError("max-audio-seconds must be positive")
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)

    source_manifest = args.manifest.expanduser().resolve()
    manifest: dict[str, Any] = json.loads(
        source_manifest.read_text(encoding="utf-8")
    )
    source_csv = Path(str(manifest["input_csv"])).expanduser().resolve()
    fields, rows = read_csv(source_csv)
    required = {"wav_name", "dialogue_key", "source_audio_path"}
    if not rows or not required.issubset(fields):
        raise ValueError(f"prepared CSV lacks {sorted(required)}: {source_csv}")

    groups: dict[str, list[tuple[int, dict[str, str], float]]] = defaultdict(list)
    for source_index, row in enumerate(rows):
        dialogue = str(row.get("dialogue_key") or row["wav_name"])
        duration = effective_seconds(
            str(row["source_audio_path"]),
            float(args.max_audio_seconds),
        )
        groups[dialogue].append((source_index, row, duration))

    # Largest-processing-time-first balancing keeps every dialogue on one GPU.
    # This preserves both neighbouring-ASR context and label-free smoothing.
    ordered_groups = sorted(
        groups.items(),
        key=lambda item: (
            -sum(value[2] for value in item[1]),
            -len(item[1]),
            item[0],
        ),
    )
    shard_groups: list[list[tuple[str, list[tuple[int, dict[str, str], float]]]]] = [
        [] for _ in range(args.num_shards)
    ]
    shard_costs = [0.0] * args.num_shards
    shard_rows = [0] * args.num_shards
    for dialogue, members in ordered_groups:
        shard = min(
            range(args.num_shards),
            key=lambda index: (shard_costs[index], shard_rows[index], index),
        )
        shard_groups[shard].append((dialogue, members))
        shard_costs[shard] += sum(value[2] for value in members)
        shard_rows[shard] += len(members)

    args.output_dir.mkdir(parents=True)
    shards: list[dict[str, Any]] = []
    assigned_names: set[str] = set()
    for shard, dialogue_groups in enumerate(shard_groups):
        shard_dir = args.output_dir / f"shard_{shard:02d}"
        shard_dir.mkdir()
        members = [
            value
            for _, group_members in dialogue_groups
            for value in group_members
        ]
        members.sort(key=lambda value: value[0])
        shard_csv = shard_dir / "input.csv"
        with shard_csv.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for _, row, _ in members:
                name = str(row["wav_name"])
                if name in assigned_names:
                    raise ValueError(f"row assigned twice: {name}")
                assigned_names.add(name)
                writer.writerow(row)

        shard_manifest = dict(manifest)
        shard_manifest.update(
            {
                "sample_count": len(members),
                "input_csv": str(shard_csv.resolve()),
                "input_csv_sha256": sha256(shard_csv),
                "shard": {
                    "index": shard,
                    "count": args.num_shards,
                    "dialogue_count": len(dialogue_groups),
                    "effective_audio_seconds": shard_costs[shard],
                    "strategy": "complete-dialogue LPT by capped audio duration",
                    "source_manifest": str(source_manifest),
                    "source_manifest_sha256": sha256(source_manifest),
                },
            }
        )
        shard_manifest_path = shard_dir / "manifest.json"
        shard_manifest_path.write_text(
            json.dumps(shard_manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        shards.append(
            {
                "index": shard,
                "rows": len(members),
                "dialogues": len(dialogue_groups),
                "effective_audio_seconds": shard_costs[shard],
                "manifest": str(shard_manifest_path.resolve()),
                "manifest_sha256": sha256(shard_manifest_path),
                "input_csv": str(shard_csv.resolve()),
                "input_csv_sha256": sha256(shard_csv),
            }
        )

    expected_names = {str(row["wav_name"]) for row in rows}
    if assigned_names != expected_names:
        raise RuntimeError("sharding did not preserve the exact input row set")

    summary = {
        "source_manifest": str(source_manifest),
        "source_manifest_sha256": sha256(source_manifest),
        "source_input_csv": str(source_csv),
        "source_input_csv_sha256": sha256(source_csv),
        "rows": len(rows),
        "dialogues": len(groups),
        "num_shards": args.num_shards,
        "max_audio_seconds": float(args.max_audio_seconds),
        "strategy": "complete-dialogue LPT by capped audio duration",
        "shards": shards,
    }
    summary_path = args.output_dir / "split_summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
