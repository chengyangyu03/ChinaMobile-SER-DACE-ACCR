#!/usr/bin/env python3
"""Audit private training provenance; emit aggregate counts and hashes only."""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path


SPLITS = {
    "train": "train_timesplit.csv",
    "validation": "val_202506.csv",
    "test": "test_202507.csv",
}
LABELS = {"中性": "A", "厌恶": "B", "愤怒": "C"}
TRAINING_RUN = "eval_outputs/abc_qwen2audio_directhead_fullr16_kd040_twoepoch_valonly_20260722"
TEACHER = "eval_outputs/abc_c25_rootclosed4_replay_step9_c25_final_trainproxy_valonly_20260709/train_oof_predictions.csv"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def source_audio(root: Path, row: dict[str, str]) -> Path:
    # Resolve the source batch under the supplied root even after relocating it.
    return root / Path(row["source_audio_path"]).parent.name / row["wav_name"]


def audit(root: Path, compare_audio: bool = False) -> dict:
    result = {"protocol": "aggregate audit of final DACE/ACCR training provenance", "splits": {}}
    frames = {}
    for split, filename in SPLITS.items():
        rows = read(root / filename)
        frames[split] = rows
        result["splits"][split] = {
            "table": filename,
            "sha256": sha256(root / filename),
            "rows": len(rows),
            "dialogues": len({r["dialogue_key"] for r in rows}),
            "source_counts": dict(sorted(Counter(r["source"] for r in rows).items())),
            "audio_directory_counts": dict(sorted(Counter(Path(r["source_audio_path"]).parent.name for r in rows).items())),
            "label_counts": dict(sorted(Counter(LABELS[r["label"]] for r in rows).items())),
            "month_counts": dict(sorted(Counter(r["month"] for r in rows).items())),
            "duplicate_wav_names": len(rows) - len({r["wav_name"] for r in rows}),
            "missing_audio": sum(not source_audio(root, r).is_file() for r in rows),
        }
    result["dialogue_overlap"] = {}
    split_names = list(SPLITS)
    for index, first in enumerate(split_names):
        for second in split_names[index + 1:]:
            result["dialogue_overlap"][f"{first}_{second}"] = len(
                {r["dialogue_key"] for r in frames[first]}
                & {r["dialogue_key"] for r in frames[second]}
            )
    result["source_tables"] = {}
    for name in ("data_1w2.csv", "data_1w5.csv"):
        rows = read(root / name)
        column = "Emotion" if name == "data_1w2.csv" else "label"
        result["source_tables"][name] = {
            "rows": len(rows), "label_counts": dict(Counter(r[column] for r in rows)),
            "sha256": sha256(root / name),
        }
    manifest = json.loads((root / TRAINING_RUN / "manifest.json").read_text(encoding="utf-8"))
    result["training_manifest_sha256"] = sha256(root / TRAINING_RUN / "manifest.json")
    result["training_input_hashes_match_manifest"] = {
        key: sha256(root / filename) == manifest["inputs"][key]["sha256"]
        for key, filename in (("train_csv", SPLITS["train"]), ("val_csv", SPLITS["validation"]), ("teacher_train", TEACHER))
    }
    teacher_path = root / TEACHER
    teachers = {Path(r["wav_name"]).name: r for r in read(teacher_path)}
    correct = wrong = missing = 0
    for row in frames["train"]:
        teacher = teachers.get(row["wav_name"])
        if teacher is None:
            missing += 1
        else:
            prediction = max("ABC", key=lambda label: float(teacher[f"prob_{label}"]))
            if prediction == LABELS[row["label"]]:
                correct += 1
            else:
                wrong += 1
    result["teacher"] = {
        "sha256": sha256(teacher_path), "rows": len(teachers),
        "correct_argmax_with_kd": correct, "wrong_argmax_ce_only": wrong,
        "missing_teacher_ce_only": missing, "total_ce_only": wrong + missing,
        "kd_alpha_for_eligible_rows": 0.4, "temperature": 2.0,
        "matches_saved_training_manifest": (
            correct == manifest["teacher_truth_agreement_filter"]["retained_correct_argmax"]
            and wrong == manifest["teacher_truth_agreement_filter"]["suppressed_wrong_argmax"]
            and missing == manifest["teacher_coverage"]["missing_truth_only"]
        ),
    }
    result["final_weights"] = []
    for step, side in ((700, "primary"), (1400, "secondary")):
        for name in ("adapter_model.safetensors", "classifier_head.pt"):
            checkpoint = root / TRAINING_RUN / f"checkpoint-{step}" / name
            release = root / "modelscope_release_dace_accr_20260724/models/shared" / f"{side}_adapter" / name
            digest = sha256(checkpoint)
            result["final_weights"].append({
                "adapter": side, "step": step, "file": name,
                "sha256": digest, "matches_modelscope_release": digest == sha256(release),
            })
    all_rows = [row for rows in frames.values() for row in rows]
    result["audio_directory"] = {
        "rows_in_merged_dataset": len(all_rows),
        "wav_files": sum(1 for _ in (root / "audio").glob("*.wav")),
        "missing_merged_files": sum(not (root / "audio" / row["wav_name"]).is_file() for row in all_rows),
        "independent_third_training_source": False,
        "direct_audio_directory_reads_in_final_training_table": sum(
            Path(row["source_audio_path"]).parent.name == "audio" for row in frames["train"]
        ),
    }
    if compare_audio:
        matched = 0
        for row in all_rows:
            if sha256(source_audio(root, row)) == sha256(root / "audio" / row["wav_name"]):
                matched += 1
        result["audio_directory"]["sha256_compared"] = len(all_rows)
        result["audio_directory"]["byte_identical_to_sources"] = matched
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--compare-audio", action="store_true")
    args = parser.parse_args()
    result = audit(args.project_root.resolve(), args.compare_audio)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(f"Wrote aggregate training audit: {args.output}")


if __name__ == "__main__":
    main()
