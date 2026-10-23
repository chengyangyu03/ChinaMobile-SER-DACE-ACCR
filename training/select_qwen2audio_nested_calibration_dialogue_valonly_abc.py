#!/usr/bin/env python3
"""Nested dialogue-group selection of zero-forward Qwen2-Audio postprocessing.

Every outer heldout dialogue is transformed by a fixed candidate chosen using
only the other dialogues.  Candidate transforms contain only class-logit bias,
temperature, and label-free dialogue probability smoothing; no extra model is
loaded.  Test-like inputs and outputs are deliberately rejected.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.model_selection import GroupKFold

import train_qwen2audio_mvef_student_valonly_abc as common


LABELS = common.LABELS
BIAS_VALUES = (-0.30, -0.15, 0.0, 0.15, 0.30)
TEMPERATURES = (0.75, 1.0, 1.25, 1.5)
ALPHAS = (0.0, 0.15, 0.30, 0.45, 0.60)
GROUP_MINS = (3, 5)
MAX_MARGINS: tuple[float | None, ...] = (None, 0.20, 0.35)
GROUP_MODES = ("dialogue", "same_side")


def read_csv(path: Path) -> list[dict[str, str]]:
    common.reject_test_path(path, "validation input")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"Empty CSV: {path}")
    return rows


def normalize_label(value: Any) -> str | None:
    raw = str(value).strip()
    return common.CN_TO_LABEL.get(raw) or common.CN_TO_LABEL.get(raw.lower())


def hash_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


def macro_f1(truth: np.ndarray, pred: np.ndarray) -> float:
    values = []
    for label in range(3):
        tp = int(np.sum((truth == label) & (pred == label)))
        fp = int(np.sum((truth != label) & (pred == label)))
        fn = int(np.sum((truth == label) & (pred != label)))
        precision = tp / max(1, tp + fp)
        recall = tp / max(1, tp + fn)
        values.append(2 * precision * recall / max(1e-12, precision + recall))
    return float(np.mean(values))


def group_codes(keys: list[str]) -> tuple[np.ndarray, int]:
    mapping: dict[str, int] = {}
    codes = []
    for key in keys:
        if key not in mapping:
            mapping[key] = len(mapping)
        codes.append(mapping[key])
    return np.asarray(codes, dtype=np.int64), len(mapping)


def group_mean(probabilities: np.ndarray, codes: np.ndarray, count: int) -> tuple[np.ndarray, np.ndarray]:
    sums = np.zeros((count, 3), dtype=np.float64)
    sizes = np.zeros(count, dtype=np.int64)
    np.add.at(sums, codes, probabilities)
    np.add.at(sizes, codes, 1)
    means = sums / np.maximum(1, sizes)[:, None]
    return means[codes], sizes[codes]


def candidate_id(candidate: tuple[Any, ...]) -> str:
    bias_a, bias_c, temperature, alpha, group_min, max_margin, mode = candidate
    margin = "none" if max_margin is None else f"{max_margin:.2f}"
    return (
        f"ba{bias_a:+.2f}_bc{bias_c:+.2f}_t{temperature:.2f}_a{alpha:.2f}_"
        f"min{group_min}_m{margin}_{mode}"
    )


def candidate_parameters(candidate: tuple[Any, ...]) -> dict[str, Any]:
    bias_a, bias_c, temperature, alpha, group_min, max_margin, mode = candidate
    return {
        "bias_a": float(bias_a),
        "bias_c": float(bias_c),
        "temperature": float(temperature),
        "alpha": float(alpha),
        "group_min": int(group_min),
        "max_margin": None if max_margin is None else float(max_margin),
        "mode": str(mode),
    }


def complexity(candidate: tuple[Any, ...]) -> tuple[float, float, float, int, int]:
    bias_a, bias_c, temperature, alpha, group_min, max_margin, mode = candidate
    return (
        abs(float(bias_a)) + abs(float(bias_c)),
        abs(float(temperature) - 1.0),
        float(alpha),
        0 if max_margin is None else 1,
        0 if mode == "dialogue" else 1,
    )


def transform(
    probabilities: np.ndarray,
    dialogue_keys: list[str],
    side_keys: list[str],
    candidate: tuple[Any, ...],
) -> np.ndarray:
    bias_a, bias_c, temperature, alpha, group_min, max_margin, mode = candidate
    logits = np.log(np.clip(probabilities, 1e-12, 1.0))
    logits += np.asarray([bias_a, 0.0, bias_c], dtype=np.float64)[None, :]
    calibrated = softmax(logits / float(temperature))
    if float(alpha) == 0.0:
        return calibrated
    keys = dialogue_keys if mode == "dialogue" else side_keys
    codes, count = group_codes(keys)
    means, sizes = group_mean(calibrated, codes, count)
    smoothed = (1.0 - float(alpha)) * calibrated + float(alpha) * means
    eligible = sizes >= int(group_min)
    if max_margin is not None:
        ordered = np.sort(calibrated, axis=1)
        eligible &= (ordered[:, -1] - ordered[:, -2]) <= float(max_margin)
    return np.where(eligible[:, None], smoothed, calibrated)


def all_candidates() -> list[tuple[Any, ...]]:
    candidates = []
    for bias_a in BIAS_VALUES:
        for bias_c in BIAS_VALUES:
            for temperature in TEMPERATURES:
                for alpha in ALPHAS:
                    # alpha=0 has no group dependence; keep one canonical copy.
                    if alpha == 0.0:
                        candidates.append((bias_a, bias_c, temperature, alpha, 5, None, "dialogue"))
                        continue
                    for group_min in GROUP_MINS:
                        for max_margin in MAX_MARGINS:
                            for mode in GROUP_MODES:
                                candidates.append((bias_a, bias_c, temperature, alpha, group_min, max_margin, mode))
    unique = {candidate_id(candidate): candidate for candidate in candidates}
    return [unique[key] for key in sorted(unique)]


def choose(
    probabilities: np.ndarray,
    truth: np.ndarray,
    dialogue_keys: list[str],
    side_keys: list[str],
    candidates: list[tuple[Any, ...]],
) -> tuple[Any, ...]:
    best: tuple[Any, ...] | None = None
    best_score: tuple[Any, ...] | None = None
    calibration_cache: dict[tuple[float, float, float], tuple[np.ndarray, np.ndarray]] = {}
    group_cache: dict[tuple[float, float, float, str], tuple[np.ndarray, np.ndarray]] = {}
    for candidate in candidates:
        bias_a, bias_c, temperature, alpha, group_min, max_margin, mode = candidate
        calibration_key = (float(bias_a), float(bias_c), float(temperature))
        if calibration_key not in calibration_cache:
            logits = np.log(np.clip(probabilities, 1e-12, 1.0))
            logits += np.asarray([bias_a, 0.0, bias_c], dtype=np.float64)[None, :]
            calibrated = softmax(logits / float(temperature))
            ordered = np.sort(calibrated, axis=1)
            margins = ordered[:, -1] - ordered[:, -2]
            calibration_cache[calibration_key] = calibrated, margins
        calibrated, margins = calibration_cache[calibration_key]
        if float(alpha) == 0.0:
            transformed = calibrated
        else:
            group_key = (*calibration_key, str(mode))
            if group_key not in group_cache:
                keys = dialogue_keys if mode == "dialogue" else side_keys
                codes, count = group_codes(keys)
                group_cache[group_key] = group_mean(calibrated, codes, count)
            means, sizes = group_cache[group_key]
            smoothed = (1.0 - float(alpha)) * calibrated + float(alpha) * means
            eligible = sizes >= int(group_min)
            if max_margin is not None:
                eligible &= margins <= float(max_margin)
            transformed = np.where(eligible[:, None], smoothed, calibrated)
        pred = transformed.argmax(axis=1)
        score = (
            int(np.sum(pred == truth)),
            macro_f1(truth, pred),
            *tuple(-value for value in complexity(candidate)),
            candidate_id(candidate),
        )
        if best_score is None or score > best_score:
            best, best_score = candidate, score
    assert best is not None
    return best


def metrics(truth: np.ndarray, pred: np.ndarray) -> dict[str, Any]:
    confusion = np.zeros((3, 3), dtype=np.int64)
    np.add.at(confusion, (truth, pred), 1)
    return {
        "rows": int(len(truth)),
        "correct": int(np.sum(truth == pred)),
        "accuracy": float(np.mean(truth == pred)),
        "macro_f1": macro_f1(truth, pred),
        "confusion_matrix": confusion.tolist(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--meta", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    for path in (args.predictions, args.meta, args.output_dir):
        common.reject_test_path(path, "nested calibration/dialogue path")
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)

    prediction_rows = read_csv(args.predictions)
    meta_rows = read_csv(args.meta)
    meta_by_wav = {Path(row["wav_name"]).name: row for row in meta_rows}
    if len(meta_by_wav) != len(meta_rows):
        raise ValueError("Duplicate wav_name in metadata")
    ordered_meta = []
    probabilities = []
    truth = []
    seen: set[str] = set()
    for row in prediction_rows:
        wav = Path(row["wav_name"]).name
        if wav in seen or wav not in meta_by_wav:
            raise ValueError(f"Duplicate or missing metadata for {wav}")
        seen.add(wav)
        ordered_meta.append(meta_by_wav[wav])
        probabilities.append([float(row[f"prob_{label}"]) for label in LABELS])
        label = normalize_label(row["label"])
        if label not in common.LABEL_TO_ID:
            raise ValueError(f"Unsupported label for {wav}")
        truth.append(common.LABEL_TO_ID[label])
    probabilities_array = np.asarray(probabilities, dtype=np.float64)
    probabilities_array /= probabilities_array.sum(axis=1, keepdims=True)
    truth_array = np.asarray(truth, dtype=np.int64)
    dialogue_keys = [str(row["dialogue_key"]) for row in ordered_meta]
    side_keys = [f"{row['dialogue_key']}::{common.side_order(row)[0]}" for row in ordered_meta]
    candidates = all_candidates()

    splitter = GroupKFold(n_splits=4)
    outer_probabilities = np.zeros_like(probabilities_array)
    fold_records = []
    raw_pred = probabilities_array.argmax(axis=1)
    for fold, (train_index, heldout_index) in enumerate(
        splitter.split(np.arange(len(truth_array)), truth_array, groups=np.asarray(dialogue_keys))
    ):
        train_dialogues = [dialogue_keys[index] for index in train_index]
        train_sides = [side_keys[index] for index in train_index]
        candidate = choose(
            probabilities_array[train_index], truth_array[train_index], train_dialogues, train_sides, candidates
        )
        heldout_dialogues = [dialogue_keys[index] for index in heldout_index]
        heldout_sides = [side_keys[index] for index in heldout_index]
        outer_probabilities[heldout_index] = transform(
            probabilities_array[heldout_index], heldout_dialogues, heldout_sides, candidate
        )
        heldout_pred = outer_probabilities[heldout_index].argmax(axis=1)
        fold_records.append({
            "fold": fold,
            "train_rows": int(len(train_index)),
            "heldout_rows": int(len(heldout_index)),
            "train_dialogue_sha256": hash_text("\n".join(sorted(set(train_dialogues)))),
            "heldout_dialogue_sha256": hash_text("\n".join(sorted(set(heldout_dialogues)))),
            "candidate": candidate_id(candidate),
            "raw_correct": int(np.sum(raw_pred[heldout_index] == truth_array[heldout_index])),
            "selected_correct": int(np.sum(heldout_pred == truth_array[heldout_index])),
            "net": int(np.sum(heldout_pred == truth_array[heldout_index]) - np.sum(raw_pred[heldout_index] == truth_array[heldout_index])),
        })
    outer_pred = outer_probabilities.argmax(axis=1)
    deployment_candidate = choose(probabilities_array, truth_array, dialogue_keys, side_keys, candidates)
    raw_metrics = metrics(truth_array, raw_pred)
    selected_metrics = metrics(truth_array, outer_pred)
    gate_pass = (
        selected_metrics["accuracy"] >= 0.81
        and selected_metrics["macro_f1"] >= 0.76
        and min(record["net"] for record in fold_records) >= 0
    )
    summary = {
        "protocol": "nested four-fold complete-dialogue class calibration and model-free smoothing",
        "test_evaluated": False,
        "candidate_count": len(candidates),
        "grid": {
            "bias_a": BIAS_VALUES, "bias_b": [0.0], "bias_c": BIAS_VALUES,
            "temperature": TEMPERATURES, "alpha": ALPHAS, "group_min": GROUP_MINS,
            "max_margin": MAX_MARGINS, "group_mode": GROUP_MODES,
        },
        "raw": raw_metrics,
        "nested_selected": selected_metrics,
        "folds": fold_records,
        "deployment_candidate_fit_on_all_validation": {
            "candidate": candidate_id(deployment_candidate),
            "parameters": candidate_parameters(deployment_candidate),
        },
        "gate": {"accuracy_at_least": 0.81, "macro_f1_at_least": 0.76, "worst_fold_net_at_least": 0},
        "gate_pass": gate_pass,
        "true_test_authorized": gate_pass,
        "average_model_forwards": 1.0,
        "single_student": True,
    }
    args.output_dir.mkdir(parents=True)
    with (args.output_dir / "nested_oof_predictions.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        fieldnames = ["wav_name", "label", "raw_pred", "pred", "correct", "prob_A", "prob_B", "prob_C"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for index, (row, raw, pred, probability) in enumerate(
            zip(prediction_rows, raw_pred, outer_pred, outer_probabilities)
        ):
            writer.writerow({
                "wav_name": Path(row["wav_name"]).name,
                "label": LABELS[int(truth_array[index])],
                "raw_pred": LABELS[int(raw)], "pred": LABELS[int(pred)],
                "correct": int(int(pred) == int(truth_array[index])),
                "prob_A": probability[0], "prob_B": probability[1], "prob_C": probability[2],
            })
    (args.output_dir / "selection.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
