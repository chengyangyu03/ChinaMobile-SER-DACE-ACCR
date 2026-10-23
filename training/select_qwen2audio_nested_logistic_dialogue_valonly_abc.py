#!/usr/bin/env python3
"""Nested tiny-logistic calibration plus model-free dialogue smoothing."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold

import select_qwen2audio_nested_calibration_dialogue_valonly_abc as base
import train_qwen2audio_mvef_student_valonly_abc as common


C_VALUES = (0.01, 0.03, 0.10, 0.30, 1.0, 3.0, 10.0)
CLASS_WEIGHTS: tuple[str | None, ...] = (None, "balanced")
ALPHAS = (0.0, 0.20, 0.40, 0.60, 0.80)
GROUP_MINS = (3, 5)
MAX_MARGINS: tuple[float | None, ...] = (None, 0.20, 0.35)
GROUP_MODES = ("dialogue", "same_side")


def features(probabilities: np.ndarray) -> np.ndarray:
    values = np.log(np.clip(probabilities, 1e-12, 1.0))
    return values - values.mean(axis=1, keepdims=True)


def smoothing_candidates() -> list[tuple[Any, ...]]:
    result = [(0.0, 0.0, 1.0, 0.0, 5, None, "dialogue")]
    result.extend(
        (0.0, 0.0, 1.0, alpha, group_min, margin, mode)
        for alpha in ALPHAS[1:]
        for group_min in GROUP_MINS
        for margin in MAX_MARGINS
        for mode in GROUP_MODES
    )
    return result


def model_config_id(c_value: float, class_weight: str | None) -> str:
    return f"C{c_value:g}_cw{class_weight or 'none'}"


def smoothing_id(candidate: tuple[Any, ...]) -> str:
    return base.candidate_id(candidate)


def fit_model(probabilities: np.ndarray, truth: np.ndarray, c_value: float, class_weight: str | None) -> LogisticRegression:
    model = LogisticRegression(
        C=float(c_value), class_weight=class_weight, solver="lbfgs",
        max_iter=1000, random_state=20260722,
    )
    model.fit(features(probabilities), truth)
    if not np.array_equal(model.classes_, np.arange(3)):
        raise ValueError(f"calibrator lost a class: {model.classes_}")
    return model


def predict_model(model: LogisticRegression, probabilities: np.ndarray) -> np.ndarray:
    result = model.predict_proba(features(probabilities)).astype(np.float64)
    result /= result.sum(axis=1, keepdims=True)
    return result


def crossfit_calibrator(
    probabilities: np.ndarray,
    truth: np.ndarray,
    groups: list[str],
    c_value: float,
    class_weight: str | None,
    splits: int,
) -> np.ndarray:
    output = np.zeros_like(probabilities)
    splitter = GroupKFold(n_splits=splits)
    for train_index, heldout_index in splitter.split(np.arange(len(truth)), truth, groups=np.asarray(groups)):
        model = fit_model(probabilities[train_index], truth[train_index], c_value, class_weight)
        output[heldout_index] = predict_model(model, probabilities[heldout_index])
    if np.any(output.sum(axis=1) <= 0):
        raise RuntimeError("incomplete logistic crossfit")
    return output


def choose_config(
    probabilities: np.ndarray,
    truth: np.ndarray,
    dialogue: list[str],
    side: list[str],
    inner_splits: int,
) -> tuple[float, str | None, tuple[Any, ...], dict[str, Any]]:
    best = None
    best_score = None
    best_audit = None
    smoothers = smoothing_candidates()
    for c_value in C_VALUES:
        for class_weight in CLASS_WEIGHTS:
            calibrated = crossfit_calibrator(
                probabilities, truth, dialogue, c_value, class_weight, inner_splits
            )
            for smoother in smoothers:
                transformed = base.transform(calibrated, dialogue, side, smoother)
                pred = transformed.argmax(axis=1)
                score = (
                    int(np.sum(pred == truth)),
                    base.macro_f1(truth, pred),
                    -int(class_weight is not None),
                    -abs(np.log10(float(c_value)) - np.log10(0.3)),
                    smoothing_id(smoother),
                )
                if best_score is None or score > best_score:
                    best = (c_value, class_weight, smoother)
                    best_score = score
                    best_audit = {
                        "crossfit_correct": int(np.sum(pred == truth)),
                        "crossfit_macro_f1": base.macro_f1(truth, pred),
                    }
    assert best is not None and best_audit is not None
    return *best, best_audit


def read_inputs(prediction_path: Path, meta_path: Path) -> tuple[Any, ...]:
    prediction_rows = base.read_csv(prediction_path)
    meta_rows = base.read_csv(meta_path)
    meta_by_wav = {Path(row["wav_name"]).name: row for row in meta_rows}
    if len(meta_by_wav) != len(meta_rows) or len(prediction_rows) != len(meta_rows):
        raise ValueError("duplicate or unequal prediction/metadata rows")
    ordered_meta = []
    probabilities = []
    truth = []
    seen = set()
    for row in prediction_rows:
        wav = Path(row["wav_name"]).name
        if wav in seen or wav not in meta_by_wav:
            raise ValueError(f"duplicate or missing metadata for {wav}")
        seen.add(wav)
        ordered_meta.append(meta_by_wav[wav])
        probabilities.append([float(row[f"prob_{label}"]) for label in common.LABELS])
        label = base.normalize_label(row["label"])
        if label not in common.LABEL_TO_ID:
            raise ValueError(f"unsupported label for {wav}")
        truth.append(common.LABEL_TO_ID[label])
    values = np.asarray(probabilities, dtype=np.float64)
    values /= values.sum(axis=1, keepdims=True)
    return prediction_rows, ordered_meta, values, np.asarray(truth, dtype=np.int64)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--meta", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    for path in (args.predictions, args.meta, args.output_dir):
        common.reject_test_path(path, "nested logistic path")
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    prediction_rows, meta, probabilities, truth = read_inputs(args.predictions, args.meta)
    dialogue = [str(row["dialogue_key"]) for row in meta]
    side = [f"{row['dialogue_key']}::{common.side_order(row)[0]}" for row in meta]
    raw_pred = probabilities.argmax(axis=1)
    outer_probability = np.zeros_like(probabilities)
    fold_records = []
    splitter = GroupKFold(n_splits=4)
    for fold, (train_index, heldout_index) in enumerate(
        splitter.split(np.arange(len(truth)), truth, groups=np.asarray(dialogue))
    ):
        train_dialogue = [dialogue[index] for index in train_index]
        train_side = [side[index] for index in train_index]
        c_value, class_weight, smoother, inner_audit = choose_config(
            probabilities[train_index], truth[train_index], train_dialogue, train_side, 3
        )
        model = fit_model(probabilities[train_index], truth[train_index], c_value, class_weight)
        heldout_calibrated = predict_model(model, probabilities[heldout_index])
        heldout_dialogue = [dialogue[index] for index in heldout_index]
        heldout_side = [side[index] for index in heldout_index]
        transformed = base.transform(heldout_calibrated, heldout_dialogue, heldout_side, smoother)
        outer_probability[heldout_index] = transformed
        pred = transformed.argmax(axis=1)
        fold_records.append({
            "fold": fold,
            "train_rows": int(len(train_index)),
            "heldout_rows": int(len(heldout_index)),
            "calibrator": model_config_id(c_value, class_weight),
            "smoother": smoothing_id(smoother),
            "inner_audit": inner_audit,
            "raw_correct": int(np.sum(raw_pred[heldout_index] == truth[heldout_index])),
            "selected_correct": int(np.sum(pred == truth[heldout_index])),
            "net": int(np.sum(pred == truth[heldout_index]) - np.sum(raw_pred[heldout_index] == truth[heldout_index])),
        })

    c_value, class_weight, smoother, deployment_audit = choose_config(
        probabilities, truth, dialogue, side, 4
    )
    deployment_model = fit_model(probabilities, truth, c_value, class_weight)
    pred = outer_probability.argmax(axis=1)
    selected_metrics = base.metrics(truth, pred)
    raw_metrics = base.metrics(truth, raw_pred)
    worst_net = min(record["net"] for record in fold_records)
    gate_pass = selected_metrics["accuracy"] >= .81 and selected_metrics["macro_f1"] >= .76 and worst_net >= 0
    summary = {
        "protocol": "outer-4/inner-3 complete-dialogue multinomial logistic calibration plus model-free smoothing",
        "test_evaluated": False,
        "candidate_count": len(C_VALUES) * len(CLASS_WEIGHTS) * len(smoothing_candidates()),
        "raw": raw_metrics,
        "nested_selected": selected_metrics,
        "folds": fold_records,
        "deployment_candidate_fit_on_all_validation": {
            "calibrator": model_config_id(c_value, class_weight),
            "C": c_value,
            "class_weight": class_weight,
            "classes": deployment_model.classes_.tolist(),
            "coef": deployment_model.coef_.tolist(),
            "intercept": deployment_model.intercept_.tolist(),
            "smoother": {
                "candidate": smoothing_id(smoother),
                "parameters": base.candidate_parameters(smoother),
            },
            "crossfit_audit": deployment_audit,
        },
        "gate": {"accuracy_at_least": .81, "macro_f1_at_least": .76, "worst_fold_net_at_least": 0},
        "gate_pass": gate_pass,
        "true_test_authorized": gate_pass,
        "average_model_forwards": 1.0,
        "single_student": True,
        "calibrator_trainable_parameters": 12,
    }
    args.output_dir.mkdir(parents=True)
    with (args.output_dir / "nested_oof_predictions.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=[
            "wav_name", "label", "raw_pred", "pred", "correct", "prob_A", "prob_B", "prob_C"
        ])
        writer.writeheader()
        for index, row in enumerate(prediction_rows):
            writer.writerow({
                "wav_name": Path(row["wav_name"]).name,
                "label": common.LABELS[int(truth[index])],
                "raw_pred": common.LABELS[int(raw_pred[index])],
                "pred": common.LABELS[int(pred[index])],
                "correct": int(pred[index] == truth[index]),
                "prob_A": outer_probability[index, 0],
                "prob_B": outer_probability[index, 1],
                "prob_C": outer_probability[index, 2],
            })
    (args.output_dir / "selection.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
