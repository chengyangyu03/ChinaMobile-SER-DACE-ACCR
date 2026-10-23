#!/usr/bin/env python3
"""Strict nested class-specific joint calibration of two raw-audio+ASR trunks."""

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
import select_qwen2audio_nested_logistic_dialogue_valonly_abc as simple
import train_qwen2audio_mvef_student_valonly_abc as common


def read(path: Path) -> list[dict[str, str]]:
    common.reject_test_path(path, "joint-logistic validation input")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"empty CSV: {path}")
    return rows


def align(omni_path: Path, q_path: Path, meta_path: Path) -> tuple[Any, ...]:
    omni_rows, q_rows, meta_rows = read(omni_path), read(q_path), read(meta_path)
    omni_by_wav = {Path(row["wav_name"]).name: row for row in omni_rows}
    meta_by_wav = {Path(row["wav_name"]).name: row for row in meta_rows}
    if len(omni_by_wav) != len(omni_rows) or len(meta_by_wav) != len(meta_rows):
        raise ValueError("duplicate input wav_name")
    if len(q_rows) != len(omni_rows) or len(q_rows) != len(meta_rows):
        raise ValueError("unequal input row counts")
    ordered_meta, p_omni, p_q, truth = [], [], [], []
    for row in q_rows:
        wav = Path(row["wav_name"]).name
        if wav not in omni_by_wav or wav not in meta_by_wav:
            raise ValueError(f"missing aligned row for {wav}")
        q_label = base.normalize_label(row.get("label"))
        o_label = base.normalize_label(omni_by_wav[wav].get("label"))
        if q_label not in common.LABEL_TO_ID or q_label != o_label:
            raise ValueError(f"label mismatch for {wav}")
        ordered_meta.append(meta_by_wav[wav])
        p_omni.append([float(omni_by_wav[wav][f"prob_{label}"]) for label in common.LABELS])
        p_q.append([float(row[f"prob_{label}"]) for label in common.LABELS])
        truth.append(common.LABEL_TO_ID[q_label])
    arrays = [np.asarray(values, dtype=np.float64) for values in (p_omni, p_q)]
    for values in arrays:
        values /= values.sum(axis=1, keepdims=True)
    return q_rows, ordered_meta, arrays[0], arrays[1], np.asarray(truth, dtype=np.int64)


def features(omni: np.ndarray, qwen: np.ndarray) -> np.ndarray:
    return np.concatenate((simple.features(omni), simple.features(qwen)), axis=1)


def fit_model(
    omni: np.ndarray, qwen: np.ndarray, truth: np.ndarray,
    c_value: float, class_weight: str | None,
) -> LogisticRegression:
    model = LogisticRegression(
        C=float(c_value), class_weight=class_weight, solver="lbfgs",
        max_iter=1000, random_state=20260722,
    )
    model.fit(features(omni, qwen), truth)
    if not np.array_equal(model.classes_, np.arange(3)):
        raise ValueError(f"joint calibrator lost a class: {model.classes_}")
    return model


def predict_model(model: LogisticRegression, omni: np.ndarray, qwen: np.ndarray) -> np.ndarray:
    result = model.predict_proba(features(omni, qwen)).astype(np.float64)
    result /= result.sum(axis=1, keepdims=True)
    return result


def crossfit(
    omni: np.ndarray, qwen: np.ndarray, truth: np.ndarray, groups: list[str],
    c_value: float, class_weight: str | None, splits: int,
) -> np.ndarray:
    output = np.zeros_like(omni)
    splitter = GroupKFold(n_splits=splits)
    for train_index, heldout_index in splitter.split(np.arange(len(truth)), truth, groups=np.asarray(groups)):
        model = fit_model(omni[train_index], qwen[train_index], truth[train_index], c_value, class_weight)
        output[heldout_index] = predict_model(model, omni[heldout_index], qwen[heldout_index])
    if np.any(output.sum(axis=1) <= 0):
        raise RuntimeError("incomplete joint-logistic crossfit")
    return output


def choose(
    omni: np.ndarray, qwen: np.ndarray, truth: np.ndarray,
    dialogue: list[str], side: list[str], splits: int,
) -> tuple[float, str | None, tuple[Any, ...], dict[str, Any]]:
    best = None
    best_score = None
    best_audit = None
    for c_value in simple.C_VALUES:
        for class_weight in simple.CLASS_WEIGHTS:
            calibrated = crossfit(omni, qwen, truth, dialogue, c_value, class_weight, splits)
            for smoother in simple.smoothing_candidates():
                transformed = base.transform(calibrated, dialogue, side, smoother)
                pred = transformed.argmax(axis=1)
                score = (
                    int(np.sum(pred == truth)), base.macro_f1(truth, pred),
                    -int(class_weight is not None),
                    -abs(np.log10(float(c_value)) - np.log10(0.3)),
                    simple.smoothing_id(smoother),
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--omni", type=Path, required=True)
    parser.add_argument("--qwen2audio", type=Path, required=True)
    parser.add_argument("--meta", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    for path in (args.omni, args.qwen2audio, args.meta, args.output_dir):
        common.reject_test_path(path, "nested joint-logistic path")
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    q_rows, meta, omni, qwen, truth = align(args.omni, args.qwen2audio, args.meta)
    dialogue = [str(row["dialogue_key"]) for row in meta]
    side = [f"{row['dialogue_key']}::{common.side_order(row)[0]}" for row in meta]
    raw_probability = 0.5 * omni + 0.5 * qwen
    raw_pred = raw_probability.argmax(axis=1)
    outer_probability = np.zeros_like(omni)
    fold_records = []
    splitter = GroupKFold(n_splits=4)
    for fold, (train_index, heldout_index) in enumerate(
        splitter.split(np.arange(len(truth)), truth, groups=np.asarray(dialogue))
    ):
        train_dialogue = [dialogue[index] for index in train_index]
        train_side = [side[index] for index in train_index]
        c_value, class_weight, smoother, inner_audit = choose(
            omni[train_index], qwen[train_index], truth[train_index], train_dialogue, train_side, 3
        )
        model = fit_model(omni[train_index], qwen[train_index], truth[train_index], c_value, class_weight)
        calibrated = predict_model(model, omni[heldout_index], qwen[heldout_index])
        heldout_dialogue = [dialogue[index] for index in heldout_index]
        heldout_side = [side[index] for index in heldout_index]
        transformed = base.transform(calibrated, heldout_dialogue, heldout_side, smoother)
        outer_probability[heldout_index] = transformed
        pred = transformed.argmax(axis=1)
        fold_records.append({
            "fold": fold,
            "train_rows": int(len(train_index)),
            "heldout_rows": int(len(heldout_index)),
            "calibrator": simple.model_config_id(c_value, class_weight),
            "smoother": simple.smoothing_id(smoother),
            "inner_audit": inner_audit,
            "raw_correct": int(np.sum(raw_pred[heldout_index] == truth[heldout_index])),
            "selected_correct": int(np.sum(pred == truth[heldout_index])),
            "net": int(np.sum(pred == truth[heldout_index]) - np.sum(raw_pred[heldout_index] == truth[heldout_index])),
        })
    c_value, class_weight, smoother, deployment_audit = choose(omni, qwen, truth, dialogue, side, 4)
    deployment_model = fit_model(omni, qwen, truth, c_value, class_weight)
    pred = outer_probability.argmax(axis=1)
    raw_metrics = base.metrics(truth, raw_pred)
    selected_metrics = base.metrics(truth, pred)
    worst_net = min(record["net"] for record in fold_records)
    gate_pass = selected_metrics["accuracy"] >= .81 and selected_metrics["macro_f1"] >= .76 and worst_net >= 0
    summary = {
        "protocol": "outer-4/inner-3 complete-dialogue class-specific joint logistic calibration of exactly two trunks",
        "test_evaluated": False,
        "candidate_count": len(simple.C_VALUES) * len(simple.CLASS_WEIGHTS) * len(simple.smoothing_candidates()),
        "raw": raw_metrics,
        "nested_selected": selected_metrics,
        "folds": fold_records,
        "deployment_candidate_fit_on_all_validation": {
            "feature_order": ["omni_logodds_A", "omni_logodds_B", "omni_logodds_C", "qwen2audio_logodds_A", "qwen2audio_logodds_B", "qwen2audio_logodds_C"],
            "calibrator": simple.model_config_id(c_value, class_weight),
            "C": c_value,
            "class_weight": class_weight,
            "classes": deployment_model.classes_.tolist(),
            "coef": deployment_model.coef_.tolist(),
            "intercept": deployment_model.intercept_.tolist(),
            "smoother": {"candidate": simple.smoothing_id(smoother), "parameters": base.candidate_parameters(smoother)},
            "crossfit_audit": deployment_audit,
        },
        "gate": {"accuracy_at_least": .81, "macro_f1_at_least": .76, "worst_fold_net_at_least": 0},
        "gate_pass": gate_pass,
        "true_test_authorized": gate_pass,
        "model_count": 2,
        "average_model_forwards": 2.0,
        "calibrator_trainable_parameters": 21,
    }
    args.output_dir.mkdir(parents=True)
    with (args.output_dir / "nested_oof_predictions.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["wav_name", "label", "raw_pred", "pred", "correct", "prob_A", "prob_B", "prob_C"])
        writer.writeheader()
        for index, row in enumerate(q_rows):
            writer.writerow({
                "wav_name": Path(row["wav_name"]).name,
                "label": common.LABELS[int(truth[index])],
                "raw_pred": common.LABELS[int(raw_pred[index])],
                "pred": common.LABELS[int(pred[index])],
                "correct": int(pred[index] == truth[index]),
                **{f"prob_{label}": outer_probability[index, j] for j, label in enumerate(common.LABELS)},
            })
    (args.output_dir / "selection.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
