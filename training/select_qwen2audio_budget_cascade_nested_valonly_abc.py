#!/usr/bin/env python3
"""Strict nested budget cascade for two Qwen2-Audio trajectory adapters.

The first adapter is always evaluated.  A second adapter is requested only
when the calibrated first-adapter confidence or top-two margin is below a
validation-frozen quantile.  All calibration, routing, and smoothing choices
are selected inside complete-dialogue outer/inner folds.
"""

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


C_VALUES = (0.03, 0.10, 0.30, 1.0, 3.0)
CALL_FRACTIONS = (0.25, 0.40, 0.55, 0.70, 0.85)
ROUTER_SIGNALS = ("confidence", "margin")
SMOOTHERS: tuple[tuple[Any, ...], ...] = (
    (0.0, 0.0, 1.0, 0.0, 5, None, "dialogue"),
    (0.0, 0.0, 1.0, 0.20, 3, None, "same_side"),
    (0.0, 0.0, 1.0, 0.40, 3, 0.20, "same_side"),
    (0.0, 0.0, 1.0, 0.60, 5, 0.35, "same_side"),
)


def read(path: Path) -> list[dict[str, str]]:
    common.reject_test_path(path, "budget-cascade validation input")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"empty CSV: {path}")
    return rows


def align(
    primary_path: Path,
    secondary_path: Path,
    meta_path: Path,
) -> tuple[list[dict[str, str]], list[dict[str, str]], np.ndarray, np.ndarray, np.ndarray]:
    primary_rows = read(primary_path)
    secondary_rows = read(secondary_path)
    meta_rows = read(meta_path)
    secondary_by_wav = {Path(row["wav_name"]).name: row for row in secondary_rows}
    meta_by_wav = {Path(row["wav_name"]).name: row for row in meta_rows}
    if len(secondary_by_wav) != len(secondary_rows) or len(meta_by_wav) != len(meta_rows):
        raise ValueError("duplicate input wav_name")
    if len(primary_rows) != len(secondary_rows) or len(primary_rows) != len(meta_rows):
        raise ValueError("unequal input row counts")

    ordered_meta: list[dict[str, str]] = []
    primary_probabilities = []
    secondary_probabilities = []
    truth = []
    seen = set()
    for row in primary_rows:
        wav = Path(row["wav_name"]).name
        if wav in seen or wav not in secondary_by_wav or wav not in meta_by_wav:
            raise ValueError(f"duplicate or missing aligned row for {wav}")
        seen.add(wav)
        secondary = secondary_by_wav[wav]
        first_label = base.normalize_label(row.get("label"))
        second_label = base.normalize_label(secondary.get("label"))
        if first_label not in common.LABEL_TO_ID or first_label != second_label:
            raise ValueError(f"label mismatch for {wav}")
        ordered_meta.append(meta_by_wav[wav])
        primary_probabilities.append(
            [float(row[f"prob_{label}"]) for label in common.LABELS]
        )
        secondary_probabilities.append(
            [float(secondary[f"prob_{label}"]) for label in common.LABELS]
        )
        truth.append(common.LABEL_TO_ID[first_label])

    arrays = [
        np.asarray(values, dtype=np.float64)
        for values in (primary_probabilities, secondary_probabilities)
    ]
    for values in arrays:
        values /= values.sum(axis=1, keepdims=True)
    return (
        primary_rows,
        ordered_meta,
        arrays[0],
        arrays[1],
        np.asarray(truth, dtype=np.int64),
    )


def joint_features(primary: np.ndarray, secondary: np.ndarray) -> np.ndarray:
    return np.concatenate(
        (simple.features(primary), simple.features(secondary)),
        axis=1,
    )


def fit_primary(
    probabilities: np.ndarray,
    truth: np.ndarray,
    c_value: float,
) -> LogisticRegression:
    model = LogisticRegression(
        C=float(c_value),
        solver="lbfgs",
        max_iter=1000,
        random_state=20260724,
    )
    model.fit(simple.features(probabilities), truth)
    if not np.array_equal(model.classes_, np.arange(3)):
        raise ValueError("primary calibrator lost a class")
    return model


def predict_primary(model: LogisticRegression, probabilities: np.ndarray) -> np.ndarray:
    result = model.predict_proba(simple.features(probabilities)).astype(np.float64)
    result /= result.sum(axis=1, keepdims=True)
    return result


def fit_joint(
    primary: np.ndarray,
    secondary: np.ndarray,
    truth: np.ndarray,
    c_value: float,
) -> LogisticRegression:
    model = LogisticRegression(
        C=float(c_value),
        solver="lbfgs",
        max_iter=1000,
        random_state=20260724,
    )
    model.fit(joint_features(primary, secondary), truth)
    if not np.array_equal(model.classes_, np.arange(3)):
        raise ValueError("joint calibrator lost a class")
    return model


def predict_joint(
    model: LogisticRegression,
    primary: np.ndarray,
    secondary: np.ndarray,
) -> np.ndarray:
    result = model.predict_proba(joint_features(primary, secondary)).astype(np.float64)
    result /= result.sum(axis=1, keepdims=True)
    return result


def crossfit_primary(
    probabilities: np.ndarray,
    truth: np.ndarray,
    groups: list[str],
    c_value: float,
    splits: int,
) -> np.ndarray:
    output = np.zeros_like(probabilities)
    splitter = GroupKFold(n_splits=splits)
    for train_index, heldout_index in splitter.split(
        np.arange(len(truth)), truth, groups=np.asarray(groups)
    ):
        model = fit_primary(probabilities[train_index], truth[train_index], c_value)
        output[heldout_index] = predict_primary(model, probabilities[heldout_index])
    if np.any(output.sum(axis=1) <= 0):
        raise RuntimeError("incomplete primary crossfit")
    return output


def crossfit_joint(
    primary: np.ndarray,
    secondary: np.ndarray,
    truth: np.ndarray,
    groups: list[str],
    c_value: float,
    splits: int,
) -> np.ndarray:
    output = np.zeros_like(primary)
    splitter = GroupKFold(n_splits=splits)
    for train_index, heldout_index in splitter.split(
        np.arange(len(truth)), truth, groups=np.asarray(groups)
    ):
        model = fit_joint(
            primary[train_index],
            secondary[train_index],
            truth[train_index],
            c_value,
        )
        output[heldout_index] = predict_joint(
            model,
            primary[heldout_index],
            secondary[heldout_index],
        )
    if np.any(output.sum(axis=1) <= 0):
        raise RuntimeError("incomplete joint crossfit")
    return output


def router_values(probabilities: np.ndarray, signal: str) -> np.ndarray:
    if signal == "confidence":
        return probabilities.max(axis=1)
    if signal == "margin":
        ordered = np.sort(probabilities, axis=1)
        return ordered[:, -1] - ordered[:, -2]
    raise ValueError(f"unsupported router signal: {signal}")


def lower_quantile(values: np.ndarray, fraction: float) -> float:
    return float(np.quantile(values, fraction, method="higher"))


def apply_cascade(
    primary: np.ndarray,
    joint: np.ndarray,
    signal: str,
    threshold: float,
) -> tuple[np.ndarray, np.ndarray]:
    called = router_values(primary, signal) <= float(threshold)
    result = np.where(called[:, None], joint, primary)
    result /= result.sum(axis=1, keepdims=True)
    return result, called


def choose_config(
    raw_primary: np.ndarray,
    raw_secondary: np.ndarray,
    truth: np.ndarray,
    dialogue: list[str],
    side: list[str],
    splits: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    primary_cache = {
        c_value: crossfit_primary(raw_primary, truth, dialogue, c_value, splits)
        for c_value in C_VALUES
    }
    joint_cache = {
        c_value: crossfit_joint(
            raw_primary, raw_secondary, truth, dialogue, c_value, splits
        )
        for c_value in C_VALUES
    }
    best_config: dict[str, Any] | None = None
    best_audit: dict[str, Any] | None = None
    best_score: tuple[Any, ...] | None = None
    for primary_c in C_VALUES:
        primary = primary_cache[primary_c]
        for joint_c in C_VALUES:
            joint = joint_cache[joint_c]
            for signal in ROUTER_SIGNALS:
                values = router_values(primary, signal)
                for fraction in CALL_FRACTIONS:
                    threshold = lower_quantile(values, fraction)
                    cascaded, called = apply_cascade(
                        primary, joint, signal, threshold
                    )
                    for smoother in SMOOTHERS:
                        transformed = base.transform(
                            cascaded, dialogue, side, smoother
                        )
                        prediction = transformed.argmax(axis=1)
                        correct = int(np.sum(prediction == truth))
                        macro = base.macro_f1(truth, prediction)
                        call_rate = float(np.mean(called))
                        score = (
                            correct,
                            macro,
                            -call_rate,
                            -abs(np.log10(primary_c) - np.log10(0.3)),
                            -abs(np.log10(joint_c) - np.log10(0.3)),
                        )
                        if best_score is None or score > best_score:
                            best_score = score
                            best_config = {
                                "primary_C": float(primary_c),
                                "joint_C": float(joint_c),
                                "router_signal": signal,
                                "target_call_fraction": float(fraction),
                                "router_threshold": threshold,
                                "smoother": smoother,
                            }
                            best_audit = {
                                "crossfit_correct": correct,
                                "crossfit_macro_f1": macro,
                                "called_second": int(np.sum(called)),
                                "call_rate": call_rate,
                                "average_model_forwards": 1.0 + call_rate,
                            }
    assert best_config is not None and best_audit is not None
    return best_config, best_audit


def model_record(model: LogisticRegression, feature_order: list[str]) -> dict[str, Any]:
    return {
        "feature_order": feature_order,
        "classes": model.classes_.tolist(),
        "coef": model.coef_.tolist(),
        "intercept": model.intercept_.tolist(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--primary", type=Path, required=True)
    parser.add_argument("--secondary", type=Path, required=True)
    parser.add_argument("--meta", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    for path in (args.primary, args.secondary, args.meta, args.output_dir):
        common.reject_test_path(path, "nested budget-cascade path")
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)

    rows, meta, raw_primary, raw_secondary, truth = align(
        args.primary, args.secondary, args.meta
    )
    dialogue = [str(row["dialogue_key"]) for row in meta]
    side = [
        f"{row['dialogue_key']}::{common.side_order(row)[0]}"
        for row in meta
    ]
    raw_prediction = raw_primary.argmax(axis=1)
    outer_probability = np.zeros_like(raw_primary)
    outer_called = np.zeros(len(truth), dtype=bool)
    fold_records = []
    splitter = GroupKFold(n_splits=4)
    for fold, (train_index, heldout_index) in enumerate(
        splitter.split(
            np.arange(len(truth)), truth, groups=np.asarray(dialogue)
        )
    ):
        train_dialogue = [dialogue[index] for index in train_index]
        train_side = [side[index] for index in train_index]
        config, inner_audit = choose_config(
            raw_primary[train_index],
            raw_secondary[train_index],
            truth[train_index],
            train_dialogue,
            train_side,
            3,
        )
        primary_model = fit_primary(
            raw_primary[train_index],
            truth[train_index],
            config["primary_C"],
        )
        joint_model = fit_joint(
            raw_primary[train_index],
            raw_secondary[train_index],
            truth[train_index],
            config["joint_C"],
        )
        fitted_train_primary = predict_primary(
            primary_model, raw_primary[train_index]
        )
        router_threshold = lower_quantile(
            router_values(
                fitted_train_primary,
                config["router_signal"],
            ),
            config["target_call_fraction"],
        )
        heldout_primary = predict_primary(
            primary_model, raw_primary[heldout_index]
        )
        heldout_joint = predict_joint(
            joint_model,
            raw_primary[heldout_index],
            raw_secondary[heldout_index],
        )
        cascaded, called = apply_cascade(
            heldout_primary,
            heldout_joint,
            config["router_signal"],
            router_threshold,
        )
        heldout_dialogue = [dialogue[index] for index in heldout_index]
        heldout_side = [side[index] for index in heldout_index]
        transformed = base.transform(
            cascaded,
            heldout_dialogue,
            heldout_side,
            config["smoother"],
        )
        outer_probability[heldout_index] = transformed
        outer_called[heldout_index] = called
        prediction = transformed.argmax(axis=1)
        raw_correct = int(
            np.sum(raw_prediction[heldout_index] == truth[heldout_index])
        )
        selected_correct = int(
            np.sum(prediction == truth[heldout_index])
        )
        fold_records.append(
            {
                "fold": fold,
                "train_rows": int(len(train_index)),
                "heldout_rows": int(len(heldout_index)),
                "primary_C": config["primary_C"],
                "joint_C": config["joint_C"],
                "router_signal": config["router_signal"],
                "target_call_fraction": config["target_call_fraction"],
                "inner_crossfit_router_threshold": config[
                    "router_threshold"
                ],
                "outer_fitted_router_threshold": router_threshold,
                "smoother": simple.smoothing_id(config["smoother"]),
                "inner_audit": inner_audit,
                "called_second": int(np.sum(called)),
                "call_rate": float(np.mean(called)),
                "average_model_forwards": 1.0 + float(np.mean(called)),
                "raw_primary_correct": raw_correct,
                "selected_correct": selected_correct,
                "net": selected_correct - raw_correct,
            }
        )

    deployment_config, deployment_audit = choose_config(
        raw_primary,
        raw_secondary,
        truth,
        dialogue,
        side,
        4,
    )
    deployment_primary = fit_primary(
        raw_primary, truth, deployment_config["primary_C"]
    )
    deployment_joint = fit_joint(
        raw_primary,
        raw_secondary,
        truth,
        deployment_config["joint_C"],
    )
    deployment_primary_probability = predict_primary(
        deployment_primary, raw_primary
    )
    deployment_router_threshold = lower_quantile(
        router_values(
            deployment_primary_probability,
            deployment_config["router_signal"],
        ),
        deployment_config["target_call_fraction"],
    )
    prediction = outer_probability.argmax(axis=1)
    raw_metrics = base.metrics(truth, raw_prediction)
    selected_metrics = base.metrics(truth, prediction)
    called_count = int(np.sum(outer_called))
    call_rate = float(np.mean(outer_called))
    average_forwards = 1.0 + call_rate
    worst_net = min(record["net"] for record in fold_records)
    gate_pass = (
        selected_metrics["accuracy"] >= 0.81
        and selected_metrics["macro_f1"] >= 0.76
        and worst_net >= 0
        and average_forwards <= 2.0
    )
    summary = {
        "protocol": (
            "outer-4/inner-3 complete-dialogue confidence cascade from "
            "step700 primary to optional step1400 secondary"
        ),
        "test_evaluated": False,
        "candidate_count_per_inner_selection": (
            len(C_VALUES)
            * len(C_VALUES)
            * len(ROUTER_SIGNALS)
            * len(CALL_FRACTIONS)
            * len(SMOOTHERS)
        ),
        "raw_primary": raw_metrics,
        "nested_selected": selected_metrics,
        "folds": fold_records,
        "called_second": called_count,
        "call_rate": call_rate,
        "average_model_forwards": average_forwards,
        "maximum_model_forwards": 2,
        "deployment_candidate_fit_on_all_validation": {
            "primary_C": deployment_config["primary_C"],
            "joint_C": deployment_config["joint_C"],
            "router_signal": deployment_config["router_signal"],
            "target_call_fraction": deployment_config[
                "target_call_fraction"
            ],
            "crossfit_router_threshold": deployment_config[
                "router_threshold"
            ],
            "fitted_validation_router_threshold": (
                deployment_router_threshold
            ),
            "smoother": {
                "candidate": simple.smoothing_id(
                    deployment_config["smoother"]
                ),
                "parameters": base.candidate_parameters(
                    deployment_config["smoother"]
                ),
            },
            "primary_calibrator": model_record(
                deployment_primary,
                ["primary_logodds_A", "primary_logodds_B", "primary_logodds_C"],
            ),
            "joint_calibrator": model_record(
                deployment_joint,
                [
                    "primary_logodds_A",
                    "primary_logodds_B",
                    "primary_logodds_C",
                    "secondary_logodds_A",
                    "secondary_logodds_B",
                    "secondary_logodds_C",
                ],
            ),
            "crossfit_audit": deployment_audit,
        },
        "gate": {
            "accuracy_at_least": 0.81,
            "macro_f1_at_least": 0.76,
            "worst_fold_net_at_least": 0,
            "average_model_forwards_at_most": 2.0,
        },
        "gate_pass": gate_pass,
        "true_test_authorized": False,
        "independently_deployable": True,
        "inputs": ["raw_audio", "ASR"],
        "large_model_count": 2,
        "maximum_calibrator_trainable_parameters": 33,
    }
    args.output_dir.mkdir(parents=True)
    output_csv = args.output_dir / "nested_oof_predictions.csv"
    with output_csv.open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "wav_name",
                "label",
                "raw_primary_pred",
                "pred",
                "correct",
                "called_second",
                "prob_A",
                "prob_B",
                "prob_C",
            ],
        )
        writer.writeheader()
        for index, row in enumerate(rows):
            writer.writerow(
                {
                    "wav_name": Path(row["wav_name"]).name,
                    "label": common.LABELS[int(truth[index])],
                    "raw_primary_pred": common.LABELS[
                        int(raw_prediction[index])
                    ],
                    "pred": common.LABELS[int(prediction[index])],
                    "correct": int(prediction[index] == truth[index]),
                    "called_second": int(outer_called[index]),
                    **{
                        f"prob_{label}": outer_probability[index, label_index]
                        for label_index, label in enumerate(common.LABELS)
                    },
                }
            )
    (args.output_dir / "selection.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
