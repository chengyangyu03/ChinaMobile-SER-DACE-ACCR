#!/usr/bin/env python3
"""Rank strict nested pairs of two frozen Qwen2-Audio adapters."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from train_qwen2audio_mvef_student_valonly_abc import sha256


def parse(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("expected NAME=RUN_DIRECTORY")
    name, path = value.split("=", 1)
    return name, Path(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", action="append", type=parse, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if len(args.candidate) < 2:
        raise ValueError("at least two adapter-pair candidates are required")
    if "test" in str(args.output).lower():
        raise ValueError("diagnostic selector refuses test-like output")
    if args.output.exists():
        raise FileExistsError(args.output)
    names = [name for name, _ in args.candidate]
    if len(names) != len(set(names)):
        raise ValueError("duplicate adapter-pair candidate")

    candidates = []
    for order, (name, run) in enumerate(args.candidate):
        if "test" in str(run).lower():
            raise ValueError("diagnostic selector refuses test-like input")
        selection_path = run / "selection.json"
        predictions_path = run / "nested_oof_predictions.csv"
        selection = json.loads(selection_path.read_text(encoding="utf-8"))
        metrics = selection["nested_selected"]
        folds = selection["folds"]
        if int(metrics["rows"]) != 3196 or len(folds) != 4:
            raise ValueError(f"incomplete strict nested pair: {run}")
        candidates.append(
            {
                "name": name,
                "order": order,
                "run": str(run),
                "metrics": metrics,
                "folds": folds,
                "worst_fold_net": min(int(fold["net"]) for fold in folds),
                "deployment_candidate": selection[
                    "deployment_candidate_fit_on_all_validation"
                ],
                "selection_sha256": sha256(selection_path),
                "predictions_sha256": sha256(predictions_path),
            }
        )
    winner = max(
        candidates,
        key=lambda item: (
            float(item["metrics"]["accuracy"]),
            float(item["metrics"]["macro_f1"]),
            int(item["worst_fold_net"]),
            -int(item["order"]),
        ),
    )
    result = {
        "protocol": (
            "adaptive diagnostic only: strict nested pairing of two frozen "
            "Qwen2-Audio adapters sharing one base architecture"
        ),
        "test_evaluated": False,
        "candidate_count": len(candidates),
        "candidates": candidates,
        "winner": winner,
        "reference_omni_pair_accuracy": 0.8016270337922403,
        "beats_reference_omni_pair": (
            float(winner["metrics"]["accuracy"]) > 0.8016270337922403
        ),
        "gate_pass": False,
        "true_test_authorized": False,
        "diagnostic_only": True,
        "model_architecture_count": 1,
        "adapter_count": 2,
        "model_forwards": 2.0,
    }
    args.output.parent.mkdir(parents=True, exist_ok=False)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
