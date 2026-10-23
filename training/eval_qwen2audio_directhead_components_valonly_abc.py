#!/usr/bin/env python3
"""Extract decomposed direct-head logits from one validation-only forward.

The deployable model is unchanged: one Qwen2-Audio backbone, one LoRA adapter,
and one four-pool residual head.  This evaluator exposes the five additive
three-class logit components (LM, answer, audio mean, audio std, text mean) so
that a tiny strictly cross-fitted readout can calibrate them separately.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from peft import PeftModel
from torch.utils.data import DataLoader
from transformers import AutoProcessor, BitsAndBytesConfig, Qwen2AudioForConditionalGeneration

import train_qwen2audio_directhead_student_valonly_abc as direct
import train_qwen2audio_mvef_student_valonly_abc as common


COMPONENTS = ("lm", "answer", "audio_mean", "audio_std", "text_mean")


@torch.no_grad()
def component_logits(
    model: direct.DirectAudioTextClassifier,
    batch: dict[str, Any],
) -> tuple[torch.Tensor, dict[str, torch.Tensor], float]:
    """Return total logits and their exact five-component decomposition."""
    device = model.bias.device
    model_inputs = {
        key: value.to(device)
        for key, value in batch.items()
        if key in {"input_ids", "attention_mask", "input_features", "feature_attention_mask"}
    }
    output = model.backbone.get_base_model().model(**model_inputs, use_cache=False)
    hidden = output.last_hidden_state.float()
    attention = model_inputs["attention_mask"].bool()
    positions = attention.long().sum(dim=1) - 1
    rows = torch.arange(hidden.shape[0], device=device)
    answer_state = hidden[rows, positions]
    audio_mask = (model_inputs["input_ids"] == model.audio_token_id) & attention
    if torch.any(audio_mask.sum(dim=1) == 0):
        raise RuntimeError("Processor produced a row without expanded audio tokens")
    text_mask = attention & ~audio_mask
    if torch.any(text_mask.sum(dim=1) == 0):
        raise RuntimeError("Processor produced a row without text tokens")

    audio_mean = model.masked_mean(hidden, audio_mask)
    audio_centered = hidden - audio_mean.unsqueeze(1)
    audio_variance = model.masked_mean(audio_centered.square(), audio_mask)
    audio_std = torch.sqrt(audio_variance.clamp_min(1e-8))
    text_mean = model.masked_mean(hidden, text_mask)
    blocks = (answer_state, audio_mean, audio_std, text_mean)
    projected = [
        model.dropout(F.gelu(projection(norm(block))))
        for block, norm, projection in zip(blocks, model.norms, model.projections)
    ]

    components: dict[str, torch.Tensor] = {
        "lm": model.log_scale.exp() * F.linear(answer_state, model.base_weight)
        + model.bias
        + model.output.bias
    }
    residual_names = COMPONENTS[1:]
    for index, (name, value) in enumerate(zip(residual_names, projected)):
        start = index * model.block_hidden
        stop = start + model.block_hidden
        components[name] = F.linear(value, model.output.weight[:, start:stop])
    total = sum(components.values())
    whole = (
        model.log_scale.exp() * F.linear(answer_state, model.base_weight)
        + model.bias
        + model.output(torch.cat(projected, dim=-1))
    )
    reconstruction_error = float((total - whole).abs().max().detach().cpu())
    return total, components, reconstruction_error


def metrics(truth: np.ndarray, prediction: np.ndarray) -> dict[str, Any]:
    confusion = np.zeros((3, 3), dtype=np.int64)
    for expected, actual in zip(truth, prediction):
        confusion[int(expected), int(actual)] += 1
    recalls = np.diag(confusion) / np.maximum(1, confusion.sum(axis=1))
    precisions = np.diag(confusion) / np.maximum(1, confusion.sum(axis=0))
    f1 = 2 * precisions * recalls / np.maximum(1e-12, precisions + recalls)
    return {
        "rows": int(len(truth)),
        "correct": int(np.trace(confusion)),
        "accuracy": float(np.trace(confusion) / len(truth)),
        "macro_f1": float(f1.mean()),
        "confusion_matrix": confusion.tolist(),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--val-csv", type=Path, default=Path("val_202506.csv"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-audio-seconds", type=float, default=16.0)
    parser.add_argument("--include-context", action="store_true")
    parser.add_argument("--max-val-rows", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for path, role in (
        (args.val_csv, "component validation"),
        (args.output_dir, "component output"),
        (args.adapter, "component adapter"),
    ):
        common.reject_test_path(path, role)
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    if not (args.adapter / "classifier_head.pt").is_file():
        raise FileNotFoundError(f"Missing direct classifier head: {args.adapter}")

    rows = common.read_table(args.val_csv, require_historical=False)
    if args.max_val_rows:
        rows = rows[: args.max_val_rows]
    processor = AutoProcessor.from_pretrained(args.model, local_files_only=True)
    processor.tokenizer.padding_side = "right"
    encoded = [processor.tokenizer.encode(label, add_special_tokens=False) for label in common.LABELS]
    if any(len(value) != 1 for value in encoded):
        raise RuntimeError(f"A/B/C are not single tokens: {encoded}")
    token_ids = torch.tensor([value[0] for value in encoded], dtype=torch.long)
    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )
    backbone = Qwen2AudioForConditionalGeneration.from_pretrained(
        args.model,
        local_files_only=True,
        device_map={"": 0},
        torch_dtype=torch.float16,
        quantization_config=quantization,
        low_cpu_mem_usage=True,
    )
    backbone.config.use_cache = False
    backbone = PeftModel.from_pretrained(backbone, args.adapter, is_trainable=False)
    model = direct.DirectAudioTextClassifier(
        backbone,
        token_ids,
        int(backbone.get_base_model().config.audio_token_id),
        initialize_from_lm_head=False,
    ).to(torch.device("cuda", 0))
    direct.load_head(model, args.adapter)
    model.eval()

    loader = DataLoader(
        common.RowDataset(rows),
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=direct.DirectCollator(processor, args.max_audio_seconds, args.include_context),
    )
    output_rows: list[dict[str, Any]] = []
    truth: list[int] = []
    predictions: list[int] = []
    maximum_reconstruction_error = 0.0
    for batch in loader:
        total, components, error = component_logits(model, batch)
        maximum_reconstruction_error = max(maximum_reconstruction_error, error)
        probabilities = torch.softmax(total, dim=-1).cpu().numpy()
        totals = total.cpu().numpy()
        component_arrays = {name: value.cpu().numpy() for name, value in components.items()}
        for row_index, wav in enumerate(batch["wav_names"]):
            expected = int(batch["class_labels"][row_index])
            predicted = int(probabilities[row_index].argmax())
            truth.append(expected)
            predictions.append(predicted)
            record: dict[str, Any] = {
                "wav_name": wav,
                "label": common.LABELS[expected],
                "pred": common.LABELS[predicted],
                "correct": int(expected == predicted),
            }
            for class_index, label in enumerate(common.LABELS):
                record[f"prob_{label}"] = float(probabilities[row_index, class_index])
                record[f"logit_{label}"] = float(totals[row_index, class_index])
                for name in COMPONENTS:
                    record[f"component_{name}_{label}"] = float(
                        component_arrays[name][row_index, class_index]
                    )
            output_rows.append(record)

    args.output_dir.mkdir(parents=True)
    output_csv = args.output_dir / "component_predictions.csv"
    with output_csv.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output_rows[0]))
        writer.writeheader()
        writer.writerows(output_rows)
    result = {
        "protocol": "one-forward exact additive direct-head component extraction",
        "test_evaluated": False,
        "model": str(args.model),
        "adapter": str(args.adapter),
        "adapter_sha256": common.sha256(args.adapter / "adapter_model.safetensors"),
        "classifier_head_sha256": common.sha256(args.adapter / "classifier_head.pt"),
        "components": list(COMPONENTS),
        "reconstruction_max_abs_error": maximum_reconstruction_error,
        "metrics": metrics(np.asarray(truth), np.asarray(predictions)),
        "model_forwards_per_row": 1.0,
        "output": str(output_csv),
        "output_sha256": common.sha256(output_csv),
    }
    (args.output_dir / "summary_metrics.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
