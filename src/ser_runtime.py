#!/usr/bin/env python3
"""Shared end-to-end runtime for the DACE and ACCR SER methods.

Both methods load one 4-bit Qwen2-Audio base and switch between two LoRA
adapters from the same training trajectory.  They consume only raw audio and
ASR at inference time.  All calibration and dialogue smoothing parameters are
frozen in the published validation-selection JSON files.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F
from peft import PeftModel
from scipy.signal import resample_poly
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoProcessor,
    BitsAndBytesConfig,
    Qwen2AudioForConditionalGeneration,
)


LABELS = ("A", "B", "C")
LABEL_TO_ID = {label: index for index, label in enumerate(LABELS)}
LABEL_TO_ZH = {"A": "中性", "B": "厌恶", "C": "愤怒"}
LABEL_MAP = {
    "a": "A",
    "b": "B",
    "c": "C",
    "中性": "A",
    "neutral": "A",
    "厌恶": "B",
    "disgust": "B",
    "disgusted": "B",
    "愤怒": "C",
    "anger": "C",
    "angry": "C",
}
FILENAME = re.compile(
    r"^(?P<dialogue>.+)_(?P<side>left|right)_(?P<order>\d+)\.[^.]+$",
    re.IGNORECASE,
)
EXPECTED_PROTOCOLS = {
    "DACE": (
        "outer-4/inner-3 complete-dialogue class-specific joint logistic "
        "calibration of exactly two trunks"
    ),
    "ACCR": (
        "outer-4/inner-3 complete-dialogue adaptive confidence-calibrated "
        "routing from "
        "step700 primary to optional step1400 secondary"
    ),
}
SYSTEM_PROMPT = (
    "你是语音情绪识别专家。A=中性（平稳或证据不足）；"
    "B=厌恶（明显负面但仍可控）；"
    "C=愤怒（仅限强烈爆发、攻击、辱骂、威胁或失控）。"
    "结合语气强度、语速、音量、停顿、重复和ASR；"
    "不要仅因投诉、脏词、拒绝或质问就判C。只回答A、B或C。"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalize_label(value: Any) -> str:
    return LABEL_MAP.get(str(value or "").strip().casefold(), "")


def clean_text(value: Any, maximum: int = 320) -> str:
    text = " ".join(str(value or "").replace("\u3000", " ").split())
    return text[:maximum]


def side_order(row: dict[str, Any]) -> tuple[str, int]:
    segment = str(row.get("segment") or "")
    parts = segment.rsplit("_", 1)
    if len(parts) == 2 and parts[1].isdigit():
        return parts[0].lower(), int(parts[1])
    match = FILENAME.match(Path(str(row["wav_name"])).name)
    if match:
        return match.group("side").lower(), int(match.group("order"))
    # Keep the exact training/evaluation prompt contract for filenames without
    # an explicit side/order suffix.  The frozen models were trained with
    # `seg0` for this fallback, not with the row's global table index.
    return "unknown", 0


def add_context(rows: list[dict[str, Any]]) -> None:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        side, order = side_order(row)
        row["side"] = side
        row["side_order"] = order
        groups[(str(row["dialogue_key"]), side)].append(row)
    for members in groups.values():
        members.sort(
            key=lambda row: (int(row["side_order"]), int(row["source_index"]))
        )
        for position, row in enumerate(members):
            row["previous_asr"] = (
                clean_text(members[position - 1]["ASR"]) if position else ""
            )
            row["current_asr"] = clean_text(row["ASR"])
            row["next_asr"] = (
                clean_text(members[position + 1]["ASR"])
                if position + 1 < len(members)
                else ""
            )


def read_input(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows: list[dict[str, Any]] = list(csv.DictReader(handle))
    required = {"wav_name", "ASR", "dialogue_key", "segment", "source_audio_path"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError(f"prepared input lacks columns {sorted(required)}: {path}")
    seen: set[str] = set()
    for index, row in enumerate(rows):
        wav = Path(str(row["wav_name"])).name
        if wav in seen:
            raise ValueError(f"duplicate wav_name: {wav}")
        seen.add(wav)
        audio = Path(str(row["source_audio_path"])).expanduser()
        if not audio.is_file():
            raise FileNotFoundError(audio)
        label = normalize_label(row.get("label"))
        row["wav_name"] = wav
        row["source_audio_path"] = str(audio.resolve())
        row["label_abc"] = label
        row["label_id"] = LABEL_TO_ID.get(label, 0)
        row["source_index"] = index
    add_context(rows)
    return rows


def load_audio(path: str, seconds: float) -> np.ndarray:
    audio, sampling_rate = sf.read(path, dtype="float32", always_2d=False)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    if sampling_rate != 16000:
        divisor = math.gcd(int(sampling_rate), 16000)
        audio = resample_poly(
            audio,
            16000 // divisor,
            int(sampling_rate) // divisor,
        ).astype(np.float32)
    maximum = int(round(float(seconds) * 16000))
    if len(audio) > maximum:
        audio = audio[:maximum]
    return np.asarray(audio, dtype=np.float32)


def make_prompt(processor: Any, row: dict[str, Any]) -> str:
    context = (
        "通话上下文如下：\n"
        f"side={row['side']} seg{row['side_order']} "
        f"has_prev={int(bool(row['previous_asr']))} "
        f"has_next={int(bool(row['next_asr']))} "
        f"[上一段]{row['previous_asr']} "
        f"[当前]{row['current_asr'] or '(空)'} "
        f"[下一段]{row['next_asr']}\n"
        "请结合语音与转写判断情绪。第一行只输出 A、B 或 C。"
    )
    conversation = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {"type": "audio", "audio": row["source_audio_path"]},
                {"type": "text", "text": context},
            ],
        },
    ]
    return processor.apply_chat_template(
        conversation,
        add_generation_prompt=True,
        tokenize=False,
    )


class RowDataset(Dataset):
    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.rows[index]


class Collator:
    def __init__(self, processor: Any, seconds: float) -> None:
        self.processor = processor
        self.seconds = float(seconds)

    def __call__(self, rows: list[dict[str, Any]]) -> dict[str, Any]:
        batch = self.processor(
            text=[make_prompt(self.processor, row) for row in rows],
            audio=[load_audio(row["source_audio_path"], self.seconds) for row in rows],
            sampling_rate=16000,
            return_tensors="pt",
            padding=True,
        )
        batch["wav_names"] = [str(row["wav_name"]) for row in rows]
        return batch


class DirectAudioTextClassifier(nn.Module):
    """Four-pool residual classifier used by both published adapters."""

    def __init__(
        self,
        backbone: nn.Module,
        token_ids: torch.Tensor,
        audio_token_id: int,
        *,
        block_hidden: int = 64,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.audio_token_id = int(audio_token_id)
        hidden_size = int(backbone.get_base_model().config.text_config.hidden_size)
        self.hidden_size = hidden_size
        self.block_hidden = int(block_hidden)
        self.num_blocks = 4
        lm_rows = backbone.get_base_model().lm_head.weight.detach()[token_ids].float()
        self.register_buffer("base_weight", lm_rows.clone())
        self.log_scale = nn.Parameter(torch.zeros((), dtype=torch.float32))
        self.bias = nn.Parameter(torch.zeros(len(LABELS), dtype=torch.float32))
        self.norms = nn.ModuleList(
            [
                nn.LayerNorm(hidden_size, dtype=torch.float32)
                for _ in range(self.num_blocks)
            ]
        )
        self.projections = nn.ModuleList(
            [
                nn.Linear(hidden_size, self.block_hidden, dtype=torch.float32)
                for _ in range(self.num_blocks)
            ]
        )
        self.dropout = nn.Dropout(float(dropout))
        self.output = nn.Linear(
            self.num_blocks * self.block_hidden,
            len(LABELS),
            dtype=torch.float32,
        )
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    @staticmethod
    def masked_mean(hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        denominator = mask.sum(dim=1, keepdim=True).clamp_min(1).to(hidden.dtype)
        return (hidden * mask.unsqueeze(-1)).sum(dim=1) / denominator

    def forward(self, batch: dict[str, Any]) -> torch.Tensor:
        device = self.bias.device
        inputs = {
            key: value.to(device)
            for key, value in batch.items()
            if key
            in {
                "input_ids",
                "attention_mask",
                "input_features",
                "feature_attention_mask",
            }
        }
        output = self.backbone.get_base_model().model(
            **inputs,
            use_cache=False,
        )
        hidden = output.last_hidden_state.float()
        attention = inputs["attention_mask"].bool()
        positions = attention.long().sum(dim=1) - 1
        batch_indices = torch.arange(hidden.shape[0], device=device)
        answer_state = hidden[batch_indices, positions]
        audio_mask = (inputs["input_ids"] == self.audio_token_id) & attention
        if torch.any(audio_mask.sum(dim=1) == 0):
            raise RuntimeError("processor produced a row without audio tokens")
        text_mask = attention & ~audio_mask
        if torch.any(text_mask.sum(dim=1) == 0):
            raise RuntimeError("processor produced a row without text tokens")
        audio_mean = self.masked_mean(hidden, audio_mask)
        audio_centered = hidden - audio_mean.unsqueeze(1)
        audio_variance = self.masked_mean(audio_centered.square(), audio_mask)
        audio_std = torch.sqrt(audio_variance.clamp_min(1e-8))
        text_mean = self.masked_mean(hidden, text_mask)
        projected = torch.cat(
            [
                self.dropout(F.gelu(projection(norm(block))))
                for block, norm, projection in zip(
                    (answer_state, audio_mean, audio_std, text_mean),
                    self.norms,
                    self.projections,
                )
            ],
            dim=-1,
        )
        base_logits = F.linear(answer_state, self.base_weight)
        return (
            self.log_scale.exp() * base_logits
            + self.bias
            + self.output(projected)
        )


def load_head(model: DirectAudioTextClassifier, checkpoint: Path) -> None:
    state = torch.load(
        checkpoint / "classifier_head.pt",
        map_location="cpu",
        weights_only=True,
    )
    if state.get("head_type") != "four_pool_residual":
        raise ValueError(f"unsupported classifier head in {checkpoint}")
    expected = {
        "audio_token_id": model.audio_token_id,
        "hidden_size": model.hidden_size,
        "block_hidden": model.block_hidden,
        "num_blocks": model.num_blocks,
    }
    actual = {key: int(state[key]) for key in expected}
    if actual != expected:
        raise ValueError(f"classifier-head mismatch: {actual} != {expected}")
    incompatible = model.load_state_dict(state["state_dict"], strict=False)
    missing = [
        key
        for key in incompatible.missing_keys
        if not key.startswith("backbone.")
    ]
    if missing or incompatible.unexpected_keys:
        raise ValueError(
            "classifier-head state mismatch: "
            f"missing={missing}, unexpected={incompatible.unexpected_keys}"
        )


@torch.inference_mode()
def infer_probabilities(
    model: DirectAudioTextClassifier,
    processor: Any,
    rows: list[dict[str, Any]],
    *,
    batch_size: int,
    max_audio_seconds: float,
) -> np.ndarray:
    if not rows:
        return np.empty((0, 3), dtype=np.float64)
    loader = DataLoader(
        RowDataset(rows),
        batch_size=int(batch_size),
        shuffle=False,
        num_workers=0,
        collate_fn=Collator(processor, max_audio_seconds),
    )
    values: list[np.ndarray] = []
    for batch in loader:
        values.append(torch.softmax(model(batch), dim=-1).cpu().numpy())
    probabilities = np.concatenate(values, axis=0).astype(np.float64)
    return probabilities / probabilities.sum(axis=1, keepdims=True)


def centered_log_probabilities(probabilities: np.ndarray) -> np.ndarray:
    values = np.log(np.clip(probabilities, 1e-12, 1.0))
    return values - values.mean(axis=1, keepdims=True)


def frozen_logistic(
    features: np.ndarray,
    record: dict[str, Any],
    expected_width: int,
) -> np.ndarray:
    classes = np.asarray(record["classes"], dtype=np.int64)
    coefficients = np.asarray(record["coef"], dtype=np.float64)
    intercept = np.asarray(record["intercept"], dtype=np.float64)
    if (
        not np.array_equal(classes, np.arange(3))
        or coefficients.shape != (3, expected_width)
        or intercept.shape != (3,)
        or features.shape[1] != expected_width
    ):
        raise ValueError("invalid frozen logistic payload")
    logits = features @ coefficients.T + intercept
    logits -= logits.max(axis=1, keepdims=True)
    values = np.exp(logits)
    return values / values.sum(axis=1, keepdims=True)


def dialogue_smooth(
    probabilities: np.ndarray,
    rows: list[dict[str, Any]],
    record: dict[str, Any],
) -> np.ndarray:
    parameters = record["parameters"]
    bias = np.asarray(
        [
            float(parameters["bias_a"]),
            0.0,
            float(parameters["bias_c"]),
        ],
        dtype=np.float64,
    )
    logits = np.log(np.clip(probabilities, 1e-12, 1.0)) + bias[None, :]
    logits /= float(parameters["temperature"])
    logits -= logits.max(axis=1, keepdims=True)
    calibrated = np.exp(logits)
    calibrated /= calibrated.sum(axis=1, keepdims=True)
    alpha = float(parameters["alpha"])
    if alpha == 0.0:
        return calibrated
    mode = str(parameters["mode"])
    keys = (
        [str(row["dialogue_key"]) for row in rows]
        if mode == "dialogue"
        else [
            f"{row['dialogue_key']}::{side_order(row)[0]}"
            for row in rows
        ]
    )
    mapping: dict[str, int] = {}
    codes: list[int] = []
    for key in keys:
        if key not in mapping:
            mapping[key] = len(mapping)
        codes.append(mapping[key])
    code_array = np.asarray(codes, dtype=np.int64)
    sums = np.zeros((len(mapping), 3), dtype=np.float64)
    sizes = np.zeros(len(mapping), dtype=np.int64)
    np.add.at(sums, code_array, calibrated)
    np.add.at(sizes, code_array, 1)
    means = sums / np.maximum(sizes[:, None], 1)
    smoothed = (
        (1.0 - alpha) * calibrated
        + alpha * means[code_array]
    )
    eligible = sizes[code_array] >= int(parameters["group_min"])
    maximum_margin = parameters["max_margin"]
    if maximum_margin is not None:
        ordered = np.sort(calibrated, axis=1)
        eligible &= (
            ordered[:, -1] - ordered[:, -2]
        ) <= float(maximum_margin)
    return np.where(eligible[:, None], smoothed, calibrated)


def classification_metrics(
    truth: np.ndarray,
    prediction: np.ndarray,
) -> dict[str, Any]:
    confusion = np.zeros((3, 3), dtype=np.int64)
    np.add.at(confusion, (truth, prediction), 1)
    recalls = np.diag(confusion) / np.maximum(1, confusion.sum(axis=1))
    precisions = np.diag(confusion) / np.maximum(1, confusion.sum(axis=0))
    f1 = 2 * precisions * recalls / np.maximum(
        1e-12,
        precisions + recalls,
    )
    return {
        "rows": int(len(truth)),
        "correct": int(np.trace(confusion)),
        "accuracy": float(np.trace(confusion) / len(truth)),
        "macro_f1": float(f1.mean()),
        "confusion_matrix_A_B_C": confusion.tolist(),
    }


def load_selection(
    method: str,
    path: Path,
    expected_sha256: str,
) -> tuple[dict[str, Any], str]:
    actual_sha256 = sha256(path)
    if actual_sha256 != expected_sha256:
        raise ValueError(
            f"{method} frozen parameter SHA256 mismatch: {actual_sha256}"
        )
    selection = json.loads(path.read_text(encoding="utf-8"))
    if (
        selection.get("test_evaluated") is not False
        or selection.get("protocol") != EXPECTED_PROTOCOLS[method]
    ):
        raise ValueError(f"unexpected {method} validation-selection protocol")
    return selection["deployment_candidate_fit_on_all_validation"], actual_sha256


def load_model(
    base_model: Path,
    primary_adapter: Path,
) -> tuple[DirectAudioTextClassifier, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("DACE/ACCR inference requires an NVIDIA CUDA GPU")
    processor = AutoProcessor.from_pretrained(
        base_model,
        local_files_only=True,
    )
    processor.tokenizer.padding_side = "right"
    encoded = [
        processor.tokenizer.encode(label, add_special_tokens=False)
        for label in LABELS
    ]
    if any(len(value) != 1 for value in encoded):
        raise RuntimeError(f"A/B/C are not single tokens: {encoded}")
    token_ids = torch.tensor(
        [value[0] for value in encoded],
        dtype=torch.long,
    )
    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )
    backbone = Qwen2AudioForConditionalGeneration.from_pretrained(
        base_model,
        local_files_only=True,
        device_map={"": 0},
        torch_dtype=torch.float16,
        quantization_config=quantization,
        low_cpu_mem_usage=True,
    )
    backbone.config.use_cache = False
    backbone = PeftModel.from_pretrained(
        backbone,
        primary_adapter,
        adapter_name="primary",
        is_trainable=False,
    )
    config = backbone.get_base_model().config
    audio_token_id = getattr(
        config,
        "audio_token_id",
        getattr(config, "audio_token_index", None),
    )
    if audio_token_id is None:
        raise ValueError("Qwen2-Audio config lacks audio token id")
    model = DirectAudioTextClassifier(
        backbone,
        token_ids,
        int(audio_token_id),
    ).to(torch.device("cuda", 0))
    load_head(model, primary_adapter)
    model.eval()
    return model, processor


def switch_to_secondary(
    model: DirectAudioTextClassifier,
    secondary_adapter: Path,
) -> None:
    model.backbone.load_adapter(
        secondary_adapter,
        adapter_name="secondary",
        is_trainable=False,
    )
    model.backbone.set_adapter("secondary")
    load_head(model, secondary_adapter)
    model.eval()


def write_predictions(
    path: Path,
    rows: list[dict[str, Any]],
    probabilities: np.ndarray,
    prediction: np.ndarray,
    called_second: np.ndarray,
) -> None:
    fields = (
        "wav_name",
        "label",
        "label_zh",
        "pred",
        "pred_zh",
        "correct",
        "called_second",
        "model_forwards",
        "confidence",
        "prob_A",
        "prob_B",
        "prob_C",
    )
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for index, row in enumerate(rows):
            label = str(row["label_abc"])
            predicted = LABELS[int(prediction[index])]
            writer.writerow(
                {
                    "wav_name": row["wav_name"],
                    "label": label,
                    "label_zh": LABEL_TO_ZH.get(label, ""),
                    "pred": predicted,
                    "pred_zh": LABEL_TO_ZH[predicted],
                    "correct": (
                        int(LABEL_TO_ID[label] == int(prediction[index]))
                        if label in LABEL_TO_ID
                        else ""
                    ),
                    "called_second": int(called_second[index]),
                    "model_forwards": 1 + int(called_second[index]),
                    "confidence": float(probabilities[index].max()),
                    **{
                        f"prob_{label_name}": float(
                            probabilities[index, class_index]
                        )
                        for class_index, label_name in enumerate(LABELS)
                    },
                }
            )


def parse_args(method: str) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=f"Run {method} from a prepared audio+ASR manifest"
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument("--primary-adapter", type=Path, required=True)
    parser.add_argument("--secondary-adapter", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--expected-selection-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-timestamp", required=True)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-audio-seconds", type=float, default=16.0)
    return parser.parse_args()


def run(method: str) -> None:
    method = method.upper()
    if method not in EXPECTED_PROTOCOLS:
        raise ValueError(f"unsupported method: {method}")
    cpu_threads = os.environ.get("SER_CPU_THREADS")
    if cpu_threads:
        thread_count = int(cpu_threads)
        if thread_count <= 0:
            raise ValueError("SER_CPU_THREADS must be positive")
        torch.set_num_threads(thread_count)
        torch.set_num_interop_threads(max(1, min(4, thread_count)))
    args = parse_args(method)
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    if args.batch_size <= 0:
        raise ValueError("batch-size must be positive")
    for path in (
        args.base_model / "config.json",
        args.primary_adapter / "adapter_model.safetensors",
        args.primary_adapter / "classifier_head.pt",
        args.secondary_adapter / "adapter_model.safetensors",
        args.secondary_adapter / "classifier_head.pt",
        args.selection,
        args.manifest,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)

    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    input_csv = Path(str(manifest["input_csv"]))
    rows = read_input(input_csv)
    payload, selection_sha = load_selection(
        method,
        args.selection,
        args.expected_selection_sha256,
    )
    input_finished = time.perf_counter()

    model, processor = load_model(args.base_model, args.primary_adapter)
    primary_loaded = time.perf_counter()
    primary_probability = infer_probabilities(
        model,
        processor,
        rows,
        batch_size=args.batch_size,
        max_audio_seconds=args.max_audio_seconds,
    )
    primary_finished = time.perf_counter()

    if method == "DACE":
        called_second = np.ones(len(rows), dtype=bool)
        secondary_rows = rows
        primary_calibrated = None
        router_threshold = None
    else:
        primary_calibrated = frozen_logistic(
            centered_log_probabilities(primary_probability),
            payload["primary_calibrator"],
            3,
        )
        router_threshold = float(payload["fitted_validation_router_threshold"])
        if str(payload["router_signal"]) != "confidence":
            raise ValueError("published ACCR router must use confidence")
        called_second = primary_calibrated.max(axis=1) <= router_threshold
        secondary_rows = [
            rows[index] for index in np.flatnonzero(called_second)
        ]
    routed = time.perf_counter()

    secondary_probability = np.empty((0, 3), dtype=np.float64)
    if secondary_rows:
        switch_to_secondary(model, args.secondary_adapter)
        secondary_loaded = time.perf_counter()
        secondary_probability = infer_probabilities(
            model,
            processor,
            secondary_rows,
            batch_size=args.batch_size,
            max_audio_seconds=args.max_audio_seconds,
        )
    else:
        secondary_loaded = time.perf_counter()
    secondary_finished = time.perf_counter()

    if method == "DACE":
        features = np.concatenate(
            (
                centered_log_probabilities(primary_probability),
                centered_log_probabilities(secondary_probability),
            ),
            axis=1,
        )
        combined = frozen_logistic(features, payload, 6)
    else:
        assert primary_calibrated is not None
        combined = primary_calibrated.copy()
        called_indices = np.flatnonzero(called_second)
        if len(called_indices):
            joint_features = np.concatenate(
                (
                    centered_log_probabilities(
                        primary_probability[called_indices]
                    ),
                    centered_log_probabilities(secondary_probability),
                ),
                axis=1,
            )
            combined[called_indices] = frozen_logistic(
                joint_features,
                payload["joint_calibrator"],
                6,
            )
    combined /= combined.sum(axis=1, keepdims=True)
    transformed = dialogue_smooth(combined, rows, payload["smoother"])
    transformed /= transformed.sum(axis=1, keepdims=True)
    prediction = transformed.argmax(axis=1)
    known_indices = np.asarray(
        [
            index
            for index, row in enumerate(rows)
            if str(row["label_abc"]) in LABEL_TO_ID
        ],
        dtype=np.int64,
    )
    metrics = None
    if len(known_indices):
        truth = np.asarray(
            [
                LABEL_TO_ID[str(rows[index]["label_abc"])]
                for index in known_indices
            ],
            dtype=np.int64,
        )
        metrics = classification_metrics(truth, prediction[known_indices])
    postprocessed = time.perf_counter()

    args.output_dir.mkdir(parents=True)
    stem = f"{method.lower()}_{args.run_timestamp}"
    predictions_path = args.output_dir / f"{stem}_predictions.csv"
    summary_path = args.output_dir / f"{stem}_summary.json"
    write_predictions(
        predictions_path,
        rows,
        transformed,
        prediction,
        called_second,
    )
    written = time.perf_counter()

    secondary_calls = int(called_second.sum())
    summary = {
        "method": method,
        "method_full_name": (
            "Dual-Adapter Calibrated Ensemble"
            if method == "DACE"
            else "Adaptive Confidence-Calibrated Routing"
        ),
        "inputs": ["raw_audio", "ASR"],
        "labels": {
            "A": "neutral/中性",
            "B": "disgust or disgusted/厌恶",
            "C": "anger or angry/愤怒",
        },
        "run_timestamp": args.run_timestamp,
        "rows": len(rows),
        "labeled_rows": int(len(known_indices)),
        "metrics": metrics,
        "base_model_load_count": 1,
        "minimum_model_forwards": 2 if method == "DACE" else 1,
        "maximum_model_forwards": 2,
        "called_second": secondary_calls,
        "call_rate": secondary_calls / len(rows),
        "average_model_forwards": 1.0 + secondary_calls / len(rows),
        "router": (
            None
            if method == "DACE"
            else {
                "signal": "calibrated_primary_confidence",
                "comparison": "<=",
                "threshold": router_threshold,
            }
        ),
        "manifest": str(args.manifest),
        "manifest_sha256": sha256(args.manifest),
        "input_csv": str(input_csv),
        "input_csv_sha256": sha256(input_csv),
        "base_model": str(args.base_model),
        "base_config_sha256": sha256(args.base_model / "config.json"),
        "primary_adapter": str(args.primary_adapter),
        "primary_adapter_sha256": sha256(
            args.primary_adapter / "adapter_model.safetensors"
        ),
        "primary_head_sha256": sha256(
            args.primary_adapter / "classifier_head.pt"
        ),
        "secondary_adapter": str(args.secondary_adapter),
        "secondary_adapter_sha256": sha256(
            args.secondary_adapter / "adapter_model.safetensors"
        ),
        "secondary_head_sha256": sha256(
            args.secondary_adapter / "classifier_head.pt"
        ),
        "selection": str(args.selection),
        "selection_sha256": selection_sha,
        "selection_protocol": EXPECTED_PROTOCOLS[method],
        "batch_size": int(args.batch_size),
        "max_audio_seconds": float(args.max_audio_seconds),
        "cpu_threads": int(torch.get_num_threads()),
        "cpu_interop_threads": int(torch.get_num_interop_threads()),
        "peak_cuda_memory_bytes": int(torch.cuda.max_memory_allocated()),
        "timing_seconds": {
            "input_and_frozen_parameter_validation": input_finished - started,
            "processor_base_primary_load": primary_loaded - input_finished,
            "primary_forward": primary_finished - primary_loaded,
            "routing": routed - primary_finished,
            "secondary_adapter_and_head_load": secondary_loaded - routed,
            "secondary_forward": secondary_finished - secondary_loaded,
            "frozen_calibration_and_dialogue_smoothing": (
                postprocessed - secondary_finished
            ),
            "artifact_write": written - postprocessed,
            "total": written - started,
        },
        "outputs": {
            "predictions": str(predictions_path),
            "predictions_sha256": sha256(predictions_path),
            "summary": str(summary_path),
        },
    }
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


__all__ = ["run"]
