#!/usr/bin/env python3
"""Train one deployable Qwen2-Audio student from raw audio, ASR and train-OOF MVEF.

The model predicts A/B/C from the next-token logits.  The script deliberately
accepts only historical train and validation tables; paths containing ``test``
are rejected.  MVEF is a training-only soft teacher and is not used at
deployment.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import re
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import torch
import torch.distributed as dist
import torch.nn.functional as F
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from scipy.signal import resample_poly
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel
from transformers import (
    AutoProcessor,
    BitsAndBytesConfig,
    Qwen2AudioForConditionalGeneration,
)


LABELS = ("A", "B", "C")
LABEL_TO_ID = {label: index for index, label in enumerate(LABELS)}
CN_TO_LABEL = {
    "A": "A", "B": "B", "C": "C",
    "中性": "A", "neutral": "A",
    "厌恶": "B", "disgust": "B", "disgusted": "B",
    "愤怒": "C", "angry": "C", "anger": "C",
}


def reject_test_path(path: Path, role: str) -> None:
    if "test" in str(path).lower():
        raise ValueError(f"{role} must not reference a test-like path: {path}")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def clean_text(value: Any, maximum: int = 320) -> str:
    text = " ".join(str(value or "").replace("\u3000", " ").split())
    return text[:maximum]


def side_order(row: dict[str, str]) -> tuple[str, int]:
    segment = str(row.get("segment") or "")
    parts = segment.rsplit("_", 1)
    if len(parts) == 2 and parts[1].isdigit():
        return parts[0], int(parts[1])
    wav = Path(row["wav_name"]).name
    match = re.search(r"_(left|right)_(\d+)\.wav$", wav)
    return (match.group(1), int(match.group(2))) if match else ("unknown", 0)


def read_table(path: Path, *, require_historical: bool) -> list[dict[str, Any]]:
    reject_test_path(path, "data table")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    required = {"wav_name", "label", "ASR", "dialogue_key", "segment", "source_audio_path"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError(f"Table lacks {sorted(required)}: {path}")
    seen: set[str] = set()
    for index, row in enumerate(rows):
        wav = Path(row["wav_name"]).name
        if wav in seen:
            raise ValueError(f"Duplicate wav_name {wav}")
        seen.add(wav)
        label = CN_TO_LABEL.get(str(row["label"]).strip().lower())
        if label not in LABEL_TO_ID:
            raise ValueError(f"Unsupported label {row['label']!r} for {wav}")
        month = int(str(row.get("month") or "0"))
        if require_historical and month > 202505:
            raise ValueError(f"Historical train row leaks month {month}: {wav}")
        audio = Path(row["source_audio_path"])
        if not audio.is_file():
            raise FileNotFoundError(audio)
        row["wav_name"] = wav
        row["label_abc"] = label
        row["label_id"] = LABEL_TO_ID[label]
        row["source_index"] = index
        row["month_int"] = month
    add_context(rows)
    return rows


def add_context(rows: list[dict[str, Any]]) -> None:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        side, order = side_order(row)
        row["side"] = side
        row["side_order"] = order
        groups[(str(row["dialogue_key"]), side)].append(row)
    for members in groups.values():
        members.sort(key=lambda row: (int(row["side_order"]), int(row["source_index"])))
        for pos, row in enumerate(members):
            row["previous_asr"] = clean_text(members[pos - 1]["ASR"]) if pos else ""
            row["current_asr"] = clean_text(row["ASR"])
            row["next_asr"] = clean_text(members[pos + 1]["ASR"]) if pos + 1 < len(members) else ""


def read_teacher(path: Path, rows: list[dict[str, Any]]) -> dict[str, int]:
    reject_test_path(path, "MVEF train-OOF teacher")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        teacher_rows = list(csv.DictReader(handle))
    by_wav: dict[str, dict[str, str]] = {}
    for teacher in teacher_rows:
        wav = Path(teacher["wav_name"]).name
        if wav in by_wav:
            raise ValueError(f"Duplicate teacher wav {wav}")
        by_wav[wav] = teacher
    extra = set(by_wav) - {row["wav_name"] for row in rows}
    if extra:
        raise ValueError(f"Teacher has {len(extra)} rows outside train table; first={sorted(extra)[0]}")
    covered = 0
    missing = 0
    for row in rows:
        teacher = by_wav.get(row["wav_name"])
        if teacher is None:
            one_hot = np.zeros(3, dtype=np.float32)
            one_hot[int(row["label_id"])] = 1.0
            row["teacher_probs"] = one_hot
            row["teacher_weight"] = 0.0
            missing += 1
            continue
        probs = np.asarray([float(teacher[f"prob_{label}"]) for label in LABELS], dtype=np.float64)
        if not np.all(np.isfinite(probs)) or probs.min() < 0 or probs.sum() <= 0:
            raise ValueError(f"Invalid teacher probabilities for {row['wav_name']}")
        row["teacher_probs"] = (probs / probs.sum()).astype(np.float32)
        row["teacher_weight"] = 1.0
        covered += 1
    return {"covered": covered, "missing_truth_only": missing}


def make_prompt(processor: Any, row: dict[str, Any], include_context: bool) -> str:
    context = ""
    if include_context:
        context = (
            f"\n同一通话同一说话方的上一段ASR：{row['previous_asr'] or '(无)'}"
            f"\n同一通话同一说话方的下一段ASR：{row['next_asr'] or '(无)'}"
        )
    instruction = (
        "请判断当前语音中说话人的情感。"
        "A=中性，B=厌恶，C=愤怒。"
        f"\n当前语音ASR：{row['current_asr'] or '(空)'}{context}"
        "\n结合原始音频和ASR，只回答一个字母A、B或C。"
    )
    conversation = [{
        "role": "user",
        "content": [
            {"type": "audio", "audio": row["source_audio_path"]},
            {"type": "text", "text": instruction},
        ],
    }]
    return processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)


def load_audio(path: str, seconds: float) -> np.ndarray:
    audio, sampling_rate = sf.read(path, dtype="float32", always_2d=False)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    if sampling_rate != 16000:
        divisor = math.gcd(int(sampling_rate), 16000)
        audio = resample_poly(audio, 16000 // divisor, int(sampling_rate) // divisor).astype(np.float32)
    limit = int(round(seconds * 16000))
    if len(audio) > limit:
        audio = audio[:limit]
    return np.asarray(audio, dtype=np.float32)


class RowDataset(Dataset):
    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.rows[index]


class Collator:
    def __init__(self, processor: Any, seconds: float, include_context: bool) -> None:
        self.processor = processor
        self.seconds = seconds
        self.include_context = include_context

    def __call__(self, rows: list[dict[str, Any]]) -> dict[str, Any]:
        texts = [make_prompt(self.processor, row, self.include_context) for row in rows]
        audios = [load_audio(row["source_audio_path"], self.seconds) for row in rows]
        batch = self.processor(
            text=texts,
            audio=audios,
            sampling_rate=16000,
            return_tensors="pt",
            padding=True,
        )
        batch["class_labels"] = torch.tensor([row["label_id"] for row in rows], dtype=torch.long)
        batch["teacher_probs"] = torch.tensor(
            np.stack([row.get("teacher_probs", np.full(3, 1 / 3, dtype=np.float32)) for row in rows]),
            dtype=torch.float32,
        )
        batch["teacher_weights"] = torch.tensor(
            [float(row.get("teacher_weight", 0.0)) for row in rows], dtype=torch.float32
        )
        batch["wav_names"] = [row["wav_name"] for row in rows]
        return batch


def class_logits(model: Any, batch: dict[str, Any], token_ids: torch.Tensor) -> torch.Tensor:
    model_inputs = {
        key: value.to(token_ids.device)
        for key, value in batch.items()
        if key in {"input_ids", "attention_mask", "input_features", "feature_attention_mask"}
    }
    output = model(**model_inputs, use_cache=False)
    attention = model_inputs["attention_mask"]
    positions = attention.sum(dim=1) - 1
    rows = torch.arange(output.logits.shape[0], device=output.logits.device)
    return output.logits[rows, positions][:, token_ids]


@torch.no_grad()
def evaluate(
    model: Any,
    loader: DataLoader,
    token_ids: torch.Tensor,
    output_csv: Path,
    labels_by_wav: dict[str, int],
) -> dict[str, Any]:
    model.eval()
    records: list[dict[str, Any]] = []
    confusion = np.zeros((3, 3), dtype=np.int64)
    for batch in loader:
        logits = class_logits(model, batch, token_ids)
        probs = torch.softmax(logits.float(), dim=-1).cpu().numpy()
        for wav, probability in zip(batch["wav_names"], probs):
            truth = labels_by_wav[wav]
            pred = int(probability.argmax())
            confusion[truth, pred] += 1
            records.append({
                "wav_name": wav,
                "label": LABELS[truth],
                "pred": LABELS[pred],
                "correct": int(pred == truth),
                **{f"prob_{label}": float(probability[index]) for index, label in enumerate(LABELS)},
            })
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    correct = int(np.trace(confusion))
    recalls = np.diag(confusion) / np.maximum(1, confusion.sum(axis=1))
    precisions = np.diag(confusion) / np.maximum(1, confusion.sum(axis=0))
    f1 = 2 * precisions * recalls / np.maximum(1e-12, precisions + recalls)
    return {
        "rows": len(records),
        "correct": correct,
        "accuracy": correct / len(records),
        "macro_f1": float(f1.mean()),
        "confusion_matrix": confusion.tolist(),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--train-csv", type=Path, default=Path("train_timesplit.csv"))
    parser.add_argument("--val-csv", type=Path, default=Path("val_202506.csv"))
    parser.add_argument("--teacher-train", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=800)
    parser.add_argument("--save-steps", default="200,400,600,800")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--kd-alpha", type=float, default=0.20)
    parser.add_argument("--temperature", type=float, default=2.0)
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--max-audio-seconds", type=float, default=16.0)
    parser.add_argument("--include-context", action="store_true")
    parser.add_argument("--seed", type=int, default=20260722)
    parser.add_argument("--max-train-rows", type=int, default=0)
    parser.add_argument("--max-val-rows", type=int, default=0)
    parser.add_argument("--eval-only-adapter", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    distributed = world_size > 1
    if distributed:
        if args.eval_only_adapter is not None:
            raise ValueError("Evaluation is deliberately single-GPU; do not use torchrun")
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl")
    device_index = local_rank if distributed else 0
    is_main = rank == 0
    for path, role in [(args.train_csv, "train"), (args.val_csv, "validation"), (args.teacher_train, "teacher")]:
        reject_test_path(path, role)
    if is_main:
        if args.output_dir.exists():
            raise FileExistsError(args.output_dir)
        args.output_dir.mkdir(parents=True)
    if distributed:
        dist.barrier()
    random.seed(args.seed + rank)
    np.random.seed(args.seed + rank)
    torch.manual_seed(args.seed + rank)

    train_rows = read_table(args.train_csv, require_historical=True)
    val_rows = read_table(args.val_csv, require_historical=False)
    teacher_coverage = read_teacher(args.teacher_train, train_rows)
    if args.max_train_rows:
        train_rows = train_rows[: args.max_train_rows]
    if args.max_val_rows:
        val_rows = val_rows[: args.max_val_rows]

    processor = AutoProcessor.from_pretrained(args.model, local_files_only=True)
    processor.tokenizer.padding_side = "right"
    encoded = [processor.tokenizer.encode(label, add_special_tokens=False) for label in LABELS]
    if any(len(value) != 1 for value in encoded):
        raise RuntimeError(f"A/B/C are not single tokens: {encoded}")

    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )
    model = Qwen2AudioForConditionalGeneration.from_pretrained(
        args.model,
        local_files_only=True,
        device_map={"": device_index},
        torch_dtype=torch.float16,
        quantization_config=quantization,
        low_cpu_mem_usage=True,
    )
    model.config.use_cache = False
    if args.eval_only_adapter is not None:
        model = PeftModel.from_pretrained(model, args.eval_only_adapter, is_trainable=False)
    else:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
        lora = LoraConfig(
            r=args.lora_rank,
            lora_alpha=args.lora_alpha,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        )
        model = get_peft_model(model, lora)
    if is_main:
        model.print_trainable_parameters()
    token_ids = torch.tensor([values[0] for values in encoded], device=torch.device("cuda", device_index))
    collator = Collator(processor, args.max_audio_seconds, args.include_context)
    val_loader = DataLoader(RowDataset(val_rows), batch_size=args.batch_size, shuffle=False, collate_fn=collator)
    labels_by_wav = {row["wav_name"]: int(row["label_id"]) for row in val_rows}

    manifest = {
        "protocol": "single Qwen2-Audio raw-audio+ASR student; train truth plus train-OOF MVEF soft KD",
        "test_evaluated": False,
        "arguments": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "inputs": {
            "train_csv": {"path": str(args.train_csv), "sha256": sha256(args.train_csv)},
            "val_csv": {"path": str(args.val_csv), "sha256": sha256(args.val_csv)},
            "teacher_train": {"path": str(args.teacher_train), "sha256": sha256(args.teacher_train)},
        },
        "label_token_ids": dict(zip(LABELS, [values[0] for values in encoded])),
        "train_rows": len(train_rows),
        "val_rows": len(val_rows),
        "teacher_coverage": teacher_coverage,
    }
    manifest["distributed"] = {"world_size": world_size, "global_micro_batch": world_size * args.batch_size}
    if is_main:
        (args.output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    if distributed:
        dist.barrier()

    if args.eval_only_adapter is not None:
        metrics = evaluate(model, val_loader, token_ids, args.output_dir / "val_predictions.csv", labels_by_wav)
        (args.output_dir / "summary_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        print(json.dumps(metrics, indent=2), flush=True)
        return

    train_generator = torch.Generator().manual_seed(args.seed + rank)
    train_dataset = RowDataset(train_rows)
    train_sampler = (
        DistributedSampler(
            train_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            seed=args.seed,
            drop_last=False,
        )
        if distributed else None
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        generator=train_generator,
        collate_fn=collator,
        num_workers=0,
    )
    if distributed:
        model = DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank)
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=args.learning_rate,
        betas=(0.9, 0.95),
        weight_decay=0.01,
    )
    save_steps = {int(value) for value in args.save_steps.split(",") if value.strip()}
    model.train()
    optimizer.zero_grad(set_to_none=True)
    data_epoch = 0
    if train_sampler is not None:
        train_sampler.set_epoch(data_epoch)
    iterator = iter(train_loader)
    start = time.time()
    recent: list[float] = []
    for step in range(1, args.steps + 1):
        for _ in range(args.gradient_accumulation):
            try:
                batch = next(iterator)
            except StopIteration:
                data_epoch += 1
                if train_sampler is not None:
                    train_sampler.set_epoch(data_epoch)
                iterator = iter(train_loader)
                batch = next(iterator)
            logits = class_logits(model, batch, token_ids)
            labels = batch["class_labels"].to(token_ids.device)
            teacher = batch["teacher_probs"].to(token_ids.device)
            teacher_weights = batch["teacher_weights"].to(token_ids.device)
            hard = F.cross_entropy(logits.float(), labels, reduction="none")
            temperature = args.temperature
            kd = F.kl_div(
                F.log_softmax(logits.float() / temperature, dim=-1),
                teacher,
                reduction="none",
            ).sum(dim=-1) * (temperature ** 2)
            row_alpha = args.kd_alpha * teacher_weights
            loss = ((1.0 - row_alpha) * hard + row_alpha * kd).mean()
            (loss / args.gradient_accumulation).backward()
            recent.append(float(loss.detach()))
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        if is_main and (step == 1 or step % 10 == 0):
            elapsed = time.time() - start
            print(json.dumps({
                "step": step,
                "loss": float(np.mean(recent[-20:])),
                "elapsed_seconds": elapsed,
                "seconds_per_step": elapsed / step,
            }), flush=True)
        if step in save_steps or step == args.steps:
            checkpoint = args.output_dir / f"checkpoint-{step}"
            if distributed:
                dist.barrier()
            if is_main:
                unwrapped = model.module if isinstance(model, DistributedDataParallel) else model
                unwrapped.save_pretrained(checkpoint)
                processor.save_pretrained(checkpoint / "processor")
            if distributed:
                dist.barrier()
    if is_main:
        (args.output_dir / "training_complete.json").write_text(
            json.dumps({"steps": args.steps, "elapsed_seconds": time.time() - start}, indent=2), encoding="utf-8"
        )
    if distributed:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
