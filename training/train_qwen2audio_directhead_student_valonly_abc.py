#!/usr/bin/env python3
"""Train one Qwen2-Audio emotion student with a tiny direct multimodal head.

The deployable model is one 4-bit Qwen2-Audio backbone, one LoRA adapter, and a
three-way classifier.  MVEF probabilities are used only as historical OOF soft
targets.  Test-like paths are deliberately rejected.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from transformers import AutoProcessor, BitsAndBytesConfig, Qwen2AudioForConditionalGeneration

import train_qwen2audio_mvef_student_valonly_abc as common


LABELS = common.LABELS
FULL_TARGETS = [
    "q_proj", "k_proj", "v_proj", "o_proj",  # language and audio attention
    "out_proj",                                # audio attention output
    "gate_proj", "up_proj", "down_proj",      # language FFN
    "fc1", "fc2",                             # audio FFN
    "linear",                                  # multimodal projector
]

RICH_SYSTEM_PROMPT = (
    "你是语音情绪识别专家。A=中性（平稳或证据不足）；"
    "B=厌恶（明显负面但仍可控）；"
    "C=愤怒（仅限强烈爆发、攻击、辱骂、威胁或失控）。"
    "结合语气强度、语速、音量、停顿、重复和ASR；"
    "不要仅因投诉、脏词、拒绝或质问就判C。只回答A、B或C。"
)


def make_directhead_prompt(processor: Any, row: dict[str, Any], include_context: bool) -> str:
    previous = row["previous_asr"] if include_context else ""
    following = row["next_asr"] if include_context else ""
    context = (
        "通话上下文如下：\n"
        f"side={row['side']} seg{row['side_order']} "
        f"has_prev={int(bool(previous))} has_next={int(bool(following))} "
        f"[上一段]{previous} [当前]{row['current_asr'] or '(空)'} [下一段]{following}\n"
        "请结合语音与转写判断情绪。第一行只输出 A、B 或 C。"
    )
    conversation = [
        {"role": "system", "content": RICH_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {"type": "audio", "audio": row["source_audio_path"]},
                {"type": "text", "text": context},
            ],
        },
    ]
    return processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)


class DirectCollator:
    def __init__(self, processor: Any, seconds: float, include_context: bool) -> None:
        self.processor = processor
        self.seconds = seconds
        self.include_context = include_context

    def __call__(self, rows: list[dict[str, Any]]) -> dict[str, Any]:
        texts = [make_directhead_prompt(self.processor, row, self.include_context) for row in rows]
        audios = [common.load_audio(row["source_audio_path"], self.seconds) for row in rows]
        batch = self.processor(
            text=texts, audio=audios, sampling_rate=16000, return_tensors="pt", padding=True
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


class DirectAudioTextClassifier(nn.Module):
    """Four-pool residual classifier over one Qwen2-Audio forward."""

    def __init__(
        self,
        backbone: nn.Module,
        token_ids: torch.Tensor,
        audio_token_id: int,
        *,
        initialize_from_lm_head: bool,
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
        self.norms = nn.ModuleList([
            nn.LayerNorm(hidden_size, dtype=torch.float32) for _ in range(self.num_blocks)
        ])
        self.projections = nn.ModuleList([
            nn.Linear(hidden_size, self.block_hidden, dtype=torch.float32) for _ in range(self.num_blocks)
        ])
        self.dropout = nn.Dropout(float(dropout))
        self.output = nn.Linear(self.num_blocks * self.block_hidden, len(LABELS), dtype=torch.float32)
        # The head starts as exactly the original A/B/C LM-row classifier.
        # Random block projections become trainable after the first non-zero
        # residual-output update, matching the proven historical multipool head.
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    @staticmethod
    def masked_mean(hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        denominator = mask.sum(dim=1, keepdim=True).clamp_min(1).to(hidden.dtype)
        return (hidden * mask.unsqueeze(-1)).sum(dim=1) / denominator

    def forward(self, batch: dict[str, Any]) -> torch.Tensor:
        device = self.bias.device
        model_inputs = {
            key: value.to(device)
            for key, value in batch.items()
            if key in {"input_ids", "attention_mask", "input_features", "feature_attention_mask"}
        }
        # Calling the combined Qwen2AudioModel avoids retaining every language
        # layer as output_hidden_states=True would, while all installed LoRA
        # modules remain active.
        combined_model = self.backbone.get_base_model().model
        output = combined_model(**model_inputs, use_cache=False)
        hidden = output.last_hidden_state.float()
        attention = model_inputs["attention_mask"].bool()
        positions = attention.long().sum(dim=1) - 1
        row_indices = torch.arange(hidden.shape[0], device=device)
        answer_state = hidden[row_indices, positions]
        audio_mask = (model_inputs["input_ids"] == self.audio_token_id) & attention
        if torch.any(audio_mask.sum(dim=1) == 0):
            raise RuntimeError("Processor produced a row without expanded audio tokens")
        text_mask = attention & ~audio_mask
        if torch.any(text_mask.sum(dim=1) == 0):
            raise RuntimeError("Processor produced a row without text tokens")
        audio_mean = self.masked_mean(hidden, audio_mask)
        audio_centered = hidden - audio_mean.unsqueeze(1)
        audio_variance = self.masked_mean(audio_centered.square(), audio_mask)
        audio_std = torch.sqrt(audio_variance.clamp_min(1e-8))
        text_mean = self.masked_mean(hidden, text_mask)
        blocks = [answer_state, audio_mean, audio_std, text_mean]
        projected = torch.cat([
            self.dropout(F.gelu(projection(norm(block))))
            for block, norm, projection in zip(blocks, self.norms, self.projections)
        ], dim=-1)
        base_logits = F.linear(answer_state, self.base_weight)
        return self.log_scale.exp() * base_logits + self.bias + self.output(projected)


def unwrap(model: nn.Module) -> DirectAudioTextClassifier:
    return model.module if isinstance(model, DistributedDataParallel) else model


def save_checkpoint(model: nn.Module, processor: Any, output: Path) -> None:
    direct = unwrap(model)
    direct.backbone.save_pretrained(output)
    torch.save(
        {
            "head_type": "four_pool_residual",
            "pooling": ["answer_state", "audio_mean", "audio_std", "text_mean"],
            "audio_token_id": direct.audio_token_id,
            "hidden_size": direct.hidden_size,
            "block_hidden": direct.block_hidden,
            "num_blocks": direct.num_blocks,
            "state_dict": {
                key: value.detach().cpu() for key, value in direct.state_dict().items()
                if not key.startswith("backbone.")
            },
        },
        output / "classifier_head.pt",
    )
    processor.save_pretrained(output / "processor")


def load_head(model: DirectAudioTextClassifier, checkpoint: Path) -> None:
    state = torch.load(checkpoint / "classifier_head.pt", map_location="cpu", weights_only=True)
    if state.get("head_type") != "four_pool_residual":
        raise ValueError(f"Unsupported classifier pooling in {checkpoint}")
    if int(state["audio_token_id"]) != model.audio_token_id:
        raise ValueError("Saved audio token id does not match the base model")
    expected = {
        "hidden_size": model.hidden_size,
        "block_hidden": model.block_hidden,
        "num_blocks": model.num_blocks,
    }
    actual = {key: int(state[key]) for key in expected}
    if actual != expected:
        raise ValueError(f"Classifier-head config mismatch: saved={actual}, expected={expected}")
    incompatible = model.load_state_dict(state["state_dict"], strict=False)
    nonbackbone_missing = [key for key in incompatible.missing_keys if not key.startswith("backbone.")]
    if nonbackbone_missing or incompatible.unexpected_keys:
        raise ValueError(
            "Classifier-head state mismatch: "
            f"missing_nonbackbone={nonbackbone_missing}, unexpected={incompatible.unexpected_keys}"
        )


@torch.no_grad()
def evaluate(
    model: DirectAudioTextClassifier,
    loader: DataLoader,
    output_csv: Path,
    labels_by_wav: dict[str, int],
) -> dict[str, Any]:
    model.eval()
    confusion = np.zeros((3, 3), dtype=np.int64)
    records: list[dict[str, Any]] = []
    for batch in loader:
        probabilities = torch.softmax(model(batch), dim=-1).cpu().numpy()
        for wav, probability in zip(batch["wav_names"], probabilities):
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
    parser.add_argument("--steps", type=int, default=1400)
    parser.add_argument("--save-steps", default="350,700,1050,1400")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=7.5e-5)
    parser.add_argument("--kd-alpha", type=float, default=0.40)
    parser.add_argument("--temperature", type=float, default=2.0)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
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
            raise ValueError("Evaluation is deliberately single-GPU")
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl")
    device_index = local_rank if distributed else 0
    is_main = rank == 0
    for path, role in [(args.train_csv, "train"), (args.val_csv, "validation"), (args.teacher_train, "teacher")]:
        common.reject_test_path(path, role)
    if is_main:
        if args.output_dir.exists():
            raise FileExistsError(args.output_dir)
        args.output_dir.mkdir(parents=True)
    if distributed:
        dist.barrier()
    random.seed(args.seed + rank)
    np.random.seed(args.seed + rank)
    torch.manual_seed(args.seed + rank)

    train_rows = common.read_table(args.train_csv, require_historical=True)
    val_rows = common.read_table(args.val_csv, require_historical=False)
    teacher_coverage = common.read_teacher(args.teacher_train, train_rows)
    teacher_agreement = {"retained_correct_argmax": 0, "suppressed_wrong_argmax": 0}
    for row in train_rows:
        if not float(row["teacher_weight"]):
            continue
        if int(np.asarray(row["teacher_probs"]).argmax()) == int(row["label_id"]):
            teacher_agreement["retained_correct_argmax"] += 1
        else:
            row["teacher_weight"] = 0.0
            teacher_agreement["suppressed_wrong_argmax"] += 1
    if args.max_train_rows:
        train_rows = train_rows[:args.max_train_rows]
    if args.max_val_rows:
        val_rows = val_rows[:args.max_val_rows]

    processor = AutoProcessor.from_pretrained(args.model, local_files_only=True)
    processor.tokenizer.padding_side = "right"
    encoded = [processor.tokenizer.encode(label, add_special_tokens=False) for label in LABELS]
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
        device_map={"": device_index},
        torch_dtype=torch.float16,
        quantization_config=quantization,
        low_cpu_mem_usage=True,
    )
    backbone.config.use_cache = False
    if args.eval_only_adapter is not None:
        backbone = PeftModel.from_pretrained(backbone, args.eval_only_adapter, is_trainable=False)
    else:
        backbone = prepare_model_for_kbit_training(backbone, use_gradient_checkpointing=True)
        backbone = get_peft_model(backbone, LoraConfig(
            r=args.lora_rank,
            lora_alpha=args.lora_alpha,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=FULL_TARGETS,
        ))
    model = DirectAudioTextClassifier(
        backbone,
        token_ids,
        int(backbone.get_base_model().config.audio_token_id),
        initialize_from_lm_head=args.eval_only_adapter is None,
    ).to(torch.device("cuda", device_index))
    if args.eval_only_adapter is not None:
        load_head(model, args.eval_only_adapter)
    if is_main:
        backbone.print_trainable_parameters()
        print(json.dumps({
            "direct_head_trainable": sum(
                p.numel() for name, p in model.named_parameters() if not name.startswith("backbone.")
            ),
            "target_modules": FULL_TARGETS,
        }), flush=True)

    collator = DirectCollator(processor, args.max_audio_seconds, args.include_context)
    val_loader = DataLoader(common.RowDataset(val_rows), batch_size=args.batch_size, shuffle=False, collate_fn=collator)
    labels_by_wav = {row["wav_name"]: int(row["label_id"]) for row in val_rows}
    manifest = {
        "protocol": "single Qwen2-Audio full-LoRA plus direct answer/audio-pooling head",
        "test_evaluated": False,
        "arguments": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "inputs": {
            "train_csv": {"path": str(args.train_csv), "sha256": common.sha256(args.train_csv)},
            "val_csv": {"path": str(args.val_csv), "sha256": common.sha256(args.val_csv)},
            "teacher_train": {"path": str(args.teacher_train), "sha256": common.sha256(args.teacher_train)},
        },
        "train_rows": len(train_rows),
        "val_rows": len(val_rows),
        "teacher_coverage": teacher_coverage,
        "teacher_truth_agreement_filter": teacher_agreement,
        "target_modules": FULL_TARGETS,
        "pooling": ["answer_state", "audio_mean", "audio_std", "text_mean"],
        "head_type": "four_pool_residual",
        "head_block_hidden": 64,
        "prompt": "rich v2 manual intensity calibration plus current/previous/next same-side ASR",
        "distributed": {"world_size": world_size, "global_micro_batch": world_size * args.batch_size},
    }
    if is_main:
        (args.output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    if distributed:
        dist.barrier()

    if args.eval_only_adapter is not None:
        metrics = evaluate(model, val_loader, args.output_dir / "val_predictions.csv", labels_by_wav)
        (args.output_dir / "summary_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        print(json.dumps(metrics, indent=2), flush=True)
        return

    sampler = DistributedSampler(
        common.RowDataset(train_rows), num_replicas=world_size, rank=rank, shuffle=True, seed=args.seed, drop_last=False
    ) if distributed else None
    train_loader = DataLoader(
        common.RowDataset(train_rows), batch_size=args.batch_size, shuffle=sampler is None,
        sampler=sampler, generator=torch.Generator().manual_seed(args.seed + rank), collate_fn=collator, num_workers=0,
    )
    if distributed:
        model = DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank)
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=args.learning_rate, betas=(0.9, 0.95), weight_decay=0.01,
    )
    save_steps = {int(value) for value in args.save_steps.split(",") if value.strip()}
    model.train()
    optimizer.zero_grad(set_to_none=True)
    epoch = 0
    if sampler is not None:
        sampler.set_epoch(epoch)
    iterator = iter(train_loader)
    start = time.time()
    recent: list[float] = []
    for step in range(1, args.steps + 1):
        for _ in range(args.gradient_accumulation):
            try:
                batch = next(iterator)
            except StopIteration:
                epoch += 1
                if sampler is not None:
                    sampler.set_epoch(epoch)
                iterator = iter(train_loader)
                batch = next(iterator)
            logits = model(batch)
            labels = batch["class_labels"].to(logits.device)
            teacher = batch["teacher_probs"].to(logits.device)
            teacher_weights = batch["teacher_weights"].to(logits.device)
            hard = F.cross_entropy(logits, labels, reduction="none")
            # The stored teacher values are T=1 probabilities.  Distillation
            # at T>1 must soften both distributions; otherwise the optimum is
            # student_logits ~= T * teacher_logits and becomes artificially
            # overconfident at deployment temperature one.
            teacher_at_temperature = F.softmax(
                torch.log(teacher.clamp_min(1e-8)) / args.temperature, dim=-1
            )
            kd = F.kl_div(
                F.log_softmax(logits / args.temperature, dim=-1),
                teacher_at_temperature,
                reduction="none",
            ).sum(dim=-1) * (args.temperature ** 2)
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
                "step": step, "loss": float(np.mean(recent[-20:])),
                "elapsed_seconds": elapsed, "seconds_per_step": elapsed / step,
            }), flush=True)
        if step in save_steps or step == args.steps:
            if distributed:
                dist.barrier()
            if is_main:
                save_checkpoint(model, processor, args.output_dir / f"checkpoint-{step}")
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
