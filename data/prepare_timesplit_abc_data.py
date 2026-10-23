#!/usr/bin/env python3
"""Prepare chronological, leakage-aware audio emotion datasets.

This script merges data_1w2.csv and data_1w5.csv, normalizes emotion labels,
creates a merged audio directory containing only valid samples, splits by month,
and writes ShareGPT-style A/B/C datasets for Qwen2.5-Omni SFT.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import pandas as pd


VALID_LABELS = {"中性", "厌恶", "愤怒"}
LABEL_MAP = {
    "neutral": "中性",
    "中性": "中性",
    "disgusted": "厌恶",
    "disgust": "厌恶",
    "厌恶": "厌恶",
    "angry": "愤怒",
    "anger": "愤怒",
    "愤怒": "愤怒",
}
LABEL_TO_ABC = {"中性": "A", "厌恶": "B", "愤怒": "C"}
FILENAME_RE = re.compile(r"^(?P<dialogue_id>[^_]+)_(?P<timestamp>\d{14})_(?P<segment>.+)\.wav$")


SYSTEM_PROMPT = """你是一位语音情绪识别专家。你的任务是根据通话语音判断用户情绪。

候选情绪只有三类：
A: 中性
B: 厌恶
C: 愤怒

判断时不要只看通话主题，要结合语气强度、语速、音量、重复、质问、辱骂、攻击性和整体情绪波动。

分类标准：
中性：语气整体平稳，可能有轻微质疑、拒绝或不会操作，但没有明显情绪激动。
厌恶：明显反感、排斥、责怪、讽刺或持续质问，负面情绪较强，但未达到失控爆发或直接攻击辱骂。
愤怒：情绪爆发、语气极激烈或失控，常伴随辱骂、威胁、侮辱或强烈敌意。

请只输出 A、B 或 C，不要输出其他内容。"""

USER_PROMPT = "<audio>请判断这段通话语音的情绪。\nA: 中性\nB: 厌恶\nC: 愤怒\n只输出 A、B 或 C。"


@dataclass(frozen=True)
class SourceSpec:
    csv_path: Path
    audio_dir: Path
    wav_col: str
    label_col: str
    text_col: str


def normalize_label(value: object) -> str | None:
    key = str(value).strip()
    if not key or key.lower() == "nan":
        return None
    return LABEL_MAP.get(key.lower(), LABEL_MAP.get(key))


def parse_wav_name(wav_name: str) -> tuple[str, datetime, str]:
    name = Path(str(wav_name).strip()).name
    match = FILENAME_RE.match(name)
    if match is None:
        raise ValueError(f"Invalid wav filename: {wav_name}")

    timestamp = datetime.strptime(match.group("timestamp"), "%Y%m%d%H%M%S")
    dialogue_key = f"{match.group('dialogue_id')}_{match.group('timestamp')}"
    return dialogue_key, timestamp, match.group("segment")


def load_source(spec: SourceSpec, source_name: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(spec.csv_path)
    rows: list[dict[str, object]] = []
    invalid_rows: list[dict[str, object]] = []
    missing_audio: list[dict[str, object]] = []

    for idx, row in df.iterrows():
        wav_name = Path(str(row[spec.wav_col]).strip()).name
        label = normalize_label(row[spec.label_col])
        text = "" if pd.isna(row[spec.text_col]) else str(row[spec.text_col])

        try:
            dialogue_key, timestamp, segment = parse_wav_name(wav_name)
        except ValueError as exc:
            invalid_rows.append(
                {
                    "source_csv": spec.csv_path.name,
                    "row_index": idx,
                    "wav_name": wav_name,
                    "raw_label": row[spec.label_col],
                    "reason": str(exc),
                }
            )
            continue

        if label not in VALID_LABELS:
            invalid_rows.append(
                {
                    "source_csv": spec.csv_path.name,
                    "row_index": idx,
                    "wav_name": wav_name,
                    "raw_label": row[spec.label_col],
                    "reason": "invalid_label",
                }
            )
            continue

        audio_path = spec.audio_dir / wav_name
        if not audio_path.exists():
            missing_audio.append(
                {
                    "source_csv": spec.csv_path.name,
                    "row_index": idx,
                    "wav_name": wav_name,
                    "label": label,
                    "source_audio_path": str(audio_path),
                    "reason": "missing_audio",
                }
            )
            continue

        rows.append(
            {
                "wav_name": wav_name,
                "label": label,
                "ASR": text,
                "timestamp": timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                "month": timestamp.strftime("%Y%m"),
                "dialogue_key": dialogue_key,
                "segment": segment,
                "source": source_name,
                "source_csv": spec.csv_path.name,
                "source_audio_path": str(audio_path),
            }
        )

    return pd.DataFrame(rows), pd.DataFrame(invalid_rows), pd.DataFrame(missing_audio)


def link_or_copy(src: Path, dst: Path) -> str:
    if dst.exists():
        if dst.stat().st_size != src.stat().st_size:
            return "exists_size_mismatch"
        return "exists"

    try:
        os.link(src, dst)
        return "hardlinked"
    except OSError:
        shutil.copy2(src, dst)
        return "copied"


def write_json_dataset(df: pd.DataFrame, path: Path) -> None:
    records: list[dict[str, object]] = []
    for row in df.itertuples(index=False):
        records.append(
            {
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": USER_PROMPT},
                    {"role": "assistant", "content": LABEL_TO_ABC[row.label]},
                ],
                "audios": [f"audio/{row.wav_name}"],
            }
        )

    with path.open("w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)


def write_dataset_info(output_dir: Path) -> None:
    dataset_info = {
        "train_set": {
            "file_name": "train_set_abc.json",
            "formatting": "sharegpt",
            "columns": {"messages": "messages", "audios": "audios"},
            "tags": {
                "system_tag": "system",
                "role_tag": "role",
                "content_tag": "content",
                "user_tag": "user",
                "assistant_tag": "assistant",
            },
        },
        "val_set": {
            "file_name": "val_set_abc.json",
            "formatting": "sharegpt",
            "columns": {"messages": "messages", "audios": "audios"},
            "tags": {
                "system_tag": "system",
                "role_tag": "role",
                "content_tag": "content",
                "user_tag": "user",
                "assistant_tag": "assistant",
            },
        },
        "test_set": {
            "file_name": "test_set_abc.json",
            "formatting": "sharegpt",
            "columns": {"messages": "messages", "audios": "audios"},
            "tags": {
                "system_tag": "system",
                "role_tag": "role",
                "content_tag": "content",
                "user_tag": "user",
                "assistant_tag": "assistant",
            },
        },
    }
    with (output_dir / "dataset_info.json").open("w", encoding="utf-8") as f:
        json.dump(dataset_info, f, ensure_ascii=False, indent=2)


def save_csv(df: pd.DataFrame, path: Path) -> None:
    df.to_csv(path, index=False, encoding="utf-8-sig")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--audio-dir", type=Path, default=Path("audio"))
    parser.add_argument("--processed-dir", type=Path, default=Path("processed_data_timesplit_abc"))
    parser.add_argument("--val-month", default="202506")
    parser.add_argument("--test-month", default="202507")
    parser.add_argument("--clean-audio", action="store_true", help="Remove wav files in output audio dir not in valid data.")
    args = parser.parse_args()

    root = args.root.resolve()
    audio_dir = (root / args.audio_dir).resolve()
    processed_dir = (root / args.processed_dir).resolve()
    audio_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    sources = [
        SourceSpec(root / "data_1w2.csv", root / "audio_1w2", "wav_path", "Emotion", "text"),
        SourceSpec(root / "data_1w5.csv", root / "audio_1w5", "wav_name", "label", "ASR"),
    ]

    valid_frames: list[pd.DataFrame] = []
    invalid_frames: list[pd.DataFrame] = []
    missing_frames: list[pd.DataFrame] = []
    for spec in sources:
        valid, invalid, missing = load_source(spec, spec.csv_path.stem)
        valid_frames.append(valid)
        invalid_frames.append(invalid)
        missing_frames.append(missing)

    data = pd.concat(valid_frames, ignore_index=True)
    data = data.sort_values(["timestamp", "dialogue_key", "wav_name"], kind="mergesort").reset_index(drop=True)

    duplicate_mask = data.duplicated("wav_name", keep=False)
    duplicates = data.loc[duplicate_mask].copy()
    if duplicate_mask.any():
        data = data.drop_duplicates("wav_name", keep="first").reset_index(drop=True)

    link_status_counts: dict[str, int] = {}
    for row in data.itertuples(index=False):
        status = link_or_copy(Path(row.source_audio_path), audio_dir / row.wav_name)
        link_status_counts[status] = link_status_counts.get(status, 0) + 1

    if args.clean_audio:
        valid_names = set(data["wav_name"])
        removed = 0
        for wav_path in audio_dir.glob("*.wav"):
            if wav_path.name not in valid_names:
                wav_path.unlink()
                removed += 1
        link_status_counts["removed_extra_from_output_audio"] = removed

    train_df = data[(data["month"] != args.val_month) & (data["month"] != args.test_month)].copy()
    val_df = data[data["month"] == args.val_month].copy()
    test_df = data[data["month"] == args.test_month].copy()

    split_by_dialogue = {
        "train_val": len(set(train_df["dialogue_key"]) & set(val_df["dialogue_key"])),
        "train_test": len(set(train_df["dialogue_key"]) & set(test_df["dialogue_key"])),
        "val_test": len(set(val_df["dialogue_key"]) & set(test_df["dialogue_key"])),
    }

    save_csv(data, root / "audio_data.csv")
    save_csv(train_df, root / "train_timesplit.csv")
    save_csv(val_df, root / "val_202506.csv")
    save_csv(test_df, root / "test_202507.csv")

    invalid_all = pd.concat(invalid_frames, ignore_index=True)
    missing_all = pd.concat(missing_frames, ignore_index=True)
    save_csv(invalid_all, root / "audio_data_invalid_rows.csv")
    save_csv(missing_all, root / "audio_data_missing_audio.csv")
    save_csv(duplicates, root / "audio_data_duplicate_wav_names.csv")

    write_json_dataset(train_df, processed_dir / "train_set_abc.json")
    write_json_dataset(val_df, processed_dir / "val_set_abc.json")
    write_json_dataset(test_df, processed_dir / "test_set_abc.json")
    write_dataset_info(processed_dir)

    summary = {
        "total_valid": int(len(data)),
        "splits": {
            "train": {"rows": int(len(train_df)), "label_counts": train_df["label"].value_counts().to_dict()},
            "val_202506": {"rows": int(len(val_df)), "label_counts": val_df["label"].value_counts().to_dict()},
            "test_202507": {"rows": int(len(test_df)), "label_counts": test_df["label"].value_counts().to_dict()},
        },
        "invalid_rows": int(len(invalid_all)),
        "missing_audio": int(len(missing_all)),
        "duplicate_wav_rows": int(len(duplicates)),
        "dialogue_key_overlap_between_splits": split_by_dialogue,
        "audio_output": str(audio_dir),
        "audio_link_status_counts": link_status_counts,
    }
    with (processed_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
