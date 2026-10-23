#!/usr/bin/env python3
"""Normalize an audio folder plus ASR CSV/XLSX for DACE and ACCR."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import pandas as pd


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
NAME_RE = re.compile(
    r"^(?P<dialogue>.+)_(?P<side>left|right)_(?P<order>\d+)\.[^.]+$",
    re.IGNORECASE,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    if path.suffix.lower() == ".xlsx":
        return pd.read_excel(path)
    raise ValueError("ASR table must be .csv or .xlsx")


def normalize_label(value: object) -> str | None:
    return LABEL_MAP.get(str(value).strip().casefold())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio-dir", type=Path, required=True)
    parser.add_argument("--asr-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--wav-column", default="wav_name")
    parser.add_argument("--asr-column", default="ASR")
    parser.add_argument("--label-column", default="label")
    parser.add_argument("--dialogue-column", default="dialogue_key")
    parser.add_argument("--segment-column", default="segment")
    args = parser.parse_args()

    audio_dir = args.audio_dir.expanduser().resolve()
    table_path = args.asr_file.expanduser().resolve()
    table = read_table(table_path)
    required = {args.wav_column, args.asr_column}
    missing = required.difference(table.columns)
    if missing:
        raise ValueError(f"ASR table is missing columns: {sorted(missing)}")
    if table[args.wav_column].astype(str).duplicated().any():
        raise ValueError("wav_name values must be unique")

    label_column_present = args.label_column in table.columns
    if label_column_present:
        complete_labels = bool(table[args.label_column].notna().all())
        empty_labels = bool(table[args.label_column].isna().all())
        if not complete_labels and not empty_labels:
            raise ValueError(
                "label column is partially empty; fill every label or remove it"
            )
        has_labels = complete_labels
    else:
        has_labels = False

    rows: list[dict[str, object]] = []
    for index, source in table.reset_index(drop=True).iterrows():
        wav_name = Path(str(source[args.wav_column])).name
        audio_path = audio_dir / wav_name
        if not audio_path.is_file():
            raise FileNotFoundError(f"missing audio for row {index}: {audio_path}")
        match = NAME_RE.match(wav_name)
        supplied_dialogue = (
            args.dialogue_column in table.columns
            and pd.notna(source[args.dialogue_column])
            and str(source[args.dialogue_column]).strip()
        )
        supplied_segment = (
            args.segment_column in table.columns
            and pd.notna(source[args.segment_column])
            and str(source[args.segment_column]).strip()
        )
        dialogue = (
            str(source[args.dialogue_column]).strip()
            if supplied_dialogue
            else match.group("dialogue")
            if match
            else Path(wav_name).stem
        )
        segment = (
            str(source[args.segment_column]).strip()
            if supplied_segment
            else f"{match.group('side').lower()}_{match.group('order')}"
            if match
            else f"unknown_{index}"
        )
        label = ""
        if has_labels:
            raw_label = source[args.label_column]
            normalized = normalize_label(raw_label)
            if normalized is None:
                raise ValueError(
                    f"unsupported label at row {index}: {raw_label!r}"
                )
            label = normalized
        asr = (
            ""
            if pd.isna(source[args.asr_column])
            else str(source[args.asr_column])
        )
        rows.append(
            {
                "sample_index": index,
                "wav_name": wav_name,
                "label": label,
                "ASR": asr,
                "dialogue_key": dialogue,
                "segment": segment,
                "source_audio_path": str(audio_path.resolve()),
            }
        )

    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError(output_dir)
    output_dir.mkdir(parents=True)
    input_csv = output_dir / "input.csv"
    pd.DataFrame(rows).to_csv(input_csv, index=False, encoding="utf-8-sig")
    manifest = {
        "schema_version": 1,
        "source_table": str(table_path),
        "source_table_sha256": sha256(table_path),
        "audio_dir": str(audio_dir),
        "sample_count": len(rows),
        "has_labels": has_labels,
        "label_mapping": {
            "A": ["A", "中性", "neutral"],
            "B": ["B", "厌恶", "disgust", "disgusted"],
            "C": ["C", "愤怒", "anger", "angry"],
        },
        "input_csv": str(input_csv),
        "input_csv_sha256": sha256(input_csv),
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
