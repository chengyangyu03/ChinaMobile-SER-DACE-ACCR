#!/usr/bin/env python3
"""Download the official Qwen2-Audio base used by DACE and ACCR."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / "models/base/Qwen2-Audio-7B-Instruct"
REPO = "Qwen/Qwen2-Audio-7B-Instruct"
EXPECTED_SHARDS = {
    "model-00001-of-00005.safetensors": 3911340848,
    "model-00002-of-00005.safetensors": 3980786080,
    "model-00003-of-00005.safetensors": 3980819456,
    "model-00004-of-00005.safetensors": 3643135696,
    "model-00005-of-00005.safetensors": 1278214288,
}


def snapshot_complete() -> bool:
    required = (
        "config.json",
        "model.safetensors.index.json",
        "preprocessor_config.json",
        "tokenizer.json",
    )
    return (
        all((TARGET / name).is_file() for name in required)
        and all(
            (TARGET / name).is_file()
            and (TARGET / name).stat().st_size == expected_size
            for name, expected_size in EXPECTED_SHARDS.items()
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--reuse-local",
        type=Path,
        help=(
            "Development-only: symlink an already downloaded official snapshot "
            "instead of downloading it again."
        ),
    )
    args = parser.parse_args()
    TARGET.parent.mkdir(parents=True, exist_ok=True)
    if args.reuse_local is not None:
        if TARGET.exists() or TARGET.is_symlink():
            raise FileExistsError(TARGET)
        source = args.reuse_local.expanduser().resolve()
        if not (source / "model.safetensors.index.json").is_file():
            raise FileNotFoundError(source / "model.safetensors.index.json")
        TARGET.symlink_to(source, target_is_directory=True)
    else:
        if TARGET.is_symlink() or (TARGET.exists() and not TARGET.is_dir()):
            raise FileExistsError(TARGET)
        TARGET.mkdir(parents=True, exist_ok=True)
        environment = os.environ.copy()
        environment.setdefault("MODELSCOPE_DOWNLOAD_PARALLELS", "8")
        environment.setdefault("MODELSCOPE_DOWNLOAD_PARALLEL_WORKERS", "8")
        command = [
            sys.executable,
            "-m",
            "modelscope.cli.cli",
            "download",
            REPO,
            "--revision",
            "master",
            "--local_dir",
            str(TARGET),
            "--max-workers",
            "8",
        ]
        for attempt in range(1, 4):
            print(f"[base download] attempt {attempt}/3", flush=True)
            subprocess.run(command, env=environment, check=True)
            if snapshot_complete():
                break
            print(
                "[base download] snapshot is incomplete; resuming the same "
                "target directory",
                flush=True,
            )
        else:
            raise RuntimeError(
                "Qwen2-Audio snapshot remains incomplete after three attempts"
            )
    subprocess.run(
        [sys.executable, str(ROOT / "scripts/verify_install.py")],
        check=True,
    )


if __name__ == "__main__":
    main()
