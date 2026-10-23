#!/usr/bin/env python3
"""Verify package files, Python dependencies, CUDA, and the base snapshot."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "models/base/Qwen2-Audio-7B-Instruct"
MODULES = (
    "accelerate",
    "bitsandbytes",
    "modelscope",
    "numpy",
    "openpyxl",
    "pandas",
    "peft",
    "scipy",
    "soundfile",
    "torch",
    "transformers",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-base-model", action="store_true")
    args = parser.parse_args()
    versions = {}
    for name in MODULES:
        module = importlib.import_module(name)
        versions[name] = str(getattr(module, "__version__", "unknown"))

    subprocess.run(
        ["sha256sum", "-c", str(ROOT / "SHA256SUMS")],
        cwd=ROOT,
        check=True,
    )
    subprocess.run(
        [sys.executable, str(ROOT / "scripts/download_project_models.py"), "--check-only"],
        check=True,
    )
    base = {"checked": False}
    if not args.skip_base_model:
        required = (
            "config.json",
            "model.safetensors.index.json",
            "model-00001-of-00005.safetensors",
            "model-00002-of-00005.safetensors",
            "model-00003-of-00005.safetensors",
            "model-00004-of-00005.safetensors",
            "model-00005-of-00005.safetensors",
            "preprocessor_config.json",
            "tokenizer.json",
        )
        missing = [name for name in required if not (BASE / name).is_file()]
        if missing:
            raise FileNotFoundError(f"base snapshot is incomplete: {missing}")
        config = json.loads((BASE / "config.json").read_text(encoding="utf-8"))
        if config.get("model_type") != "qwen2_audio":
            raise ValueError("base model is not Qwen2-Audio")
        subprocess.run(
            [
                "sha256sum",
                "-c",
                str(ROOT / "BASE_MODEL_SHA256SUMS"),
            ],
            cwd=BASE,
            check=True,
        )
        base = {
            "checked": True,
            "path": str(BASE),
            "config_sha256": sha256(BASE / "config.json"),
            "index_sha256": sha256(BASE / "model.safetensors.index.json"),
            "manifest_sha256": sha256(ROOT / "BASE_MODEL_SHA256SUMS"),
        }
    torch = importlib.import_module("torch")
    sys.path.insert(0, str(ROOT / "src"))
    runtime = importlib.import_module("ser_runtime")
    fallback = runtime.side_order(
        {
            "wav_name": "call_without_number_right.wav",
            "segment": "right",
            "source_index": 987,
        }
    )
    if fallback != ("unknown", 0):
        raise ValueError(
            f"runtime prompt fallback drifted from training contract: {fallback}"
        )
    report = {
        "versions": versions,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_device_count": int(torch.cuda.device_count()),
        "base_model": base,
        "runtime_contract": {
            "unnumbered_segment_fallback": list(fallback),
            "expected": ["unknown", 0],
            "checked": True,
        },
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
