#!/usr/bin/env python3
"""Download the four published project weights and verify their pinned hashes.

This script uses only the Python standard library. The base Qwen2-Audio model
is fetched separately by download_base_model.py.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import time
from urllib.parse import quote
from urllib.request import Request, urlopen


ROOT = Path(__file__).resolve().parents[1]


def verify(path: Path, artifact: dict) -> bool:
    if not path.is_file() or path.stat().st_size != artifact["size"]:
        return False
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest() == artifact["sha256"]


def download(manifest: dict, destination: Path, *, check_only: bool = False) -> None:
    for artifact in manifest["artifacts"]:
        relative = Path(artifact["path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Artifact paths must stay inside the destination")
        target = destination / relative
        if verify(target, artifact):
            print(f"OK {relative}", flush=True)
            continue
        if check_only:
            raise ValueError(f"Missing or mismatched project weight: {relative}")
        target.parent.mkdir(parents=True, exist_ok=True)
        url = (
            f"https://modelscope.cn/models/{quote(manifest['model_id'], safe='/')}"
            f"/resolve/{quote(manifest['revision'], safe='')}/"
            f"{quote(relative.as_posix(), safe='/')}"
        )
        temporary = target.with_name(target.name + ".part")
        for attempt in range(3):
            try:
                print(f"Downloading {relative} (attempt {attempt + 1}/3)", flush=True)
                request = Request(url, headers={"User-Agent": "ChinaMobile-SER-download/1"})
                with urlopen(request, timeout=120) as response, temporary.open("wb") as handle:
                    shutil.copyfileobj(response, handle, length=1024 * 1024)
                if not verify(temporary, artifact):
                    raise ValueError(f"Downloaded size/SHA256 mismatch: {relative}")
                os.replace(temporary, target)
                break
            except Exception:
                temporary.unlink(missing_ok=True)
                if attempt == 2:
                    raise
                time.sleep(2)
        print(f"OK {relative}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", type=Path, default=ROOT)
    parser.add_argument("--manifest", type=Path, default=ROOT / "MODEL_ARTIFACTS.json")
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    download(manifest, args.destination.resolve(), check_only=args.check_only)


if __name__ == "__main__":
    main()
