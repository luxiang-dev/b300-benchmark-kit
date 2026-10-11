#!/usr/bin/env python3
"""Download or hash-verify the optional pinned DeepSeek checkpoint (about 395 GiB)."""
import argparse
import concurrent.futures
import json
from pathlib import Path
import shutil
import sys
import download_models as downloads

LOCK = Path(__file__).with_name("deepseek-model.lock.json")


def catalog(workspace):
    models = json.loads(LOCK.read_text(encoding="utf-8"))["models"]
    return workspace / "build", [{**m, "name": m["path"].removeprefix("build/models/")} for m in models]


def verify(workspace):
    root, models = catalog(Path(workspace))
    return downloads.verify(root, models)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=["download", "verify"])
    p.add_argument("--workspace", type=Path, required=True, help="Prepared closed/NVIDIA directory")
    p.add_argument("--workers", type=int, default=2)
    args = p.parse_args()
    root, models = catalog(args.workspace.resolve())
    if not (args.workspace / "Makefile").is_file() or not (root / "models").is_dir():
        p.error("Use the prepared closed/NVIDIA directory with existing build/models (make link_dirs first)")
    if args.workers < 1 or args.workers > 8:
        p.error("--workers must be 1..8")
    total = sum(f["size"] for m in models for f in m["files"])
    print(f"Pinned FP4 weights + original tokenizer/config: {total / 2**30:.2f} GiB", flush=True)
    if args.action == "download":
        needed = 0
        for m in models:
            for f in m["files"]:
                target = downloads.safe_path(root / "models" / m["name"], f["path"])
                if downloads.valid_file(target, f):
                    continue
                part = target.with_name(target.name + ".part")
                offset = part.stat().st_size if part.is_file() else 0
                needed += max(0, f["size"] - offset) if offset <= f["size"] else f["size"]
        if shutil.disk_usage((root / "models").resolve()).free < needed + 2 * 2**30:
            raise RuntimeError(f"Need {needed / 2**30:.2f} GiB + 2 GiB reserve for the optional weights")
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
            tasks = [pool.submit(downloads.download_one, root, m, f) for m in models for f in m["files"]]
            for task in concurrent.futures.as_completed(tasks):
                task.result()
    failures = downloads.verify(root, models)
    if failures:
        print("Missing/corrupt DeepSeek files: " + ", ".join(failures), file=sys.stderr)
        return 1
    print("DeepSeek model/tokenizer hashes verified; dataset/runtime still need separate preparation.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
