#!/usr/bin/env python3
"""Seal or verify a complete offline bundle. CPU-only, standard library."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
from download_models import verify, write_json

ROOT = Path(__file__).resolve().parent
CODE = ["Dockerfile", ".dockerignore", "requirements.txt", "models.json", "models.lock.json", "download_models.py",
        "bundle.py", "gpu_test.py", "report.py", "image_constraints.py", "run.sh", "prepare_offline.sh", "copy_to_disk.ps1",
        "README_使用说明.md", "public_benchmark.py", "public_baselines.json", "build_nccl_tests.py",
        "vendor/nccl-tests-v2.17.6.tar.gz", "vendor/README.md", "deepseek_mlperf.py", "run_deepseek_mlperf.sh",
        "README_DeepSeek_MLPerf.md", "deepseek-model.lock.json", "deepseek_assets.py", "prepare_deepseek_accuracy.py",
        "README.md", "docs/使用总览.md", "images/runtime.inspect.json", "images/b300-test.tar"]


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=["seal", "verify", "status"])
    p.add_argument("--root", type=Path, default=ROOT)
    args = p.parse_args()
    root = args.root.resolve()
    missing = [name for name in CODE if not (root / name).is_file()]
    if args.action == "status":
        lock = root / "models.lock.json"
        models = json.loads(lock.read_text(encoding="utf-8"))["models"] if lock.exists() else []
        for m in models:
            expected = sum(x["size"] for x in m["files"])
            present = sum((root / "models" / m["name"] / x["path"]).stat().st_size
                          for x in m["files"] if (root / "models" / m["name"] / x["path"]).is_file())
            print(f"{m['name']}: {present / 2**30:.2f}/{expected / 2**30:.2f} GiB (size only; not hash verified)")
        print("Missing package files:", ", ".join(missing) or "none")
        print("Sealed:", (root / "bundle.sha256.json").exists())
        return 0
    if missing:
        raise RuntimeError("Offline bundle incomplete: " + ", ".join(missing))
    models = json.loads((root / "models.lock.json").read_text(encoding="utf-8"))["models"]
    failures = verify(root, models)
    if failures:
        raise RuntimeError("Model files missing/corrupt: " + ", ".join(failures))
    inspect = json.loads((root / "images/runtime.inspect.json").read_text(encoding="utf-8"))[0]
    if args.action == "seal":
        write_json(root / "bundle.sha256.json", {
            "schema": 1, "image_id": inspect["Id"], "architecture": inspect["Architecture"],
            "files": {name: sha(root / name) for name in CODE}})
        print("SEALED: models verified; complete runtime archive present; SHA256 recorded.")
    else:
        seal = json.loads((root / "bundle.sha256.json").read_text(encoding="utf-8"))
        if set(seal["files"]) != set(CODE):
            raise RuntimeError("Seal file inventory mismatch")
        for name, expected in seal["files"].items():
            if sha(root / name) != expected:
                raise RuntimeError("Package SHA256 mismatch: " + name)
        if seal["image_id"] != inspect["Id"] or seal["architecture"] != inspect["Architecture"]:
            raise RuntimeError("Image metadata mismatch")
        print("VERIFIED: complete sealed offline bundle.")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
