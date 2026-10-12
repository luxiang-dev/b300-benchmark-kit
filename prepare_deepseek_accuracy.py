#!/usr/bin/env python3
"""Populate the evaluator's pinned Hugging Face cache from already verified local metadata."""
import argparse
import os
from pathlib import Path
import shutil
import subprocess
import deepseek_assets
import download_models


def prepare(workspace):
    workspace = Path(workspace).resolve()
    _, models = deepseek_assets.catalog(workspace)
    model = next(m for m in models if m["repo"] == "deepseek-ai/DeepSeek-R1")
    source = workspace / "build/models" / model["name"]
    destination = workspace / "build/hf-cache/hub/models--deepseek-ai--DeepSeek-R1/snapshots" / model["revision"]
    selected = [item for item in model["files"] if item["path"] in
                ("config.json", "tokenizer.json", "tokenizer_config.json")]
    if len(selected) != 3 or any(not download_models.valid_file(source / item["path"], item) for item in selected):
        raise RuntimeError("Download and verify the pinned original tokenizer metadata first")
    destination.mkdir(parents=True, exist_ok=True)
    for item in selected:
        target = destination / item["path"]
        if not download_models.valid_file(target, item):
            shutil.copyfile(source / item["path"], target)
        if not download_models.valid_file(target, item):
            raise RuntimeError("Tokenizer cache copy verification failed: " + item["path"])
    return destination


def verify_cache(workspace):
    workspace = Path(workspace).resolve()
    _, models = deepseek_assets.catalog(workspace)
    model = next(m for m in models if m["repo"] == "deepseek-ai/DeepSeek-R1")
    destination = workspace / "build/hf-cache/hub/models--deepseek-ai--DeepSeek-R1/snapshots" / model["revision"]
    selected = [item for item in model["files"] if item["path"] in ("config.json", "tokenizer.json", "tokenizer_config.json")]
    if len(selected) != 3 or any(not download_models.valid_file(destination / item["path"], item) for item in selected):
        raise RuntimeError("Missing/corrupt offline tokenizer cache; run prepare_deepseek_accuracy.py during preparation")


def warm_evaluator(workspace):
    workspace = Path(workspace).resolve()
    env = {**os.environ, "HF_HOME": str(workspace / "build/hf-cache"),
           "HF_HUB_CACHE": str(workspace / "build/hf-cache/hub"),
           "HF_DATASETS_CACHE": str(workspace / "build/hf-cache/datasets"),
           "HF_HUB_OFFLINE": "0", "TRANSFORMERS_OFFLINE": "0", "HF_DATASETS_OFFLINE": "0"}
    for key in ("TRANSFORMERS_CACHE",):
        env.pop(key, None)
    script = """import runpy, sys
evaluation = runpy.run_path(sys.argv[1], run_name='prepare_evaluator_cache')
assert evaluation['evaluate_math500']('2', '2')
assert evaluation['load_lcb_benchmark']()
"""
    subprocess.run(["/work/.dsr1-acc-venv/bin/python3", "-c", script,
                    str(workspace / "3rdparty/mlc-inference/language/deepseek-r1/eval_accuracy.py")],
                   cwd=workspace, env=env, check=True, timeout=1800)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--warm-evaluator", action="store_true", help="Online: load official PRM/LiveCodeBench dependencies and dataset cache")
    args = parser.parse_args()
    print(prepare(args.workspace))
    if args.warm_evaluator:
        warm_evaluator(args.workspace)
