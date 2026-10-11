#!/usr/bin/env python3
"""Optional DeepSeek-R1 MLPerf v6.0 Offline adapter for a prepared NVIDIA container."""
import argparse
import ast
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request

SOURCE_REVISION = "4d3916ac9cf474b679cdfcf492d43a0559418ad1"
MLCOMMONS_REVISION = "175ea7ba24a826946458f72ec4a2221215f52802"
SYSTEM = "B300-SXM-270GBx8"
REFERENCE_ID = "mlperf-6.0-0073-deepseek-r1-offline"
CONFIG = Path(f"configs/{SYSTEM}/Offline/deepseek-r1.py")
MODEL = Path("build/models/deepseek-r1/fp4-quantized-modelopt/deepseek_r1-torch-fp4")
ORIGINAL = Path("build/models/deepseek-r1/deepseek-r1")
DATA = Path("build/preprocessed_data/deepseek-r1")
REFERENCE_ACCURACY = 81.3582
REFERENCE_OUTPUT_LENGTH = 3886.2274
BASE_ARGS = "--benchmarks=deepseek-r1 --scenarios=Offline --core_type=trtllm_endpoint"


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 2**20), b""):
            h.update(block)
    return h.hexdigest()


def field(text, key):
    matches = re.findall(r"^\s*" + re.escape(key) + r"\s*:\s*(.*?)\s*$", text, re.M)
    if len(matches) != 1:
        raise ValueError(f"Missing/duplicate LoadGen field: {key}")
    return matches[0]


def positive(value):
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise ValueError("Non-finite/non-positive MLPerf metric")
    return value


def parse_performance(text):
    if field(text, "Scenario") != "Offline" or field(text, "Mode") != "PerformanceOnly":
        raise ValueError("Expected MLPerf Offline PerformanceOnly")
    if field(text, "Result is") != "VALID":
        raise ValueError("LoadGen result is not VALID")
    for key in ("Min duration satisfied", "Min queries satisfied"):
        if field(text, key) != "Yes":
            raise ValueError(f"MLPerf validity constraint failed: {key}")
    if int(field(text, "min_duration (ms)")) < 600000:
        raise ValueError("Shortened MLPerf run: minimum duration must be 600000 ms")
    for key in ("performance_sample_count", "accuracy_sample_count"):
        if int(field(text, key)) != 4388:
            raise ValueError("Expected the full 4388-sample DeepSeek-R1 QSL")
    if int(field(text, "samples_per_query")) < 105312:
        raise ValueError("Expected the v6.0 Offline workload with at least 105312 issued samples")
    if "No errors encountered during test." not in text:
        raise ValueError("Missing zero-error LoadGen completion evidence")
    return dict(tokens_per_second=positive(field(text, "Tokens per second")),
                samples_per_second=positive(field(text, "Samples per second")),
                scenario="Offline", mode="PerformanceOnly", validity="VALID", qsl_samples=4388)


def parse_accuracy(text):
    matches = re.findall(r"^\s*(\{[^\n]*['\"]exact_match['\"][^\n]*\})\s*$", text, re.M)
    if len(matches) != 1:
        raise ValueError("Missing/duplicate official DeepSeek accuracy result dictionary")
    result = ast.literal_eval(matches[0])
    accuracy = positive(result["exact_match"])
    length = positive(result["tokens_per_sample"])
    if accuracy > 100 or result["num-samples"] != 4388:
        raise ValueError("Accuracy evaluation must cover all 4388 samples")
    if accuracy < REFERENCE_ACCURACY * 0.99:
        raise ValueError("DeepSeek exact match does not reach 99% of the v6.0 reference")
    if not 0.9 * REFERENCE_OUTPUT_LENGTH <= length <= 1.1 * REFERENCE_OUTPUT_LENGTH:
        raise ValueError("Mean generated length differs from the v6.0 reference by more than 10%")
    return dict(exact_match=accuracy, tokens_per_sample=length, samples=4388,
                minimum_exact_match=REFERENCE_ACCURACY * 0.99)


def parse_detail(path):
    versions = []
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if not line.startswith(":::MLLOG "):
                continue
            entry = json.loads(line.removeprefix(":::MLLOG "))
            if entry.get("metadata", {}).get("is_error"):
                raise ValueError("LoadGen detail contains an error event")
            if entry.get("key") == "loadgen_version":
                versions.append(entry.get("value"))
    if len(versions) != 1 or not isinstance(versions[0], str) or not re.match(r"^6\.0(?:[.\s]|$)", versions[0]):
        raise ValueError("Expected one actual LoadGen v6.0 version in the detail log")
    return versions[0]


def only_file(folder, name):
    matches = [p for p in folder.rglob(name) if "deepseek-r1" in p.parts and "Offline" in p.parts]
    if len(matches) != 1:
        raise ValueError(f"Expected one fresh deepseek-r1/Offline/{name} in {folder}")
    return matches[0]


def load_evidence(out):
    """Only the adapter's successful run with intact raw performance+accuracy logs can compare."""
    out = Path(out).resolve()
    path = out / "deepseek-mlperf.json"
    if not path.exists():
        return None
    result = read_json(path)
    if result.get("status") != "PASS":
        raise ValueError("DeepSeek MLPerf adapter did not complete successfully")
    context = result["context"]
    expected = dict(suite="MLPerf Inference v6.0", model="deepseek-r1", scenario="Offline",
                    precision="fp4", gpus=8, nodes=1, tensor_parallelism=8, expert_parallelism=8,
                    speculative_decoding=False, source_revision=SOURCE_REVISION,
                    mlcommons_revision=MLCOMMONS_REVISION)
    if any(context.get(k) != v for k, v in expected.items()):
        raise ValueError("DeepSeek model/scenario/precision/topology/source mismatch")
    for name, checksum in result["evidence_sha256"].items():
        p = (out / name).resolve()
        if not p.is_relative_to(out) or not p.is_file() or sha(p) != checksum:
            raise ValueError(f"DeepSeek evidence missing or checksum mismatch: {name}")
    perf = only_file(out / "deepseek-performance", "mlperf_log_summary.txt")
    detail = only_file(out / "deepseek-performance", "mlperf_log_detail.txt")
    accuracy = only_file(out / "deepseek-accuracy", "accuracy.txt")
    accuracy_log = only_file(out / "deepseek-accuracy", "mlperf_log_accuracy.json")
    accuracy_summary = only_file(out / "deepseek-accuracy", "mlperf_log_summary.txt")
    required = [perf, detail, accuracy, accuracy_log, accuracy_summary]
    if any(p.relative_to(out).as_posix() not in result["evidence_sha256"] or p.stat().st_size == 0 for p in required):
        raise ValueError("Missing mandatory DeepSeek performance/accuracy evidence hashes")
    atext = accuracy_summary.read_text(encoding="utf-8")
    if field(atext, "Scenario") != "Offline" or field(atext, "Mode") != "AccuracyOnly":
        raise ValueError("Expected separate Offline AccuracyOnly run")
    entries = read_json(accuracy_log)
    if not isinstance(entries, list) or len(entries) != 4388 or {e.get("qsl_idx") for e in entries} != set(range(4388)):
        raise ValueError("Raw accuracy responses do not cover all 4388 unique QSL samples")
    if any(not isinstance(e.get("data"), str) or not re.fullmatch(r"(?:[0-9a-fA-F]{2})+", e["data"]) for e in entries):
        raise ValueError("Missing/invalid generated token data in accuracy responses")
    return dict(performance=parse_performance(perf.read_text(encoding="utf-8")),
                accuracy=parse_accuracy(accuracy.read_text(encoding="utf-8")), context=context,
                loadgen_version=parse_detail(detail),
                note="同模型、FP4、MLPerf v6.0 Offline 测量及准确率条件已验证；系统和软件差异仍需核对。仅为本地复现，不是 MLCommons 认证提交。")


def check(workspace):
    workspace = Path(workspace).resolve()
    required = ["Makefile", str(CONFIG), "code/deepseek-r1/tensorrt/generation_config.json",
                "3rdparty/mlc-inference/loadgen/mlperf.conf",
                "3rdparty/mlc-inference/language/deepseek-r1/eval_accuracy.py",
                "3rdparty/mlc-inference/language/deepseek-r1/submodules/prm800k/setup.py",
                "3rdparty/mlc-inference/tools/submission/submission_checker/constants.py",
                "code/deepseek-r1/tensorrt/requirements.accuracy.txt", str(ORIGINAL / "config.json"),
                str(ORIGINAL / "tokenizer.json"), str(ORIGINAL / "tokenizer_config.json"),
                str(MODEL / "config.json"), str(MODEL / "hf_quant_config.json"),
                str(MODEL / "model.safetensors.index.json"), str(DATA / "input_ids_padded.npy"),
                str(DATA / "input_lens.npy"), "build/data/deepseek-r1/mlperf_deepseek_r1_dataset_4388_fp8_eval.pkl"]
    missing = [x for x in required if not (workspace / x).is_file()]
    if missing:
        raise RuntimeError("DeepSeek MLPerf assets missing: " + ", ".join(missing))
    import deepseek_assets
    print("Hash-verifying pinned DeepSeek FP4 weights (about 395 GiB)...", flush=True)
    corrupt = deepseek_assets.verify(workspace)
    if corrupt:
        raise RuntimeError("Pinned model/tokenizer hash check failed: " + ", ".join(corrupt))
    venv = Path("/work/.dsr1-acc-venv")
    marker = venv / ".requirements_hash"
    if not (venv / "bin/python3").is_file() or not marker.is_file() or marker.read_text().strip() != sha(workspace / "code/deepseek-r1/tensorrt/requirements.accuracy.txt"):
        raise RuntimeError("Prepare the official /work/.dsr1-acc-venv in the image before going offline")
    if not (workspace / "3rdparty/trtllm").is_dir():
        raise RuntimeError("Prepare the official 3rdparty/trtllm checkout")
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=workspace, text=True).strip()
    if revision != SOURCE_REVISION:
        raise RuntimeError(f"Use the pinned official source revision {SOURCE_REVISION}; found {revision}")
    subprocess.run(["git", "diff", "--exit-code", "HEAD", "--", "code", "src", "configs", "Makefile*"],
                   cwd=workspace, check=True, stdout=subprocess.DEVNULL)
    mlc = workspace / "3rdparty/mlc-inference"
    mlc_revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=mlc, text=True).strip()
    if mlc_revision != MLCOMMONS_REVISION:
        raise RuntimeError("Use the pinned v6.0 MLCommons dependency: " + MLCOMMONS_REVISION)
    subprocess.run(["git", "diff", "--exit-code", "HEAD", "--", "loadgen", "tools/submission/submission_checker",
                    "language/deepseek-r1/eval_accuracy.py"], cwd=mlc, check=True, stdout=subprocess.DEVNULL)
    model = read_json(workspace / MODEL / "config.json")
    if model.get("model_type") != "deepseek_v3" or model.get("num_hidden_layers") != 61 or model.get("n_routed_experts") != 256:
        raise RuntimeError("Requires full DeepSeek-R1; distilled/small models are not this benchmark")
    index = read_json(workspace / MODEL / "model.safetensors.index.json")
    shards = set(index["weight_map"].values())
    if not shards:
        raise RuntimeError("Empty DeepSeek checkpoint index")
    for name in shards:
        p = (workspace / MODEL / name).resolve()
        if not p.is_relative_to((workspace / MODEL).resolve()) or not p.is_file() or p.stat().st_size == 0:
            raise RuntimeError("Incomplete/unsafe DeepSeek checkpoint shard: " + name)
    import numpy as np
    lengths = np.load(workspace / DATA / "input_lens.npy", mmap_mode="r", allow_pickle=False)
    inputs = np.load(workspace / DATA / "input_ids_padded.npy", mmap_mode="r", allow_pickle=False)
    if lengths.shape[0] != 4388 or inputs.shape[0] != 4388:
        raise RuntimeError("Preprocessed dataset must contain 4388 samples")
    os.environ["SYSTEM_NAME"] = SYSTEM
    sys.path.insert(0, str(workspace))
    if "code" in sys.modules and not hasattr(sys.modules["code"], "__path__"):
        del sys.modules["code"]
    for name in ("mlperf_loadgen", "tensorrt", "tensorrt_llm", "nvmitten"):
        importlib.import_module(name)
    # The NVIDIA loader selects the latest version in this checkout. Refuse a
    # later ruleset accidentally mixed into the pinned v6.0 submission.
    from code.common.mlcommons import loadgen
    if loadgen._vstr != "v6.0":
        raise RuntimeError("The prepared MLCommons rules/LoadGen checkout must select v6.0")
    import torch
    if torch.cuda.device_count() != 8 or any(not re.search(r"\bB300\b", torch.cuda.get_device_name(i)) for i in range(8)):
        raise RuntimeError("Requires eight physical B300 GPUs")
    return dict(suite="MLPerf Inference v6.0", model="deepseek-r1", scenario="Offline", precision="fp4",
                gpus=8, nodes=1, tensor_parallelism=8, expert_parallelism=8, speculative_decoding=False,
                source_revision=revision, config_sha256=sha(workspace / CONFIG),
                model_config_sha256=sha(workspace / MODEL / "config.json"),
                dataset_sha256={p.name: sha(p) for p in (workspace / DATA / "input_ids_padded.npy", workspace / DATA / "input_lens.npy")},
                cuda=torch.version.cuda, torch=torch.__version__,
                tensorrt=getattr(importlib.import_module("tensorrt"), "__version__", "unknown"),
                tensorrt_llm=getattr(importlib.import_module("tensorrt_llm"), "__version__", "unknown"),
                mlcommons_revision=mlc_revision,
                model_checkpoint="nvidia/DeepSeek-R1-FP4@4bedb8a695a119b1a38d16a675c4665e58708aea",
                official_accuracy_targets=loadgen.model_config["accuracy-target"]["deepseek-r1"])


def command(target, phase, foreground=False):
    args = BASE_ARGS + f" --test_mode={phase}"
    if foreground:
        args += " --server_in_foreground=True"
    # Assets and existing build links are checked above; avoid recreating links
    # to container-external scratch paths during the offline run.
    return ["make", "--assume-old=link_dirs", target, "ENV=release", "NO_DISPLAY_RESULTS=1", "RUN_ARGS=" + args]


def stop(proc):
    try:
        os.killpg(proc.pid, signal.SIGTERM)
        proc.wait(timeout=30)
    except ProcessLookupError:
        pass
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid, signal.SIGKILL)
        proc.wait()


def run_phase(workspace, out, phase, startup_timeout, test_timeout):
    folder = out / ("deepseek-performance" if phase == "PerformanceOnly" else "deepseek-accuracy")
    folder.mkdir()
    env = {**os.environ, "SYSTEM_NAME": SYSTEM, "LOG_DIR": str(folder),
           "MODEL_DIR": str(workspace / "build/models"), "DATA_DIR": str(workspace / "build/data"),
           "PREPROCESSED_DATA_DIR": str(workspace / "build/preprocessed_data"),
           "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}
    endpoint = "http://127.0.0.1:30000/health"
    try:
        with urllib.request.urlopen(endpoint, timeout=2):
            raise RuntimeError("Endpoint already running; require a fresh server for each phase")
    except (OSError, urllib.error.URLError):
        pass
    with (folder / "server.log").open("w", encoding="utf-8") as server_log:
        proc = subprocess.Popen(command("run_llm_server", phase, True), cwd=workspace, env=env,
                                stdout=server_log, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            deadline = time.monotonic() + startup_timeout
            while True:
                if proc.poll() is not None:
                    raise RuntimeError(f"DeepSeek endpoint exited ({proc.returncode}); inspect server.log")
                try:
                    with urllib.request.urlopen(endpoint, timeout=5) as response:
                        if response.status == 200:
                            break
                except (OSError, urllib.error.URLError):
                    pass
                if time.monotonic() >= deadline:
                    raise TimeoutError("DeepSeek endpoint startup timed out")
                time.sleep(5)
            with (folder / "harness.log").open("w", encoding="utf-8") as stream:
                subprocess.run(command("run_harness", phase), cwd=workspace, env=env,
                               stdout=stream, stderr=subprocess.STDOUT, timeout=test_timeout, check=True)
        finally:
            stop(proc)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=["check", "run"])
    p.add_argument("--workspace", type=Path, required=True)
    p.add_argument("--out", type=Path)
    p.add_argument("--startup-timeout", type=int, default=7200)
    p.add_argument("--test-timeout", type=int, default=43200)
    args = p.parse_args()
    if args.action == "check":
        print(json.dumps(check(args.workspace), indent=2))
        return 0
    if not args.out or min(args.startup_timeout, args.test_timeout) <= 0:
        p.error("run requires --out and positive timeouts")
    out, workspace = args.out.resolve(), args.workspace.resolve()
    out.mkdir(parents=True, exist_ok=True)
    if any((out / name).exists() for name in ("deepseek-mlperf.json", "deepseek-performance", "deepseek-accuracy")):
        p.error("Use a new DeepSeek result directory")
    result = dict(schema=1, status="FAIL", origin="local_run",
                  time_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    try:
        result["context"] = check(workspace)
        for phase in ("PerformanceOnly", "AccuracyOnly"):
            print(f"DeepSeek-R1 MLPerf v6.0 Offline: {phase}", flush=True)
            run_phase(workspace, out, phase, args.startup_timeout, args.test_timeout)
        files = [only_file(out / "deepseek-performance", name) for name in ("mlperf_log_summary.txt", "mlperf_log_detail.txt")]
        files += [only_file(out / "deepseek-accuracy", name) for name in ("accuracy.txt", "mlperf_log_accuracy.json", "mlperf_log_summary.txt")]
        result.update(status="PASS", evidence_sha256={p.relative_to(out).as_posix(): sha(p) for p in files})
        write_json(out / "deepseek-mlperf.json", result)
        validated = load_evidence(out)
        result.update(performance=validated["performance"], accuracy=validated["accuracy"])
    except Exception as exc:
        result.update(status="FAIL", error=str(exc))
    write_json(out / "deepseek-mlperf.json", result)
    print(json.dumps({k: result[k] for k in ("status", "error", "performance", "accuracy") if k in result}, ensure_ascii=False), flush=True)
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
