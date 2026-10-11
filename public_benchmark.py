#!/usr/bin/env python3
"""Run/import official nccl-tests and compare against versioned public data (offline)."""
import argparse
import hashlib
import html
import json
import math
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
BASELINES = ROOT / "public_baselines.json"
PARAMS = ["-b", "512k", "-e", "8G", "-f", "2", "-g", "8", "-t", "1",
          "-n", "20", "-w", "1", "-m", "1", "-c", "1", "-G", "0", "-d", "float", "-o", "sum"]
HEADER = re.compile(r"nThread (\d+) nGpus (\d+) minBytes (\d+) maxBytes (\d+) "
                    r"step: (\d+)\(factor\) warmup iters: (\d+) iters: (\d+) "
                    r"agg iters: (\d+) validation: (\d+) graph: (\d+)")
HEADER_KEYS = ("threads", "gpus_per_thread", "min_bytes", "max_bytes", "step_factor",
               "warmup", "iterations", "aggregation", "validation", "cuda_graph")
DEVICE = re.compile(r"#\s+Rank\s+(\d+)\s+Group\s+(\d+)\s+Pid\s+\d+\s+on\s+(\S+)"
                    r"\s+device\s+(\d+)\s+\[[^\]]+\]\s+(.+)")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def positive(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise ValueError("Metric must be finite and positive")
    return value


def nccl_version(code):
    return f"{code // 10000}.{code % 10000 // 100}.{code % 100}"


def parse_nccl(log):
    """Reject truncated logs, wrong devices/groups, disabled checks, and corrupted data."""
    if len(HEADER.findall(log)) != 1:
        raise ValueError("Expected one nccl-tests run header")
    workload = dict(zip(HEADER_KEYS, map(int, HEADER.search(log).groups())))
    devices = [dict(rank=int(r), group=int(g), host=h, device=int(d), gpu=name.strip())
               for r, g, h, d, name in DEVICE.findall(log)]
    if {d["rank"] for d in devices} != set(range(8)) or len(devices) != 8:
        raise ValueError("Log must identify all eight GPU ranks exactly once")
    if any(not re.search(r"\bB300(?:\b|_)", d["gpu"]) for d in devices):
        raise ValueError("Expected B300 GPUs; GB300 and other models are different systems")
    if len({d["group"] for d in devices}) != 1 or len({d["host"] for d in devices}) != 1:
        raise ValueError("Expected one node and one collective group")
    if len({d["device"] for d in devices}) != 8 or workload["threads"] * workload["gpus_per_thread"] != 8:
        raise ValueError("Expected eight distinct devices in one process")
    workload.update(nodes=1, gpus=8, collective="all_reduce_perf")
    version = re.search(r"nccl-headers=(\d+) nccl-library=(\d+)", log)
    if not version:
        raise ValueError("Missing actual NCCL header/library versions")
    tool = re.search(r"nccl-tests version (\S+)", log)
    concluded = re.search(r"Collective test concluded: (?:\S*/)?all_reduce_perf\b", log)
    started = re.search(r"Collective test starting: (?:\S*/)?all_reduce_perf\b", log)
    bounds = re.findall(r"Out of bounds values\s*:\s*(\d+)\s+(\S+)", log)
    if not started or not concluded or bounds != [("0", "OK")]:
        raise ValueError("Run did not conclude with zero validation errors")
    if workload["validation"] < 1:
        raise ValueError("Correctness validation was disabled")
    rows = []
    for line in log.splitlines():
        fields = line.split()
        if not fields or not fields[0].isdigit():
            continue
        if len(fields) != 13:
            raise ValueError("Malformed NCCL measurement row")
        size, count = int(fields[0]), int(fields[1])
        if fields[2:4] != ["float", "sum"] or count * 4 != size:
            raise ValueError("Expected FP32 SUM with consistent byte count")
        row = dict(size_bytes=size, dtype="float", operation="sum")
        for placement, offset in (("out_of_place", 5), ("in_place", 9)):
            latency, alg, bus = [positive(float(x)) for x in fields[offset:offset + 3]]
            wrong = int(fields[offset + 3])
            if wrong != 0:
                raise ValueError("NCCL data validation failed")
            if abs(bus / alg - 1.75) > 0.025:
                raise ValueError("Bandwidth factor is inconsistent with eight-GPU all-reduce")
            row[placement] = dict(latency_us=latency, algorithm_gbps=alg, bus_gbps=bus, wrong=wrong)
        rows.append(row)
    expected, size = [], workload["min_bytes"]
    if size < 1 or not size <= workload["max_bytes"] <= 64 * 2**30 or workload["step_factor"] < 2:
        raise ValueError("Invalid NCCL size sweep")
    while size <= workload["max_bytes"]:
        expected.append(size)
        size *= workload["step_factor"]
    if [r["size_bytes"] for r in rows] != expected:
        raise ValueError("Missing, duplicate or unexpected message sizes")
    return dict(schema=1, status="PASS", workload=workload, devices=devices, measurements=rows,
                runtime=dict(nccl=nccl_version(int(version[2])), nccl_headers=nccl_version(int(version[1])),
                             nccl_tests=tool[1] if tool else "unknown"))


def validate_baselines(data):
    if data.get("schema") != 1 or not isinstance(data.get("references"), list):
        raise ValueError("Unsupported public baseline schema")
    seen = set()
    for ref in data["references"]:
        if ref["id"] in seen:
            raise ValueError("Duplicate public baseline ID")
        seen.add(ref["id"])
        if ref["system"]["gpus"] != 8 or ref["system"]["nodes"] != 1 or not re.search(r"\bB300\b", ref["system"]["gpu"]):
            raise ValueError("Public reference must be one node with eight B300 GPUs")
        if not ref["source"]["url"].startswith("https://") or not re.fullmatch(r"[0-9a-f]{40}", ref["source"]["revision"]):
            raise ValueError("Public reference requires an HTTPS source and pinned revision")
        if ref["kind"] not in ("nccl-tests", "context"):
            raise ValueError("Unsupported public reference kind")
        if ref["kind"] == "nccl-tests":
            if set(ref["workload"]) != set(HEADER_KEYS) | {"nodes", "gpus", "collective"}:
                raise ValueError("NCCL reference requires every workload field")
            if ref["workload"]["collective"] != "all_reduce_perf":
                raise ValueError("Only all-reduce references are supported")
            if not isinstance(ref["size_bytes"], int) or ref["size_bytes"] <= 0:
                raise ValueError("Reference message byte size is required")
        if not ref["metrics"]:
            raise ValueError("Public reference has no metrics")
        if ref["id"] == "mlperf-6.0-0073-deepseek-r1-offline" and (
                len(ref["metrics"]) != 1 or any(ref["metrics"][0].get(k) != v for k, v in
                dict(key="tokens_per_second", unit="Tokens/s", direction="higher").items())):
            raise ValueError("DeepSeek reference requires whole-system output Tokens/s")
        for metric in ref["metrics"]:
            positive(metric["value"])
            if metric["direction"] not in ("higher", "lower"):
                raise ValueError("Metric direction is required")
            if ref["kind"] == "nccl-tests":
                units = dict(latency_us=("us", "lower"), algorithm_gbps=("GB/s", "higher"), bus_gbps=("GB/s", "higher"))
                if metric["placement"] not in ("in_place", "out_of_place") or units.get(metric["key"]) != (metric["unit"], metric["direction"]):
                    raise ValueError("NCCL reference metric/unit/direction mismatch")
    return data


def compare(out, baselines=None):
    out = Path(out)
    path = Path(baselines) if baselines else out / "public-baselines.json"
    if not path.exists() and not baselines:
        path = BASELINES
    data = validate_baselines(read_json(path))
    measured = None
    measurement_error = None
    import deepseek_mlperf
    deepseek, deepseek_error = None, None
    try:
        deepseek = deepseek_mlperf.load_evidence(out)
    except Exception as exc:
        deepseek_error = str(exc)
    # Reparse original evidence instead of trusting cached performance numbers.
    if (out / "nccl-public.json").exists():
        try:
            cached = read_json(out / "nccl-public.json")
            raw = (out / "nccl-public.log").read_bytes()
            if cached.get("status") != "PASS" or cached.get("log_sha256") != hashlib.sha256(raw).hexdigest():
                raise ValueError("Failed test or raw log checksum mismatch")
            measured = parse_nccl(raw.decode("utf-8", errors="replace"))
            for key in ("cuda", "torch", "nccl_tests_revision", "binary_sha256"):
                if key in cached.get("runtime", {}):
                    measured["runtime"][key] = cached["runtime"][key]
        except Exception as exc:
            measurement_error = str(exc)
    comparisons = []
    for ref in data["references"]:
        mismatches, differences = [], []
        status = "REFERENCE_ONLY" if ref["kind"] == "context" else "NOT_RUN"
        reason = ref.get("note", "")
        found = None
        is_deepseek = ref["id"] == deepseek_mlperf.REFERENCE_ID
        if is_deepseek and deepseek_error:
            status, reason = "INVALID", deepseek_error
        elif is_deepseek and deepseek:
            for key in ("suite", "model", "precision", "scenario"):
                if ref["workload"].get(key) != deepseek["context"].get(key):
                    mismatches.append(f"{key}: 本次={deepseek['context'].get(key)}，公测={ref['workload'].get(key)}")
            if ref["workload"].get("metric_scope") != "whole_system":
                mismatches.append("公测指标不是整机吞吐")
            status = "CONFIG_MISMATCH" if mismatches else "WORKLOAD_MATCHED"
            found = deepseek["performance"]
            reason = deepseek["note"]
            differences += [f"本次准确率 exact_match={deepseek['accuracy']['exact_match']:.4f}；输出长度均值={deepseek['accuracy']['tokens_per_sample']:.4f}；样本数=4388。",
                            "FP4 权重采用官方固定版本；公测权重具体版本、CPU、固件、软件、功耗及拓扑仍需核对。",
                            "此对比是本地性能参考，未执行 MLPerf 审计与正式提交认证。"]
        elif ref["kind"] == "nccl-tests" and measurement_error:
            status, reason = "INVALID", measurement_error
        elif ref["kind"] == "nccl-tests" and measured:
            mismatches = [f"{k}: 本次={measured['workload'].get(k)}，公测={v}"
                          for k, v in ref["workload"].items() if measured["workload"].get(k) != v]
            found = next((r for r in measured["measurements"] if r["size_bytes"] == ref["size_bytes"]), None)
            if not found:
                mismatches.append("缺少对应消息大小")
            for key, value in ref["runtime"].items():
                actual = measured["runtime"].get(key, "unknown")
                if value is None or actual != value:
                    differences.append(f"{key}: 本次={actual}，公测={value if value is not None else '未披露'}")
            differences.append("服务器 CPU、固件、拓扑、功耗及散热配置需结合硬件快照人工核对。")
            status = "CONFIG_MISMATCH" if mismatches else "WORKLOAD_MATCHED"
            reason = "测试口径一致；百分比用于性能参考，软件/硬件差异仍需核对。" if not mismatches else "测试口径不同，不计算百分比。"
        metrics = []
        for m in ref["metrics"]:
            actual = None
            if found and status == "WORKLOAD_MATCHED":
                actual = found[m["key"]] if is_deepseek else found[m["placement"]][m["key"]]
            metrics.append({**m, "measured": actual,
                            "relative_percent": actual / m["value"] * 100 if actual is not None else None,
                            "delta_percent": (actual / m["value"] - 1) * 100 if actual is not None else None})
        comparisons.append(dict(reference_id=ref["id"], kind=ref["kind"], title=ref["title"], source=ref["source"],
                                system=ref["system"], workload=ref["workload"], reference_runtime=ref.get("runtime", {}),
                                status=status, reason=reason, mismatches=mismatches, differences=differences, metrics=metrics))
    return dict(schema=1, baselines_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                measurement_error=measurement_error or deepseek_error, deepseek_error=deepseek_error,
                measured_deepseek=deepseek, measured_runtime=measured["runtime"] if measured else None,
                comparisons=comparisons,
                note="百分比=本次值/公测值×100；差距=(本次值/公测值−1)×100。延迟越低越好，带宽/吞吐越高越好。公测差距不参与硬件 PASS/FAIL。")


def render(comparison):
    esc = lambda x: html.escape(str(x), quote=True)
    labels = dict(WORKLOAD_MATCHED="口径一致，配置差异待核对", REFERENCE_ONLY="仅供参考：当前负载不可比",
                  NOT_RUN="未运行公测对比负载", INVALID="结果证据无效", CONFIG_MISMATCH="测试口径不一致")
    pieces = ["<section><h2>8×B300 公测对比</h2><p>" + esc(comparison["note"]) + "</p>"]
    if comparison.get("error"):
        pieces.append("<p>公测基线不可用：" + esc(comparison["error"]) + "</p>")
    if comparison.get("measured_deepseek"):
        pieces.append("<details><summary>本次 DeepSeek-R1 配置及准确率证据</summary><pre>" +
                      esc(json.dumps(comparison["measured_deepseek"], indent=2, ensure_ascii=False)) + "</pre></details>")
    for ref in comparison["comparisons"]:
        source = ref["source"]
        pieces.append(f'<h3>{esc(ref["title"])}</h3><p>{esc(labels[ref["status"]])}。{esc(ref["reason"])}</p>'
                      f'<p><a href="{esc(source["url"])}">{esc(source["publisher"])}</a> · 核实日期 {esc(source["verified_on"])}'
                      f' · 源版本 {esc(source["revision"][:12])}</p>')
        pieces.append("<table><tr><th>指标</th><th>公测值</th><th>本次值</th><th>本次/公测</th><th>差距</th></tr>")
        for m in ref["metrics"]:
            actual = "—" if m["measured"] is None else f'{m["measured"]:,.2f}'
            ratio = "—" if m["relative_percent"] is None else f'{m["relative_percent"]:.1f}%'
            delta = "—" if m["delta_percent"] is None else f'{m["delta_percent"]:+.1f}%'
            pieces.append(f'<tr><td>{esc(m["label"])} ({esc(m["unit"])})</td><td>{m["value"]:,.2f}</td>'
                          f'<td>{actual}</td><td>{ratio}</td><td>{delta}</td></tr>')
        pieces.append("</table><ul>" + "".join("<li>" + esc(x) + "</li>" for x in ref["mismatches"] + ref["differences"]) + "</ul>")
        pieces.append("<details><summary>公测硬件、软件及测试口径</summary><pre>" +
                      esc(json.dumps({k: ref[k] for k in ("system", "workload", "reference_runtime")}, indent=2, ensure_ascii=False)) + "</pre></details>")
    pieces.append("</section>")
    return "".join(pieces)


def capture(out, log, runtime=None):
    raw = log.read_bytes()
    try:
        result = parse_nccl(raw.decode("utf-8", errors="replace"))
        result["runtime"].update(runtime or {})
    except Exception as exc:
        result = dict(schema=1, status="FAIL", error=str(exc))
    result.update(log_sha256=hashlib.sha256(raw).hexdigest(),
                  time_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    write_json(out / "nccl-public.json", result)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="action", required=True)
    for name in ("run", "import-nccl", "compare"):
        parser = sub.add_parser(name)
        parser.add_argument("--out", type=Path, required=True)
        if name == "run":
            parser.add_argument("--binary", default="/opt/nccl-tests/build/all_reduce_perf")
            parser.add_argument("--timeout", type=int, default=1800)
        elif name == "import-nccl":
            parser.add_argument("--log", type=Path, required=True)
        else:
            parser.add_argument("--baselines", type=Path)
    args = p.parse_args()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    if args.action == "compare":
        data = compare(out, args.baselines)
        write_json(out / "public-comparison.json", data)
        page = '<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>8×B300 公测对比</title>' \
               '<style>body{max-width:1100px;margin:32px auto;padding:20px;font:16px system-ui}' \
               'table{border-collapse:collapse;width:100%}td,th{padding:10px;text-align:left;border-bottom:1px solid #ddd}' \
               'pre{white-space:pre-wrap;word-break:break-word}</style>' + render(data) + '</html>'
        (out / "public-comparison.html").write_text(page, encoding="utf-8")
        print(out / "public-comparison.html")
        return 1 if data["measurement_error"] else 0
    # Refuse accidental replacement of evidence from an earlier run.
    if (out / "nccl-public.json").exists() or (out / "nccl-public.log").exists():
        p.error("Use a directory without previous nccl-public evidence")
    shutil.copyfile(BASELINES, out / "public-baselines.json")
    log = out / "nccl-public.log"
    rc, error, runtime = 0, None, {}
    if args.action == "import-nccl":
        shutil.copyfile(args.log, log)
    else:
        if args.timeout <= 0:
            p.error("--timeout must be positive")
        command = [args.binary] + PARAMS
        runtime_path = Path("/opt/nccl-tests/build-manifest.json")
        if runtime_path.exists():
            runtime.update(read_json(runtime_path))
        print("Running eight-GPU public workload: " + " ".join(command), flush=True)
        try:
            with log.open("w", encoding="utf-8") as stream:
                rc = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT,
                                    timeout=args.timeout, check=False).returncode
        except Exception as exc:
            rc, error = 1, str(exc)
    result = capture(out, log, runtime)
    result["origin"] = "imported_log" if args.action == "import-nccl" else "local_run"
    if args.action == "run":
        result.update(command=command, exit_code=rc)
    if rc:
        result.update(status="FAIL", error=error or f"nccl-tests exit code {rc}")
    write_json(out / "nccl-public.json", result)
    print(json.dumps(result, ensure_ascii=False), flush=True)
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
