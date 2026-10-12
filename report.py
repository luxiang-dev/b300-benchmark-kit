#!/usr/bin/env python3
"""Generate a portable HTML/JSON report; missing evidence never becomes PASS."""
import csv
from datetime import datetime
import html
import json
import math
from pathlib import Path
import re
import sys
import xml.etree.ElementTree as ET
import public_benchmark


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def ecc_snapshot(path):
    result = {}
    for gpu in ET.parse(path).getroot().findall("gpu"):
        uuid = gpu.findtext("uuid") or gpu.get("id")
        counters = {}
        volatile = gpu.find("ecc_errors/volatile")
        if volatile is not None:
            def walk(node, prefix=""):
                key = prefix + "/" + node.tag
                if len(node):
                    for child in node:
                        walk(child, key)
                elif "uncorrectable" in key or "double_bit" in key:
                    try:
                        counters[key] = int(node.text)
                    except (ValueError, TypeError):
                        pass
            walk(volatile)
        result[uuid] = counters
    return result


def hardware_issues(out):
    failures, review = [], []
    try:
        before, after = ecc_snapshot(out / "before.xml"), ecc_snapshot(out / "after.xml")
        if set(before) != set(after) or len(after) != 8:
            failures.append("测试前后 GPU 列表发生变化，或 GPU 数量不是 8。")
        for uuid, counters in after.items():
            if not counters:
                review.append(f"{uuid}: 无法读取不可纠正 ECC 计数。")
            for key, value in counters.items():
                baseline = before.get(uuid, {}).get(key)
                if baseline is None:
                    review.append(f"{uuid}: {key} 缺少测试前基线。")
                elif value > baseline:
                    failures.append(f"{uuid}: {key} 增加 {value - baseline}。")
                elif value < baseline:
                    review.append(f"{uuid}: {key} 计数下降，可能发生重置。")
                elif value > 0:
                    review.append(f"{uuid}: {key} 存在历史不可纠正错误 {value}。")
        for gpu in ET.parse(out / "after.xml").getroot().findall("gpu"):
            pending = gpu.findtext("retired_pages/pending_retirement")
            if pending in ("Yes", "Pending"):
                failures.append(f"{gpu.findtext('uuid')}: 有显存页等待退役。")
            remapping = gpu.findtext("remapped_rows/remapping_failure_occurred")
            if remapping == "Yes":
                failures.append(f"{gpu.findtext('uuid')}: 显存行重映射失败。")
    except Exception as exc:
        review.append(f"硬件快照不可用: {exc}")
    kernel = out / "kernel-during.log"
    logs = kernel.read_text(encoding="utf-8", errors="replace") if kernel.exists() else ""
    if not logs.strip() or "-- No entries --" in logs:
        review.append("未获取有效内核日志，Xid 检查需要人工复核。")
    if re.search(r"NVRM.*Xid|Xid \(", logs, re.IGNORECASE):
        failures.append("测试期间内核日志出现 NVIDIA Xid，检查 kernel-during.log。")
    if (out / "telemetry.stderr").exists() and (out / "telemetry.stderr").stat().st_size:
        review.append("GPU 遥测出现读取错误，检查 telemetry.stderr。")
    return failures, review


def telemetry_evidence(out, plan):
    failures, review, rows = [], [], []
    by_gpu, invalid = {}, 0
    try:
        expected = set(ecc_snapshot(out / "before.xml"))
        if len(expected) != plan["gpus"]:
            raise ValueError("GPU UUID snapshot is incomplete")
        with (out / "telemetry.csv").open(encoding="utf-8") as stream:
            for raw in csv.DictReader(stream, skipinitialspace=True):
                try:
                    uuid, rank = raw["uuid"].strip(), int(raw["index"])
                    row = {k: float(raw[k]) for k in ("temperature_c", "power_w", "memory_used_mib", "gpu_util_percent")}
                    timestamp = datetime.strptime(raw["timestamp"].strip(), "%Y/%m/%d %H:%M:%S.%f").timestamp()
                    if uuid not in expected or not 0 <= rank < plan["gpus"] or not all(math.isfinite(v) for v in row.values()):
                        raise ValueError("Invalid UUID, index or non-finite telemetry")
                    if not 0 < row["temperature_c"] < 120 or min(row["power_w"], row["memory_used_mib"]) < 0 or not 0 <= row["gpu_util_percent"] <= 100:
                        raise ValueError("Out-of-range telemetry")
                    rows.append(row)
                    by_gpu.setdefault(uuid, []).append(dict(row, index=rank, timestamp=timestamp))
                except (ValueError, KeyError, TypeError):
                    invalid += 1
        if set(by_gpu) != expected:
            review.append("遥测未覆盖全部预期 GPU UUID。")
        indices = {uuid: {r["index"] for r in samples} for uuid, samples in by_gpu.items()}
        if any(len(v) != 1 for v in indices.values()) or len({next(iter(v)) for v in indices.values()}) != len(indices):
            review.append("遥测 GPU UUID 与索引映射不一致。")
        window = load_json(out / "telemetry-window.json")
        start, end = float(window["start_epoch"]), float(window["end_epoch"])
        if not all(math.isfinite(v) for v in (start, end)) or end < start:
            raise ValueError("Invalid telemetry collection window")
        for uuid, samples in by_gpu.items():
            times = [s["timestamp"] for s in samples]
            if len(times) < 2 or any(b <= a or b - a > 30 for a, b in zip(times, times[1:])) or abs(times[0] - start) > 30 or abs(end - times[-1]) > 30:
                review.append(f"{uuid}: 遥测采样不足、存在超过 30 秒的间断或未覆盖测试窗口。")
        if invalid:
            review.append(f"遥测有 {invalid} 行无效数据。")
    except Exception as exc:
        review.append(f"遥测证据不完整: {exc}")
    limit = float(plan.get("max_gpu_temp", 85))
    for uuid, samples in by_gpu.items():
        if max(s["temperature_c"] for s in samples) >= limit:
            failures.append(f"{uuid}: 遥测温度达到停止阈值 {limit:g}°C。")
    summary = dict(samples=len(rows), invalid_samples=invalid,
                   peak_temperature_c=max((x["temperature_c"] for x in rows), default=None),
                   peak_power_w=max((x["power_w"] for x in rows), default=None),
                   per_gpu={uuid: dict(samples=len(samples), peak_temperature_c=max(r["temperature_c"] for r in samples),
                                      peak_power_w=max(r["power_w"] for r in samples)) for uuid, samples in by_gpu.items()})
    return summary, failures, review


def generate(out):
    plan = load_json(out / "plan.json")
    stages = []
    path = out / "stages.tsv"
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            name, status, code = line.split("\t")
            stages.append(dict(name=name, status=status, exit_code=int(code)))
    results, parse_errors = [], []
    for path in sorted(out.glob("*-gpu*.json")):
        try:
            result = load_json(path)
            if not isinstance(result, dict) or result.get("status") not in ("PASS", "FAIL") or not isinstance(result.get("rank"), int) or result.get("test") not in ("environment", "stress", "nccl", "llm", "image"):
                raise ValueError("Invalid GPU result schema")
            result["file"] = path.name
            results.append(result)
        except Exception as exc:
            parse_errors.append(f"{path.name}: {exc}")
    by_name = {x["name"]: x for x in stages}
    incomplete = []
    if plan["profile"] == "inventory":
        incomplete.append("仅采集清单，未运行压力、通信或模型测试。")
    for label in plan["required"]:
        stage = by_name.get(label)
        if stage is None or stage["status"] == "SKIP":
            incomplete.append(f"{label}: 未运行或跳过。")
        if label not in ("dcgm", "nccl-public", "deepseek-r1-mlperf"):
            matches = [x for x in results if x["test"] + ("-" + Path(x["model"]).name if x.get("model") else "") == label]
            ranks = {x["rank"] for x in matches if x.get("status") == "PASS"}
            if ranks != set(range(plan["gpus"])):
                incomplete.append(f"{label}: 成功结果未覆盖全部 {plan['gpus']} 张 GPU。")
    failures, review = hardware_issues(out)
    try:
        public = public_benchmark.compare(out)
        if public["measurement_error"]:
            failures.append("公测对比结果证据无效: " + public["measurement_error"])
        if "nccl-public" in plan["required"] and not any(
                x["kind"] == "nccl-tests" and x["status"] == "WORKLOAD_MATCHED" for x in public["comparisons"]):
            incomplete.append("nccl-public: 缺少有效且口径一致的 8 卡公测对比结果。")
        if "deepseek-r1-mlperf" in plan["required"] and not any(
                x["reference_id"] == "mlperf-6.0-0073-deepseek-r1-offline" and x["status"] == "WORKLOAD_MATCHED" for x in public["comparisons"]):
            incomplete.append("deepseek-r1-mlperf: 缺少有效的 Offline 性能或完整准确率证据。")
    except Exception as exc:
        public = dict(schema=1, error=str(exc), comparisons=[], note="公测参考不可用。")
        review.append("公测基线无法解析: " + str(exc))
        for label in ("nccl-public", "deepseek-r1-mlperf"):
            if label in plan["required"]:
                incomplete.append(label + ": 无法完成公测对比。")
    public_benchmark.write_json(out / "public-comparison.json", public)
    if by_name.get("dcgm", {}).get("status") == "PASS":
        try:
            # DCGM JSON schemas vary by release; inspect status fields recursively.
            raw = (out / "dcgm.json").read_text(encoding="utf-8", errors="replace")
            dcgm = json.JSONDecoder().raw_decode(raw[raw.index("{"):])[0]
            statuses = []
            def walk(node):
                if isinstance(node, dict):
                    for key, value in node.items():
                        if key.lower() == "status" and isinstance(value, str):
                            statuses.append(value.lower())
                        else:
                            walk(value)
                elif isinstance(node, list):
                    for value in node:
                        walk(value)
            walk(dcgm)
            if any(x in ("fail", "failed", "error") for x in statuses):
                failures.append("DCGM JSON 内含失败结果，检查 dcgm.json。")
            if not any(x == "pass" for x in statuses):
                incomplete.append("DCGM JSON 没有可识别的通过结果。")
            if any(x != "pass" for x in statuses):
                review.append("DCGM 存在跳过、警告或无法识别的状态，检查 dcgm.json。")
        except Exception as exc:
            incomplete.append(f"DCGM JSON 无法解析: {exc}")
    failures += parse_errors
    failures += [f"{x['name']}: 退出码 {x['exit_code']}" for x in stages if x["status"] == "FAIL"]
    failures += [f"{x['file']}: {x.get('error', '测试失败')}" for x in results if x.get("status") != "PASS"]
    telemetry, tfailures, treview = telemetry_evidence(out, plan)
    failures += tfailures
    review += treview
    overall = "FAIL" if failures else "INCOMPLETE" if incomplete else "REVIEW" if review else "PASS"
    summary = dict(overall=overall, profile=plan["profile"], expected_gpus=plan["gpus"], stages=stages,
                   failures=failures, incomplete=incomplete, review=review, results=results, public_comparison=public,
                   telemetry=telemetry)
    (out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    esc = lambda value: html.escape(str(value))
    table = "".join(f"<tr><td>{esc(x['name'])}</td><td>{esc(x['status'])}</td><td>{x['exit_code']}</td></tr>" for x in stages)
    details = "".join(f"<details><summary>{esc(x['test'])} · GPU {x['rank']} · {esc(x['status'])}</summary><pre>{esc(json.dumps(x, indent=2, ensure_ascii=False))}</pre></details>" for x in results)
    images = "".join(f'<a href="{esc(p.name)}"><img src="{esc(p.name)}" alt="{esc(p.name)}"></a>' for p in sorted(out.glob("sdxl-gpu*.png")))
    notes = "".join(f"<li>{esc(x)}</li>" for x in failures + incomplete + review)
    explanation = ("DeepSeek-R1 使用完整 FP4 模型、TP8/EP8，显示 MLPerf v6.0 Offline 整机输出 Tokens/s。"
                   "性能与准确率各启动新的 endpoint；准确率响应、长度和原始日志须通过验证。公测差距仅为性能参考，本地复现未完成审计或认证。"
                   if plan["profile"] == "deepseek" else
                   "LLM 是 8 个独立副本，吞吐包含预填充与解码，未测首 Token 延迟，也未使用张量并行。SDXL 按每卡记录生成时间。"
                   "GEMM 是 BF16 合成负载；PyTorch NCCL 使用 MAX 性能测量与 SUM 正确性检查。nccl-public 单独使用官方 FP32 SUM all_reduce_perf 口径。"
                   "带宽不能直接证明流量走了 NVLink；应对照同型号、同软件、同功耗配置的基线。")
    page = f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>B300 8卡测试报告</title><style>body{{max-width:1100px;margin:32px auto;padding:0 20px;font:16px system-ui;color:#172538;background:#f3f6fa}}section,details{{background:white;padding:18px;margin:16px 0;border-radius:10px}}table{{border-collapse:collapse;width:100%}}td,th{{padding:10px;text-align:left;border-bottom:1px solid #ddd}}pre{{white-space:pre-wrap;word-break:break-word;font-size:13px}}img{{width:200px;margin:6px;border-radius:8px}}.status{{font-size:30px;font-weight:bold}}</style>
<h1>NVIDIA B300 · 8卡测试报告</h1><section><div class="status">{overall}</div><p>模式：{esc(plan['profile'])}；预期 GPU：{plan['gpus']} 张。</p>
<p>PASS 仅表示本次已运行项目通过。短时测试不能证明长期稳定性，也不能替代厂商验收或离线现场诊断。</p><ul>{notes}</ul></section>
<section><h2>测试阶段</h2><table><tr><th>项目</th><th>结果</th><th>退出码</th></tr>{table}</table></section>
{public_benchmark.render(public)}
<section><h2>遥测</h2><pre>{esc(json.dumps(summary['telemetry'],indent=2,ensure_ascii=False))}</pre><p>完整采样见 telemetry.csv；PCIe/NVLink 拓扑见 topology.txt 和 nvlink.txt。</p></section>
<section><h2>图像输出</h2>{images}</section><h2>每卡结果与模型输出</h2>{details}
<section><h2>指标解释</h2><p>{esc(explanation)}</p></section></html>'''
    (out / "report.html").write_text(page, encoding="utf-8")
    return summary


if __name__ == "__main__":
    status = generate(Path(sys.argv[1]).resolve())["overall"]
    print(status)
    sys.exit(0 if status == "PASS" else 1 if status == "FAIL" else 2)
