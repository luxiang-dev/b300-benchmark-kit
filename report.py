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


def hardware_issues(out, expected_gpus=8, strict=False):
    failures, review = [], []
    try:
        before, after = ecc_snapshot(out / "before.xml"), ecc_snapshot(out / "after.xml")
        if set(before) != set(after) or len(after) != expected_gpus:
            failures.append(f"测试前后 GPU 列表发生变化，或 GPU 数量不是 {expected_gpus}。")
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
    if strict:
        try:
            record = load_json(out / 'collection-status.json')['kernel-during.log']
            stderr = (out / 'kernel-during.log.stderr').read_text(encoding='utf-8')
            argv = record['argv']
            if record.get('returncode') != 0 or record.get('error') or stderr.strip() or argv[:2] != ['journalctl', '-k'] or '--since' not in argv:
                review.append('kernel-during.log: 内核日志读取有权限/退出状态/窗口问题，不能证明没有 Xid/RAS。')
            else:
                expected_start = float(load_json(out / 'plan.json')['start_epoch'])
                observed_start = float(argv[argv.index('--since')+1].removeprefix('@'))
                if not math.isfinite(expected_start) or not math.isfinite(observed_start) or abs(observed_start-expected_start) > 2:
                    review.append('kernel-during.log: 采集窗口与本次运行开始时间不一致。')
        except (OSError, ValueError, KeyError, TypeError, IndexError):
            review.append('kernel-during.log: 无法确认本次内核日志读取完整性。')
    if not logs.strip() or "-- No entries --" in logs:
        try:
            collected = load_json(out / 'collection-status.json')['kernel-during.log']
            stderr = (out / 'kernel-during.log.stderr').read_text(encoding='utf-8')
            proven_empty = collected['returncode'] == 0 and collected.get('error') is None and not stderr.strip() and collected['argv'][:2] == ['journalctl', '-k']
        except (OSError, ValueError, KeyError, TypeError):
            proven_empty = False
        if not proven_empty:
            review.append("未获取有效内核日志，Xid 检查需要人工复核。")
    if re.search(r"NVRM.*Xid|Xid \(", logs, re.IGNORECASE):
        failures.append("测试期间内核日志出现 NVIDIA Xid，检查 kernel-during.log。")
    if re.search(r'(?:EDAC|MCE|Hardware Error|AER|PCIe).*?(?:uncorrectable|uncorrected|fatal error)', logs, re.IGNORECASE):
        failures.append('测试期间有主机内存/CPU/PCIe 不可纠正或致命硬件错误，检查 kernel-during.log。')
    elif re.search(r'Hardware Error|EDAC.*(?:CE|error)|AER.*error', logs, re.IGNORECASE):
        review.append('测试期间有主机 RAS/PCIe 错误记录，需 OEM 复核。')
    if (out / "telemetry.stderr").exists() and (out / "telemetry.stderr").stat().st_size:
        review.append("GPU 遥测出现读取错误，检查 telemetry.stderr。")
    import hardware_health
    extra_failures, extra_review = hardware_health.evaluate(out, strict)
    return list(dict.fromkeys(failures + extra_failures)), list(dict.fromkeys(review + extra_review))


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
    stage_errors = []
    path = out / "stages.tsv"
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                name, status, code = line.split("\t")
                code = int(code)
                if status not in ('RUNNING', 'PASS', 'FAIL', 'SKIP') or (status in ('PASS', 'SKIP', 'RUNNING') and code != 0):
                    raise ValueError('Inconsistent stage status/exit code')
                stages.append(dict(name=name, status=status, exit_code=code))
            except ValueError as error:
                stage_errors.append(f'无效阶段记录: {line}: {error}')
    results, parse_errors = [], []
    for path in sorted(out.glob("*-gpu*.json")):
        if path.name.startswith('identity-gpu'):
            continue
        try:
            result = load_json(path)
            if not isinstance(result, dict) or result.get("status") not in ("PASS", "FAIL") or type(result.get("rank")) is not int or not 0 <= result['rank'] < plan['gpus'] or result.get("test") not in ("environment", "stress", "memory", "nccl", "llm", "image"):
                raise ValueError("Invalid GPU result schema")
            result["file"] = path.name
            if result['status'] == 'PASS' and plan.get('result_contract') == 2:
                import gpu_evidence
                result['_verification_review'] = gpu_evidence.validate_gpu_result(result, plan, out)
            elif result['test'] == 'memory' and result['status'] == 'PASS':
                from memory_probe import validate_evidence
                validate_evidence(result)
            results.append(result)
        except Exception as exc:
            parse_errors.append(f"{path.name}: {exc}")
    by_name = {x["name"]: x for x in stages}
    incomplete = []
    if plan["profile"] == "inventory":
        incomplete.append("仅采集清单，未运行压力、通信或模型测试。")
    for label in plan["required"]:
        stage = by_name.get(label)
        if stage is None or stage["status"] in ("SKIP", "RUNNING"):
            incomplete.append(f"{label}: 未运行或跳过。")
        if label not in ("dcgm", "nccl-public", "deepseek-r1-mlperf", "identity", "server", "server-health-before", "server-health-after"):
            matches = [x for x in results if x["test"] + ("-" + Path(x["model"]).name if x.get("model") else "") == label]
            ranks = {x["rank"] for x in matches if x.get("status") == "PASS"}
            if ranks != set(range(plan["gpus"])):
                incomplete.append(f"{label}: 成功结果未覆盖全部 {plan['gpus']} 张 GPU。")
            if plan.get('result_contract') == 2:
                passing = [x for x in matches if x.get('status') == 'PASS']
                physical = set(ecc_snapshot(out / 'before.xml'))
                if len(passing) != plan['gpus'] or {x.get('uuid') for x in passing} != physical:
                    incomplete.append(f'{label}: 本次结果未唯一覆盖全部物理 GPU UUID。')
    failures, review = hardware_issues(out, plan['gpus'], strict=plan.get('result_contract') == 2)
    for result in results:
        review += [result['file'] + ': ' + item for item in result.pop('_verification_review', [])]
    identity = None
    if 'identity' in plan['required']:
        try:
            import authenticity
            identity = authenticity.evaluate_identity(out, plan['gpus'])
            if identity['status'] == 'FAIL':
                failures.append('身份交叉核验出现矛盾，检查 identity-summary.json。')
            elif identity['status'] != 'CONSISTENT':
                review.append('身份核验存在缺失或需复核的证据，检查 identity-summary.json。')
        except Exception as error:
            incomplete.append('身份证据无法复核: ' + str(error))
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
            if plan.get('result_contract') == 2:
                import dcgm_evidence
                review += dcgm_evidence.validate_dcgm_result(dcgm, plan, out)
            if any(x in ("fail", "failed", "error") for x in statuses):
                failures.append("DCGM JSON 内含失败结果，检查 dcgm.json。")
            if not any(x == "pass" for x in statuses):
                incomplete.append("DCGM JSON 没有可识别的通过结果。")
            if any(x != "pass" for x in statuses):
                review.append("DCGM 存在跳过、警告或无法识别的状态，检查 dcgm.json。")
        except Exception as exc:
            incomplete.append(f"DCGM JSON 无法解析: {exc}")
    server = None
    if 'server' in plan['required']:
        try:
            import server_probe
            server = server_probe.read_result(out)
            if server['mode'] != 'full':
                incomplete.append('服务器仅采集清单，未执行 CPU/RAM/结果盘抽查。')
            if server['status'] == 'FAIL':
                failures.append('服务器功能抽查失败，检查 server-summary.json。')
            elif server['status'] != 'PASS':
                review += ['服务器清单/抽查需复核: ' + item for item in server['findings']]
                if any(c['status'] != 'PASS' for c in server['checks'].values()):
                    review.append('服务器有功能抽查跳过，检查 server-summary.json。')
        except Exception as error:
            incomplete.append('服务器证据无效或缺失: ' + str(error))
    failures += parse_errors + stage_errors
    server_health = None
    if 'server-health-before' in plan['required'] or 'server-health-after' in plan['required']:
        try:
            import server_health as health_reader
            server_health = health_reader.evaluate(out)
            failures += ['整机健康: ' + item for item in server_health['failures']]
            review += ['整机健康: ' + item for item in server_health['review']]
        except Exception as error:
            incomplete.append('整机 SMART/BMC 健康证据无效: ' + str(error))
    failures += [f"{x['name']}: 退出码 {x['exit_code']}" for x in stages if x["status"] == "FAIL"]
    failures += [f"{x['file']}: {x.get('error', '测试失败')}" for x in results if x.get("status") != "PASS"]
    telemetry, tfailures, treview = telemetry_evidence(out, plan)
    failures += tfailures
    review += treview
    overall = "FAIL" if failures else "INCOMPLETE" if incomplete else "REVIEW" if review else "PASS"
    summary = dict(overall=overall, profile=plan["profile"], expected_gpus=plan["gpus"], stages=stages,
                   failures=failures, incomplete=incomplete, review=review, results=results, public_comparison=public,
                   telemetry=telemetry, identity_verification=identity, server_verification=server, server_health=server_health)
    import purchase_acceptance
    summary['acceptance'] = purchase_acceptance.evaluate(summary)
    import inventory_summary
    summary['observed_inventory'] = inventory_summary.derive(out)
    (out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    esc = lambda value: html.escape(str(value))
    table = "".join(f"<tr><td>{esc(x['name'])}</td><td>{esc(x['status'])}</td><td>{x['exit_code']}</td></tr>" for x in stages)
    details = "".join(f"<details><summary>{esc(x['test'])} · GPU {x['rank']} · {esc(x['status'])}</summary><pre>{esc(json.dumps(x, indent=2, ensure_ascii=False))}</pre></details>" for x in results)
    images = "".join(f'<a href="{esc(p.name)}"><img src="{esc(p.name)}" alt="{esc(p.name)}"></a>' for p in sorted(out.glob("sdxl-gpu*.png")))
    notes = "".join(f"<li>{esc(x)}</li>" for x in failures + incomplete + review)
    identity_section = ('<section><h2>硬件身份交叉核验</h2><pre>' + esc(json.dumps(identity, indent=2, ensure_ascii=False)) +
                        '</pre><p>CONSISTENT 仅表示本次软件可观察到的身份和计算证据一致；仍需将序列号、板标签、采购记录及 BMC 清单交给厂商核对。</p></section>') if identity else ''
    server_section = ('<section><h2>整机清单与有限功能抽查</h2><pre>' + esc(json.dumps(server, indent=2, ensure_ascii=False)) +
                      '</pre><p>默认 RAM 仅测试 256 MiB、结果盘仅写读 128 MiB 临时文件；不覆盖全部内存/介质，不测网卡远端线速，也不替代 OEM 验收。</p></section>') if server else ''
    if server_health:
        server_section += '<section><h2>物理磁盘 SMART 与 BMC 健康复核</h2><pre>' + esc(json.dumps(server_health, indent=2, ensure_ascii=False)) + '</pre></section>'
    explanation = ("DeepSeek-R1 使用完整 FP4 模型、TP8/EP8，显示 MLPerf v6.0 Offline 整机输出 Tokens/s。"
                   "性能与准确率各启动新的 endpoint；准确率响应、长度和原始日志须通过验证。公测差距仅为性能参考，本地复现未完成审计或认证。"
                   if plan["profile"] == "deepseek" else
                   "LLM 是 8 个独立副本，吞吐包含预填充与解码，未测首 Token 延迟，也未使用张量并行。SDXL 按每卡记录生成时间。"
                   "GEMM 是 BF16 合成负载；PyTorch NCCL 使用 MAX 性能测量与 SUM 正确性检查。nccl-public 单独使用官方 FP32 SUM all_reduce_perf 口径。"
                   "带宽不能直接证明流量走了 NVLink；应对照同型号、同软件、同功耗配置的基线。")
    if plan['profile'] in ('identity', 'authenticity'):
        explanation = ('身份核验交叉检查 PCI、CUDA 架构、UUID、容量和真实计算。HBM 地址模式验证同时分配的实际容量，'
                       '累计读验字节不是显存容量；85%目标覆盖不是全显存覆盖。GEMM/NCCL/DCGM 为功能与稳定性证据，'
                       '不能替代板标签、采购记录和厂商认证。公测性能差距本身不等于假卡。')
    page = f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>B300 {plan['gpus']}卡测试报告</title><style>body{{max-width:1100px;margin:32px auto;padding:0 20px;font:16px system-ui;color:#172538;background:#f3f6fa}}section,details{{background:white;padding:18px;margin:16px 0;border-radius:10px}}table{{border-collapse:collapse;width:100%}}td,th{{padding:10px;text-align:left;border-bottom:1px solid #ddd}}pre{{white-space:pre-wrap;word-break:break-word;font-size:13px}}img{{width:200px;margin:6px;border-radius:8px}}.status{{font-size:30px;font-weight:bold}}</style>
<h1>NVIDIA B300 · {plan['gpus']}卡测试报告</h1><section><div class="status">{overall}</div><p>模式：{esc(plan['profile'])}；预期 GPU：{plan['gpus']} 张。</p>
<p>PASS 仅表示本次已运行项目通过。短时测试不能证明长期稳定性，也不能替代厂商验收或离线现场诊断。</p><ul>{notes}</ul></section>
<section><h2>采购验收状态：{esc(summary['acceptance']['status'])}</h2><p>{esc(summary['acceptance']['statement'])}</p><ul>{''.join('<li>待完成：' + esc(item['title']) + '</li>' for item in summary['acceptance']['pending_checks'])}</ul></section>
<section><h2>测试阶段</h2><table><tr><th>项目</th><th>结果</th><th>退出码</th></tr>{table}</table></section>
{identity_section}
{server_section}
<section><h2>实测配置清单（不是采购承诺）</h2><p>请将 observed-inventory.json 交给卖方核对。缺失字段保留 UNKNOWN；此清单不会自动完成配置验收。</p><pre>{esc(json.dumps(summary['observed_inventory'], indent=2, ensure_ascii=False))}</pre></section>
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
