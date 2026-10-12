"""Strict validation of locally generated GPU evidence, not product certification.

Version 2 results bind each worker to the run and physical inventory. These
checks reject incomplete, stale or internally inconsistent result files. They
cannot authenticate a hostile operating system, firmware or fabricated log.
"""
import hashlib
import json
import math
from pathlib import Path
import re

import authenticity
import memory_probe


def _integer(value, label, minimum=1):
    if type(value) is not int or value < minimum:
        raise ValueError(f'{label}: expected integer >= {minimum}')
    return value


def _number(value, label, minimum=0, positive=True):
    if type(value) not in (int, float) or not math.isfinite(value) or value < minimum or (positive and value == minimum):
        raise ValueError(f'{label}: missing, non-finite or out-of-range number')
    return value


def _equal_number(value, expected, label, rel_tol=1e-6, abs_tol=1e-8):
    _number(value, label, positive=False)
    if not math.isclose(value, expected, rel_tol=rel_tol, abs_tol=abs_tol):
        raise ValueError(f'{label}: does not agree with the measured inputs')


def _temperature(result, plan):
    peak = _number(result.get('peak_temperature_c'), 'peak_temperature_c')
    limit = _number(plan.get('max_gpu_temp', 85), 'plan.max_gpu_temp')
    if peak >= min(limit, 120):
        raise ValueError('peak_temperature_c reaches the configured stop threshold')


def _common(result, plan, out):
    if type(result.get('schema_version')) is not int or result['schema_version'] != 2:
        raise ValueError('schema_version: a version 2 result is required')
    run_id = plan.get('run_id')
    if not isinstance(run_id, str) or not run_id or result.get('run_id') != run_id:
        raise ValueError('run_id: result is not bound to this run')
    count = _integer(plan.get('gpus'), 'plan.gpus')
    rank = _integer(result.get('rank'), 'rank', 0)
    if rank >= count or result.get('world_size') != count or type(result.get('world_size')) is not int:
        raise ValueError('worker rank/world_size does not match the requested GPU count')
    snapshot = Path(out) / 'before.xml'
    if result.get('hardware_snapshot_sha256') != hashlib.sha256(snapshot.read_bytes()).hexdigest():
        raise ValueError('hardware_snapshot_sha256: stale or changed physical inventory')
    inventory = authenticity.parse_smi_xml(snapshot.read_bytes())['gpus']
    uuids = [authenticity.normalize_uuid(gpu['uuid']) for gpu in inventory]
    if len(inventory) != count or None in uuids or len(set(uuids)) != count:
        raise ValueError('physical GPU UUID inventory is incomplete or duplicated')
    gpu_uuid = authenticity.normalize_uuid(result.get('uuid'))
    if gpu_uuid is None or gpu_uuid not in uuids:
        raise ValueError('uuid: worker is not part of this physical GPU inventory')
    gpu = inventory[uuids.index(gpu_uuid)]
    bus = authenticity.normalize_bus(result.get('pci_bus_id'))
    if bus is None or bus != authenticity.normalize_bus(gpu['pci_bus_id']):
        raise ValueError('pci_bus_id: worker and physical GPU disagree')
    if result.get('capability') != authenticity.EXPECTED_CAPABILITY or any(type(x) is not int for x in result['capability']):
        raise ValueError('capability: B300 compute capability must be 10.3')
    total = _integer(result.get('total_memory_bytes'), 'total_memory_bytes')
    nvml = gpu['memory_total_bytes']
    if type(nvml) is not int or nvml <= 0 or abs(total - nvml) > max(2 * 2**30, nvml * .02):
        raise ValueError('total_memory_bytes: CUDA and physical inventory disagree')
    host = Path(out) / 'identity-host.json'
    if 'identity' in plan.get('required', []):
        if not host.is_file() or result.get('identity_host_sha256') != hashlib.sha256(host.read_bytes()).hexdigest():
            raise ValueError('identity_host_sha256: identity snapshot is absent, stale or changed')
        matches = []
        for file in Path(out).glob('identity-gpu*.json'):
            record = json.loads(file.read_text(encoding='utf-8'))
            if authenticity.normalize_uuid(record.get('uuid')) == gpu_uuid:
                matches.append(record)
        if len(matches) != 1:
            raise ValueError('identity CUDA worker mapping is missing or duplicated')
        probe = matches[0]
        if probe.get('host_sha256') != result['identity_host_sha256'] or probe.get('status') != 'PASS':
            raise ValueError('identity CUDA evidence is stale or failed')
        probe_total = probe.get('total_memory_bytes')
        memory_tolerance = 16 * 2**20 if result.get('test') == 'memory' else 0
        if type(probe_total) is not int or abs(probe_total - total) > memory_tolerance or authenticity.normalize_bus(probe.get('pci_bus_id')) != bus or probe.get('capability') != result['capability']:
            raise ValueError('identity CUDA and workload GPU observations disagree')
    elif host.is_file() and result.get('identity_host_sha256') != hashlib.sha256(host.read_bytes()).hexdigest():
        raise ValueError('identity_host_sha256: an existing identity snapshot is not bound')
    return gpu_uuid, total


def _environment(result, total):
    if result.get('bf16_correctness') is not True:
        raise ValueError('environment.bf16_correctness evidence is missing or failed')
    if not re.search(r'\bB300\b', str(result.get('gpu', ''))):
        raise ValueError('environment.gpu does not identify B300')
    _equal_number(result.get('memory_gib'), total / 2**30, 'environment.memory_gib')
    for field in ('torch', 'cuda'):
        if not isinstance(result.get(field), str) or not result[field].strip():
            raise ValueError('environment.' + field + ' version is missing')
    versions = result.get('nccl')
    if not isinstance(versions, list) or len(versions) != 3 or any(type(x) is not int or x < 0 for x in versions) or versions[0] <= 0:
        raise ValueError('environment.nccl version is missing or invalid')


def _stress(result, plan, total):
    reviews = []
    seconds = _number(plan.get('seconds'), 'plan.seconds')
    if result.get('requested_seconds') != seconds or type(result.get('requested_seconds')) not in (int, float):
        raise ValueError('stress.requested_seconds differs from this run plan')
    if _number(result.get('duration_s'), 'stress.duration_s') < seconds:
        raise ValueError('stress.duration_s is shorter than the requested test')
    matrix = _integer(result.get('matrix'), 'stress.matrix', 256)
    if matrix != plan.get('matrix', 8192) or matrix > 16384 or matrix % 256 or result.get('dtype') != 'BF16':
        raise ValueError('stress matrix or dtype differs from the requested workload')
    iterations = _integer(result.get('gemm_iterations'), 'stress.gemm_iterations')
    if iterations % 20:
        raise ValueError('stress.gemm_iterations differs from the complete batch contract')
    _number(result.get('gemm_tflops'), 'stress.gemm_tflops')
    _temperature(result, plan)
    if result.get('memory_patterns') != [17, -31]:
        raise ValueError('stress.memory_patterns is missing or invalid')
    gib = _number(result.get('memory_tested_gib'), 'stress.memory_tested_gib')
    allocated = gib * 2**30
    if not math.isclose(allocated / memory_probe.CHUNK_BYTES, round(allocated / memory_probe.CHUNK_BYTES), abs_tol=1e-8) or allocated > total:
        raise ValueError('stress.memory_tested_gib does not represent bounded complete chunks')
    fraction = _number(plan.get('stress_memory_fraction', .65), 'plan.stress_memory_fraction')
    if not .05 <= fraction <= .8:
        raise ValueError('stress memory fraction is out of range')
    target = int(total * fraction) // memory_probe.CHUNK_BYTES * memory_probe.CHUNK_BYTES
    if allocated > target:
        raise ValueError('stress allocation exceeds the requested capacity')
    if allocated < target:
        reviews.append('stress: 实际 HBM 压力覆盖低于请求目标，检查空闲容量及其他进程。')
    return reviews


def _memory(result, plan, total):
    memory_probe.validate_evidence(result)
    _temperature(result, plan)
    expected = plan.get('memory_fraction', .85)
    if type(expected) not in (int, float) or result['requested_memory_fraction'] != expected:
        raise ValueError('memory.requested_memory_fraction differs from this run plan')
    # CUDA properties and cudaMemGetInfo total should describe the same device;
    # retain a small tolerance for allocation-granularity reporting differences.
    if abs(result['total_memory_bytes'] - total) > 16 * 2**20:
        raise ValueError('memory capacity differs from the bound CUDA device')
    chunk_count = result['allocated_bytes'] // memory_probe.CHUNK_BYTES
    positions = result.get('cpu_reference_positions_per_chunk')
    checks = result.get('cpu_reference_checks_per_pass')
    if type(positions) is not int or positions != 3 or type(checks) is not int or checks != chunk_count * 3:
        raise ValueError('memory: independent CPU reference coverage is missing or inconsistent')
    for phase in result['passes']:
        if type(phase.get('cpu_reference_checks')) is not int or phase['cpu_reference_checks'] != checks:
            raise ValueError('memory: a pattern has incomplete independent CPU reference checks')
    elapsed = sum(phase['write_seconds'] + phase['verify_seconds'] for phase in result['passes'])
    if result['duration_s'] + 1e-6 < elapsed:
        raise ValueError('memory.duration_s is shorter than its measured pattern phases')
    target = int(total * expected) // memory_probe.CHUNK_BYTES * memory_probe.CHUNK_BYTES
    if result['free_memory_limited'] or result['allocated_bytes'] < target:
        return ['memory: 空闲显存限制导致实际 HBM 地址模式覆盖低于请求目标，验货需补测。']
    return []


def _nccl(result, plan):
    count = plan['gpus']
    if type(result.get('iterations')) is not int or result['iterations'] != plan.get('nccl_iterations', 50):
        raise ValueError('nccl.iterations differs from the requested workload')
    measurements = result.get('measurements')
    if not isinstance(measurements, list) or len(measurements) != 3:
        raise ValueError('nccl: all three message sizes must have measurements')
    sizes = []
    for row in measurements:
        if not isinstance(row, dict):
            raise ValueError('nccl measurement is not an object')
        size = _integer(row.get('size_mib'), 'nccl.size_mib')
        sizes.append(size)
        if row.get('operation') != 'MAX' or row.get('sum_correctness') != 'PASS':
            raise ValueError('nccl: measured operation or SUM correctness evidence is invalid')
        seconds = _number(row.get('latency_ms'), 'nccl.latency_ms') / 1000
        alg = size * 2**20 / seconds / 1e9
        _equal_number(row.get('algorithm_gbps'), alg, 'nccl.algorithm_gbps')
        _equal_number(row.get('bus_gbps'), alg * 2 * (count - 1) / count, 'nccl.bus_gbps', abs_tol=1e-8)
    if sorted(sizes) != [16, 64, 256]:
        raise ValueError('nccl: missing, duplicated or unknown message size')


def _latencies(result, label):
    values = result.get('latency_s')
    if not isinstance(values, list) or not values:
        raise ValueError(label + '.latency_s: measurements are missing')
    for value in values:
        _number(value, label + '.latency_s')
    return values


def _model(result, total, out):
    mode = result['test']
    if not isinstance(result.get('model'), str) or not result['model'].strip():
        raise ValueError(mode + ': model identifier is missing')
    allocated = _number(result.get('peak_allocated_gib'), mode + '.peak_allocated_gib')
    if allocated * 2**30 > total:
        raise ValueError(mode + ': peak allocated memory exceeds physical capacity')
    latencies = _latencies(result, mode)
    if mode == 'llm':
        iterations = _integer(result.get('iterations'), 'llm.iterations')
        batch = _integer(result.get('batch'), 'llm.batch')
        tokens = _integer(result.get('forced_output_tokens'), 'llm.forced_output_tokens')
        _integer(result.get('input_tokens_per_request'), 'llm.input_tokens_per_request')
        if len(latencies) != iterations or result.get('dtype') != 'BF16':
            raise ValueError('llm: requested iterations or dtype is inconsistent')
        _equal_number(result.get('tokens_per_second'), iterations * batch * tokens / sum(latencies), 'llm.tokens_per_second')
        _number(result.get('aggregate_tokens_per_second'), 'llm.aggregate_tokens_per_second')
        if not isinstance(result.get('sample'), str) or not result['sample'].strip():
            raise ValueError('llm: generated text is absent')
    else:
        if result.get('resolution') != [1024, 1024] or result.get('dtype') != 'FP16':
            raise ValueError('image: resolution or dtype is inconsistent')
        _integer(result.get('steps'), 'image.steps')
        _equal_number(result.get('images_per_second'), len(latencies) / sum(latencies), 'image.images_per_second')
        for index in range(len(latencies)):
            image = Path(out) / f"sdxl-gpu{result['rank']}-{index}.png"
            if not image.is_file() or image.stat().st_size <= 24:
                raise ValueError('image: generated PNG is missing or empty')
            with image.open('rb') as stream:
                if stream.read(8) != b'\x89PNG\r\n\x1a\n':
                    raise ValueError('image: generated artifact is not a PNG')


def validate_gpu_result(result, plan, out):
    """Return review findings, or raise for invalid v2 passing evidence."""
    if plan.get('result_contract') != 2:
        return []
    if not isinstance(result, dict) or result.get('status') != 'PASS':
        raise ValueError('GPU evidence must be a passing result object')
    _uuid, total = _common(result, plan, out)
    mode = result.get('test')
    if mode == 'environment':
        _environment(result, total)
    elif mode == 'stress':
        return _stress(result, plan, total)
    elif mode == 'memory':
        return _memory(result, plan, total)
    elif mode == 'nccl':
        _nccl(result, plan)
    elif mode in ('llm', 'image'):
        _model(result, total, out)
    else:
        raise ValueError('Unknown GPU evidence test')
    return []
