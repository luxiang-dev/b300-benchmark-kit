"""Conservative validation of DCGM 4.x entity-centric diagnostic JSON.

Field names follow NVIDIA/DCGM's Diag.cpp and NvvsJsonStrings.h. Unrecognized
formats or missing requested GPU coverage remain incomplete evidence.
"""
from pathlib import Path
import re

import authenticity


CORE_TESTS = frozenset(('pcie', 'memory', 'memory_bandwidth', 'diagnostic', 'targeted_stress', 'targeted_power'))
SOURCE = 'https://github.com/NVIDIA/DCGM/blob/master/dcgmi/Diag.cpp'


def _device_id(value):
    if type(value) is int:
        return f'{value:04x}'
    value = str(value or '').strip().lower().removeprefix('0x')
    return value if re.fullmatch(r'[0-9a-f]{4}', value) else None


def _gpu_group(value):
    return isinstance(value, dict) and value.get('entity_group') == 'GPU' and type(value.get('entity_group_id')) is int and value['entity_group_id'] == 1


def _id(value, count):
    if type(value) is not int or not 0 <= value < count:
        raise ValueError('DCGM: GPU entity ID is missing, duplicated or outside this run')
    return value


def _status(node, label, reviews):
    if not isinstance(node, dict) or not isinstance(node.get('status'), str):
        raise ValueError('DCGM: missing status for ' + label)
    status = node['status'].lower()
    if status in ('fail', 'failed', 'error'):
        raise ValueError('DCGM: failed result for ' + label)
    if status != 'pass':
        reviews.append('DCGM: ' + label + ' 状态为 ' + node['status'] + '，验货需补测或厂商复核。')
    if node.get('warnings'):
        reviews.append('DCGM: ' + label + ' 包含警告，检查原始诊断详情。')


def validate_dcgm_result(result, plan, out):
    """Return review findings or raise if entity/test coverage is not provable."""
    if plan.get('result_contract') != 2:
        return []
    reviews = []
    if not isinstance(result, dict) or not isinstance(result.get('DCGM Diagnostic'), dict):
        raise ValueError('DCGM: unsupported diagnostic schema; entity coverage cannot be confirmed')
    diagnostic = result['DCGM Diagnostic']
    if diagnostic.get('runtime_error') or result.get('runtime_error'):
        raise ValueError('DCGM: runtime error in diagnostic output')
    def retained_warnings(value):
        if isinstance(value, dict):
            for key, item in value.items():
                if key.lower() in ('warning', 'warnings') and item:
                    reviews.append('DCGM: 原始输出包含警告，检查完整诊断结果。')
                retained_warnings(item)
        elif isinstance(value, list):
            for item in value:
                retained_warnings(item)
    retained_warnings(result)
    count = plan.get('gpus')
    if type(count) is not int or count <= 0:
        raise ValueError('DCGM: invalid requested GPU count')
    groups = result.get('entity_groups')
    if not isinstance(groups, list):
        raise ValueError('DCGM: physical entity inventory is missing')
    gpu_groups = [group for group in groups if _gpu_group(group)]
    if len(gpu_groups) != 1 or not isinstance(gpu_groups[0].get('entities'), list):
        raise ValueError('DCGM: GPU entity group is missing or duplicated')
    entities = gpu_groups[0]['entities']
    ids = []
    physical = authenticity.parse_smi_xml((Path(out) / 'before.xml').read_bytes())['gpus']
    if len(physical) != count:
        raise ValueError('DCGM: physical inventory does not match the requested count')
    for entity in entities:
        if not isinstance(entity, dict):
            raise ValueError('DCGM: malformed entity inventory')
        index = _id(entity.get('entity_id'), count)
        ids.append(index)
        device = _device_id(entity.get('device_id'))
        host_device = authenticity.unpack_pci_id(physical[index]['pci_device_id']).get('device')
        if device and host_device and device != host_device:
            raise ValueError('DCGM: device ID differs from the physical inventory')
        if not device:
            reviews.append(f'DCGM: GPU {index} 缺少可识别的device ID。')
        serial = entity.get('serial_num')
        host_serial = physical[index]['serial']
        if serial and host_serial and host_serial not in ('N/A', 'Not Supported') and str(serial) != host_serial:
            raise ValueError('DCGM: serial number differs from the physical inventory')
        if not serial:
            reviews.append(f'DCGM: GPU {index} 缺少serial，需对照设备清单。')
    if len(ids) != count or set(ids) != set(range(count)):
        raise ValueError('DCGM: entity inventory does not cover each physical GPU exactly once')
    categories = diagnostic.get('test_categories')
    if not isinstance(categories, list) or not categories:
        raise ValueError('DCGM: test categories are missing')
    tests = {}
    for category in categories:
        if not isinstance(category, dict) or not isinstance(category.get('tests'), list):
            raise ValueError('DCGM: malformed test category')
        for test in category['tests']:
            if not isinstance(test, dict) or not isinstance(test.get('name'), str):
                raise ValueError('DCGM: test name is missing')
            name = test['name'].strip().lower().replace(' ', '_')
            if name == 'gpu_memory':
                name = 'memory'
            if name in tests:
                raise ValueError('DCGM: repeated test result: ' + name)
            tests[name] = test
            summary = test.get('test_summary')
            if summary is not None:
                _status(summary, name + ' test summary', reviews)
            rows = test.get('results')
            if not isinstance(rows, list):
                raise ValueError('DCGM: per-entity test results are missing: ' + name)
            gpu_ids = []
            for row in rows:
                if not isinstance(row, dict):
                    raise ValueError('DCGM: malformed test result')
                _status(row, name + ' entity ' + str(row.get('entity_id', 'global')), reviews)
                if _gpu_group(row):
                    gpu_ids.append(_id(row.get('entity_id'), count))
            if name in CORE_TESTS and (len(gpu_ids) != count or set(gpu_ids) != set(range(count))):
                raise ValueError('DCGM: ' + name + ' does not cover each requested GPU exactly once')
    missing = CORE_TESTS - tests.keys()
    if missing:
        raise ValueError('DCGM: requested core tests are missing: ' + ', '.join(sorted(missing)))
    return list(dict.fromkeys(reviews))
