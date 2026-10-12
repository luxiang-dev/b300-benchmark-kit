#!/usr/bin/env python3
"""Derive an observed inventory from hash-checked evidence, never a purchase BOM.

No hardware commands, downloads, privilege changes or procurement approval are
performed. Missing, malformed and unavailable observations stay unknown.
"""
import argparse
from datetime import datetime, timezone
import json
import hashlib
from pathlib import Path
import re
import sys
import xml.etree.ElementTree as ET

import authenticity
from evidence import digest
import server_probe

PURPOSE = 'OBSERVED_NOT_PURCHASE_BOM'
STATEMENT = ('This is a software-observed inventory, not a purchase BOM or OEM confirmation. '
             'No expected configuration is inferred and no procurement check is cleared. '
             'Compare with independent seller/OEM specifications and physical labels.')
MAX_RAW_BYTES = 16 * 2**20
UNKNOWN_TEXT = {'unknown', 'not specified', 'not available', 'not present', 'not reported',
                'not provided', 'none', 'n/a', 'not supported', 'default string',
                'to be filled by o.e.m.', 'to be filled by oem', 'invalid'}


def known_text(value):
    if not isinstance(value, str):
        return None
    value = value.strip()
    return value if value and value.lower() not in UNKNOWN_TEXT else None


def positive_integer(value):
    if type(value) is int and value > 0:
        return value
    if isinstance(value, str) and re.fullmatch(r'[0-9]+', value.strip()):
        return int(value) if int(value) > 0 else None
    return None


def parse_lscpu(raw):
    value = json.loads(raw)
    if not isinstance(value, dict) or not isinstance(value.get('lscpu'), list):
        raise ValueError('Unrecognized lscpu JSON')
    fields = {}
    wanted = {'architecture': 'architecture', 'model name': 'model_name',
              'socket(s)': 'sockets', 'cpu(s)': 'logical_cpus'}
    for row in value['lscpu']:
        if not isinstance(row, dict) or not isinstance(row.get('field'), str):
            raise ValueError('Invalid lscpu field')
        key = wanted.get(row['field'].rstrip(':').strip().lower())
        if key:
            item = positive_integer(row.get('data')) if key in ('sockets', 'logical_cpus') else known_text(row.get('data'))
            if key in fields and fields[key] != item:
                raise ValueError('Conflicting lscpu fields')
            fields[key] = item
    return {key: fields.get(key) for key in wanted.values()}


def parse_free(raw):
    lines = raw.splitlines()
    header = next((line.split() for line in lines if line.split() and line.split()[0] == 'total'), None)
    memory = next((line.split() for line in lines if line.split() and line.split()[0] == 'Mem:'), None)
    if not header or not memory or len(memory) < 2 or positive_integer(memory[1]) is None:
        raise ValueError('Unrecognized free -b memory total')
    return {'os_reported_total_bytes': positive_integer(memory[1]),
            'installed_dimm_bytes': None,
            'scope': 'OS-visible Mem total from free -b; not installed DIMM capacity or tested RAM coverage'}


def parse_smbios(raw):
    blocks = re.split(r'(?m)(?=^Handle\s+0x[0-9a-fA-F]+,\s+DMI type\s+\d+,)', raw)
    system, dimms = [], []
    recognized = False
    for block in blocks:
        match = re.match(r'^Handle\s+(0x[0-9a-fA-F]+),\s+DMI type\s+(\d+),', block)
        if not match or int(match[2]) not in (1, 17):
            continue
        recognized = True
        fields = {}
        for line in block.splitlines()[1:]:
            field = re.match(r'^\s+([^:]+):\s*(.*)$', line)
            if field:
                fields[field[1].strip()] = field[2].strip()
        if int(match[2]) == 1:
            system.append({'manufacturer': known_text(fields.get('Manufacturer')),
                           'product_name': known_text(fields.get('Product Name')),
                           'serial_number': known_text(fields.get('Serial Number')),
                           'version': known_text(fields.get('Version')), 'handle': match[1]})
        else:
            size = fields.get('Size')
            empty = size == 'No Module Installed'
            size_match = re.fullmatch(r'([0-9]+)\s+(kB|KB|MB|GB|TB)', size or '')
            populated = False if empty else True if size_match and int(size_match[1]) > 0 else None
            speed_raw = fields.get('Configured Memory Speed') or fields.get('Configured Clock Speed')
            speed = re.fullmatch(r'([0-9]+)\s+(MT/s|MHz)', speed_raw or '')
            dimms.append({'handle': match[1], 'locator': known_text(fields.get('Locator')),
                          'bank_locator': known_text(fields.get('Bank Locator')), 'populated': populated,
                          'size_raw': size, 'size_value': int(size_match[1]) if size_match else None,
                          'size_unit': size_match[2] if size_match else None,
                          'manufacturer': known_text(fields.get('Manufacturer')),
                          'part_number': known_text(fields.get('Part Number')),
                          'serial_number': known_text(fields.get('Serial Number')),
                          'configured_speed_raw': speed_raw,
                          'configured_speed': {'value': int(speed[1]), 'unit': speed[2]} if speed and int(speed[1]) > 0 else None})
    if not recognized:
        raise ValueError('No recognized SMBIOS type 1 or 17 records')
    return {'systems': system, 'dimms': dimms,
            'scope': 'SMBIOS reported records and literal capacity/speed units; not independently verified physical installation'}


def parse_lsblk(raw):
    value = json.loads(raw)
    if not isinstance(value, dict) or not isinstance(value.get('blockdevices'), list):
        raise ValueError('Unrecognized lsblk JSON')
    disks, names = [], set()
    def walk(entries, depth=0):
        if depth > 16 or not isinstance(entries, list):
            raise ValueError('Invalid block-device hierarchy')
        for item in entries:
            if not isinstance(item, dict):
                raise ValueError('Invalid block-device entry')
            if item.get('type') == 'disk':
                name = known_text(item.get('path')) or known_text(item.get('name'))
                if name and name in names:
                    raise ValueError('Duplicate disk entry')
                if name:
                    names.add(name)
                disks.append({'name': known_text(item.get('name')), 'path': known_text(item.get('path')),
                              'model': known_text(item.get('model')), 'serial_number': known_text(item.get('serial')),
                              'size_bytes': positive_integer(item.get('size')), 'transport': known_text(item.get('tran')),
                              'rotational_reported': item.get('rota') if type(item.get('rota')) in (bool, int) and item.get('rota') in (0, 1) else None,
                              'scope': 'OS disk entry; RAID/virtual devices may hide or differ from physical drives'})
            if 'children' in item and item['children'] is not None:
                walk(item['children'], depth + 1)
    walk(value['blockdevices'])
    return {'disks': disks, 'observed_disk_entries': len(disks), 'physical_ssd_count': None,
            'scope': 'OS block devices with type=disk, not a physical SSD count or full-media verification'}


def parse_interfaces(raw):
    value = json.loads(raw)
    if not isinstance(value, list) or any(not isinstance(item, dict) for item in value):
        raise ValueError('Unrecognized ip -j link JSON')
    interfaces = []
    for item in value:
        linkinfo = item.get('linkinfo') if isinstance(item.get('linkinfo'), dict) else {}
        kind = known_text(linkinfo.get('info_kind'))
        link_type = known_text(item.get('link_type'))
        interfaces.append({'ifindex': positive_integer(item.get('ifindex')), 'name': known_text(item.get('ifname')),
                           'mac_address_reported': known_text(item.get('address')), 'link_type': link_type,
                           'operstate': known_text(item.get('operstate')), 'mtu': positive_integer(item.get('mtu')),
                           'virtual_kind': kind,
                           'is_known_virtual': True if link_type == 'loopback' or kind in {'bridge', 'bond', 'veth', 'vlan', 'vxlan', 'tun', 'tap', 'dummy', 'macvlan', 'ipvlan', 'vrf', 'wireguard'} else None})
    return {'interfaces': interfaces, 'observed_interface_entries': len(interfaces), 'physical_nic_card_count': None,
            'scope': 'OS interfaces include virtual/loopback/bond/VLAN devices; neither NIC card count nor port line rate is established'}


def derive(out):
    """Write observed-inventory.json. No PASS/acceptance verdict is produced."""
    out = Path(out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    missing, sources = [], {}
    result = {'schema_version': 1, 'test': 'observed-inventory', 'purpose': PURPOSE,
              'comparison_status': 'NOT_COMPARED', 'procurement_checks_cleared': [],
              'time_utc': datetime.now(timezone.utc).isoformat(), 'statement': STATEMENT,
              'system': {'status': 'UNKNOWN', 'records': []}, 'cpu': {'status': 'UNKNOWN', 'architecture': None, 'model_name': None, 'sockets': None, 'logical_cpus': None},
              'ram': {'status': 'UNKNOWN', 'os_reported_total_bytes': None, 'installed_dimm_bytes': None,
                      'scope': 'OS-visible memory only; installed DIMM capacity and tested coverage are not inferred'},
              'dimms': {'status': 'UNKNOWN', 'devices': [], 'scope': 'SMBIOS observations; not independent physical DIMM verification'},
              'storage': {'status': 'UNKNOWN', 'disks': [], 'physical_ssd_count': None,
                          'scope': 'OS disk entries, not physical SSD count or tested media capacity'},
              'interfaces': {'status': 'UNKNOWN', 'interfaces': [], 'physical_nic_card_count': None,
                             'scope': 'OS interfaces may be virtual; not physical NIC count or line-rate measurement'},
              'gpus': {'status': 'UNKNOWN', 'devices': [], 'observed_count': None,
                       'expected_gpus_from_plan': None, 'identity_verdict': None,
                       'scope': 'Hashed nvidia-smi XML and re-evaluated identity evidence; not OEM authentication'}}
    def gap(component, detail):
        missing.append({'component': component, 'status': 'UNKNOWN', 'detail': str(detail)})
    try:
        server = server_probe.read_result(out)
    except (OSError, ValueError, TypeError, KeyError, AttributeError) as error:
        server = None
        gap('server', 'Server raw evidence cannot be hash-validated: ' + str(error))
    def stream(label):
        if server is None:
            raise ValueError('Validated server inventory is unavailable')
        command = server['commands'][label]
        if command.get('returncode') != 0 or command.get('error') or command.get('status') != 'COLLECTED':
            raise ValueError('Inventory command unavailable: ' + str(command.get('error') or command.get('returncode')))
        def checked_bytes(record):
            path = out / record['path']
            if path.is_symlink() or not path.resolve().is_relative_to(out) or path.stat().st_size > MAX_RAW_BYTES:
                raise ValueError('Inventory stream escaped directory or exceeds parser size limit')
            raw = path.read_bytes()
            if hashlib.sha256(raw).hexdigest() != record['sha256']:
                raise ValueError('Inventory stream changed after validation')
            return raw
        if checked_bytes(command['files']['stderr']).strip():
            raise ValueError('Inventory stderr is not empty; inspect permission/tool errors')
        record = command['files']['stdout']
        raw = checked_bytes(record).decode('utf-8')
        if not raw.strip():
            raise ValueError('Inventory stdout is empty')
        sources[record['path']] = record['sha256']
        return raw
    for key, label, parser, required in (
        ('cpu', 'cpu', parse_lscpu, ('architecture', 'model_name', 'sockets', 'logical_cpus')),
        ('ram', 'ram', parse_free, ('os_reported_total_bytes',)),
        ('storage', 'storage', parse_lsblk, ('disks',)),
        ('interfaces', 'nic-link', parse_interfaces, ('interfaces',)),
    ):
        try:
            parsed = parser(stream(label))
            unknown = [name for name in required if not parsed.get(name)]
            observed = any(parsed.get(name) for name in required)
            result[key] = {'status': 'UNKNOWN' if not observed else 'PARTIAL' if unknown else 'OBSERVED', **parsed}
            if unknown:
                gap(key, 'Missing recognized fields: ' + ', '.join(unknown))
            entries = parsed.get('disks') or parsed.get('interfaces') or []
            for index, item in enumerate(entries):
                fields = ('name', 'model', 'serial_number', 'size_bytes') if key == 'storage' else ('name', 'ifindex')
                absent = [name for name in fields if item.get(name) is None]
                if absent:
                    result[key]['status'] = 'PARTIAL'
                    gap(key, f'Entry {index} missing: ' + ', '.join(absent))
        except (OSError, ValueError, TypeError, KeyError, AttributeError) as error:
            gap(key, error)
    try:
        smbios = parse_smbios(stream('smbios'))
        result['system'] = {'status': 'OBSERVED' if smbios['systems'] else 'UNKNOWN', 'records': smbios['systems']}
        result['dimms'] = {'status': 'OBSERVED' if smbios['dimms'] else 'UNKNOWN', 'devices': smbios['dimms'], 'scope': smbios['scope']}
        for key, entries, fields in [('system', smbios['systems'], ('manufacturer', 'product_name', 'serial_number')),
                                     ('dimms', smbios['dimms'], ('populated',))]:
            if not entries:
                gap(key, 'No recognized SMBIOS records for this component')
            for index, item in enumerate(entries):
                wanted = fields + (('manufacturer', 'part_number', 'serial_number', 'configured_speed') if key == 'dimms' and item.get('populated') is True else ())
                absent = [name for name in wanted if item.get(name) is None]
                if absent:
                    result[key]['status'] = 'PARTIAL'
                    gap(key, f'Record {index} missing: ' + ', '.join(absent))
    except (OSError, ValueError, TypeError, KeyError, AttributeError) as error:
        gap('smbios', error)
    try:
        host_path = out / 'identity-host.json'
        if host_path.is_symlink():
            raise ValueError('Identity host metadata is a symbolic link')
        host = json.loads(host_path.read_text(encoding='utf-8'))
        authenticity.validate_host_shape(host)
        command = host['commands']['nvidia-smi-xml']
        if command.get('argv') != ['nvidia-smi', '-q', '-x'] or command.get('returncode') != 0 or command.get('error'):
            raise ValueError('nvidia-smi XML command did not succeed')
        if authenticity.evidence_bytes(out, command['files']['stderr']).strip():
            raise ValueError('nvidia-smi XML stderr is not empty')
        parsed = authenticity.parse_smi_xml(authenticity.evidence_bytes(out, command['files']['stdout']))
        expected = None
        try:
            plan_path = out / 'plan.json'
            if plan_path.is_symlink():
                raise ValueError('Plan is a symbolic link')
            candidate = json.loads(plan_path.read_text(encoding='utf-8')).get('gpus')
            if type(candidate) is int and 1 <= candidate <= 8:
                expected = candidate
            else:
                raise ValueError('Plan GPU count is unavailable or invalid')
        except (OSError, ValueError, TypeError, AttributeError) as error:
            gap('gpus', 'Expected GPU count is not established by plan: ' + str(error))
        checked = authenticity.evaluate_identity(out, expected if expected is not None else 8)
        devices = []
        for item in parsed['gpus']:
            devices.append({'index': item['index'], 'model_name': known_text(item['name']),
                            'uuid': known_text(item['uuid']), 'serial_number': known_text(item['serial']),
                            'pci_bus_id': authenticity.normalize_bus(item['pci_bus_id']),
                            'memory_total_raw': item['memory_total_raw'], 'driver_reported_memory_bytes': item['memory_total_bytes'],
                            'board_part_number': known_text(item['board_part_number']),
                            'gpu_part_number': known_text(item['gpu_part_number']), 'vbios_version': known_text(item['vbios_version'])})
        result['gpus'].update(status='OBSERVED' if devices else 'UNKNOWN', devices=devices,
                              observed_count=len(devices), expected_gpus_from_plan=expected, identity_verdict=checked['status'])
        sources['identity-host.json'] = digest(host_path)
        sources[command['files']['stdout']['path']] = command['files']['stdout']['sha256']
        if checked['status'] != 'CONSISTENT':
            result['gpus']['status'] = 'PARTIAL'
            gap('gpus', 'Re-evaluated identity requires review: ' + checked['status'])
        if not devices:
            gap('gpus', 'No GPU records in validated nvidia-smi XML')
        for index, item in enumerate(devices):
            absent = [name for name in ('model_name', 'uuid', 'serial_number', 'pci_bus_id', 'driver_reported_memory_bytes') if item[name] is None]
            if absent:
                result['gpus']['status'] = 'PARTIAL'
                gap('gpus', f'GPU {index} missing: ' + ', '.join(absent))
    except (OSError, ValueError, TypeError, KeyError, AttributeError, ET.ParseError) as error:
        gap('gpus', 'GPU raw identity evidence cannot be validated: ' + str(error))
    result.update(status='PARTIAL' if missing else 'OBSERVED', missing_evidence=missing, source_evidence_sha256=sources)
    authenticity.write_json(out / 'observed-inventory.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['derive'])
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(derive(args.out), ensure_ascii=False, allow_nan=False), flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
