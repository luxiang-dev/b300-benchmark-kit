#!/usr/bin/env python3
"""Read-only physical block-device SMART and current BMC health evidence.

No privilege escalation, SMART self-test, raw-device write, or log clearing.
Visible TYPE=disk nodes may still hide RAID members or virtual backing media;
these checks cannot replace an OEM physical inventory or acceptance procedure.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import subprocess
import sys
from evidence import digest

MAX_DISKS = 32
TIMEOUT = 8
DEVICE = re.compile(r'^/dev/(?:sd[a-z]+|hd[a-z]+|nvme\d+n\d+|vd[a-z]+|xvd[a-z]+|mmcblk\d+|pmem\d+|cciss/c\d+d\d+)$')
COMMANDS = {
    'lsblk': ['lsblk', '-J', '-b', '-o', 'NAME,PATH,TYPE,SIZE,MODEL,SERIAL'],
    'bmc-sensors': ['ipmitool', 'sdr', 'elist'],
    'bmc-events': ['ipmitool', 'sel', 'elist'],
    'sensors': ['sensors', '-j'],
}
SOURCES = [
    'https://github.com/smartmontools/smartmontools/blob/main/src/smartctl.8.in',
    'https://github.com/smartmontools/smartmontools/blob/main/src/nvmeprint.cpp',
    'https://github.com/ipmitool/ipmitool/blob/master/lib/ipmi_sdr.c',
    'https://github.com/ipmitool/ipmitool/blob/master/include/ipmitool/ipmi_sel.h',
]
LIMITATION = ('Read-only SMART/BMC snapshots are health evidence, not provenance or procurement approval. '
              'RAID members, inaccessible disks and absent tools require OEM/controller evidence. '
              'Historical wear/events require OEM review; no self-tests or log clearing were performed.')


def save(path, data):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(path)


def capture(out, phase, label, argv):
    stdout, stderr, rc, error = b'', b'', None, None
    try:
        result = subprocess.run(argv, capture_output=True, timeout=TIMEOUT, check=False)
        stdout, stderr, rc = result.stdout, result.stderr, result.returncode
    except subprocess.TimeoutExpired as failure:
        stdout, stderr = failure.stdout or b'', failure.stderr or b''
        error = 'Command exceeded the 8 second read-only timeout'
    except OSError as failure:
        error = str(failure)
    files = {}
    for kind, content in [('stdout', stdout), ('stderr', stderr)]:
        if isinstance(content, str):
            content = content.encode('utf-8')
        path = out / 'server-health-evidence' / f'{phase}-{label}.{kind}.txt'
        path.write_bytes(content)
        files[kind] = dict(path=path.relative_to(out).as_posix(), size=len(content), sha256=digest(path))
    return dict(argv=argv, returncode=rc, error=error, files=files)


def disk_inventory(raw):
    root = json.loads(raw)
    if not isinstance(root, dict) or not isinstance(root.get('blockdevices'), list):
        raise ValueError('lsblk JSON has no blockdevices array')
    disks, review = {}, []
    def walk(nodes):
        if not isinstance(nodes, list):
            raise ValueError('Invalid lsblk children')
        for node in nodes:
            if not isinstance(node, dict):
                raise ValueError('Invalid lsblk device record')
            if node.get('type') == 'disk':
                path, name, size = node.get('path'), node.get('name'), node.get('size')
                if isinstance(size, str) and re.fullmatch(r'[0-9]+', size):
                    size = int(size)
                if not isinstance(path, str) or not DEVICE.fullmatch(path) or not isinstance(name, str) or path.split('/')[-1] != name:
                    review.append('Unsafe or unsupported disk node was not queried: ' + str(path))
                elif type(size) is not int or size <= 0:
                    review.append(path + ': missing/invalid capacity; SMART remains available for review')
                    disks[path] = dict(path=path, name=name, size_bytes=None, model=node.get('model'), serial=node.get('serial'))
                elif path in disks:
                    review.append(path + ': repeated block-device mapping requires controller review')
                else:
                    disks[path] = dict(path=path, name=name, size_bytes=size, model=node.get('model'), serial=node.get('serial'))
            if 'children' in node:
                walk(node['children'])
    walk(root['blockdevices'])
    values = [disks[path] for path in sorted(disks)]
    if not values:
        review.append('No safe TYPE=disk nodes were enumerated; physical media coverage is unverified')
    if len(values) > MAX_DISKS:
        review.append(f'Only the first {MAX_DISKS} of {len(values)} visible disks were queried')
    for disk in values:
        if not disk['model'] or not disk['serial']:
            review.append(disk['path'] + ': lsblk model/serial is incomplete')
        if re.match(r'/dev/(?:x?vd|cciss/)', disk['path']):
            review.append(disk['path'] + ': virtual/RAID node; underlying physical media needs OEM/controller evidence')
    return values, review


def verified_stream(out, phase, label, record, kind):
    metadata = record['files'][kind]
    expected = f'server-health-evidence/{phase}-{label}.{kind}.txt'
    if metadata.get('path') != expected:
        raise ValueError('Unexpected health raw evidence path')
    path = out / expected
    if path.is_symlink() or path.parent.is_symlink() or not path.resolve().is_relative_to(out):
        raise ValueError('Health raw evidence path escaped or is linked')
    content = path.read_bytes()
    # Parse the exact same bytes whose hash is checked; do not hash once and
    # reopen a potentially changed file for semantic evaluation.
    if type(metadata.get('size')) is not int or len(content) != metadata['size'] or hashlib.sha256(content).hexdigest() != metadata.get('sha256'):
        raise ValueError('Health raw evidence hash/size changed')
    return content.decode('utf-8', errors='replace')


def read_record(out, phase, label, record, argv):
    if not isinstance(record, dict) or record.get('argv') != argv or set(record.get('files', {})) != {'stdout', 'stderr'}:
        raise ValueError('Health command or raw streams are incomplete')
    stdout = verified_stream(out, phase, label, record, 'stdout')
    stderr = verified_stream(out, phase, label, record, 'stderr')
    rc = record.get('returncode')
    if rc is not None and type(rc) is not int:
        raise ValueError('Invalid health command return code')
    return stdout, stderr, rc, record.get('error')


def integer(value):
    return type(value) is int and value >= 0


def smart_health(raw, disk, rc):
    data = json.loads(raw)
    if not isinstance(data, dict):
        raise ValueError('SMART JSON is not an object')
    failures, review, counters = [], [], {}
    path = disk['path']
    if rc is None or not 0 <= rc <= 255:
        review.append('SMART command did not return a usable bitmask')
    else:
        if rc & 7:
            review.append('SMART command/open/read failure bits are set; coverage is incomplete')
        if rc & 24:
            failures.append('SMART reports a currently failing health/attribute threshold')
        if rc & 224:
            review.append('SMART reports historical attribute/error/self-test records; obtain OEM review')
    reported_rc = data.get('smartctl', {}).get('exit_status')
    if reported_rc is not None and (type(reported_rc) is not int or reported_rc != rc):
        review.append('SMART JSON and process exit status differ')
    status = data.get('smart_status', {}).get('passed')
    if status is False:
        failures.append('SMART overall health is failing')
    elif status is not True:
        review.append('SMART overall health is missing/unreadable')
    serial = data.get('serial_number')
    if not isinstance(serial, str) or not serial.strip():
        review.append('SMART serial number is unavailable')
    elif disk.get('serial') and serial.strip() != str(disk['serial']).strip():
        failures.append('SMART and lsblk serial numbers disagree')
    device_name = data.get('device', {}).get('name')
    if device_name is not None and device_name != path:
        review.append('SMART returned a different device name; verify the controller mapping')
    log = data.get('nvme_smart_health_information_log')
    protocol = str(data.get('device', {}).get('protocol', '')).upper()
    if log is not None or protocol == 'NVME' or path.startswith('/dev/nvme'):
        if not isinstance(log, dict):
            review.append('NVMe SMART/health information log is missing')
        else:
            for key in ('critical_warning', 'media_errors'):
                value = log.get(key)
                if not integer(value) or (key == 'critical_warning' and value > 255):
                    review.append('Missing/invalid NVMe ' + key)
                else:
                    counters[key] = value
                    if value:
                        failures.append('NVMe ' + key + ' is nonzero: ' + str(value))
            for key in ('percentage_used', 'power_on_hours', 'unsafe_shutdowns', 'num_err_log_entries'):
                value = log.get(key)
                if not integer(value):
                    review.append('Missing/invalid NVMe lifetime field: ' + key)
                else:
                    counters[key] = value
                    if value:
                        review.append('NVMe historical/lifetime ' + key + '=' + str(value) + '; OEM review, not a counterfeit verdict')
            spare, threshold = log.get('available_spare'), log.get('available_spare_threshold')
            if integer(spare) and integer(threshold) and spare <= 100 and threshold <= 100:
                if spare < threshold:
                    failures.append('NVMe available spare is below its reported threshold')
            else:
                review.append('NVMe spare/threshold information is incomplete')
    elif protocol in ('ATA', 'SATA') or 'ata_smart_attributes' in data:
        attributes = data.get('ata_smart_attributes', {}).get('table')
        if not isinstance(attributes, list) or not attributes:
            review.append('ATA attribute table is missing; lifetime/media coverage needs OEM review')
        else:
            known = {5: 'reallocated_sectors', 9: 'power_on_hours', 187: 'reported_uncorrectable',
                     197: 'pending_sectors', 198: 'offline_uncorrectable', 199: 'interface_crc_errors'}
            for attribute in attributes:
                if not isinstance(attribute, dict):
                    review.append('Malformed ATA attribute record')
                    continue
                key = known.get(attribute.get('id'))
                if key:
                    value = attribute.get('raw', {}).get('value')
                    if not integer(value):
                        review.append('ATA raw counter unavailable: ' + key)
                    else:
                        counters[key] = value
                        if value and key in ('pending_sectors', 'offline_uncorrectable'):
                            failures.append('ATA ' + key + ' is nonzero: ' + str(value))
                        elif value:
                            review.append('ATA historical/vendor-specific ' + key + '=' + str(value) + '; OEM interpretation required')
                if str(attribute.get('when_failed', '')).lower() == 'now':
                    failures.append('ATA attribute currently below threshold: ' + str(attribute.get('name')))
    elif protocol in ('SCSI', 'SAS'):
        grown = data.get('scsi_grown_defect_list')
        if integer(grown):
            counters['grown_defect_list'] = grown
            if grown:
                review.append('SCSI grown defect count is nonzero; obtain OEM review')
        else:
            review.append('SCSI grown defect count is unavailable')
        error_log = data.get('scsi_error_counter_log', {})
        for operation in ('read', 'write', 'verify'):
            value = error_log.get(operation, {}).get('total_uncorrected_errors')
            if integer(value):
                counters[operation + '_uncorrected_errors'] = value
                if value:
                    failures.append('SCSI ' + operation + ' uncorrected errors are nonzero: ' + str(value))
            else:
                review.append('SCSI ' + operation + ' error counters are unavailable')
    else:
        review.append('Unrecognized disk protocol; physical media health needs OEM/controller diagnostics')
    return dict(path=path, model=data.get('model_name') or data.get('product'), serial=serial,
                protocol=protocol or None, counters=counters, failures=failures, review=review)


def bmc_sensors(raw):
    rows, failures, review = [], [], []
    alarm_codes = {'cr', 'nr', 'nc', 'lnr', 'unr', 'lcr', 'ucr', 'lnc', 'unc'}
    discrete_alarms = {'failure detected', 'predictive failure', 'ac lost', 'power supply ac lost', 'configuration error',
                       'drive fault', 'in critical array', 'in failed array', 'uncorrectable ecc',
                       'memory scrub failed', 'thermal trip'}
    known_normal = {'presence detected', 'device present', 'drive presence', 'connected', 'fully redundant',
                    'redundancy fully restored', 'system working', 'processor presence detected', '0x00'}
    for line in raw.splitlines():
        if not line.strip():
            continue
        parts = [part.strip() for part in line.split('|')]
        if len(parts) < 4 or not parts[0]:
            review.append('Unrecognized BMC SDR row: ' + line[:160])
            continue
        state = parts[2].lower()
        rows.append(dict(name=parts[0], status=state, reading=parts[-1]))
        if state in alarm_codes:
            failures.append(parts[0] + ': current BMC threshold alarm ' + state)
        elif state != 'ok':
            review.append(parts[0] + ': BMC state ' + state + ' is not a proven healthy reading')
        tokens = {token.strip().lower() for token in parts[-1].split(',') if token.strip()}
        if tokens & discrete_alarms:
            failures.append(parts[0] + ': current BMC discrete alarm ' + parts[-1])
        elif state == 'ok' and (not tokens or not tokens.issubset(known_normal)):
            numeric = re.fullmatch(r'([+-]?(?:\d+(?:\.\d*)?|\.\d+))\s+.+', parts[-1])
            if not numeric or not math.isfinite(float(numeric[1])):
                review.append(parts[0] + ': current discrete/empty BMC reading needs OEM interpretation: ' + parts[-1])
    if not rows:
        review.append('No valid current BMC SDR sensor readings')
    return rows, failures, review


def sensor_alarms(raw):
    root = json.loads(raw)
    if not isinstance(root, dict):
        raise ValueError('sensors JSON is not an object')
    failures, review = [], []
    def walk(node, location=''):
        if isinstance(node, dict):
            for key, value in node.items():
                if key.lower().endswith('_alarm'):
                    if type(value) not in (int, float, bool) or not math.isfinite(value) or value < 0:
                        review.append('Unreadable sensor alarm: ' + location + '/' + key)
                    elif value:
                        failures.append('Current sensor alarm is asserted: ' + location + '/' + key)
                else:
                    walk(value, location + '/' + key)
        elif isinstance(node, list):
            for value in node:
                walk(value, location)
    walk(root)
    return failures, review


def read_phase(out, phase):
    target = out / f'server-health-{phase}.json'
    if not target.exists():
        return None
    if target.is_symlink():
        raise ValueError('Linked health phase metadata')
    record = json.loads(target.read_text(encoding='utf-8'))
    if not isinstance(record, dict) or record.get('schema') != 1 or record.get('test') != 'server-health' or record.get('phase') != phase:
        raise ValueError('Invalid health phase schema')
    commands = record.get('commands')
    if not isinstance(commands, dict) or not set(COMMANDS).issubset(commands):
        raise ValueError('Missing health command records')
    failures, review, disks, health, sensors, events = [], [], [], [], [], []
    try:
        collected = datetime.fromisoformat(record['time_utc'])
        if collected.tzinfo is None or not math.isfinite(collected.timestamp()):
            raise ValueError('Unbound health snapshot timestamp')
    except (ValueError, KeyError, TypeError) as failure:
        raise ValueError('Invalid health snapshot timestamp') from failure
    streams = {}
    for label, argv in COMMANDS.items():
        stdout, stderr, rc, error = read_record(out, phase, label, commands[label], argv)
        streams[label] = stdout
        if label == 'sensors' and (error or rc != 0 or not stdout.strip()):
            # Optional board sensors; mandatory BMC coverage is checked separately.
            continue
        if error or rc != 0 or not stdout.strip():
            review.append(label + ': command unavailable/empty; inspect original stderr and OEM health records')
        if stderr.strip() and label != 'sensors':
            review.append(label + ': diagnostic stderr requires review')
    try:
        disks, notes = disk_inventory(streams['lsblk'])
        review += notes
    except (ValueError, TypeError) as failure:
        review.append('lsblk inventory is unreadable: ' + str(failure))
    expected_smart = {f'smart-{index:03d}': disk for index, disk in enumerate(disks[:MAX_DISKS])}
    if set(commands) - set(COMMANDS) - set(expected_smart):
        failures.append('Unexpected SMART command records not matching the raw disk inventory')
    for label, disk in expected_smart.items():
        if label not in commands:
            review.append(disk['path'] + ': SMART raw evidence missing')
            continue
        stdout, stderr, rc, error = read_record(out, phase, label, commands[label], ['smartctl', '-a', '-j', disk['path']])
        if error or rc is None or not stdout.strip():
            review.append(disk['path'] + ': SMART collection unavailable; permissions/RAID/OEM follow-up required')
        if stderr.strip():
            review.append(disk['path'] + ': SMART stderr requires review')
        try:
            parsed = smart_health(stdout, disk, rc)
            health.append(parsed)
            failures += [disk['path'] + ': ' + item for item in parsed['failures']]
            review += [disk['path'] + ': ' + item for item in parsed['review']]
        except (ValueError, TypeError, AttributeError) as failure:
            review.append(disk['path'] + ': SMART JSON unavailable/malformed: ' + str(failure))
    sensors, alerts, notes = bmc_sensors(streams['bmc-sensors'])
    failures += alerts
    review += notes
    sel = streams['bmc-events'].strip()
    if sel and sel.lower() not in ('sel has no entries', 'sel has no entries.'):
        events = [line.strip() for line in sel.splitlines() if line.strip()]
        review.append(f'BMC SEL has {len(events)} historical/unclassified entries; retain and obtain OEM interpretation')
    if streams['sensors'].strip():
        try:
            alerts, notes = sensor_alarms(streams['sensors'])
            failures += alerts
            review += notes
        except (ValueError, TypeError):
            review.append('Optional board sensors JSON cannot be interpreted')
    return dict(phase=phase, failures=failures, review=review, disks=disks, disk_health=health,
                bmc_sensors=sensors, sel_events=events, collected_at=record.get('time_utc'))


def evaluate(out):
    out = Path(out).resolve()
    failures, review, phases = [], [], {}
    for phase in ('before', 'after'):
        try:
            parsed = read_phase(out, phase)
            if parsed is None:
                review.append('Missing server health ' + phase + ' snapshot')
            else:
                phases[phase] = parsed
                failures += [phase + ': ' + item for item in parsed['failures']]
                review += [phase + ': ' + item for item in parsed['review']]
        except (OSError, ValueError, KeyError, TypeError, AttributeError) as failure:
            failures.append(phase + ': server health raw evidence cannot be verified: ' + str(failure))
    changes = []
    if set(phases) == {'before', 'after'}:
        before, after = phases['before'], phases['after']
        if datetime.fromisoformat(after['collected_at']) <= datetime.fromisoformat(before['collected_at']):
            review.append('Health snapshot timing is reversed/equal; test-window coverage requires review')
        original = {disk['path']: disk for disk in before['disks']}
        current = {disk['path']: disk for disk in after['disks']}
        if set(original) != set(current):
            failures.append('Visible disk inventory changed between health snapshots')
        for path in set(original) & set(current):
            for field in ('serial', 'size_bytes'):
                if original[path][field] != current[path][field]:
                    failures.append(path + ': ' + field + ' changed between health snapshots')
        old_health = {disk['path']: disk for disk in before['disk_health']}
        new_health = {disk['path']: disk for disk in after['disk_health']}
        media_counters = {'media_errors', 'reallocated_sectors', 'reported_uncorrectable', 'pending_sectors',
                          'offline_uncorrectable', 'grown_defect_list', 'read_uncorrected_errors',
                          'write_uncorrected_errors', 'verify_uncorrected_errors'}
        for path in set(old_health) & set(new_health):
            if old_health[path]['serial'] != new_health[path]['serial']:
                failures.append(path + ': SMART serial changed between snapshots')
            for key, value in new_health[path]['counters'].items():
                old = old_health[path]['counters'].get(key)
                if old is not None and value != old:
                    changes.append(dict(path=path, counter=key, before=old, after=value, delta=value-old))
                    if value > old and key in media_counters:
                        failures.append(path + ': media/error counter increased during testing: ' + key)
                    elif value < old:
                        review.append(path + ': SMART counter decreased/reset: ' + key)
                    else:
                        review.append(path + ': SMART lifetime/interface counter changed: ' + key + '; OEM review')
        new_events = set(after['sel_events']) - set(before['sel_events'])
        if new_events:
            review.append(f'BMC SEL added {len(new_events)} event rows during testing; OEM interpretation required')
    return dict(schema=1, test='server-health', status='FAIL' if failures else 'REVIEW' if review else 'PASS',
                failures=failures, review=review, inventory=phases, counter_changes=changes,
                limitation=LIMITATION, sources=SOURCES)


def collect(out, phase):
    if phase not in ('before', 'after'):
        raise ValueError('Health phase must be before or after')
    out = Path(out).resolve()
    raw_directory = out / 'server-health-evidence'
    if raw_directory.is_symlink():
        raise RuntimeError('Health raw evidence directory cannot be linked')
    raw_directory.mkdir(parents=True, exist_ok=True)
    target = out / f'server-health-{phase}.json'
    if target.exists() or target.is_symlink() or any(raw_directory.glob(phase + '-*')):
        raise RuntimeError('Health phase already exists; retain evidence and use a fresh result directory')
    commands = {label: capture(out, phase, label, argv) for label, argv in COMMANDS.items()}
    try:
        stdout = verified_stream(out, phase, 'lsblk', commands['lsblk'], 'stdout')
        disks, _ = disk_inventory(stdout)
    except (OSError, ValueError, TypeError):
        disks = []
    for index, disk in enumerate(disks[:MAX_DISKS]):
        label = f'smart-{index:03d}'
        commands[label] = capture(out, phase, label, ['smartctl', '-a', '-j', disk['path']])
    save(target, dict(schema=1, test='server-health', phase=phase,
                     time_utc=datetime.now(timezone.utc).isoformat(), commands=commands,
                     max_disks=MAX_DISKS, sources=SOURCES, limitation=LIMITATION))
    return evaluate(out)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['collect', 'evaluate'])
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--phase', choices=['before', 'after'])
    args = parser.parse_args()
    if args.mode == 'collect' and (sys.platform != 'linux' or not args.phase):
        parser.error('Collection requires Linux and --phase before|after')
    result = collect(args.out, args.phase) if args.mode == 'collect' else evaluate(args.out)
    print(json.dumps(result, ensure_ascii=False, allow_nan=False), flush=True)
    return 1 if result['failures'] else 0


if __name__ == '__main__':
    sys.exit(main())
