#!/usr/bin/env python3
"""Read-only server inventory and bounded CPU/RAM/result-disk spot checks.

No privilege escalation, raw-device writes, remote traffic or factory verdict.
Default checks cover 256 MiB RAM and a 128 MiB temporary result-disk file only.
"""
import argparse
from array import array
from datetime import datetime, timezone
import hashlib
import json
import math
import mmap
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from evidence import digest

MIB = 2**20
LIMITATION = ('CPU known-answer and small RAM/result-file checks are functional spot checks. '
              'They do not stress all CPU cores, cover all RAM or disk media, measure NIC line rate, '
              'validate BMC health or certify hardware provenance. Obtain OEM acceptance separately.')
COMMANDS = {
    'cpu': ['lscpu', '--json'], 'ram': ['free', '-b'], 'storage': ['lsblk', '-J', '-b', '-O'],
    'filesystems': ['df', '-B1'], 'pci': ['lspci', '-D', '-nn', '-k'],
    'nic-address': ['ip', '-j', 'address'], 'nic-link': ['ip', '-j', '-s', 'link'],
    'route': ['ip', '-j', 'route'], 'numa': ['numactl', '--hardware'],
    'rdma': ['rdma', 'link', 'show'], 'sensors': ['sensors', '-j'],
    'smbios': ['dmidecode', '-t', 'system', '-t', 'memory'],
    'bmc-fru': ['ipmitool', 'fru', 'print'], 'bmc-sensors': ['ipmitool', 'sdr', 'elist'],
    'bmc-events': ['ipmitool', 'sel', 'elist'],
}


def save(path, data):
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8')


def capture(out, label, command):
    error, rc, stdout, stderr = None, None, b'', b''
    try:
        result = subprocess.run(command, capture_output=True, timeout=20, check=False)
        rc, stdout, stderr = result.returncode, result.stdout, result.stderr
    except (OSError, subprocess.TimeoutExpired) as failure:
        error = str(failure)
    files = {}
    for kind, content in (('stdout', stdout), ('stderr', stderr)):
        path = out / 'server-evidence' / f'{label}.{kind}.txt'
        path.write_bytes(content)
        files[kind] = dict(path=path.relative_to(out).as_posix(), sha256=digest(path))
    return dict(argv=command, returncode=rc, error=error, files=files,
                status='COLLECTED' if rc == 0 and stdout.strip() else 'UNAVAILABLE')


def available_ram():
    """Respect both host availability and a container memory budget, if present."""
    fields = {}
    for line in Path('/proc/meminfo').read_text().splitlines():
        key, value = line.split(':', 1)
        fields[key] = int(value.strip().split()[0]) * 1024
    available = fields.get('MemAvailable', fields.get('MemFree', 0))
    for limit_path, used_path in (
        ('/sys/fs/cgroup/memory.max', '/sys/fs/cgroup/memory.current'),
        ('/sys/fs/cgroup/memory/memory.limit_in_bytes', '/sys/fs/cgroup/memory/memory.usage_in_bytes'),
    ):
        try:
            limit, used = Path(limit_path).read_text().strip(), int(Path(used_path).read_text())
            if limit != 'max':
                available = min(available, max(0, int(limit) - used))
        except (OSError, ValueError):
            pass
    return available


def cpu_check():
    start = time.perf_counter()
    count = 10000
    actual = hashlib.sha256(b'abc').hexdigest()
    expected = 'ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad'
    checks = dict(sha256_known_answer=actual == expected,
                  integer_sum_of_squares=sum(i*i for i in range(count+1)) == count*(count+1)*(2*count+1)//6,
                  floating_point_sum=math.fsum([0.25] * 4096) == 1024.0)
    if not all(checks.values()):
        raise RuntimeError('CPU known-answer mismatch')
    return dict(status='PASS', checks=checks, duration_s=time.perf_counter()-start,
                coverage='Known-answer functional check; not all-core stress')


def ram_check(size):
    if type(size) is not int or not MIB <= size <= 256*MIB or size % MIB:
        raise ValueError('RAM spot-check size must be whole MiB in 1..256')
    available = available_ram()
    if available < size + 512*MIB:
        return dict(status='SKIP', reason='Insufficient available RAM with 512 MiB headroom',
                    requested_bytes=size, available_bytes=available)
    start = time.perf_counter()
    patterns = ('address', 'inverted-address')
    # mmap is private anonymous memory. Every allocated byte is written/read.
    with mmap.mmap(-1, size) as allocation:
        view = memoryview(allocation).cast('I')
        try:
            chunk_words = MIB // 4
            for pattern in patterns:
                mask = 0x5A5A5A5A if pattern == 'address' else 0xA5A5A5A5
                for base in range(0, len(view), chunk_words):
                    words = array('I', ((base+i) ^ mask for i in range(chunk_words)))
                    view[base:base+chunk_words] = words
                for index, word in enumerate(view):
                    if word != (index ^ mask):
                        raise RuntimeError(f'RAM data mismatch at word {index}: {pattern}')
        finally:
            view.release()
    return dict(status='PASS', allocated_bytes=size, verified_bytes_per_pass=size,
                total_verified_bytes=size*len(patterns), patterns=list(patterns),
                duration_s=time.perf_counter()-start,
                coverage='Only this anonymous allocation; not whole host RAM')


def disk_check(out, size):
    if type(size) is not int or not MIB <= size <= 128*MIB or size % MIB:
        raise ValueError('Disk spot-check size must be whole MiB in 1..128')
    start = time.perf_counter()
    expected, actual = hashlib.sha256(), hashlib.sha256()
    scratch = None
    try:
        with tempfile.NamedTemporaryFile(prefix='.b300-disk-probe-', dir=out, delete=False) as stream:
            scratch = Path(stream.name)
            for index in range(size // MIB):
                block = hashlib.sha256(index.to_bytes(8, 'little')).digest() * (MIB // 32)
                stream.write(block)
                expected.update(block)
            stream.flush()
            import os
            os.fsync(stream.fileno())
        observed_size = scratch.stat().st_size
        with scratch.open('rb') as stream:
            for block in iter(lambda: stream.read(MIB), b''):
                actual.update(block)
        if observed_size != size or actual.digest() != expected.digest():
            raise RuntimeError('Result-disk file write/read mismatch')
        return dict(status='PASS', written_bytes=size, read_bytes=observed_size,
                    expected_sha256=expected.hexdigest(), actual_sha256=actual.hexdigest(),
                    fsync=True, duration_s=time.perf_counter()-start,
                    coverage='Temporary file on result filesystem; page cache may serve reads, not raw-media or endurance diagnosis')
    finally:
        if scratch is not None:
            scratch.unlink(missing_ok=True)


def collect(out, full=True, ram_bytes=256*MIB, disk_bytes=128*MIB):
    out = Path(out).resolve()
    (out / 'server-evidence').mkdir(parents=True, exist_ok=True)
    commands = {label: capture(out, label, argv) for label, argv in COMMANDS.items()}
    findings = [f'{label}: inventory unavailable or empty; inspect original stderr/OEM records'
                for label, record in commands.items() if record['status'] != 'COLLECTED']
    checks, files = {}, {}
    if full:
        for label, function in [('cpu', cpu_check), ('ram', lambda: ram_check(ram_bytes)),
                                ('result-disk', lambda: disk_check(out, disk_bytes))]:
            try:
                checks[label] = function()
            except Exception as failure:
                checks[label] = dict(status='FAIL', error=str(failure))
            path = out / 'server-evidence' / f'check-{label}.json'
            save(path, checks[label])
            files[label] = dict(path=path.relative_to(out).as_posix(), sha256=digest(path))
    else:
        findings.append('Inventory only: no CPU/RAM/disk spot checks were performed')
    status = ('FAIL' if any(c['status'] == 'FAIL' for c in checks.values()) else
              'REVIEW' if findings or any(c['status'] != 'PASS' for c in checks.values()) else 'PASS')
    summary = dict(schema=1, test='server', mode='full' if full else 'inventory', status=status,
                   time_utc=datetime.now(timezone.utc).isoformat(), commands=commands, checks=checks,
                   check_files=files, findings=findings, limitation=LIMITATION)
    save(out / 'server-summary.json', summary)
    return read_result(out)


def read_result(out):
    out = Path(out).resolve()
    result = json.loads((out / 'server-summary.json').read_text(encoding='utf-8'))
    if not isinstance(result, dict) or result.get('schema') != 1 or result.get('test') != 'server' or result.get('mode') not in ('full', 'inventory'):
        raise ValueError('Invalid server evidence schema')
    if set(result.get('commands', {})) != set(COMMANDS):
        raise ValueError('Server inventory incomplete')
    def verified_record(record):
        relative = record['path']
        path = out / relative
        if path.is_symlink() or not path.resolve().is_relative_to(out) or digest(path) != record['sha256']:
            raise ValueError('Server raw evidence changed or escaped its directory')
        return path
    findings = []
    for label, command in result['commands'].items():
        if command.get('argv') != COMMANDS[label]:
            raise ValueError('Unexpected inventory command')
        if set(command.get('files', {})) != {'stdout', 'stderr'}:
            raise ValueError('Missing raw server command streams')
        for record in command['files'].values():
            verified_record(record)
        stdout = verified_record(command['files']['stdout']).read_bytes()
        stderr = verified_record(command['files']['stderr']).read_bytes()
        if command['returncode'] != 0 or command.get('error') or command.get('status') != 'COLLECTED' or not stdout.strip() or stderr.strip():
            findings.append(label + ': unavailable inventory')
    checks = result.get('checks', {})
    if result['mode'] == 'full':
        if set(checks) != {'cpu', 'ram', 'result-disk'} or set(result['check_files']) != set(checks):
            raise ValueError('Server functional checks incomplete')
        for label, check in checks.items():
            raw = json.loads(verified_record(result['check_files'][label]).read_text(encoding='utf-8'))
            if raw != check or check.get('status') not in ('PASS', 'SKIP', 'FAIL'):
                raise ValueError('Server check differs from original evidence')
            if check['status'] != 'PASS':
                continue
            duration = check.get('duration_s')
            if type(duration) not in (int, float) or not math.isfinite(duration) or duration <= 0:
                raise ValueError('Invalid server check timing')
            if label == 'cpu':
                answers = check.get('checks', {})
                if set(answers) != {'sha256_known_answer', 'integer_sum_of_squares', 'floating_point_sum'} or any(value is not True for value in answers.values()):
                    raise ValueError('CPU correctness evidence is incomplete')
            elif label == 'ram':
                size = check.get('allocated_bytes')
                if type(size) is not int or not MIB <= size <= 256*MIB or size % MIB or check.get('patterns') != ['address', 'inverted-address'] or check.get('verified_bytes_per_pass') != size or check.get('total_verified_bytes') != size*2:
                    raise ValueError('RAM coverage evidence is inconsistent')
            else:
                size = check.get('written_bytes')
                if type(size) is not int or not MIB <= size <= 128*MIB or check.get('read_bytes') != size or check.get('fsync') is not True or check.get('expected_sha256') != check.get('actual_sha256') or len(check.get('actual_sha256', '')) != 64:
                    raise ValueError('Disk coverage evidence is inconsistent')
    else:
        findings.append('Inventory only; no functional spot checks')
    result['status'] = ('FAIL' if any(c['status'] == 'FAIL' for c in checks.values()) else
                        'REVIEW' if findings or any(c['status'] != 'PASS' for c in checks.values()) else 'PASS')
    result['findings'] = findings
    result['limitation'] = LIMITATION
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['inventory', 'full'])
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if sys.platform != 'linux':
        parser.error('Server probes require Linux')
    result = collect(args.out, full=args.mode == 'full')
    print(json.dumps(result, ensure_ascii=False), flush=True)
    return 1 if result['status'] == 'FAIL' else 0


if __name__ == '__main__':
    sys.exit(main())
